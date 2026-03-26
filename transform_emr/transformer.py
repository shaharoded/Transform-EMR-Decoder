"""
transformer.py
==============

GPT wrapper that plugs into the project-wide Time2Vec + context
embedding defined in embedding.py and the batch structure produced
by dataset.py.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from pathlib import Path
from tqdm.auto import tqdm

# ───────── local code ─────────────────────────────────────────────────── #
from transform_emr.embedder import EMREmbedding
from transform_emr.config.model_config import *
from transform_emr.utils import *
from transform_emr.loss import MaskedFocalBCE, MaskedSetCE
from transform_emr.schedulers import LambdaScheduleController


# ───────── components ─────────────────────────────────────────────────────────── #
class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention (no rotary/ALiBi; same math as GPT-2)."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg["embed_dim"] % cfg["n_head"] == 0
        self.n_head = cfg["n_head"]
        self.n_embd = cfg["embed_dim"]

        self.qkv   = nn.Linear(cfg["embed_dim"], 3 * cfg["embed_dim"], bias=cfg["bias"])
        self.proj  = nn.Linear(cfg["embed_dim"], cfg["embed_dim"],    bias=cfg["bias"])
        self.attn_dropout  = nn.Dropout(cfg["dropout"])
        self.resid_dropout = nn.Dropout(cfg["dropout"])
    
    def forward(self, x, key_pad_mask=None):
            B, T, C = x.shape
            qkv = self.qkv(x)                      # [B, T, 3C]
            q, k, v = qkv.split(C, dim=-1)         # each [B, T, C]

            # -- reshape to (B, h, T, d) so SDPA attends along time --
            hd = C // self.n_head
            q = q.view(B, T, self.n_head, hd).permute(0, 2, 1, 3)
            k = k.view(B, T, self.n_head, hd).permute(0, 2, 1, 3)
            v = v.view(B, T, self.n_head, hd).permute(0, 2, 1, 3)

            if key_pad_mask is not None:
                # --- Manual Masking Path (Required when padding is present) ---
                # SDPA throws RuntimeError if we pass both attn_mask and is_causal=True.
                # We must construct a combined mask [B, 1, T, T] containing both:
                # 1. Padding constraints (Cols)
                # 2. Causal constraints (Upper Triangle)

                # 1. Padding Mask: [B, 1, 1, T] -> Broadcast to [B, 1, T, T]
                # key_pad_mask is True(Keep), False(Pad). ~key_pad_mask is True(Mask Out).
                padding_mask = (~key_pad_mask).unsqueeze(1).unsqueeze(2)
                
                # 2. Causal Mask: [1, 1, T, T] 
                # triu(1) gives upper triangle (future) as True -> Mask Out.
                causal_mask = torch.ones((T, T), device=x.device, dtype=torch.bool).triu(1).view(1, 1, T, T)
                
                # 3. Combine: Mask out if it's Padding OR Future
                combined_mask = padding_mask | causal_mask
                
                # 4. Convert to float mask for SDPA (-inf for mask, 0.0 for keep)
                # Using zeros_like ensures we match dtype (e.g. float16/bfloat16)
                attn_mask = torch.zeros_like(combined_mask, dtype=q.dtype).masked_fill(combined_mask, float("-inf"))
                
                attn = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_dropout.p if self.training else 0.0,
                    is_causal=False  # We handled causality manually in the mask
                )
            else:
                # --- Optimized Path (No padding, e.g. Inference) ---
                # Here we can let SDPA handle the causal mask efficiently
                attn = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=self.attn_dropout.p if self.training else 0.0,
                    is_causal=True
                )

            y = attn.transpose(1, 2).contiguous().view(B, T, C)         # -> [B, T, C]
            y = self.proj(y)
            return self.resid_dropout(y)


class MLP(nn.Module):
    """
    SwiGLU MLP (SiLU Gating).
    Projects to a higher dimension, splits into Gate and Value, and multiplies them.
    Standard optimization for modern LLMs (LLaMA-style).
    """
    def __init__(self, cfg):
        super().__init__()
        # Standard GPT uses 4x expansion. 
        # For SwiGLU, we project to 2 * (4 * dim) so we can split it into two 4x vectors.
        hidden_dim = 4 * cfg["embed_dim"]
        
        self.w1 = nn.Linear(cfg["embed_dim"], 2 * hidden_dim, bias=cfg["bias"])
        self.w2 = nn.Linear(hidden_dim, cfg["embed_dim"], bias=cfg["bias"])
        self.drop = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        # 1. Project to double width
        projected = self.w1(x)
        
        # 2. Split into x (content) and g (gate)
        x_val, x_gate = projected.chunk(2, dim=-1)
        
        # 3. Gating: Val * SiLU(Gate)
        out = x_val * F.silu(x_gate)
        
        # 4. Project back
        return self.drop(self.w2(out))


class AdaLNBlock(nn.Module):
    """
    Transformer block with AdaLN-Zero conditioning.
    
    Instead of adding [CTX] to the sequence, we inject patient context into
    the normalization layers of *every* block.
    
    Mechanism:
        1. Project context_emb -> (scale, shift, gate)
        2. Norm(x) = (x - mu)/sigma * (1 + scale) + shift
        3. Output = x + gate * Block(Norm(x))
    
    Zero Initialization:
        We initialize the modulation projection to 0. This ensures that at the 
        start of training, the block acts as an identity function (ignoring context),
        which stabilizes deep training.
    """
    def __init__(self, cfg):
            super().__init__()
            self.att = CausalSelfAttention(cfg)
            self.mlp = MLP(cfg)

            # 1. Norms WITHOUT affine parameters (we predict them from context)
            self.ln1 = nn.LayerNorm(cfg["embed_dim"], elementwise_affine=False, eps=1e-6)
            self.ln2 = nn.LayerNorm(cfg["embed_dim"], elementwise_affine=False, eps=1e-6)

            # 2. Modulation Head
            # Predicts 6 params: (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cfg["embed_dim"], 6 * cfg["embed_dim"], bias=True)
            )

            # 3. Zero Init
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, cond_emb, key_pad_mask=None):
        """
        Calculate modulation parameters [B, 6*D] -> 6 x [B, D] (broadcast over T)
        x: [B, T, D]
        cond_emb: [B, D] (Patient Context)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(cond_emb).chunk(6, dim=1)
        )

        # -- Attention Sub-block --
        # modulate(ln(x))
        norm_x = self.ln1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        x = x + gate_msa.unsqueeze(1) * self.att(norm_x, key_pad_mask=key_pad_mask)

        # -- MLP Sub-block --
        norm_x = self.ln2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(norm_x)

        return x


# ───────── the GPT wrapper that consumes EMREmbedding ─────────────────────── #
class GPT(nn.Module):
    """
    GPT-style decoder that takes an *external* EMREmbedding instead of its own
    token/positional embeddings.

    The model learns the contextual connections between events in the EMR, and generates a
    predicted stream of events, from which the expected complications are derived.

    Parameters
    ----------
    cfg            : dict - hyper-parameters (block_size, n_layer, n_head, dropout, ...)
    embedder       : EMREmbedding - fully initialised shared embedding module
    use_checkpoint : bool - continue training from last checkpoint
    """

    def __init__(self, cfg: dict, embedder: EMREmbedding, use_checkpoint: bool=True):
        super().__init__()

        assert cfg["embed_dim"] == embedder.output_dim, (
            "Config embed_dim must equal EMREmbedding.output_dim"
        )

        self.cfg      = cfg
        self.embedder = embedder
        self.use_checkpoint = use_checkpoint

        # ─── Sanity checks ─────────────────────────────────────────────────────────────
        vocab_size = self.embedder.decoder.out_features

        assert hasattr(self.embedder.tokenizer, "id2token"), "[GPT] Embedder missing id2token map"
        assert len(self.embedder.tokenizer.id2token) == vocab_size, (
            f"[GPT] id2token size mismatch: got {len(self.embedder.tokenizer.id2token)}, expected {vocab_size}"
        )
        assert len(self.embedder.tokenizer.token2id) == self.embedder.position_embed.num_embeddings, \
            f"[GPT] Mismatch between tokenizer (len={len(self.embedder.tokenizer.token2id)}) and position_embed ({self.embedder.position_embed.num_embeddings})"

        # ─── Build layers ─────────────────────────────────────────────────────────────
        self.drop = nn.Dropout(cfg["dropout"])
        self.blocks = nn.ModuleList([AdaLNBlock(cfg) for _ in range(cfg["n_layer"])])
        self.ln_f = nn.LayerNorm(cfg["embed_dim"], eps=1e-5) # Final Norm (Standard LayerNorm with learnable affine)

        # Next token prediction head (What will be the next event?)
        self.lm_head = nn.Linear(cfg["embed_dim"], vocab_size, bias=False)
        self.lm_head.weight = self.embedder.position_embed.weight  # weight tying
        assert self.lm_head.weight.shape[0] == vocab_size, (
            f"[GPT] lm_head output dim ({self.lm_head.weight.shape[0]}) "
            f"does not match embedder.position_embed ({vocab_size})"
        )

        # Outcome Head (designed to propogate signal to the hidden state that there is an expected outcome soon)
        # This head maps the embedding dimension to the unique outcome classes (e.g., Death, Sepsis, Hypoglycemia, Release)
        all_config_outcomes = sorted(list(set(OUTCOMES + TERMINAL_OUTCOMES)))
        valid_outcomes, missing_outcomes = [], []
        for n in all_config_outcomes:
            if n in self.embedder.tokenizer.token2id:
                valid_outcomes.append(n)
            else:
                missing_outcomes.append(n)
        if missing_outcomes:
            print(f"[GPT] The following configured outcomes were NOT found in the tokenizer and will be ignored: {missing_outcomes}")
            
        # Hard Error if nothing is left (Training cannot proceed)
        if not valid_outcomes:
            raise ValueError(
                f"[GPT] No valid outcomes found! Configured outcomes: {all_config_outcomes}. "
                "None of these exist in the tokenizer vocabulary. Check your dataset configuration or tokenizer build."
            )
        self.outcome_names = valid_outcomes
        self.num_outcomes = len(self.outcome_names)
        
        # A simple MLP classifier
        self.outcome_head = nn.Sequential(
            nn.Linear(cfg["embed_dim"], cfg["embed_dim"]),
            nn.ReLU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(cfg["embed_dim"], self.num_outcomes) 
            # Note: No Sigmoid here if using BCEWithLogitsLoss later
        )

        # Δt prediction head (for regression of Δt from admission at each step -> When will each event occur?)
        # --- TIME HEAD (monotone-by-design, allows Δt=0) ---
        self.abs_t_head = nn.Sequential(
            nn.Linear(cfg["embed_dim"], cfg["time2vec_dim"]),
            nn.ReLU(),
            nn.Linear(cfg["time2vec_dim"], 1)  
            # Output: scalar abs_t, raw per-step delta logits
        )

        self.apply(self._init_weights)
        # slightly smaller init for res projections as in gpt‑2
        for n, p in self.named_parameters():
            if n.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg["n_layer"]))

        print(f"[GPT]: Total params: {self.get_num_params()/1e6:.2f} M")


    # -------------------------------------------------------- helpers ------- #
    def _init_weights(self, module):
        """
        Custom initialization to ensure stable training.
        Method based on GPT2 initialization.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        """
        Utility method to get the number of parameters in the chosen architecture.
        """
        return sum(p.numel() for p in self.parameters())

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        """
        Configures the optimizer:
        - Applies weight decay to transformer weights (dim ≥ 2)
        - No weight decay on biases / norms
        - Applies same LR everywhere, but scales embedder LR by 0.1
        """
        embedder_params = list(self.embedder.parameters())
        embedder_param_ids = set(id(p) for p in embedder_params)

        decay, no_decay = [], []

        for n, p in self.named_parameters():
            if not p.requires_grad or id(p) in embedder_param_ids:
                continue  # Embedder handled separately
            (decay if p.dim() >= 2 else no_decay).append(p)

        optim_groups = [
            {"params": decay, "weight_decay": weight_decay, "lr": learning_rate},
            {"params": no_decay, "weight_decay": 0.0, "lr": learning_rate},
            {"params": embedder_params, "weight_decay": 0.0, "lr": learning_rate * 0.1} # Lower LR for embedder tweaks
        ]

        return torch.optim.AdamW(optim_groups, betas=betas)
    
    def configure_scheduler(self, optimizer, training_settings, train_dl):
        """
        Configures the learning rate scheduler for phase 2 training.

        Uses the OneCycleLR scheduler to warm up and then anneal the learning rate
        over the course of training. Scheduler parameters (like total steps and
        learning rate) are derived from the provided training settings.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer instance to schedule.
            training_settings (dict): Dictionary containing training hyperparameters,
                including 'phase2_learning_rate' and 'phase2_n_epochs'.
            train_dl (DataLoader): Training dataloader used to compute steps per epoch.

        Returns:
            torch.optim.lr_scheduler.OneCycleLR: Configured learning rate scheduler.
        """
        # derive warmup fraction from warmup_epochs
        total_epochs = training_settings["phase2_n_epochs"]
        warmup_epochs = training_settings["warmup_epochs"]
        pct = max(1e-6, min(0.9, warmup_epochs / float(total_epochs)))  # e.g., 5/100 = 0.05

        # match max_lr list to your 3 param groups: decay, no_decay, embedder(0.1x)
        base_lr = training_settings["phase2_learning_rate"]
        max_lrs = [base_lr, base_lr, base_lr * 0.1]

        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lrs,                         # list per group
            epochs=total_epochs,                    # prefer epochs+steps_per_epoch to total_steps
            steps_per_epoch=len(train_dl),
            pct_start=pct,                          # warm-up exactly = warmup_epochs
            anneal_strategy="cos",
            cycle_momentum=False,                   # important for AdamW
            div_factor=10,                          # init LR = max_lr/10
            final_div_factor=10,                    # floor‑LR = 1e‑5 as well (cos tail reaches this)
        )
    

# ---------------------------------------------------- forward ---- #
    def forward(self, parent_raw_ids, concept_ids, value_ids, position_ids,
            abs_ts, context_vec=None):
        """
        All tensors come straight from `collate_emr`:
            parent_raw_ids (torch.Tensor)    - padded raw_concept ids, (B, T, D)
            concept_ids (torch.Tensor)       - padded concepts ids, (B, T)
            value_ids (torch.Tensor)         - padded concept_value ids, (B, T)
            position_ids (torch.Tensor)      - padded token ids, (B, T)
            abs_ts (torch.Tensor)            - relative start times from ADMISSION (hours), (B, T)
            context_vec (torch.Tensor)       - age/gender or [] if not used, (B, C)
        
        Returns:
            logits (FloatTensor): [B, T, V]
                - Next-token logits for the sequence.
                - logits[b, t] predicts input position_ids[b, t+1].
                - The last logit logits[b, -1] predicts the future after the sequence ends.
            abs_t_pred (FloatTensor): [B, T]
                - Predicted absolute time for the *next* event.
                - abs_t_pred[b, t] predicts the time for the event at t+1.
            outcome_logits (FloatTensor): [B, T, self.num_outcomes]
                - Logits over the predefined outcomes in the task.
                - Logits are propagated to the hidden layers, informing the model on expected outcomes.
        """
        # 1. Embed (x=[B, T, D], cond=[B, D])
        # The embedder now returns the sequence and the separate context embedding for AdaLN blocks
        x, cond_emb, pad_mask = self.embedder(
            parent_raw_ids, concept_ids, value_ids, position_ids,
            abs_ts, context_vec, return_mask=True
        )
        x = self.drop(x)

        # 2. Blocks with Conditioning (AdaLN)
        # We pass cond_emb to every block to modulate the normalization layers
        for blk in self.blocks:
            if self.training and self.use_checkpoint:
                x = checkpoint.checkpoint(
                    lambda x, c, m: blk(x, c, key_pad_mask=m), 
                    x, cond_emb, pad_mask, 
                    use_reentrant=False
                )
            else:
                x = blk(x, cond_emb, key_pad_mask=pad_mask)
        
        x = self.ln_f(x)                     # [B, T, D]
        
        # 3. Main next-token prediction head
        logits = self.lm_head(x)             # [B, T, V]

        # 4. Outcome Prediction Head (Auxiliary Task)
        # Input: Contextual embedding at step t (representing history 0..t)
        # Output: Unnormalized logits for each outcome class [B, T, Num_Outcomes]
        outcome_logits = self.outcome_head(x)

        # 5. Absolute time prediction (monotonic)
        # We predict the time delta to the *next* event based on the current state.
        # x[t] contains info up to t. We predict time for t+1.
        delta_raw = self.abs_t_head(x).squeeze(-1)     # [B,T]
        
        # Enforce non-negative deltas
        delta_pos = F.softplus(delta_raw)
        
        # Autoregressive time prediction:
        # Predicted_Time[t+1] = Actual_Time[t] + Predicted_Delta
        abs_t_pred = abs_ts + delta_pos                # [B, T]

        return logits, abs_t_pred, outcome_logits
    

    def save(self, path, epoch=None, best_val=None, optimizer=None, scheduler=None):
        ckpt = {
            "model_state": self.state_dict(),
            "config": self.cfg,
            "vocab_size": self.embedder.decoder.out_features,
            "outcome_names": self.outcome_names,
            "num_outcomes": self.num_outcomes,
        }
        if epoch is not None:
            ckpt["epoch"] = epoch
        if best_val is not None:
            ckpt["best_val"] = best_val
        if optimizer is not None:
            ckpt["optim_state"] = optimizer.state_dict()
        if scheduler is not None:
            ckpt["scheduler_state"] = scheduler.state_dict()
        torch.save(ckpt, path)

    
    @classmethod
    def load(cls, path, embedder, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location, weights_only=True)

        # === Vocab safety check ===
        expected_vocab = ckpt["vocab_size"]
        actual_vocab = embedder.decoder.out_features
        if expected_vocab != actual_vocab:
            raise ValueError(
                f"[GPT.load] Embedder vocab size mismatch: checkpoint={expected_vocab}, embedder={actual_vocab}"
            )
        
        # === Outcome configuration check ===
        if "outcome_names" not in ckpt or "num_outcomes" not in ckpt:
            raise ValueError(
                "[GPT.load] Invalid checkpoint: missing 'outcome_names' or 'num_outcomes'. "
                "Checkpoint was saved with an older version of the code."
            )
        
        expected_outcome_names = set(ckpt["outcome_names"])
        expected_num_outcomes = ckpt["num_outcomes"]
        
        # Validate that current config matches checkpoint
        current_outcomes = set(OUTCOMES + TERMINAL_OUTCOMES)
        if expected_outcome_names != current_outcomes:
            raise ValueError(
                f"[GPT.load] Outcome configuration mismatch!\n"
                f"  Checkpoint outcomes: {sorted(expected_outcome_names)}\n"
                f"  Current config outcomes: {sorted(current_outcomes)}\n"
                f"  Please use the same OUTCOMES and TERMINAL_OUTCOMES config that was used during training."
            )
        
        # Reconstruct model
        model = cls(cfg=ckpt["config"], embedder=embedder)
        
        # Final sanity check
        if model.num_outcomes != expected_num_outcomes:
            raise ValueError(
                f"[GPT.load] Architecture mismatch: checkpoint has {expected_num_outcomes} outcomes, "
                f"but reconstructed model has {model.num_outcomes}."
            )
        
        model.load_state_dict(ckpt["model_state"])

        # Return full training state if available
        return (
        model,
        ckpt.get("epoch", 0),
        ckpt.get("best_val", float("inf")),
        ckpt.get("optim_state", None),
        ckpt.get("scheduler_state", None)
        )


def train_transformer(model, train_dl, val_dl, resume=True, checkpoint_path=TRANSFORMER_CHECKPOINT, training_settings=TRAINING_SETTINGS):
    """
    Trains a Transformer-based EMR sequence model in Phase 2 (decoder stage),
    using a pretrained embedder and structured multi-loss optimization.
    Total Loss = λ1 * BCE + λ2 * CE + λ3 * Outcome prediction + λpen * Penalty + λt * Time Loss (τt)

    The auxiliary losses are applied gradually to stabilize training:
    -  Next-token BCE loss encourages accurate event prediction (foundational task, start at epoch 0 with no schedule).
    -  Next-token CE loss provides a complementary signal for token prediction (foundational task, start at epoch 0 with no schedule).
    -  Time prediction MSE loss guides the model to predict event timings (foundational task, start at epoch 0 with no schedule).
    -  Penalty loss encourages valid event sequences (mid-foundational task, start at the middle of the foundational phase with schedule).
    -  Outcome prediction BCE loss guides the model to predict clinical outcomes (post-foundational task, start after the foundational phase with schedule).

    Gradually applying curriculum and CBM during `warmup_epochs`, then implements early stopping based on best total loss. 
    Supports resume-from-checkpoint training. The CBM is applied during the foundational training phase only.

    Logits are masked to ensure only legal tokens are predicted, targets are masked to ensure proper loss denominator,
    and penalties are applied to encourage valid event sequences.

    Args:
        model (nn.Module): GPT decoder with attached EMREmbedding.
        train_dl (DataLoader): Training data loader.
        val_dl (DataLoader): Validation data loader.
        resume (bool): Resume from latest checkpoint if found.
        checkpoint_path (str): Path to save the best model and state.
        training_settings (dict): A settings dictionary, imported from model_config.

    Returns:
        None. Saves model checkpoints and plots training curves.

    NOTE: Despite intention, this function currently not implementing curriculum. in order to do so, one must add after 
    pred_ids are calculated (in run_epoch) the following additions:
    if train_flag -> mix_with_predictions() from utils.py -> rebuild raw/concept/value from predicted position (define LUT?) ->
    apply mix_mask to all modalities -> Second forward pass on mixed inputs.
    """
    # Create global training lookup Tensors once the tokenizer is available and move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    luts = build_luts(model.embedder.tokenizer)
    luts = {k: v.to(device) if torch.is_tensor(v) else v for k,v in luts.items()}

    # Allow embedder weights to update starting epoch 1
    set_embedder_frozen(model, freeze=False)
    model.to(device)
    
    optimizer = model.configure_optimizers(
        weight_decay=training_settings["weight_decay"],
        learning_rate=training_settings["phase2_learning_rate"],
        betas=(0.9, 0.95)
    )

    scheduler = model.configure_scheduler(optimizer, training_settings, train_dl)

    # Training Dynamics
    warmup_epochs = training_settings["warmup_epochs"]
    foundational_epochs = training_settings["foundational_epochs"]

    # Get outcomes weights (from Tokenizer) for pos_weight in BCE loss
    valid_outcomes = [n for n in model.outcome_names if n in model.embedder.tokenizer.token2id]
    outcome_token_ids = [model.embedder.tokenizer.token2id[n] for n in valid_outcomes]
    full_weights = model.embedder.tokenizer.outcome_weights.to(device)
    pos_weights  = full_weights[outcome_token_ids].to(device)
    
    # Build criterions once (for next token BCE and CE losses + aux outcome BCE)
    BCEcriterion = MaskedFocalBCE.from_counts(
        counts=model.embedder.tokenizer.token_counts,
        token_weights=model.embedder.tokenizer.token_weights,
        beta=0.999, min_count=5, clip_max=8.0,
        gamma=1.3,         # focal strength
        tau=0.85,           # pos/neg balance anchor
        neg_bounds=(0.03, 0.3),   # clamp for stability
        label_smoothing=0.0,     # optional
        hard_neg_k=0            # or e.g., 64 for hard-neg mining
    ).to(device)

    CEcriterion = MaskedSetCE(
        label_smoothing=0.0,     # optional
    ).to(device)

    OutcomeCriterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction='none')

    ckpt_path = Path(checkpoint_path).resolve()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_last = ckpt_path.parent / "ckpt_last.pt"

    start_epoch = 0
    best_val = float("inf")
    bad_epochs = 0

    if resume and ckpt_last.exists():
        print(f"[GPT]: Loading model from checkpoint: {ckpt_last}")
        loaded_model, start_epoch, best_val, opt_state, sch_state  = GPT.load(ckpt_last, embedder=model.embedder, map_location=device)
        model.load_state_dict(loaded_model.state_dict())
        optimizer.load_state_dict(opt_state)
        
        # Scheduler needs to be re-initiated with the recovered optimizer before loading it's state dict
        scheduler = model.configure_scheduler(optimizer, training_settings, train_dl)
        scheduler.load_state_dict(sch_state)
        start_epoch += 1
        print(f"[Phase-2]: Resumed training at epoch {start_epoch} (best val_loss so far: {best_val:.4f})")
    else:
        print("[Phase-2]: Starting transformer training loop...")

    schedule_controller = LambdaScheduleController(
        training_settings=training_settings,
        start_epoch=start_epoch
    )

    train_losses, val_losses = [], []
    pr_tracker = OutcomePRTracker(
        tokenizer=model.embedder.tokenizer,
        thresholds=(0.05, 0.10, 0.20, 0.30, 0.50),
        device=device
    ) # For epoch evaluation

    def run_epoch(loader, epoch, train_flag=False):
        if train_flag:
            model.train()
        else:
            model.eval()

        total_loss = total_bce = total_ce = total_penalty = total_outcome = total_dt = 0.0
        total_ce_raw = total_penalty_raw = total_outcome_raw = total_dt_raw = 0.0
        with torch.set_grad_enabled(train_flag):
            for batch in tqdm(loader, desc="Training" if train_flag else "Validation", leave=False, mininterval=1.0, miniters=10, dynamic_ncols=True):
                batch = {k: v.to(device) for k, v in batch.items()}

                # === Apply CBM on training batchs ===
                if train_flag:
                    # Starting at epoch 0, ramping through the foundational_epochs
                    p = linear_schedule(epoch=epoch, 
                                        start_epoch=0, 
                                        end_epoch=foundational_epochs, 
                                        max_val=0.25)
                    
                    batch = apply_cbm(batch=batch, 
                                      tokenizer=model.embedder.tokenizer, 
                                      forbid_ids=luts["forbid_mask_ids"], 
                                      p=p)

                # === Original logits from Model ===
                logits, abs_t_pred, outcome_logits = model(
                    parent_raw_ids=batch["parent_raw_ids"],
                    concept_ids=batch["concept_ids"],
                    value_ids=batch["value_ids"],
                    position_ids=batch["position_ids"],
                    abs_ts=batch["abs_ts"],
                    context_vec=batch["context_vec"]
                )

                # logits is [B, T, V]
                # abs_t_pred: [B, T]
                # outcome_logits: [B, T, K]

                # Slice for Autoregressive Training (Predict Next Token)
                # Input at t predicts Target at t+1.
                # We drop the LAST input (nothing to predict after it) and the FIRST target (nothing predicts it)

                # Slicing Logits (Inputs 0 to T-2)
                pred_logits = logits[:, :-1, :]       # [B, T-1, V]

                # Slicing Targets (Targets 1 to T-1)
                full_targets = batch["targets"]
                target_ids   = full_targets[:, 1:]    # [B, T-1] Used for loss calculation
                
                # === legality masks from Ground Truth + Targets ===
                # Compute on FULL targets to preserve state history (t=0)
                full_illegal, full_bonus = compute_legality_masks_tf(
                                                full_targets,
                                                luts["is_start"],
                                                luts["is_end"],
                                                luts["base_id"],
                                                luts["start_ids_per_base"],
                                                luts["end_ids_per_base"],
                                                luts["meal_rank"],
                                                luts["meal_pred_rank"],
                                                luts["K_meals"],
                                                luts["conflict_mat"],
                                                luts["predict_block"]
                                            )
                
                # Slice the masks to align with target_ids (drop t=0)
                illegal_mask = full_illegal[:, 1:, :] 
                bonus_mask   = full_bonus[:, 1:, :]

                # Soft illegal-mass penalty (pre-mask)
                # Penalize putting ANY mass on illegal tokens (using UNMASKED logits)
                nonpad = (target_ids != model.embedder.padding_idx)                # [B,T]
                logits_pre_mask = pred_logits

                # Apply masks BEFORE BCE so gradients learn legality only
                pred_logits = apply_masks_to_logits(
                    pred_logits.clone(), illegal_mask, bonus_mask
                )

                # Outcome Head Slicing
                # Input at t predicts future relative to t.
                # We align with the same truncated input sequence (0 to T-2).
                outcome_pred = outcome_logits[:, :-1, :] # [B, T-1, K]

                # Time Slicing
                # abs_t_pred[t] is the predicted time for token[t+1]
                # So abs_t_pred[0] should match abs_ts[1]
                pred_abs = abs_t_pred[:, :-1]     # [B, T-1]
                true_abs = batch["abs_ts"][:, 1:] # [B, T-1] (Shifted true times)

                # === Loss: BCE with logits (next token generation task) ===
                # Multi-hot targets
                multi_hot = get_multi_hot_targets(
                    position_ids=target_ids,
                    padding_idx=model.embedder.padding_idx,
                    vocab_size=pred_logits.size(-1),
                    k=training_settings["bce_k_window"]
                ).masked_fill(illegal_mask, 0.0) # zero‑out illegal targets, no in-place torch operation  
                
                # mask out illegal classes AND PAD steps from the denominator
                valid_pos = nonpad.unsqueeze(-1)            # [B,T,1] bool
                allowed   = (~illegal_mask) & valid_pos     # [B,T,V] bool           

                # Calculate BCE loss (only valid positions) 
                loss_bce, _ = BCEcriterion(pred_logits, multi_hot, allowed)
                loss_bce = loss_bce * training_settings["phase2_bce_weight"] # Applying weight
                
                # === Loss: CE nudge (next token generation task) ===
                # Foundational task to complement BCE, start at epoch 0, no schedule
                loss_ce_raw, _ = CEcriterion(pred_logits, multi_hot, allowed)
                lambdas = schedule_controller.get_lambdas(epoch)
                loss_ce = loss_ce_raw * lambdas["ce"]

                # === Loss: Δt (time) ===
                # Foundational task, start at epoch 0, no schedule
                # Predict abs_ts using model abs_t_head (already returned as abs_t_pred, shape [B,T-1])
                abs_t_loss_raw = F.mse_loss(pred_abs[nonpad], true_abs[nonpad], reduction="mean")
                abs_t_loss = abs_t_loss_raw * lambdas["dt"]

                # === Loss: Structural penalties on output ===
                # Mid-foundational task, starts halfway through foundational phase
                
                # 1. Soft illegal-mass penalty (pre-mask)
                # Penalize putting ANY mass on illegal tokens (using UNMASKED logits)
                p_illegal = soft_illegal_mass_penalty(
                    logits_pre_mask=logits_pre_mask,
                    illegal_mask=illegal_mask,
                    nonpad_mask=nonpad,
                    margin=0.04,
                    power=1.0
                )
                # 2. Global Closure Violations
                # Penalize leaving intervals open at the end (using MASKED logits)
                # This teaches "If you started it, you must finish it"
                # Load and normalize each penalty (∈ [0, 1]) -> Active grad on penalty functions
                p_unclosed = soft_unclosed_interval_penalty(
                                    pred_logits,             # Masked logits
                                    allowed,
                                    luts["start_ids_per_base"],
                                    luts["end_ids_per_base"]
                                )

                # Average the penalties to bound in [0, 1]
                # A little more agressive on teaching against illegal steps, heuristic.
                generative_penalty_raw = (2 * p_illegal + p_unclosed) / 3.0
                generative_penalty = lambdas["penalty"] * generative_penalty_raw

                # === Outcome Loss Calculation === 
                # Post-foundational task, starts after foundational phase               
                # Targets: "Does outcome K happen in future relative to t?"
                all_outcome_targets = get_future_outcome_targets(
                    full_targets,         # [B, T]
                    outcome_token_ids     # [K]
                )
                
                # Slice to align with inputs 0..T-2 (dropping the last step T-1)
                outcome_targets = all_outcome_targets[:, :-1, :] # [B, T-1, K]
                
                # Only learn from valid (non-pad) time steps.
                loss_outcome_raw = OutcomeCriterion(outcome_pred, outcome_targets) # [B, T-1, K]
                loss_outcome_raw = (loss_outcome_raw * valid_pos).sum() / valid_pos.sum().clamp(min=1.0)
                loss_outcome = lambdas["outcome"] * loss_outcome_raw

                # === Loss: Total Loss ===
                loss = loss_bce + loss_ce + generative_penalty + loss_outcome + abs_t_loss

                # === Backprop and Log ===
                if train_flag:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                else:
                    # Update validation classification metric
                    pr_tracker.update(pred_logits, allowed, target_ids, nonpad)

                # Update loss
                total_loss += loss.item()
                total_bce += loss_bce.item()
                total_ce += loss_ce.item()
                total_penalty += generative_penalty.item()
                total_outcome += loss_outcome.item()
                total_dt += abs_t_loss.item()
                total_ce_raw += loss_ce_raw.item()
                total_penalty_raw += generative_penalty_raw.item()
                total_outcome_raw += loss_outcome_raw.item()
                total_dt_raw += abs_t_loss_raw.item()
        
        if train_flag:
            val_eval = None
        else:
            val_eval = pr_tracker.compute()   # dict with per-class max-F1 and P/R at that point
            pr_tracker.reset()

        n_batches = len(loader)

        return (
            total_loss / n_batches,
            total_bce / n_batches,
            total_ce / n_batches,
            total_penalty / n_batches,
            total_outcome / n_batches,
            total_dt / n_batches,
            total_ce_raw / n_batches,
            total_penalty_raw / n_batches,
            total_outcome_raw / n_batches,
            total_dt_raw / n_batches,
            val_eval
        )
    
    for epoch in range(start_epoch, training_settings.get("phase2_n_epochs")):
        tr_loss, tr_bce, tr_ce, tr_pen, tr_outcome, tr_dt, _, _, _, _, _ = run_epoch(train_dl, epoch=epoch, train_flag=True)
        vl_loss, vl_bce, vl_ce, vl_pen, vl_outcome, vl_dt, vl_ce_raw, vl_pen_raw, vl_outcome_raw, vl_dt_raw, val_eval = run_epoch(val_dl, epoch=epoch, train_flag=False)

        # convenience values for printing
        rel_f1 = val_eval["per_class_maxF1"].get("RELEASE", 0.0)
        dth_f1 = val_eval["per_class_maxF1"].get("DEATH", 0.0)
        cmp_names = [n for n in val_eval["per_class_maxF1"] if n not in ("RELEASE", "DEATH")]
        cmp_f1 = (sum(val_eval["per_class_maxF1"][n] for n in cmp_names) / max(len(cmp_names), 1)) if cmp_names else 0.0

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        print(f"""[Phase-2]: Epoch {epoch:02d}
        --> Train={tr_loss:.4f} (BCE={tr_bce:.4f}, CE={tr_ce:.4f}, Pen={tr_pen:.4f}, Out={tr_outcome:.4f}, Δt={tr_dt:.4f})
        --> Val={vl_loss:.4f} (BCE={vl_bce:.4f}, CE={vl_ce:.4f}, Pen={vl_pen:.4f}, Out={vl_outcome:.4f}, Δt={vl_dt:.4f})
        --> Val-F1  RELEASE:{rel_f1:.4f}  DEATH:{dth_f1:.4f}  COMPLICATION:{cmp_f1:.4f}""")

        schedule_events = schedule_controller.update(
            epoch=epoch,
            vl_main=vl_bce,
            vl_ce_raw=vl_ce_raw,
            vl_pen_raw=vl_pen_raw,
            vl_out_raw=vl_outcome_raw,
            vl_dt_raw=vl_dt_raw,
        )
        for msg in schedule_events:
            print(msg)
        if schedule_controller.dynamic_enabled:
            print(schedule_controller.status_line(epoch))


        # Save latest
        model.save(ckpt_last, epoch, best_val, optimizer, scheduler)

        # Save best model 
        warmup_gate = schedule_controller.current_warmup_end_epoch()

        if (vl_loss < best_val - 1e-4) and (epoch >= warmup_gate):
            best_val = vl_loss
            model.save(ckpt_path, epoch, best_val, optimizer, scheduler)
            print("[Phase-2]: Current best model saved.")
            bad_epochs = 0
        elif epoch >= warmup_gate:
            bad_epochs += 1
            if bad_epochs >= training_settings["early-stop-patience"]:
                print("[Phase-2]: Early stopping triggered.")
                break
        else:
            # If warmup isn't complete - do nothing.
            continue

    plot_losses(train_losses, val_losses)
    return model, train_losses, val_losses