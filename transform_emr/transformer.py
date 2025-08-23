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
from tqdm import tqdm

# ───────── local code ─────────────────────────────────────────────────── #
from transform_emr.embedder import EMREmbedding
from transform_emr.config.model_config import *
from transform_emr.utils import *
from transform_emr.loss import MaskedFocalBCE


# ───────── helpers ─────────────────────────────────────────────────────────── #
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias   = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


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

        # pre‑built causal mask (triangular) – trimmed in forward
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(cfg["block_size"], cfg["block_size"]))
            .view(1, 1, cfg["block_size"], cfg["block_size"])
        )
    def _scaled_dot_product_attention(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v)
    
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, self.n_head, 3 * (C // self.n_head))
        q, k, v = qkv.chunk(3, dim=-1)   # (B, T, h, d)

        # PyTorch 2.1 optimized attention OR fallback
        if hasattr(F, "scaled_dot_product_attention"):
            attn = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True
            )
        else:
            attn = self._scaled_dot_product_attention(q, k, v)

        y = self.proj(attn.reshape(B, T, C))
        return self.resid_dropout(y)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w1 = nn.Linear(cfg["embed_dim"], 2 * cfg["embed_dim"], bias=cfg["bias"])
        self.w2 = nn.Linear(   cfg["embed_dim"],     cfg["embed_dim"], bias=cfg["bias"])
        self.drop = nn.Dropout(cfg["dropout"])
    def forward(self, x):
        x, gate = self.w1(x).chunk(2, dim=-1)
        return self.drop(self.w2(F.gelu(x) * gate))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ln1 = LayerNorm(cfg["embed_dim"], bias=cfg["bias"])
        self.att = CausalSelfAttention(cfg)
        self.ln2 = LayerNorm(cfg["embed_dim"], bias=cfg["bias"])
        self.mlp = MLP(cfg)

    def forward(self, x):
        res_scale = 1 / math.sqrt(2 * self.cfg["n_layer"])
        x = x + res_scale * self.att(self.ln1(x))
        x = x + res_scale * self.mlp(self.ln2(x))
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
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layer"])])
        self.ln_f  = LayerNorm(cfg["embed_dim"], bias=cfg["bias"])

        # Next token prediction head (What will be the next event?)
        self.lm_head = nn.Linear(cfg["embed_dim"], vocab_size, bias=False)
        self.lm_head.weight = self.embedder.position_embed.weight  # weight tying
        assert self.lm_head.weight.shape[0] == vocab_size, (
            f"[GPT] lm_head output dim ({self.lm_head.weight.shape[0]}) "
            f"does not match embedder.position_embed ({vocab_size})"
        )

        # Δt prediction head (for regression of Δt from admission at each step -> When will each event occur?)
        self.abs_t_head = nn.Sequential(
            nn.Linear(cfg["embed_dim"], 16 * cfg["time2vec_dim"]),
            nn.ReLU(),
            nn.Linear(16 * cfg["time2vec_dim"], 1),  # Output: scalar abs_t
            nn.Sigmoid()  # ← Bound output to [0,1]
        )

        self.apply(self._init_weights)
        # slightly smaller init for res projections as in gpt‑2
        for n, p in self.named_parameters():
            if n.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg["n_layer"]))

        print(f"[GPT]: Total params: {self.get_num_params()/1e6:.2f} M")
        
        if cfg.get("compile", False):
            if hasattr(torch, "compile"):
                print("[GPT]: Compiling model with torch.compile()")
                self = torch.compile(self)
            else:
                print("[GPT]: torch.compile() is not available in this PyTorch version. Skipping.")
        

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
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=training_settings["phase2_learning_rate"],
            total_steps=training_settings["phase2_n_epochs"] * len(train_dl),
            pct_start=0.2,
            anneal_strategy="cos",
            div_factor  = 10,       # ← start‑LR = LR / 10
            final_div_factor = 10,  # floor‑LR = 1e‑5 as well (cos tail reaches this)
        )

    # ---------------------------------------------------- forward & loss ---- #
    def forward(self, raw_concept_ids, concept_ids, value_ids, position_ids,
            abs_ts, context_vec=None):
        """
        All tensors come straight from `collate_emr`:
            raw_concept_ids (torch.Tensor)   - padded raw_concept ids, (B, T)
            concept_ids (torch.Tensor)       - padded concepts ids, (B, T)
            value_ids (torch.Tensor)         - padded concept_value ids, (B, T)
            position_ids (torch.Tensor)      - padded token ids, (B, T)
            abs_ts (torch.Tensor)            - relative start times from ADMISSION (hours), (B, T)
            context_vec (torch.Tensor)       - age/gender or [] if not used, (B, C)
        """
        def _forward(block, x):
            """Allows gradient checkpointing on blocks -> Memory efficient"""
            return block(x)
        
        x = self.drop(self.embedder(raw_concept_ids, concept_ids, value_ids, position_ids,
            abs_ts, context_vec, return_mask=False))  # (B, T+1, D)
        
        for blk in self.blocks:
            if self.training and self.use_checkpoint:
                x = checkpoint.checkpoint(_forward, blk, x, use_reentrant=False)
            else:
                x = blk(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)            # (B, T+1, V)
        abs_t_pred = self.abs_t_head(x)     # (B, T+1, 1)

        return logits, abs_t_pred.squeeze(-1)  # loss is computed in train.py, squeeze for easier MSE
    

    def save(self, path, epoch=None, best_val=None, optimizer=None, scheduler=None):
        ckpt = {
            "model_state": self.state_dict(),
            "config": self.cfg,
            "vocab_size": self.embedder.decoder.out_features,
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
        ckpt = torch.load(path, map_location=map_location)

        # === Vocab safety check ===
        expected_vocab = ckpt["vocab_size"]
        actual_vocab = embedder.decoder.out_features
        if expected_vocab != actual_vocab:
            raise ValueError(
                f"[GPT.load] Embedder vocab size mismatch: expected {expected_vocab}, got {actual_vocab}"
            )
        
        model = cls(cfg=ckpt["config"], embedder=embedder)
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
    Total Loss = λ1 * BCE + λpen * Penalty + λt * Time Loss (τt) + λt_pen * Time penalty

    Gradually applying curriculum and CBM during `warmup_epochs`, then implements early stopping based on best total loss. 
    Supports resume-from-checkpoint training.

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
    
    # Build criterion once
    criterion = MaskedFocalBCE.from_counts(
        counts=model.embedder.tokenizer.token_counts,
        token_weights=model.embedder.tokenizer.token_weights,
        beta=0.999, min_count=5, clip_max=8.0,
        gamma=1.2,         # focal strength
        tau=0.8,           # pos/neg balance anchor
        neg_bounds=(0.05, 0.5),   # clamp for stability
        label_smoothing=0.01,     # optional
        hard_neg_k=64               # or e.g., 64 for hard-neg mining
    ).to(device)

    ckpt_path = Path(checkpoint_path).resolve()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_last = ckpt_path.parent / "ckpt_last.pt"

    start_epoch = 0
    best_val = float("inf")
    bad_epochs = 0

    if resume and ckpt_last.exists():
        print(f"[GPT]: Resuming from checkpoint: {ckpt_last}")
        loaded_model, start_epoch, best_val, opt_state, sch_state  = GPT.load(ckpt_last, embedder=model.embedder, map_location=device)
        model.load_state_dict(loaded_model.state_dict())
        optimizer.load_state_dict(opt_state)
        
        # Scheduler needs to be re-initiated with the recovered optimizer before loading it's state dict
        scheduler = model.configure_scheduler(optimizer, training_settings, train_dl)
        scheduler.load_state_dict(sch_state)
        start_epoch += 1
        print(f"[GPT]: Resumed training at epoch {start_epoch} (best val_loss so far: {best_val:.4f})")
    else:
        print("[GPT]: Starting transformer training loop...")

    train_losses, val_losses = [], []

    def run_epoch(loader, epoch, train_flag=False):
        if train_flag:
            model.train()
        else:
            model.eval()
            metric = F1Aggregator(tokenizer=model.embedder.tokenizer, device=device) # For epoch evaluation

        total_loss = total_bce = total_penalty = total_dt = 0.0
        with torch.set_grad_enabled(train_flag):
            for batch in tqdm(loader, desc="Training" if train_flag else "Validation", leave=False):
                batch = {k: v.to(device) for k, v in batch.items()}

                # === Apply CBM on training batchs ===
                if train_flag:
                    batch = apply_cbm(batch, 
                                      epoch, 
                                      training_settings["warmup_epochs"], 
                                      model.embedder.tokenizer, 
                                      luts["forbid_mask_ids"], 
                                      max_p=0.25
                                      )

                # === Original logits from Model ===
                logits, abs_t_pred = model(
                    raw_concept_ids=batch["raw_concept_ids"],
                    concept_ids=batch["concept_ids"],
                    value_ids=batch["value_ids"],
                    position_ids=batch["position_ids"],
                    abs_ts=batch["abs_ts"],
                    context_vec=batch["context_vec"]
                )

                # logits is [B, T+1, V] due to [CTX] token prepending
                # We want to predict tokens 1 to T given context + tokens 0 to T-1
                pred_logits = logits[:, 1:, :]            # [B, T, V] - predictions for positions 1 to T (no [CTX])
                target_ids = batch["targets"]          # [B, T] - targets for positions 1 to T
                
                # === legality masks from Ground Truth + Targets ===
                illegal_mask, bonus_mask = compute_legality_masks_tf(
                                                target_ids,
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
                
                # Soft illegal-mass penalty (pre-mask)
                nonpad = (target_ids != model.embedder.padding_idx)                # [B,T]
                p_illegal = soft_illegal_mass_penalty(
                    logits_pre_mask=pred_logits,
                    illegal_mask=illegal_mask,
                    nonpad_mask=nonpad,
                    margin=0.02,           # good starting point (see notes)
                    power=1.0
                )

                # Apply masks BEFORE BCE so gradients learn legality only
                pred_logits = apply_masks_to_logits(
                    pred_logits, illegal_mask, bonus_mask
                )

                # Get predicted token IDs
                pred_ids = pred_logits.argmax(dim=-1) # [B, T] — single token prediction per timestep, full block

                # === Loss: BCE with logits ===
                # Multi-hot targets
                multi_hot = get_multi_hot_targets(
                    position_ids=target_ids,
                    padding_idx=model.embedder.padding_idx,
                    vocab_size=pred_logits.size(-1),
                    k=training_settings["bce_k_window"]
                ).masked_fill_(illegal_mask, 0.0)    # zero‑out illegal targets  
                
                # mask out illegal classes AND PAD steps from the denominator
                valid_pos = nonpad.unsqueeze(-1)            # [B,T,1] bool
                allowed   = (~illegal_mask) & valid_pos     # [B,T,V] bool               

                # Calculate BCE loss (only valid positions) 
                loss_bce, _ = criterion(pred_logits, multi_hot, allowed)
                loss_bce = loss_bce * training_settings["phase2_bce_weight"] # Applying weight

                # === Loss: Structural penalties on output ===
                # Load and normalize each penalty (∈ [0, 1]) -> Active grad on penalty functions
                p_struct = soft_interval_penalty(
                    pred_logits,             # keep grads
                    allowed,
                    luts["start_ids_per_base"],
                    luts["end_ids_per_base"],
                    luts["conflict_mat"],
                    alpha=10.0               # sharpness; 8–12 worked in tests
                )
                
                p_meal = soft_meal_order_penalty(
                    pred_logits,             # keep grads
                    allowed,
                    luts["meal_rank"],
                    decay=0.9,               # recency memory (0.8–0.95 reasonable)
                    beta=8.0                 # “seen” squashing sharpness
                )

                # Average the penalties to bound in [0, 1]
                lambda_pen = linear_schedule(
                    epoch,
                    training_settings['warmup_epochs'],
                    training_settings["phase2_penalty_weight"]
                )
                generative_penalty = (p_illegal + p_struct + p_meal) / 3.0
                generative_penalty = lambda_pen * generative_penalty

                # === Loss: Δt + monotonicity ===
                # Predict abs_ts[:, 1:] using model abs_t_head
                true_delta = torch.clamp(batch["abs_ts"], min=0.0, max=1.0)  # [B, T], range [0,1]
                pred_delta = torch.clamp(abs_t_pred[:, 1:], min=0.0, max=1.0)  # [B, T], range [0,1]

                mask = (target_ids != model.embedder.padding_idx).float()  # [B, T]

                # Base MSE loss (no scheduler on MSE)
                abs_t_loss = F.mse_loss(pred_delta, true_delta, reduction='none')  # [B, T]
                abs_t_loss = (abs_t_loss * mask).sum() / mask.sum().clamp(min=1)
                abs_t_loss = abs_t_loss * training_settings["phase2_dt_weight"]

                # Temporal Monotonicity Penalty (with scheduler) ===
                # Penalize predicted time going backwards: max(0, prev - curr)
                delta_diff = pred_delta[:, 1:] - pred_delta[:, :-1]  # [B, T-1]
                monotonic_penalty = F.relu(-delta_diff)              # only penalize decreases
                monotonic_mask = mask[:, 1:] * mask[:, :-1]          # valid when both t and t-1 are not PAD
                monotonic_penalty = (monotonic_penalty * monotonic_mask).sum() / monotonic_mask.sum().clamp(min=1)
                lambda_t_monotonic = linear_schedule(epoch, 
                        training_settings['warmup_epochs'], 
                        training_settings["phase2_dt_monotonic_penalty"])
                abs_t_loss += lambda_t_monotonic * monotonic_penalty

                # === Loss: Total Loss ===
                loss = loss_bce + generative_penalty + abs_t_loss

                # === Backprop and Log ===
                if train_flag:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                else:
                    # Update validation classification metric
                    metric.update(pred_ids, target_ids)


                # Update loss
                total_loss += loss.item()
                total_bce += loss_bce.item()
                total_penalty += generative_penalty.item()
                total_dt += abs_t_loss.item()
        
        val_f1 = None if train_flag else metric.compute()
        n_batches = len(loader)

        return (
            total_loss / n_batches,
            total_bce / n_batches,
            total_penalty / n_batches,
            total_dt / n_batches,
            val_f1
        )
    
    for epoch in range(start_epoch, training_settings.get("phase2_n_epochs")):
        tr_loss, tr_bce, tr_pen, tr_dt, _ = run_epoch(train_dl, epoch=epoch, train_flag=True)
        vl_loss, vl_bce, vl_pen, vl_dt, val_f1 = run_epoch(val_dl, epoch=epoch, train_flag=False)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        print(f"""[Training Transformer]: Epoch {epoch:02d}
        --> Train={tr_loss:.4f} (BCE={tr_bce:.4f}, Pen={tr_pen:.4f}, Δt={tr_dt:.4f})
        --> Val={vl_loss:.4f} (BCE={vl_bce:.4f}, Pen={vl_pen:.4f}, Δt={vl_dt:.4f})
        --> Val-F1  RELEASE:{val_f1['REL']:.4f}  DEATH:{val_f1['DTH']:.4f}  COMPLICATION:{val_f1['CMP']:.4f}""")


        # Save latest
        model.save(ckpt_last, epoch, best_val, optimizer, scheduler)

        # Save best model 
        if (vl_loss < best_val - 1e-4) and (epoch >= training_settings["warmup_epochs"]):
            best_val = vl_loss
            model.save(ckpt_path, epoch, best_val, optimizer, scheduler)
            bad_epochs = 0
        elif epoch >= training_settings["warmup_epochs"]:
            bad_epochs += 1
            if bad_epochs >= training_settings["patience"]:
                print("[GPT]: Early stopping triggered.")
                break
        else:
            # If warmup isn't complete - do nothing.
            continue

    plot_losses(train_losses, val_losses)
    return model, train_losses, val_losses