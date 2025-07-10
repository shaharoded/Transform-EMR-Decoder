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

        # Delta T prediction head (for regression of delta t from admission at each step -> When will each event occur?)
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
        config = ckpt["config"]

        # === Vocab safety check ===
        expected_vocab = ckpt["vocab_size"]
        actual_vocab = embedder.decoder.out_features
        if expected_vocab != actual_vocab:
            raise ValueError(
                f"[GPT.load] Embedder vocab size mismatch: expected {expected_vocab}, got {actual_vocab}"
            )

        model = cls(cfg=config, embedder=embedder)
        model.load_state_dict(ckpt["model_state"])

        # Return full training state if available
        return model, ckpt.get("epoch", 0), ckpt.get("best_val", float("inf")), ckpt.get("optim_state"), ckpt.get("scheduler_state")


def train_transformer(model, train_dl, val_dl, resume=True, checkpoint_path=TRANSFORMER_CHECKPOINT, training_settings=TRAINING_SETTINGS):
    """
    Trains a Transformer-based EMR sequence model in Phase 2 (decoder stage),
    using a pretrained embedder and structured multi-loss optimization.
    Total Loss = BCE + λ1 * Penalty + λ2 * Time Loss (τt)

    Freezes the embedder for the initial `warmup_epochs`, then unfreezes and reconfigures
    the optimizer to include it. Supports resume-from-checkpoint training.

    Args:
        model (nn.Module): GPT decoder with attached EMREmbedding.
        train_dl (DataLoader): Training data loader.
        val_dl (DataLoader): Validation data loader.
        resume (bool): Resume from latest checkpoint if found.
        checkpoint_path (str): Path to save the best model and state.
        training_settings (dict): A settings dictionary, imported from model_config.

    Returns:
        None. Saves model checkpoints and plots training curves.
    """
    def set_embedder_frozen(model, freeze: bool):
        for p in model.embedder.parameters():
            p.requires_grad = not freeze
        model.embedder.eval() if freeze else model.embedder.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Freeze embedder if requested
    # If not freeze, temporarly freeze for warmup phase
    freeze_epochs = training_settings["warmup_epochs"]
    set_embedder_frozen(model, freeze=True)

    model.to(device)
    model.embedder.tokenizer.token_weights = model.embedder.tokenizer.token_weights.to(device)
    
    optimizer = model.configure_optimizers(
        weight_decay=training_settings["weight_decay"],
        learning_rate=training_settings["phase2_learning_rate"],
        betas=(0.9, 0.95)
    )

    steps_per_epoch = len(train_dl)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=training_settings["phase2_learning_rate"],
        total_steps=training_settings["phase2_n_epochs"] * steps_per_epoch,
        pct_start=0.1,  # 10% warmup
        anneal_strategy='cos',
        cycle_momentum=False
    )

    ckpt_path = Path(checkpoint_path).resolve()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_last = ckpt_path.parent / "ckpt_last.pt"

    start_epoch = 0
    best_val = float("inf")
    patience = training_settings.get("patience", 5)
    wait = 0

    if resume and ckpt_last.exists():
        print(f"[GPT]: Resuming from checkpoint: {ckpt_last}")
        loaded_model, start_epoch, best_val, opt_state, sch_state = GPT.load(ckpt_last, embedder=model.embedder, map_location=device)
        model.load_state_dict(loaded_model.state_dict())
        optimizer.load_state_dict(opt_state)
        scheduler.load_state_dict(sch_state)
        start_epoch += 1
    else:
        print("[GPT]: Starting transformer training loop...")

    train_losses, val_losses = [], []

    def run_epoch(loader, train_flag=False):
        model.train() if train_flag else model.eval()
        total_loss = 0.0
        total_bce, total_penalty, total_dt = 0.0, 0.0, 0.0
        with torch.set_grad_enabled(train_flag):
            for batch in tqdm(loader, desc="Training" if train_flag else "Validation", leave=False):
                batch = {k: v.to(device) for k, v in batch.items()}
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
                pred_logits = logits[:, 1:, :]            # [B, T, V] - predictions for positions 1 to T
                target_tokens = batch["targets"]          # [B, T] - targets for positions 1 to T

                # Multi-hot targets
                multi_hot = get_multi_hot_targets(
                    position_ids=target_tokens,
                    padding_idx=model.embedder.padding_idx,
                    vocab_size=logits.size(-1),
                    k=training_settings["k_window"]
                )

                # Main loss: BCE with logits
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=model.embedder.tokenizer.token_weights.to(logits.device))
                bce = loss_fn(pred_logits, multi_hot) # [B, T, V] vs. [B, T, V]

                # Get predicted token IDs
                pred_ids = pred_logits.argmax(dim=-1) # [B, T] — single token prediction per timestep, full block

                # Debug:
                if pred_ids.max().item() >= len(model.embedder.tokenizer.token2id):
                    print("ERROR: Model is predicting invalid token IDs!")
                    print("Check your model architecture - lm_head output size mismatch!")

                # Load and normalize each penalty (∈ [0, 1])
                penalty = torch.tensor(0.0, device=logits.device)
                p1 = penalty_meal_order(pred_ids, model.embedder.tokenizer.id2token, logits.device)
                p2 = penalty_hallucinated_intervals(pred_ids, target_tokens, model.embedder.tokenizer.id2token, device=logits.device)
                p3 = penalty_false_positives(
                    predictions=pred_logits,
                    targets=multi_hot,
                    token_weights=model.embedder.tokenizer.token_weights,
                    important_token_ids=model.embedder.tokenizer.important_token_ids
                )
                # Average the penalties to bound in [0, 1] + smooth
                penalty = torch.log1p((p1 + p2 + p3) / 3.0)
                penalty = training_settings["penalty_weight"] * penalty # Apply penalty weight

                # Predict abs_ts[:, 1:] using model abs_t_head
                true_delta = torch.clamp(batch["abs_ts"], min=0.0, max=1.0)  # [B, T], range [0,1]
                pred_delta = torch.clamp(abs_t_pred[:, 1:], min=0.0, max=1.0) # shape: [B, T], range [0,1]
                
                mask = (target_tokens != model.embedder.padding_idx).float()  # [B, T]
                abs_t_loss = F.mse_loss(pred_delta, true_delta, reduction='none')  # [B, T]
                abs_t_loss = (abs_t_loss * mask).sum() / mask.sum().clamp(min=1)
                abs_t_loss = training_settings["abs_t_weight"] * abs_t_loss

                # Combine all
                loss = bce + penalty + abs_t_loss

                if train_flag:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()

                total_loss += loss.item()
                total_bce += bce.item()
                total_penalty += penalty.item()
                total_dt += abs_t_loss.item()
        
        n_batches = len(loader)

        return (
            total_loss / n_batches,
            total_bce / n_batches,
            total_penalty / n_batches,
            total_dt / n_batches
        )
    
    for epoch in range(start_epoch, training_settings.get("phase2_n_epochs")):
        # Handle unfreezing the embedder
        if epoch == freeze_epochs:
            print(f"[GPT]: Unfreezing embedder after {freeze_epochs} epochs.")
            set_embedder_frozen(model, freeze=False)
            optimizer = model.configure_optimizers(
                weight_decay=training_settings["weight_decay"],
                learning_rate=training_settings["phase2_learning_rate"],
                betas=(0.9, 0.95)
            )
        tr_loss, tr_bce, tr_pen, tr_dt = run_epoch(train_dl, train_flag=True)
        vl_loss, vl_bce, vl_pen, vl_dt = run_epoch(val_dl, train_flag=False)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        print(f"[Training Transformer]: Epoch {epoch:02d} | "
            f"\n--> Train={tr_loss:.4f} (BCE={tr_bce:.4f}, Pen={tr_pen:.4f}, Δt={tr_dt:.4f}) | "
            f"\n--> Val={vl_loss:.4f} (BCE={vl_bce:.4f}, Pen={vl_pen:.4f}, Δt={vl_dt:.4f})")

        # Save latest
        model.save(ckpt_last, epoch, best_val, optimizer, scheduler)

        if vl_loss < best_val - 1e-3:
            best_val = vl_loss
            model.save(ckpt_path, epoch, best_val, optimizer, scheduler)
            wait = 0
        else:
            if epoch >= training_settings["warmup_epochs"]:
                wait += 1
                if wait >= patience:
                    print("[GPT]: Early stopping triggered.")
                    break

    plot_losses(train_losses, val_losses)