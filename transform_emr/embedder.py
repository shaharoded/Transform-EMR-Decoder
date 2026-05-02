import copy
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.preprocessing
import math
from pathlib import Path
from tqdm.auto import tqdm

# ───────── local code ─────────────────────────────────────────────────── #
from transform_emr.dataset import EMRTokenizer
from transform_emr.config.model_config import *
from transform_emr.config.dataset_config import OUTCOMES, TERMINAL_OUTCOMES
from transform_emr.utils import compute_legality_masks_tf, get_temporal_multi_hot_targets, build_mlm, plot_losses, build_luts, logger
from transform_emr.schedulers import LambdaScheduleController

torch.serialization.add_safe_globals([
    EMRTokenizer,
    sklearn.preprocessing.StandardScaler,
    numpy._core.multiarray.scalar,
    numpy._core.multiarray._reconstruct,
    numpy.ndarray,
    numpy.dtype,
    numpy.dtypes.Int64DType,
    numpy.dtypes.Float64DType,
    numpy.float64,
    numpy.int64,
    numpy.int32,
    numpy.float32,
    numpy.bool_,
    numpy.ufunc,
])


class Time2Vec(nn.Module):
    """
    Time2Vec layer for encoding continuous time intervals (Δt) into fixed-size vectors.

    This layer captures both linear trends and periodic patterns in the time between events.

    Output dimensions:
        - 1 dimension for linear time progression (trend)
        - k-1 dimensions for periodic representation using learnable sine functions

    Args:
        out_dim (int): Output dimensionality of the time embedding (must be >= 2)

    Input:
        t (Tensor): FloatTensor of shape [B, T] or [B*T], representing time deltas in days (as float representing partial days too)

    Output:
        Tensor of shape [B, T, out_dim] or [B*T, out_dim], where out_dim = 1 + k
    """

    def __init__(self, out_dim):
        super().__init__()
        if out_dim < 2:
            raise ValueError("Time2Vec out_dim must be >= 2")
        self.linear = nn.Linear(1, 1, bias=True)          # ω0 t + b0 -> Linear component
        self.freq   = nn.Linear(1, out_dim - 1, bias=True) # ω_i t + b_i -> Sinusiodal component

    def forward(self, t):
        """
        t: [B, T] or [B*T] float tensor of time deltas
        Returns: [B, T, out_dim] or [B*T, out_dim]
        """
        t = t.unsqueeze(-1)                     # [B, T, 1]
        linear_out = self.linear(t)             # [B, T, 1]
        periodic_out = torch.sin(self.freq(t))  # [B, T, k-1]
        return torch.cat([linear_out, periodic_out], dim=-1)


class EMREmbedding(nn.Module):
    """
    Embedding layer for electronic medical record (EMR) sequences.

    This module creates time-aware, context-enhanced event representations suitable
    for Transformer-based models. It replaces traditional token and positional embeddings
    by explicitly decomposing each event into structured components:
      - Parent Raw Concept IDs (e.g., "GLUCOSE") - All source raw concepts for this Concept
      - Concept ID (e.g., "GLUCOSE_STATE")
      - Concept + Value ID (e.g., "GLUCOSE_STATE_Low")
      - Concept + Value + Position ID (e.g., "GLUCOSE_STATE_Low_START")
      - Absolute time since admission (Δt abs)
      - Patient-level context vector (e.g., age, sex, No. prior admissions in 6 months...)

    These components are embedded, concatenated, and projected into a shared
    fixed-size embedding space. A [CTX] projection is added to each sequence to incorporate patient-level context.

    All embeddings are regularized with dropout and normalized with LayerNorm.

    In addeition, a tiny MLP head (self.time_head) is attached solely for Phase-1 pre-training.  
    It encourages Time2Vec to encode absolute-time information by regressing the normalised Δt  ∈ [0,1].  
    The head is ignored / discarded in Phase-2.

    Args:
        tokenizer (EMRTokenizer): The tokenizer object managing vocabularies and token metadata.
        ctx_dim (int): Dimensionality of the patient context vector.
        time2vec_dim (int): Output dimension of each Time2Vec component (must be ≥ 2).
        embed_dim (int): Final embedding size for each event.
        dropout (float): Dropout rate applied to the combined embeddings.

    Attributes:
        tokenizer (EMRTokenizer): Stores the vocab and special token mappings.
        decoder (nn.Linear): Tied to the position embedding for predicting next token.
        output_dim (int): Final embedding size (matches `embed_dim`).
        padding_idx (int): Token index reserved for padding ([PAD]).
    """

    def __init__(self, tokenizer, ctx_dim, time2vec_dim=8, embed_dim=128, dropout=0.1):
        super().__init__()

        # --- for compatibility -------------------------------------------------
        self.tokenizer = tokenizer # keep public attr for compatibility
        self.padding_idx = tokenizer.pad_token_id
        self.output_dim = embed_dim  # keep public attr for compatibility

        # --- Token-level embeddings ---
        self.raw_concept_embed = nn.Embedding(len(tokenizer.rawconcept2id), embed_dim) # Embed for "GLUCOSE_MEASURE"
        self.concept_embed = nn.Embedding(len(tokenizer.concept2id), embed_dim) # Embed for "GLUCOSE_MEASURE_STATE"
        self.value_embed = nn.Embedding(len(tokenizer.value2id), embed_dim) # Embed for "GLUCOSE_MEASURE_STATE_High"
        self.position_embed = nn.Embedding(len(tokenizer.token2id), embed_dim) # Embed for "GLUCOSE_MEASURE_STATE_High_Start" -> the full vocab size

        # --- Time embeddings ---
        self.time2vec_abs = Time2Vec(time2vec_dim)

        # --- Time projection + supervised head ---
        self.time_proj = nn.Linear(time2vec_dim, embed_dim, bias=False)
        
        hidden = max(4, time2vec_dim // 2) # Small regression head for Δt supervision (only used in phase‑1)
        self.time_head = nn.Sequential(
            nn.Linear(time2vec_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid() # bound to [0,1]
        )

        # --- patient‑context slot ----------------------------------------
        # NOTE: In phase-1, this projection's output is added directly to the event
        # embeddings as an additive bias (forward_with_decoder / forward_with_mlm),
        # so it trains alongside the decoder via those gradients.
        # In phase-2, forward() returns it separately as `cond` — the transformer's
        # AdaLN blocks use it for shift/scale/gate conditioning (no direct addition).
        self.context_proj = nn.Linear(ctx_dim, embed_dim, bias=False)

        # --- Final projection ---
        concat_dim = 5 * embed_dim  # raw + concept + value + pos + time
        self.final_proj = nn.Linear(concat_dim, embed_dim)

        # --- regularisation ----------------------------------------------
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(embed_dim)
        self.layernorm = nn.LayerNorm(embed_dim)

        self.output_dim = embed_dim

        # --- Decoder tied to token embeddings ----------------------------
        self.decoder = nn.Linear(embed_dim, len(tokenizer.token2id), bias=False)
        self.decoder.weight = self.position_embed.weight  # weight tying

        # --- MLM tied to token embeddings ----------------------------
        self.mlm_head = nn.Linear(embed_dim, len(tokenizer.token2id), bias=False)
        self.mlm_head.weight = self.position_embed.weight  # weight tying

    def predict_time(self, abs_ts):
        """
        Helper used during Phase-1 to supervise Time2Vec.
        abs_ts must be the same tensor given to forward().
        Returns: [B, T, 1] values ∈ [0,1]
        """
        t_abs = self.time2vec_abs(abs_ts)      # [B, T, k]
        return self.time_head(t_abs)           # [B, T, 1]


    def forward(self, parent_raw_ids, concept_ids, value_ids, position_ids,
                 abs_ts, patient_contexts, return_mask=False):
        """
        Build time-aware event embeddings for a full sequence.

        Args:
            parent_raw_ids (LongTensor):  [B, T, P] (P=max_parents)
            concept_ids (LongTensor):     [B, T]
            value_ids (LongTensor):       [B, T]
            position_ids (LongTensor):    [B, T]
            abs_ts (FloatTensor):         [B, T]
            patient_contexts (FloatTensor): [B, ctx_dim]
            return_mask (bool): Whether to return an attention mask

        Returns:
                    seq:  [B, T, D]  (Event embeddings)
                    cond: [B, D]     (Patient context for AdaLN blocks of the transformer)
                    mask: [B, T]     (Padding mask, if return_mask=True)
        """
        # --- Token lookups ---
        
        # parent_raw_ids: [B, T, P]
        p_emb = self.raw_concept_embed(parent_raw_ids)   # [B, T, P, D]

        # mask PAD parents (PAD id == tokenizer.pad_token_id)
        mask = (parent_raw_ids != self.padding_idx).unsqueeze(-1)  # [B, T, P, 1]
        p_emb = p_emb * mask

        den = mask.sum(dim=2).clamp_min(1.0)            # [B, T, 1]
        r_emb = p_emb.sum(dim=2) / den                  # [B, T, D]
        
        c_emb = self.concept_embed(concept_ids)     # [B, T, D]
        v_emb = self.value_embed(value_ids)
        p_emb = self.position_embed(position_ids)

        # --- Time encoding ---
        t_abs = self.time2vec_abs(abs_ts)           # [B, T, k] (k = time2vec_dim)
        t_emb = self.time_proj(t_abs)               # [B, T, D]

        # --- Combine all token-wise pieces ---
        combined = torch.cat([r_emb, c_emb, v_emb, p_emb, t_emb], dim=-1)  # [B, T, 5D]
        ev_vec = self.final_proj(combined)                                 # [B, T, D]
        ev_vec = self.dropout(ev_vec) / self.scale                         # [B, 1, D]

        # --- Sequence Norm ---
        seq = self.layernorm(ev_vec) # [B, T, D]

        # --- Patient Context (Separated for AdaLN) ---
        cond = self.context_proj(patient_contexts)      # [B, D]

        if return_mask:
            pad_mask = (position_ids != self.padding_idx)
            return seq, cond, pad_mask

        return seq, cond
    
    def forward_with_decoder(self, batch: dict, context_dropout_prob=0.15):
        """
        Runs full forward pass + decoding (for training phase 1).
        Unpack tuple (ignore cond for simple decoding task, or use it if decoder depended on it)
        In Phase 1, we just predict tokens from embeddings. 
        We are using the context projection in order to bias the embeddings with patient context.
        Context Dropout is applied here to improve robustness when patient context is missing (inference from phase-1 to phase-2).

        Returns:
            logits: [B, T, vocab_size] — scores for next-token prediction
        """
        seq, cond = self.forward(
        parent_raw_ids=batch["parent_raw_ids"],
        concept_ids=batch["concept_ids"],
        value_ids=batch["value_ids"],
        position_ids=batch["position_ids"],
        abs_ts=batch["abs_ts"],
        patient_contexts=batch["context_vec"],
        return_mask=False,
        )  # → returns only [B,T,D]

        # Context Dropout (Crucial for Phase 2 compatibility)
        # Keep tensor-side branch for stable, vectorized execution.
        if self.training:
            B = cond.size(0)
            drop = (torch.rand(B, 1, device=cond.device) < context_dropout_prob)  # [B, 1] per-sample
            cond_for_addition = torch.where(drop, torch.zeros_like(cond), cond)
        else:
            cond_for_addition = cond

        # Additive Interaction
        # Broadcast context [B, D] -> [B, 1, D] and add to sequence.
        # This biases the event representations based on patient static data.
        combined_embedding = seq + cond_for_addition.unsqueeze(1)

        return self.decoder(combined_embedding)  # [B,T,V], Predict next token at each step
    

    def forward_with_mlm(self, batch: dict, mlm_mask=None, masked_pos_ids=None, context_dropout_prob=0.15):
        """
        Runs full forward pass + MLM (for training phase 1).
        Same inputs as forward_with_decoder, plus:
            mlm_mask - bool tensor [B,T] where True → this position was masked.
        
        Context Dropout is applied here to improve robustness when patient context is missing (inference from phase-1 to phase-2).
        Returns:
            mlm_logits [B, T, vocab_size]
        """
        seq, cond = self.forward(
        parent_raw_ids=batch["parent_raw_ids"],
        concept_ids=batch["concept_ids"],
        value_ids=batch["value_ids"],
        position_ids=masked_pos_ids,  # masked positions used here
        abs_ts=batch["abs_ts"],
        patient_contexts=batch["context_vec"],
        return_mask=False,
        )    # [B,T,D]

        # Context Dropout (Crucial for Phase 2 compatibility)
        # Keep tensor-side branch for stable, vectorized execution.
        if self.training:
            B = cond.size(0)
            drop = (torch.rand(B, 1, device=cond.device) < context_dropout_prob)  # [B, 1] per-sample
            cond_for_addition = torch.where(drop, torch.zeros_like(cond), cond)
        else:
            cond_for_addition = cond

        combined_embedding = seq + cond_for_addition.unsqueeze(1)
        logits = self.mlm_head(combined_embedding)
        
        if mlm_mask is not None:
            logits = logits[mlm_mask] # flatten to [N_masked,V]
        return logits
    
    def save(self, epoch, best_val, optimizer, scheduler, path, lambda_schedule_state=None, bad_epochs=0, training_settings=None):
        ckpt = {
            "epoch": epoch,
            "best_val": best_val,
            "bad_epochs": bad_epochs,
            "optim_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "lambda_schedule_state": lambda_schedule_state,
            "model_state": self.state_dict(),
            "config": {
                "ctx_dim": self.context_proj.in_features,
                "time2vec_dim": self.time2vec_abs.freq.out_features + 1,
                "embed_dim": self.output_dim,
                "dropout": self.dropout.p,
                "vocab_size": self.position_embed.num_embeddings,
            },
            # Keep a copy of training settings used to produce this checkpoint.
            "training_settings": copy.deepcopy(training_settings),
        }
        torch.save(ckpt, path)
    
    @classmethod
    def load(cls, path, tokenizer, map_location="cpu"):
        """
        Load EMREmbedding from a checkpoint.

        Args:
            path (str or Path): Path to checkpoint file.
            tokenizer (EMRTokenizer): Tokenizer to verify against saved model config.
            map_location (str): Device map for torch.load (default: 'cpu').

        Returns:
            model: Loaded EMREmbedding instance
            epoch (int): Last epoch saved in checkpoint
            best_val (float): Best validation loss
            optimizer_state (dict): State dict for optimizer
            scheduler_state (dict): State dict for LR scheduler
        """
        ckpt = torch.load(path, map_location=map_location, weights_only=True)
        config = ckpt.get("config")
        if config is None:
            raise ValueError(
                "[EMREmbedding.load] Invalid checkpoint: missing 'config'."
            )

        # === Safety check: tokenizer vocab consistency ===
        expected_vocab = config["vocab_size"]
        actual_vocab = len(tokenizer.token2id)
        if expected_vocab != actual_vocab:
            raise ValueError(
                f"[EMREmbedding.load] Tokenizer vocab size mismatch: "
                f"expected {expected_vocab}, got {actual_vocab}"
            )

        model = cls(
            tokenizer=tokenizer,
            ctx_dim=config["ctx_dim"],
            time2vec_dim=config["time2vec_dim"],
            embed_dim=config["embed_dim"],
            dropout=config["dropout"]
        )
        model.load_state_dict(ckpt["model_state"])

        # Helpful metadata for callers that want to fully restore prior training settings.
        model.checkpoint_model_config = copy.deepcopy(config)
        model.checkpoint_training_settings = copy.deepcopy(ckpt.get("training_settings"))

        return (
            model,
            ckpt.get("epoch", 0),
            ckpt.get("best_val", float("inf")),
            ckpt.get("optim_state"),
            ckpt.get("scheduler_state"),
            ckpt.get("lambda_schedule_state"),
            ckpt.get("bad_epochs", 0),
        )


@logger
def train_embedder(embedder, train_loader, val_loader, resume=True, checkpoint_path=PHASE1_CHECKPOINT,
                   training_settings=TRAINING_SETTINGS):
    """
    Trains an EMREmbedding model using temporal multi-hot BCE, masked language modelling (MLM),
    and Δt regression.
    Total Loss = λ1 * BCE(temporal multi-hot) + λ2 * MLM(cross-entropy) + λ3 * Δt MSE

    Legality-aware BCE notes:
     - We compute BCE per element (reduction="none") and then mask out illegal classes.
     - Illegal classes are excluded from both numerator and denominator, so they produce no gradient.
     - We also zero targets at illegal indices to avoid any accidental positive supervision there.
     - pos_weight is still applied element-wise by BCEWithLogits; masking happens AFTER the per-element loss,
       so class balancing remains intact but is computed only over the allowed set at each (b,t).
     - The final normalization divides by weights.sum() (the count of allowed elements), not B*T*V.

    Because masking happens after per-element BCE, class rebalancing via pos_weight is preserved but applied 
    only over the legal subset at each (b, t). This prevents the model from learning to suppress tokens merely because 
    they were illegal at that step, aligning Phase-1 supervision with Phase-2 decoding constraints.

    Args:
        embedder (EMREmbedding): The embedding model with decoder.
        train_loader (DataLoader): Training dataloader.
        val_loader (DataLoader): Validation dataloader.
        resume (bool): Resume from last checkpoint if available.
        checkpoint_path (str): Path to save the best model and state.
        training_settings (dict): A settings dictionary, imported from model_config.

    Returns:
        Tuple: (trained model, train_losses, val_losses)
    """
    # Create global training lookup Tensors once the tokenizer is available and move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    luts = build_luts(embedder.tokenizer)
    luts = {k: v.to(device) if torch.is_tensor(v) else v for k,v in luts.items()}
    embedder.to(device)
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()

    ckpt_path = Path(checkpoint_path).resolve()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_last = ckpt_path.parent / "ckpt_last.pt"

    # If resuming, prefer checkpoint-saved settings to avoid config mismatch.
    if resume and ckpt_last.exists():
        pre_ckpt = torch.load(ckpt_last, map_location="cpu", weights_only=True)
        if pre_ckpt.get("training_settings") is not None:
            training_settings = pre_ckpt["training_settings"]

    # ----- Loss & Optimizer -----
    optimizer = torch.optim.AdamW(embedder.parameters(), lr=training_settings["phase1_learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=embedder.tokenizer.token_weights.to(device), reduction="none")

    # All outcome + terminal token ids — used for 1-hot override in multi-hot targets
    _all_outcome_names = sorted(set(OUTCOMES + TERMINAL_OUTCOMES))
    p1_outcome_ids = torch.tensor(
        [embedder.tokenizer.token2id[n] for n in _all_outcome_names if n in embedder.tokenizer.token2id],
        dtype=torch.long, device=device,
    )

    # ----- Resume logic -----
    train_losses, val_losses = [], []
    start_epoch = 1
    best_val = float("inf")
    bad_epochs = 0
    
    lambda_schedule_state = None
    if resume and ckpt_last.exists():
        print(f"[Phase-1] Resuming from checkpoint: {ckpt_last}")
        embedder, start_epoch, best_val, optim_state, scheduler_state, lambda_schedule_state, bad_epochs = EMREmbedding.load(ckpt_last, tokenizer=embedder.tokenizer, map_location=device)
        embedder.to(device)
        if optim_state is not None:
            optimizer.load_state_dict(optim_state)
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)

        # Prefer checkpoint-saved settings to avoid resume/config mismatch issues.
        if getattr(embedder, "checkpoint_training_settings", None) is not None:
            training_settings = embedder.checkpoint_training_settings
        start_epoch += 1

    schedule_controller = LambdaScheduleController(
        schedule_config=training_settings["phase1_scheduler"],
        start_epoch=start_epoch
    )
    if lambda_schedule_state is not None:
        schedule_controller.load_state_dict(lambda_schedule_state)

    # ----- Epoch function -----
    def run_epoch(loader, epoch, train_flag=False):
        """
        Returns weighted and raw component losses for logging and scheduler calibration.
        """
        embedder.train() if train_flag else embedder.eval()
        total_loss, total_bce, total_dt, total_mlm = 0.0, 0.0, 0.0, 0.0
        total_dt_raw, total_mlm_raw = 0.0, 0.0

        for batch in tqdm(loader, desc="Training" if train_flag else "Validation", leave=False, mininterval=5.0, miniters=10, dynamic_ncols=True):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            if train_flag:
                optimizer.zero_grad(set_to_none=True)

            mlm_input_pos_ids = batch["position_ids"].clone() # To avoid in place modifications of the batch
            masked_pos_ids, mlm_mask = build_mlm(
                mlm_input_pos_ids,
                tokenizer=embedder.tokenizer,
                p=0.15
            ) # MLM mask
            
            # BCE Logits + Loss
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                bce_logits = embedder.forward_with_decoder(batch)  # [B, T, V]
            bce_logits = bce_logits.float()

            # build legality with teacher forcing (same LUTs as phase-2)
            illegal = compute_legality_masks_tf(
                position_ids=batch["position_ids"],
                is_start=luts["is_start"],
                is_end=luts["is_end"],
                base_id=luts["base_id"],
                start_ids_per_base=luts["start_ids_per_base"],
                end_ids_per_base=luts["end_ids_per_base"],
                meal_rank=luts["meal_rank"],
                meal_pred_rank=luts["meal_pred_rank"],
                K_meals=luts["K_meals"],
                conflict_mat=luts["conflict_mat"],
                predict_block=luts["predict_block"],   # PAD/MASK/CTX only
            )

            # Temporal: all tokens within phase1_bce_window_hours
            _ABS_TS_SCALE = 336.0
            _P1_WIN = training_settings.get("phase1_bce_window_hours", 3.0) / _ABS_TS_SCALE
            # next_token_ids for phase-1: query positions cover full T, so shift by 1
            # and pad the last column with padding_idx (no next token after last position).
            _pos_ids = batch["position_ids"]
            _B = _pos_ids.size(0)
            _pad_col = torch.full((_B, 1), embedder.padding_idx, dtype=torch.long, device=_pos_ids.device)
            _next_tok = torch.cat([_pos_ids[:, 1:], _pad_col], dim=1)  # [B, T]
            multi_hot_targets = get_temporal_multi_hot_targets(
                target_ids=_pos_ids,
                all_abs_ts=batch["abs_ts"],
                query_abs_ts=batch["abs_ts"],
                padding_idx=embedder.padding_idx,
                vocab_size=bce_logits.size(-1),
                window_size=_P1_WIN,
                outcome_ids=p1_outcome_ids,
                next_token_ids=_next_tok,
            )
            multi_hot_targets = multi_hot_targets.masked_fill(illegal, 0.0)

            raw = loss_fn(bce_logits, multi_hot_targets)       # [B, T, V], Per-element loss
            weights = ((~illegal) & (batch["position_ids"] != embedder.padding_idx).unsqueeze(-1)).float()  # [B,T,V]
            den = weights.sum().clamp_min(1.0)
            bce_loss = (raw * weights).sum() / den

            # MLM Logits + Loss
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                mlm_logits = embedder.forward_with_mlm(
                                                    batch,
                                                    mlm_mask=mlm_mask,
                                                    masked_pos_ids=masked_pos_ids)
            mlm_logits = mlm_logits.float()
            mlm_labels = batch["position_ids"][mlm_mask]          # ground truth
            mlm_raw = F.cross_entropy(
                mlm_logits,
                mlm_labels,
                reduction="mean"
            )
            lambdas = schedule_controller.get_lambdas(epoch)
            mlm_loss = mlm_raw * lambdas["mlm"]
            
            # Δt regression supervision
            # Predict normalised absolute time for every step (non-padding only)
            nonpad = (batch["position_ids"] != embedder.padding_idx)  # [B,T]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                pred_t = embedder.predict_time(batch["abs_ts"])            # [B,T,1]
            pred_t = pred_t.float()
            dt_raw = F.mse_loss(
                pred_t.squeeze(-1)[nonpad],                            # [N_real]
                batch["abs_ts"][nonpad],                               # [N_real]
                reduction='mean'
            )
            time_loss = dt_raw * lambdas["dt"]
            loss = bce_loss + time_loss + mlm_loss

            if train_flag:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(embedder.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            total_bce  += bce_loss.item()
            total_dt   += time_loss.item()
            total_mlm   += mlm_loss.item()
            total_dt_raw += dt_raw.item()
            total_mlm_raw += mlm_raw.item()

        n = len(loader)
        return (
            total_loss / n,
            total_bce / n,
            total_dt / n,
            total_mlm / n,
            total_dt_raw / n,
            total_mlm_raw / n,
        )

    # ----- Training loop -----
    for epoch in range(start_epoch, training_settings["phase1_n_epochs"] + 1):
        tr_tot, tr_bce, tr_dt, tr_mlm, tr_dt_raw, tr_mlm_raw = run_epoch(train_loader, epoch=epoch, train_flag=True)
        vl_tot, vl_bce, vl_dt, val_mlm, _, _ = run_epoch(val_loader, epoch=epoch, train_flag=False)

        # Update auxiliary scheduler:
        #   vl_total  → plateau detection
        #   tr_main   → calibration denominator (training BCE)
        #   mlm/dt    → calibration numerator (training raw aux losses)
        schedule_events = schedule_controller.update(
            epoch=epoch,
            vl_total=vl_tot,
            tr_main=tr_bce,
            mlm=tr_mlm_raw,
            dt=tr_dt_raw,
        )
        for msg in schedule_events:
            print(msg)

        # Step the plateau scheduler on the validation total
        scheduler.step(vl_tot)

        # Collect losses
        train_losses.append(tr_tot)
        val_losses.append(vl_tot)

        print(f"""[Phase-1] Epoch {epoch:03d}
            --> Train={tr_tot:.4f} (BCE={tr_bce:.4f}  MLM={tr_mlm:.4f}  Δt={tr_dt:.4f})
            --> Val={vl_tot:.4f} (BCE={vl_bce:.4f}  MLM={val_mlm:.4f}  Δt={vl_dt:.4f})""")

        # Save best model only after aux-scheduler warmup is complete.
        warmup_gate = schedule_controller.current_warmup_end_epoch()

        if (vl_tot < best_val - 1e-4) and (epoch >= warmup_gate):
            best_val = vl_tot
            bad_epochs = 0
            embedder.save(epoch, best_val, optimizer, scheduler, ckpt_path,
                          lambda_schedule_state=schedule_controller.state_dict(), bad_epochs=bad_epochs,
                          training_settings=training_settings)
            print("[Phase-1]: Current best model saved.")
        elif epoch >= warmup_gate:
            bad_epochs += 1
            if bad_epochs >= training_settings["early-stop-patience"]:
                # Save last checkpoint before stopping
                embedder.save(epoch, best_val, optimizer, scheduler, ckpt_last,
                              lambda_schedule_state=schedule_controller.state_dict(), bad_epochs=bad_epochs,
                              training_settings=training_settings)
                print("[Phase-1]: Early stopping triggered.")
                break
        else:
            # If warmup isn't complete - do nothing.
            continue

        # Save last checkpoint (after bad_epochs is updated)
        embedder.save(epoch, best_val, optimizer, scheduler, ckpt_last,
                      lambda_schedule_state=schedule_controller.state_dict(), bad_epochs=bad_epochs,
                      training_settings=training_settings)
    
    plot_losses(train_losses, val_losses)
    return embedder, train_losses, val_losses