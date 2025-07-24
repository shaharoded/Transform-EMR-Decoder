import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.preprocessing
import math
from pathlib import Path
from tqdm import tqdm

# ───────── local code ─────────────────────────────────────────────────── #
from transform_emr.dataset import EMRTokenizer
from transform_emr.config.model_config import *
from transform_emr.utils import get_multi_hot_targets, build_mlm

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
      - Raw Concept ID (e.g., "GLUCOSE")
      - Concept ID (e.g., "GLUCOSE_STATE")
      - Concept + Value ID (e.g., "GLUCOSE_STATE_Low")
      - Concept + Value + Position ID (e.g., "GLUCOSE_STATE_Low_START")
      - Absolute time since admission (Δt abs)
      - Patient-level context vector (e.g., age, sex, No. prior admissions in 6 months)

    These components are embedded, concatenated, and projected into a shared
    fixed-size embedding space. A special [CTX] token is prepended to each sequence
    to incorporate patient-level context at the start of modeling.

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
        self.padding_idx = 0 # Hard coded. Should never change.
        self.output_dim = embed_dim  # keep public attr for compatibility
        self.tokenizer = tokenizer # keep public attr for compatibility

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
        self.ctx_token  = nn.Parameter(torch.randn(embed_dim)) # learnable [CTX] token
        self.context_proj = nn.Linear(ctx_dim, embed_dim, bias=False)

        # --- Final projection ---
        concat_dim = 5 * embed_dim  # concept + value + pos + time
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
        Helper used during Phase‑1 to supervise Time2Vec.
        abs_ts must be the same tensor given to forward().
        Returns: [B, T, 1] values ∈ [0,1]
        """
        t_abs = self.time2vec_abs(abs_ts)      # [B, T, k]
        return self.time_head(t_abs)           # [B, T, 1]


    def forward(self, raw_concept_ids, concept_ids, value_ids, position_ids,
                 abs_ts, patient_contexts, return_mask=False):
        """
        Build time-aware event embeddings for a full sequence.

        Args:
            raw_concept_ids (LongTensor): [B, T]
            concept_ids (LongTensor):     [B, T]
            value_ids (LongTensor):       [B, T]
            position_ids (LongTensor):    [B, T]
            abs_ts (FloatTensor):         [B, T]
            patient_contexts (FloatTensor): [B, ctx_dim]
            return_mask (bool): Whether to return an attention mask

        Returns:
            embeddings:   [B, T+1, D] — [CTX] prepended
            attention_mask (optional): [B, T+1] (True for real tokens)
        """
        # --- Token lookups ---
        r_emb = self.raw_concept_embed(raw_concept_ids)     # [B, T, D]
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

        # --- [CTX] slot ---
        # Time embedding for context token with abs_ts = 0
        ctx_time = self.time_proj(self.time2vec_abs(torch.zeros_like(abs_ts[:, :1])))  # [B, 1, D]
        ctx_vec = self.ctx_token + self.context_proj(patient_contexts) + ctx_time.squeeze(1)  # [B, D]
        ctx_vec = ctx_vec.unsqueeze(1)                                  # [B, 1, D]

        seq = torch.cat([ctx_vec, ev_vec], dim=1)                       # [B, T+1, D]
        seq = self.layernorm(seq)

        if return_mask:
            pad_mask = (position_ids != self.padding_idx)
            pad_mask = torch.cat([torch.ones_like(pad_mask[:, :1]), pad_mask], dim=1)
            return seq, pad_mask

        return seq
    
    def forward_with_decoder(self, batch: dict):
        """
        Runs full forward pass + decoding (for training phase 1).
        Same inputs as forward_with_decoder

        Returns:
            logits: [B, T, vocab_size] — scores for next-token prediction
        """
        seq = self.forward(
        raw_concept_ids=batch["raw_concept_ids"],
        concept_ids=batch["concept_ids"],
        value_ids=batch["value_ids"],
        position_ids=batch["position_ids"],
        abs_ts=batch["abs_ts"],
        patient_contexts=batch["context_vec"],
        return_mask=False
        )  # → returns only [B,T+1,D]

        return self.decoder(seq[:, :-1, :])  # Predict next token at each step
    

    def forward_with_mlm(self, batch: dict, mlm_mask=None, masked_pos_ids=None):
        """
        Runs full forward pass + MLM (for training phase 1).
        Same inputs as forward_with_decoder, plus:
            mlm_mask - bool tensor [B,T] where True → this position was masked.
        Returns:
            mlm_logits [B, T, vocab_size]
        """
        seq = self.forward(
        raw_concept_ids=batch["raw_concept_ids"],
        concept_ids=batch["concept_ids"],
        value_ids=batch["value_ids"],
        position_ids=masked_pos_ids,  # masked positions used here
        abs_ts=batch["abs_ts"],
        patient_contexts=batch["context_vec"],
        return_mask=False
        )    # [B,T+1,D]
        logits = self.mlm_head(seq[:, 1:, :])                  # drop [CTX]
        if mlm_mask is not None:
            logits = logits[mlm_mask]                          # flatten to [N_masked,V]
        return logits
    
    def save(self, epoch, best_val, optimizer, scheduler, path):
        torch.save({
            "epoch": epoch,
            "best_val": best_val,
            "optim_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "model_state": self.state_dict(),
            "config": {
                "ctx_dim": self.context_proj.in_features,
                "time2vec_dim": self.time2vec_abs.freq.out_features + 1,
                "embed_dim": self.output_dim,
                "dropout": self.dropout.p,
                "vocab_size": self.position_embed.num_embeddings,
            }
        }, path)
    
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
        ckpt = torch.load(path, map_location=map_location)
        config = ckpt["config"]

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

        return model, ckpt["epoch"], ckpt["best_val"], ckpt["optim_state"], ckpt["scheduler_state"]


def train_embedder(embedder, train_loader, val_loader, resume=True, checkpoint_path=EMBEDDER_CHECKPOINT, 
                   training_settings=TRAINING_SETTINGS):
    """
    Trains an EMREmbedding model using weighted k-step prediction loss, to allow for a softer loss penalty.
    IDEA: The exact order of the token is not really important, only the existance of important tokens and patterns.
    Total Loss = λ1 * BCE + λ2 * MLM + λ3 * Time Loss (τt)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder.to(device)

    ckpt_path = Path(checkpoint_path).resolve()
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_last = ckpt_path.parent / "ckpt_last.pt"

    # ----- Loss & Optimizer -----
    optimizer = torch.optim.AdamW(embedder.parameters(), lr=training_settings["phase1_learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=embedder.tokenizer.token_weights.to(device))

    # ----- Resume logic -----
    train_losses, val_losses = [], []
    start_epoch = 1
    best_val = float("inf")
    bad_epochs = 0
    
    if resume and ckpt_last.exists():
        print(f"[Phase 1] Resuming from checkpoint: {ckpt_last}")
        embedder, start_epoch, best_val, optim_state, scheduler_state = EMREmbedding.load(ckpt_last, tokenizer=embedder.tokenizer, map_location=device)
        embedder.to(device)
        optimizer.load_state_dict(optim_state)
        scheduler.load_state_dict(scheduler_state)
        start_epoch += 1
    
    # ----- Epoch function -----
    def run_epoch(loader, train_flag=False):
        """
        Returns Tuple: total_loss, bce_loss, time_loss, mlm_loss (for logging in train_embedder and validation)
        """
        embedder.train() if train_flag else embedder.eval()
        total_loss, total_bce, total_dt, total_mlm = 0.0, 0.0, 0.0, 0.0

        for batch in tqdm(loader, desc="Training" if train_flag else "Validation", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            if train_flag:
                optimizer.zero_grad()
            
            mlm_input_pos_ids = batch["position_ids"].clone() # To avoid in place modifications of the batch
            masked_pos_ids, mlm_mask = build_mlm(
                mlm_input_pos_ids,
                tokenizer=embedder.tokenizer,
                p=0.15
            ) # MLM mask
            
            # BCE Logits + Loss
            bce_logits = embedder.forward_with_decoder(batch)  # [B, T, V]

            multi_hot_targets = get_multi_hot_targets(
                                                    position_ids=batch["position_ids"], 
                                                    padding_idx=embedder.padding_idx, 
                                                    vocab_size=bce_logits.size(-1), 
                                                    k=training_settings["bce_k_window"]
                                                    )

            bce_loss = loss_fn(bce_logits, multi_hot_targets)
            bce_loss *= training_settings["phase1_bce_weight"] # Applying weight
            
            # MLM Logits + Loss
            mlm_logits = embedder.forward_with_mlm(
                                                    batch,
                                                    mlm_mask=mlm_mask,
                                                    masked_pos_ids=masked_pos_ids
                                                )
            mlm_labels = batch["position_ids"][mlm_mask]          # ground truth
            mlm_loss = F.cross_entropy(
                mlm_logits,
                mlm_labels,
                reduction="mean"
            )
            mlm_loss *= training_settings["phase1_mlm_weight"]
            
            # Δt regression supervision
            # Predict normalised absolute time for every step
            pred_t = embedder.predict_time(batch["abs_ts"])            # [B,T,1]
            time_loss = F.mse_loss(
                pred_t.squeeze(-1),                                    # [B,T]
                batch["abs_ts"],                                       # [B,T]
                reduction='mean'
            )
            time_loss *= training_settings["phase1_dt_weight"] # Scale
            loss = bce_loss + time_loss + mlm_loss

            if train_flag:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(embedder.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            total_bce  += bce_loss.item()
            total_dt   += time_loss.item()
            total_mlm   += mlm_loss.item()

        n = len(loader)
        return (total_loss / n, total_bce / n, total_dt / n, total_mlm / n)

    # ----- Training loop -----
    for epoch in range(start_epoch, training_settings["phase1_n_epochs"] + 1):
        tr_tot, tr_bce, tr_dt, tr_mlm = run_epoch(train_loader, train_flag=True)
        vl_tot, vl_bce, vl_dt, val_mlm = run_epoch(val_loader,   train_flag=False)

        train_losses.append(tr_tot)
        val_losses.append(vl_tot)

        print(f"""[Phase-1] Epoch {epoch:03d}
            --> Train={tr_tot:.4f} (BCE={tr_bce:.4f}  MLM={tr_mlm:.4f}  Δt={tr_dt:.4f})
            --> Val={vl_tot:.4f} (BCE={vl_bce:.4f}  MLM={val_mlm:.4f}  Δt={vl_dt:.4f})""")

        # Save last checkpoint
        embedder.save(epoch, best_val, optimizer, scheduler, ckpt_last)

        # Save best model        
        if (vl_tot < best_val - 1e-4) and (epoch >= training_settings["warmup_epochs"]):
            best_val = vl_tot
            embedder.save(epoch, best_val, optimizer, scheduler, ckpt_path)
            bad_epochs = 0
        elif epoch >= training_settings["warmup_epochs"]:
            bad_epochs += 1
            if bad_epochs >= training_settings["patience"]:
                print("[Phase 1]: Early stopping triggered.")
                break
        else:
            # If warmup isn't complete - do nothing.
            continue

    return embedder, train_losses, val_losses