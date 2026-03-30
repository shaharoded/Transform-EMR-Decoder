"""
Fixed data preparation and evaluation for EMR autoresearch.

This file is NOT modified by the agent. It defines:
  - Data loading (load_data)
  - Fixed evaluation metric (evaluate_val_ce)

The metric is cross-entropy on next-event prediction — lower is better,
and it is independent of auxiliary-loss hyperparameters so experiments
are always fairly compared.

Usage (from root):
    from prepare import load_data, evaluate_val_ce, CHECKPOINT_DIR, ...
"""

import os
import sys
import math
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths — all relative to this file's directory
# ---------------------------------------------------------------------------

PROJECT_ROOT   = os.path.dirname(os.path.abspath(__file__))
EMR_MODEL_DIR  = os.path.join(PROJECT_ROOT, "emr_model")

# Make transform_emr importable without pip-installing the package
if EMR_MODEL_DIR not in sys.path:
    sys.path.insert(0, EMR_MODEL_DIR)

from transform_emr.dataset import DataProcessor, EMRTokenizer, EMRDataset, collate_emr, get_dataloader
from transform_emr.config.dataset_config import TAK_REPO_PATH
from transform_emr.utils import get_multi_hot_targets

# ---------------------------------------------------------------------------
# Constants (fixed — do not modify)
# ---------------------------------------------------------------------------

DATA_DIR               = os.path.join(EMR_MODEL_DIR, "data", "source")
TEMPORAL_DATA_FILE     = os.path.join(DATA_DIR, "temporal_events.csv")
CONTEXT_DATA_FILE      = os.path.join(DATA_DIR, "context_data.csv")

CHECKPOINT_DIR         = os.path.join(EMR_MODEL_DIR, "checkpoints")
EMBEDDER_CHECKPOINT    = os.path.join(CHECKPOINT_DIR, "phase1", "ckpt_best.pt")
TRANSFORMER_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "phase2", "ckpt_best.pt")
TOKENIZER_PATH         = os.path.join(CHECKPOINT_DIR, "tokenizer.pt")

VAL_SPLIT    = 0.2
RANDOM_SEED  = 42

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(sample=None, batch_size=64):
    """
    Load and prepare EMR data from source CSVs.

    Parameters
    ----------
    sample : int or None
        If set, use only this many randomly-sampled patients (useful for
        quick smoke-tests; use None for full training).
    batch_size : int
        Batch size for all dataloaders.

    Returns
    -------
    embedder_train_dl   : DataLoader  (unshuffled, for Phase-1 embedder)
    transformer_train_dl: DataLoader  (oversampled, for Phase-2 GPT)
    val_dl              : DataLoader  (unshuffled validation)
    tokenizer           : EMRTokenizer
    """
    print("[Data]: Loading temporal events and context data...")
    temporal_df = pd.read_csv(TEMPORAL_DATA_FILE, low_memory=False)
    ctx_df      = pd.read_csv(CONTEXT_DATA_FILE)

    if sample is not None:
        pids   = temporal_df["PatientID"].unique()
        rng    = np.random.RandomState(RANDOM_SEED)
        chosen = rng.choice(pids, size=min(sample, len(pids)), replace=False)
        temporal_df = temporal_df[temporal_df["PatientID"].isin(chosen)]
        ctx_df      = ctx_df[ctx_df["PatientID"].isin(chosen)]

    tokenizer_path = Path(TOKENIZER_PATH)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)

    if tokenizer_path.exists():
        print("[Data]: Loading tokenizer from cache...")
        processor   = DataProcessor(temporal_df, ctx_df, scaler=None, tak_repo_path=TAK_REPO_PATH)
        temporal_df, ctx_df = processor.run()
        tokenizer   = EMRTokenizer.load(str(tokenizer_path))
    else:
        print("[Data]: Building tokenizer (one-time, may take a few minutes)...")
        processor   = DataProcessor(temporal_df, ctx_df, scaler=None, tak_repo_path=TAK_REPO_PATH)
        temporal_df, ctx_df = processor.run()
        tokenizer   = EMRTokenizer.from_processed_df(temporal_df)
        tokenizer.save(str(tokenizer_path))
        print(f"[Data]: Tokenizer saved to {tokenizer_path}")

    pids = temporal_df["PatientID"].unique()
    train_ids, val_ids = train_test_split(pids, test_size=VAL_SPLIT, random_state=RANDOM_SEED)

    train_df  = temporal_df[temporal_df.PatientID.isin(train_ids)].copy()
    val_df    = temporal_df[temporal_df.PatientID.isin(val_ids)].copy()
    train_ctx = ctx_df.loc[ctx_df.index.isin(train_ids)]
    val_ctx   = ctx_df.loc[ctx_df.index.isin(val_ids)]

    train_ds  = EMRDataset(train_df, train_ctx, tokenizer=tokenizer)
    val_ds    = EMRDataset(val_df,   val_ctx,   tokenizer=tokenizer)

    print(f"[Data]: {len(train_ids)} train / {len(val_ids)} val patients  "
          f"({len(train_ds.tokens_df):,} train records, {len(val_ds.tokens_df):,} val records)")

    embedder_train_dl    = get_dataloader(train_ds, batch_size=batch_size, collate_fn=collate_emr, oversample=False)
    transformer_train_dl = get_dataloader(train_ds, batch_size=batch_size, collate_fn=collate_emr, oversample=True)
    val_dl               = get_dataloader(val_ds,   batch_size=batch_size, collate_fn=collate_emr, oversample=False)

    return embedder_train_dl, transformer_train_dl, val_dl, tokenizer

# ---------------------------------------------------------------------------
# Fixed evaluation metric (DO NOT CHANGE — this is the ground truth)
# ---------------------------------------------------------------------------

EVAL_K_WINDOW = 5  # fixed k for evaluation — independent of training bce_k_window

@torch.no_grad()
def evaluate_val_bce(model, val_dl, device="cuda"):
    """
    BCE validation loss with a fixed k=5 look-ahead window.

    BCE with multi-hot targets is the right metric for EMR data: multiple
    events can occur simultaneously (or within a few steps), so penalising
    the model for not picking a single "correct" token (as CE would do) is
    inappropriate.

    The window is pinned at EVAL_K_WINDOW=5 regardless of the bce_k_window
    training hyperparameter, so experiments with different k values remain
    directly comparable.

    Lower is better. Padding positions are excluded.

    Returns
    -------
    float : mean BCE loss over the validation set
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    padding_idx   = model.embedder.padding_idx
    total_loss    = 0.0
    total_batches = 0

    for batch in val_dl:
        batch = {key: v.to(device) if torch.is_tensor(v) else v for key, v in batch.items()}

        logits, _, _ = model(
            parent_raw_ids=batch["parent_raw_ids"],
            concept_ids   =batch["concept_ids"],
            value_ids     =batch["value_ids"],
            position_ids  =batch["position_ids"],
            abs_ts        =batch["abs_ts"],
            context_vec   =batch["context_vec"],
        )

        # Autoregressive: logit at t predicts token t+1
        pred_logits = logits[:, :-1, :].float()   # [B, T-1, V]
        target_ids  = batch["targets"][:, 1:]      # [B, T-1]
        nonpad      = target_ids != padding_idx    # [B, T-1]

        if nonpad.sum() == 0:
            continue

        vocab_size = pred_logits.size(-1)
        multi_hot  = get_multi_hot_targets(
            position_ids=target_ids,
            padding_idx =padding_idx,
            vocab_size  =vocab_size,
            k           =EVAL_K_WINDOW,
        ).to(device)                               # [B, T-1, V]

        # Average BCE over valid (non-pad) positions only
        loss_per_elem = F.binary_cross_entropy_with_logits(
            pred_logits, multi_hot, reduction="none"
        )                                          # [B, T-1, V]
        valid_mask = nonpad.unsqueeze(-1).float()  # [B, T-1, 1]
        loss = (loss_per_elem * valid_mask).sum() / (valid_mask.sum().clamp(min=1) * vocab_size)

        total_loss    += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)
