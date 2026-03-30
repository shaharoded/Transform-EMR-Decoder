"""
EMR Autoresearch training script.

This is the file the agent modifies. Architecture, hyperparameters, and
training settings — everything in here is fair game.

Usage:
    python train.py
    python train.py > run.log 2>&1   (redirect all output to log)

The only file you must NOT modify is prepare.py.
"""

import os
import sys
import time
import shutil
from pathlib import Path

# Add emr_model to path (prepare.py also does this, but be explicit here)
PROJECT_ROOT  = os.path.dirname(os.path.abspath(__file__))
EMR_MODEL_DIR = os.path.join(PROJECT_ROOT, "emr_model")
if EMR_MODEL_DIR not in sys.path:
    sys.path.insert(0, EMR_MODEL_DIR)

import torch

from prepare import (
    load_data, evaluate_val_bce,
    CHECKPOINT_DIR, EMBEDDER_CHECKPOINT, TRANSFORMER_CHECKPOINT,
)
from transform_emr.embedder import EMREmbedding, train_embedder
from transform_emr.transformer import GPT, train_transformer

# ===========================================================================
# HYPERPARAMETERS — the agent edits this section
# ===========================================================================

# Model architecture
MODEL_CONFIG = {
    # ctx_dim is set automatically from the data — do not edit manually
    "ctx_dim":       2,     # will be overwritten by prepare.load_data()
    "time2vec_dim":  32,    # dimension of each Time2Vec component (>= 2)
    "embed_dim":     64,    # shared embedding dimension for tokens & GPT
    "block_size":    512,   # max sequence length (tokens per patient window)
    "n_head":        4,     # number of attention heads (embed_dim % n_head == 0)
    "n_layer":       4,     # number of transformer decoder blocks
    "dropout":       0.1,   # dropout applied throughout
    "bias":          True,  # use bias in linear layers
}

# Training settings
TRAINING_SETTINGS = {
    # Epoch budgets (early stopping may terminate earlier)
    "phase1_n_epochs": 30,
    "phase2_n_epochs": 50,

    # Phase-2 curriculum masking ramp-up
    "cbm_ramp_epochs":  5,

    # Warm-up: do not save "best" until this many epochs have passed
    "warmup_epochs": 5,

    # Early stopping patience (epochs without improvement)
    "early-stop-patience": 10,

    # Learning rates
    "phase1_learning_rate": 3e-4,
    "phase2_learning_rate": 5e-4,
    "weight_decay":         1e-3,

    # Batch size (reduce if OOM; affects training speed not final quality much)
    "batch_size": 64,

    # BCE k-step window: how many future tokens count as "correct" targets
    "bce_k_window": 10,

    # -----------------------------------------------------------------------
    # Phase-1 auxiliary loss scheduler
    # BCE trains alone for bce_only_epochs, then MLM + Δt are activated.
    # Lambda for each aux loss is calibrated once so it contributes at most
    # aux_fraction_caps[key] × BCE_loss at the calibration epoch.
    # -----------------------------------------------------------------------
    "phase1_scheduler": {
        "bce_only_epochs": 3,
        "aux_fraction_caps": {
            "mlm": 0.20,
            "dt":  0.20,
        },
        "order":       [["mlm", "dt"]],
        "ramp_epochs": {"mlm": 1, "dt": 1},
    },

    # -----------------------------------------------------------------------
    # Phase-2 auxiliary loss scheduler (multi-stage curriculum)
    #   Stage 0: [ce, dt]   — active after bce_only_epochs
    #   Stage 1: [penalty]  — unlocked on stage-0 plateau
    #   Stage 2: [outcome]  — unlocked on stage-1 plateau
    # -----------------------------------------------------------------------
    "phase2_scheduler": {
        "bce_only_epochs": 3,
        "aux_fraction_caps": {
            "ce":      0.20,
            "dt":      0.20,
            "penalty": 0.20,
            "outcome": 0.50,
        },
        "order": [["ce", "dt"], ["penalty"], ["outcome"]],
        "ramp_epochs": {
            "ce":      1,
            "dt":      1,
            "penalty": 5,
            "outcome": 5,
        },
        "plateau_min_delta": 1e-4,
        "plateau_patience":  [3, 3],
    },
}

# ===========================================================================
# Setup
# ===========================================================================

t_start = time.time()
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Clear model checkpoints for a fresh run (tokenizer is preserved)
for phase_dir in ["phase1", "phase2"]:
    phase_path = Path(CHECKPOINT_DIR) / phase_dir
    if phase_path.exists():
        shutil.rmtree(phase_path)
    phase_path.mkdir(parents=True, exist_ok=True)

# Load data — tokenizer is built once and cached
embedder_train_dl, transformer_train_dl, val_dl, tokenizer = load_data(
    batch_size=TRAINING_SETTINGS["batch_size"]
)

# Auto-detect context vector size from the first batch
for _batch in embedder_train_dl:
    MODEL_CONFIG["ctx_dim"] = _batch["context_vec"].shape[-1]
    break

print(f"Model config: {MODEL_CONFIG}")

# ===========================================================================
# Phase 1 — Train embedder (token + time + context representations)
# ===========================================================================

embedder = EMREmbedding(
    tokenizer    = tokenizer,
    ctx_dim      = MODEL_CONFIG["ctx_dim"],
    time2vec_dim = MODEL_CONFIG["time2vec_dim"],
    embed_dim    = MODEL_CONFIG["embed_dim"],
    dropout      = MODEL_CONFIG["dropout"],
)

embedder, _, _ = train_embedder(
    embedder          = embedder,
    train_loader      = embedder_train_dl,
    val_loader        = val_dl,
    resume            = False,
    checkpoint_path   = EMBEDDER_CHECKPOINT,
    training_settings = TRAINING_SETTINGS,
)

# ===========================================================================
# Phase 2 — Train GPT transformer over learned embeddings
# ===========================================================================

model = GPT(cfg=MODEL_CONFIG, embedder=embedder)

model, _, val_losses = train_transformer(
    model             = model,
    train_dl          = transformer_train_dl,
    val_dl            = val_dl,
    resume            = False,
    checkpoint_path   = TRANSFORMER_CHECKPOINT,
    training_settings = TRAINING_SETTINGS,
)

# ===========================================================================
# Evaluation — fixed metric from prepare.py (do not change)
# ===========================================================================

# Load best saved checkpoint; fall back to last if best wasn't written
best_ckpt    = Path(TRANSFORMER_CHECKPOINT)
fallback_ckpt = best_ckpt.parent / "ckpt_last.pt"

if best_ckpt.exists():
    best_model, *_ = GPT.load(str(best_ckpt), embedder=embedder)
elif fallback_ckpt.exists():
    best_model, *_ = GPT.load(str(fallback_ckpt), embedder=embedder)
else:
    best_model = model

val_bce = evaluate_val_bce(best_model, val_dl, device=str(device))

# ===========================================================================
# Summary  (grep-friendly format — one key per line)
# ===========================================================================

t_end        = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
num_params   = (sum(p.numel() for p in best_model.parameters()) +
                sum(p.numel() for p in embedder.parameters()))
phase2_best  = min(val_losses) if val_losses else float("nan")
phase2_epochs = len(val_losses)

print("---")
print(f"val_bce_loss:     {val_bce:.6f}")
print(f"phase2_best_val:  {phase2_best:.6f}")
print(f"phase2_epochs:    {phase2_epochs}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"embed_dim:        {MODEL_CONFIG['embed_dim']}")
print(f"n_layer:          {MODEL_CONFIG['n_layer']}")
print(f"n_head:           {MODEL_CONFIG['n_head']}")
print(f"block_size:       {MODEL_CONFIG['block_size']}")
print(f"num_params:       {num_params:,}")
