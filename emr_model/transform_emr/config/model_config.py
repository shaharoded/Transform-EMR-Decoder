import os

# Get project root (2 levels up from config/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Checkpoint paths
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, 'checkpoints')
EMBEDDER_CHECKPOINT = os.path.join(CHECKPOINT_PATH, 'phase1', 'ckpt_best.pt')
TRANSFORMER_CHECKPOINT = os.path.join(CHECKPOINT_PATH, 'phase2', 'ckpt_best.pt')

# MODEL_CONFIG = {
#       "ctx_dim": 45, # Fill manually once defined your context data.
#       "time2vec_dim": 64,
#       "embed_dim": 264,
#       "block_size": 512,  # //e.g. sequence length, number of tokens processed concurrently
#       "n_head": 8,
#       "n_layer": 8,
#       "dropout": 0.1,
#       "bias": True
#     }

MODEL_CONFIG = {
      "ctx_dim": 2, # Fill manually once defined your context data.
      "time2vec_dim": 32,
      "embed_dim": 64,
      "block_size": 512,  # //e.g. sequence length, number of tokens processed concurrently
      "n_head": 4,
      "n_layer": 4,
      "dropout": 0.1,
      "bias": True
    }

TRAINING_SETTINGS = {
    "phase1_n_epochs": 100,
    "phase2_n_epochs": 100,
    # Controls how fast CBM (curriculum batch masking) ramps up its masking
    # probability during phase-2 training. Independent of the aux-loss schedule.
    "cbm_ramp_epochs": 5,
    "warmup_epochs": 10,
    "early-stop-patience": 10,

    "phase1_learning_rate": 3e-4,
    "phase2_learning_rate": 5e-4,
    "weight_decay": 1e-3,

    "batch_size": 64, # Number of patients processed concurrently
    "bce_k_window": 10, # For soft targets per token on BCE loss, number of next tokens to predict jointly.

    # Phase-1 auxiliary scheduler.
    # Single stage: mlm and dt activate after bce_only_epochs of pure BCE training.
    # Lambda max is calibrated ONCE from training losses at the first active epoch,
    # then kept fixed. Weighted contribution is capped to `fraction` of training BCE.
    "phase1_scheduler": {
        "bce_only_epochs": 3,     # Run BCE alone first so calibration uses a trained model
        "aux_fraction_caps": {
            "mlm": 0.20,  # MLM auxiliary capped to 20% of BCE at calibration epoch
            "dt":  0.20,  # Time regression auxiliary capped to 20% of BCE at calibration epoch
        },
        "order": [["mlm", "dt"]],  # Single stage: both active together after bce_only_epochs
        "ramp_epochs": {
            "mlm": 1,  # No ramp (immediate full lambda after calibration)
            "dt":  1,
        },
    },

    # Phase-2 auxiliary scheduler.
    # Multi-stage curriculum: stages unlock sequentially based on plateau detection.
    #   Stage 0: [ce, dt]      — active after bce_only_epochs, ramp immediately
    #   Stage 1: [penalty]     — unlocked when stage-0 objectives plateau (after ramp)
    #   Stage 2: [outcome]     — unlocked when stage-1 objectives plateau (after ramp)
    # Plateau is measured on vl_total (total weighted validation loss) and only checked
    # once the current stage's ramp has completed.
    # Warmup ends after the outcome ramp completes (dynamic, set by scheduler).
    "phase2_scheduler": {
        "bce_only_epochs": 3,     # Run BCE alone first so calibration uses a trained model
        "aux_fraction_caps": {
            "ce":      0.20,  # Next-token CE nudge cap
            "dt":      0.20,  # Time regression cap
            "penalty": 0.20,  # Structural legality penalty cap
            "outcome": 0.50,  # Future-outcome auxiliary cap
        },
        "order": [["ce", "dt"], ["penalty"], ["outcome"]],
        "ramp_epochs": {
            "ce":      1,  # No ramp
            "dt":      1,  # No ramp
            "penalty": 5,  # Gradual ramp after unlocking
            "outcome": 5,  # Gradual ramp after unlocking
        },
        # Plateau detection settings (applied per stage transition, in order)
        "plateau_min_delta": 1e-4,
        "plateau_patience":  [3, 3],  # Patience per transition: [0→1, 1→2]
    },
}
