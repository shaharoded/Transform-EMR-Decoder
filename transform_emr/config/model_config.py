import os

# Get project root (2 levels up from config/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Checkpoint paths
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, 'checkpoints')
PHASE1_CHECKPOINT = os.path.join(CHECKPOINT_PATH, 'phase1', 'ckpt_best.pt')
PHASE2_CHECKPOINT = os.path.join(CHECKPOINT_PATH, 'phase2', 'ckpt_best.pt')
PHASE3_CHECKPOINT = os.path.join(CHECKPOINT_PATH, 'phase3', 'ckpt_best.pt')

MODEL_CONFIG = {
      "time2vec_dim": 32,
      "embed_dim": 256,
      "block_size": 512,  # //e.g. sequence length, number of tokens processed concurrently
      "n_head": 4,
      "n_layer": 4,
      "dropout": 0.1,
      "bias": True
    }

TRAINING_SETTINGS = {
    "phase1_n_epochs": 10,
    "phase2_n_epochs": 10,
    "phase3_n_epochs": 10,

    # Phase-2 optimizer LR warmup (OneCycleLR pct_start).
    # This controls optimizer step size ramp-up, not auxiliary-loss lambda warmup.
    "lr_warmup_epochs": 5,
    "early-stop-patience": 5,

    "phase1_learning_rate": 3e-4,
    "phase2_learning_rate": 5e-4,
    "phase3_learning_rate":  1e-4,
    "weight_decay": 1e-3,

    "batch_size": 64, # Number of patients processed concurrently
    "phase1_bce_window_hours": 3.0,
    "phase2_bce_window_hours": 12.0,

    # Phase-1 auxiliary scheduler.
    # Single stage: mlm and dt activate after bce_only_epochs of pure BCE training.
    # Lambda max is calibrated ONCE from training losses at the first active epoch,
    # then kept fixed. Weighted contribution is capped to `fraction` of training BCE.
    # Increase fractions if loss doesn't change during training (e.g. if MLM loss is very small, increase its fraction to give it more weight).
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
    #   Stage 0: [ce, dt]  — active after bce_only_epochs, ramp immediately
    #   Stage 1: [outcome] — unlocked when stage-0 objectives plateau (after ramp)
    # Plateau is measured on vl_total (total weighted validation loss) and only checked
    # once the current stage's ramp has completed.
    # Warmup ends after the outcome ramp completes (dynamic, set by scheduler).
    "phase2_scheduler": {
        # Run BCE alone first so auxiliary lambda calibration uses a trained BCE baseline.
        # This value is also used to align early curricula (CBM ramp from epoch 0 and LR warmup).
        # You can decouple these by setting a separate `warmup_epochs` for LR in the scheduler and keeping this as the BCE-only period for curriculum and lambda warmup.
        "bce_only_epochs": 5,
        "aux_fraction_caps": {
            "ce":      2.00,    # Next-token CE nudge cap
            "dt":      0.20,    # Time regression cap
            "outcome": 10.00,   # Future-outcome auxiliary cap
        },
        "order": [["ce", "dt"], ["outcome"]],
        "ramp_epochs": {
            "ce":      1,  # No ramp
            "dt":      1,  # No ramp
            "outcome": 5,  # Gradual ramp after unlocking
        },
        # Plateau detection settings (applied per stage transition, in order)
        "plateau_min_delta": 1e-4,
        "plateau_patience":  [3],  # Patience per transition: [0→1]
    },

    # Outcome head — time-decayed soft labels.
    # For each position t the target for outcome k is:
    # sum_s { exp(-dt(t,s) / tau) * 1[token_s == outcome_k] }.clamp(0, 1)
    # where dt is the time gap (in hours, then normalised by 336) to future step s.
    # tau controls the decay rate: at dt=tau the weight is ~0.37; at 3*tau it is ~0.05.
    # outcome_horizon_hours hard-zeros any contribution beyond that horizon.
    "outcome_decay_tau_hours":  12.0,   # half-life-ish decay constant (hours)
    "outcome_horizon_hours":    48.0,   # keep in sync with eval horizon
}
