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
      "n_head": 4,
      "n_layer": 4,
      "dropout": 0.10,
      "bias": True,
    }

TRAINING_SETTINGS = {
    "phase1_n_epochs": 50,
    "phase2_n_epochs": 100,
    "phase3_n_epochs": 100,
    "sample": None,  # set to int (e.g. 50) for a quick smoke-test

    # Phase-2 optimizer LR warmup (OneCycleLR pct_start).
    # This controls optimizer step size ramp-up, not auxiliary-loss lambda warmup.
    "lr_warmup_epochs": 5,
    "early-stop-patience": 5,
    "early-stop-min-delta-rel": 1e-3,  # relative improvement threshold (0.1%)

    "phase1_learning_rate": 3e-4,
    "phase2_learning_rate": 3e-4,
    "phase3_learning_rate":       1e-4,
    "phase3_backbone_lr_factor":  0.01,  # M-256 baseline setting
    "phase3_weight_decay":        1e-3,  # weight decay for outcome_head in P3 (matches backbone)
    "weight_decay": 1e-3,

    "batch_size": 16, # Number of patients processed concurrently (effective batch=64 via grad accumulation)
    "grad_accumulation_steps": 4, # Accumulate gradients over N steps before optimizer.step()
    "phase1_bce_window_hours": 3.0,
    # Soft-kernel horizon for the Phase-2 LM-head BCE. The kernel decay constant
    # tau is learnable per token class (model.log_tau_lm); this value is both the
    # init for terminal tokens and the hard outer horizon beyond which the kernel
    # contribution is zero.
    "phase2_terminal_bce_window_hours": 168.0,

    # Phase-1 auxiliary scheduler.
    # Single stage: dt activates after bce_only_epochs of pure BCE training.
    # Lambda max is calibrated ONCE from training losses at the first active epoch,
    # then kept fixed. Weighted contribution is capped to `fraction` of training BCE.
    "phase1_scheduler": {
        "bce_only_epochs": 3,     # Run BCE alone first so calibration uses a trained model
        "aux_fraction_caps": {
            "dt":  0.40,  # Time regression auxiliary capped to 40% of BCE at calibration epoch
        },
        "order": [["dt"]],  # Single stage: dt active after bce_only_epochs
        "ramp_epochs": {
            "dt":  0,
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
        "bce_only_epochs": 4,
        "aux_fraction_caps": {
            "ce":      0.50,    # Next-token CE nudge cap
            "dt":      0.50,    # Time regression cap
            "ranking": 0.20,    # Pairwise AUROC-proxy ranking loss on the outcome head
        },
        "order": [["ce", "dt"], ["ranking"]],
        "ramp_epochs": {
            "ce":      0,
            "dt":      0,
            "ranking": 3,  # Gradual ramp avoids destabilising the backbone when stage 1 unlocks
        },
        "plateau_min_delta": 1e-3,
        "plateau_patience":  [2],  # Patience per transition: [0→1]
    },

    # Outcome head — time-decayed soft labels.
    # For each position t the target for outcome k is:
    # sum_s { exp(-dt(t,s) / tau_k) * 1[token_s == outcome_k] }.clamp(0, 1)
    # tau_k is a per-outcome learnable parameter (model.outcome_log_tau), initialised
    # at log(12 / 336). outcome_horizon_hours hard-zeros any contribution beyond that
    # horizon (kept in sync with the eval window family).
    "outcome_horizon_hours": 48.0,
    # Phase 3 uses a longer horizon so late-developing outcomes (CARDIO, NEUROVASC)
    # receive training signal. Phase 2 is intentionally kept at 48h to preserve
    # backbone token-timing behavior (api.py retrains Phase 2 from scratch each run;
    # changing this setting breaks gen_median_steps).
    "phase3_outcome_horizon_hours": 168.0,
}
