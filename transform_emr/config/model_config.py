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
    "foundational_epochs": 5, # Number of epochs considered as foundational training phase (only for phase 2 where conflicting tasks exist).
    "warmup_epochs": 10,
    "early-stop-patience": 10,
    
    "phase1_learning_rate": 3e-4,
    "phase2_learning_rate": 5e-4,
    "weight_decay": 1e-3,

    # Default cap used by auxiliary schedulers when no component-specific cap is provided.
    # Interpretation: at calibration epoch, weighted auxiliary contribution is targeted
    # to be at most this fraction of validation BCE (main task loss).
    "aux_max_fraction_default": 0.20,
    
    "batch_size": 64, # Number of patients processed concurrently
    "bce_k_window": 10, # For soft targets per token on BCE loss, number of next tokens to predict jointly.
    
    # Phase-1 loss settings
    "phase1_bce_weight": 1.0, # Main objective anchor. Keep as 1.0.
    # For each auxiliary component, cap weighted contribution relative to BCE.
    # Lambda max is calibrated ONCE when first available (using validation losses),
    # then kept fixed for the rest of training.
    "phase1_aux_fraction_caps": {
        "mlm": 0.20, # MLM auxiliary capped to 20% of BCE at calibration point
        "dt": 0.20,  # Time regression auxiliary capped to 20% of BCE at calibration point
    },

    # Phase-1 scheduler settings (same pattern as phase-2, but minimal by default).
    # Keep both at 1 to make activation nearly immediate after first calibration.
    "phase1_dynamic_schedule": {
        "mlm_ramp_epochs": 1, # Ramp MLM lambda to its frozen calibrated max
        "dt_ramp_epochs": 1,  # Ramp Δt lambda to its frozen calibrated max
    },
    
    # Phase-2 loss settings
    "phase2_bce_weight": 1.0, # Main objective anchor. Keep as 1.0.
    # Same fixed-at-calibration cap policy for phase-2 auxiliaries.
    # Stage gating/ramp decides when each aux becomes active; these set each aux max.
    "phase2_aux_fraction_caps": {
        "ce": 0.20,       # Next-token CE nudge cap
        "penalty": 0.20,  # Structural legality penalty cap
        "outcome": 0.20,  # Future-outcome auxiliary cap
        "dt": 0.20,       # Time regression cap
    },

    # Dynamic curriculum for phase-2:
    # - Unlock penalty only after base task plateaus.
    # - Unlock outcome only after penalty-augmented task plateaus.
    # - Warmup ends dynamically after the outcome ramp completes.
    "phase2_dynamic_schedule": {
        "enabled": True,          # If False, use static epoch-based schedule
        "plateau_min_delta": 1e-4, # Minimum improvement to reset plateau counter
        "base_plateau_patience": 3, # Patience before unlocking penalty stage
        "penalty_plateau_patience": 3, # Patience before unlocking outcome stage

        # Guardrails so transitions don't happen too early.
        "min_base_epochs": 5,    # Minimum epochs before penalty can unlock
        "min_penalty_epochs": 5, # Minimum epochs before outcome can unlock

        # Ramp lengths after each stage is unlocked.
        "penalty_ramp_epochs": 5, # Ramp penalty lambda from 0 to calibrated max
        "outcome_ramp_epochs": 5, # Ramp outcome lambda from 0 to calibrated max

        # Optional ramps for always-available phase-2 auxiliaries.
        # Set to 1 for near-immediate activation after first calibration.
        "ce_ramp_epochs": 1,
        "dt_ramp_epochs": 1,
    },
}