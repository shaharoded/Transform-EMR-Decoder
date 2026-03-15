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
    
    "batch_size": 64, # Number of patients processed concurrently
    "bce_k_window": 10, # For soft targets per token on BCE loss, number of next tokens to predict jointly.
    
    # Phase-1 auxiliary settings
    "phase1_bce_weight": 1.0, # BCE loss weight, should be 1.
    "phase1_mlm_weight": 0.05, # MLM loss regulizer weight on the phase1 training task (used as a nudge to the BCE.)
    "phase1_dt_weight": 0.1, # Weight for time regression loss component during phase 1
    
    # Phase-2 auxiliary settings
    "phase2_bce_weight": 1.0, # BCE loss weight, should be 1.
    "phase2_ce_weight": 0.15, # Cross-entropy loss weight, used as a nudge to the BCE.
    "phase2_outcome_weight": 0.15, # Weight for outcome bce loss component during phase 2
    # Balance each penalty to be 20% - 30% of the BCE loss
    "phase2_penalty_weight": 0.22, # Weight for special penalties given on next token loss function (phase 2).
    "phase2_dt_weight": 0.2, # Weight loss on the abs_t prediction, which is combined with regular loss. Currently as calculated (phase 2).
}