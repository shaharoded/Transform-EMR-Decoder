import os

# Get project root (2 levels up from config/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Checkpoint paths
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, 'checkpoints')
EMBEDDER_CHECKPOINT = os.path.join(CHECKPOINT_PATH, 'phase1', 'best_embedder.pt')
TRANSFORMER_CHECKPOINT = os.path.join(CHECKPOINT_PATH, 'phase2', 'best_model.pt')

MODEL_CONFIG = {
      "ctx_dim": 2, # Fill manually once defined your context data.
      "time2vec_dim": 16,
      "embed_dim": 256,
      "block_size": 256,  # //e.g. sequence length, number of tokens processed concurrently
      "n_head": 8,
      "n_layer": 4,
      "dropout": 0.1,
      "bias": True,
      "compile": True # Allows JIT compile for the model - Better memory and speed.
    }

TRAINING_SETTINGS = {
    "phase1_n_epochs": 100,
    "phase2_n_epochs": 80,
    "warmup_epochs": 5,
    "patience": 5,
    "phase1_learning_rate": 5e-4,
    "phase2_learning_rate": 3e-4,
    "weight_decay": 1e-3,
    "max_cbm_ratio": 0.15, # Maximum ratio of [MASK] (curriculum) within the model's context window at batch.
    "batch_size": 8, # Number of patients processed concurrently
    "bce_k_window": 10, # For soft targets per token on BCE loss, number of next tokens to predict jointly.
    "phase1_bce_weight": 1.0, # BCE loss weight, should be 1.
    "phase1_mlm_weight": 0.2, # MLM loss regulizer weight on the phase1 training task (= phase1_bce_weight / bce_k_window)
    "phase1_dt_weight": 0.1, # Weight for time regression loss component during phase 1
    "phase2_bce_weight": 1.0, # BCE loss weight, should be 1.
    "phase2_penalty_weight": 0.1, # Weight for special penalties given on next token loss function (phase 2).
    "phase2_dt_weight": 1.0, # Weight loss on the abs_t prediction, which is combined with regular loss. Currently as calculated (phase 2).
    "phase2_dt_monotonic_penalty": 0.1, # Weight for penalties given on time MSE if predicted time is not monotonically increasing (phase 2).
}