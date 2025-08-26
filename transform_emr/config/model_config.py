import os

# Get project root (2 levels up from config/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Checkpoint paths
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, 'checkpoints')
EMBEDDER_CHECKPOINT = os.path.join(CHECKPOINT_PATH, 'phase1', 'best_embedder.pt')
TRANSFORMER_CHECKPOINT = os.path.join(CHECKPOINT_PATH, 'phase2', 'best_model.pt')

MODEL_CONFIG = {
      "ctx_dim": 20, # Fill manually once defined your context data.
      "time2vec_dim": 512,
      "embed_dim": 512,
      "block_size": 1536,  # //e.g. sequence length, number of tokens processed concurrently
      "n_head": 8,
      "n_layer": 8,
      "dropout": 0.1,
      "bias": True,
      "compile": True # Allows JIT compile for the model - Better memory and speed.
    }

TRAINING_SETTINGS = {
    "phase1_n_epochs": 100,
    "phase2_n_epochs": 100,
    "warmup_epochs": 3,
    "patience": 10,
    
    "phase1_learning_rate": 3e-4,
    "phase2_learning_rate": 5e-4,
    "weight_decay": 1e-3,
    
    "batch_size": 64, # Number of patients processed concurrently
    "bce_k_window": 12, # For soft targets per token on BCE loss, number of next tokens to predict jointly.
    
    # Phase-1 auxiliary settings
    "phase1_bce_weight": 1.0, # BCE loss weight, should be 1.
    "phase1_mlm_weight": 0.2, # MLM loss regulizer weight on the phase1 training task (= phase1_bce_weight / bce_k_window)
    "phase1_dt_weight": 0.1, # Weight for time regression loss component during phase 1
    
    # Phase-2 auxiliary settings
    "phase2_bce_weight": 1.0, # BCE loss weight, should be 1.
    "phase2_ce_weight": 0.05, # Cross-entropy loss weight, used as a nudge to the BCE.
    # Balance each penalty to be 20% - 30% of the BCE loss
    "phase2_penalty_weight": 0.15, # Weight for special penalties given on next token loss function (phase 2).
    "phase2_dt_weight": 1.0, # Weight loss on the abs_t prediction, which is combined with regular loss. Currently as calculated (phase 2).
}