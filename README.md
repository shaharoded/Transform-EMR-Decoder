# Event Prediction in EMRs

This repository implements a two-phase deep learning pipeline for modeling longitudinal Electronic Medical Records (EMRs). The architecture combines temporal embeddings, patient context, and Transformer-based sequence modeling to predict or impute patient events over time.

<img src="images\Model Sceme.png" width="100%">

This repo is part of an unpublished thesis and will be finalized post-submission. **Please do not reuse without permission**.

The results shown here (in `evaluation.ipynb`) are on random data, as my research dataset is private. This model will be used on actual EMR data, stored in a closed environment. For that, it is organized as a package that can be installed:

```bash
transform-emr/
│
├── transform_emr/                     # Core Python package
│   ├── config/                        # Configuration modules
│   │   ├── __init__.py
│   │   ├── tak-repo-portable.json     # TAKRepository object from Mediator (see related project)
│   │   ├── dataset_config.py
│   │   └── model_config.py
│   ├── __init__.py                    
│   ├── dataset.py                     # Dataset, DataPreprocess and Tokenizer
│   ├── embedder.py                    # Embedding model (EMREmbedding) + training
│   ├── transformer.py                 # Transformer architecture (GPT) + training
│   ├── train.py                       # Full training pipeline (3-phase)
│   ├── inference.py                   # Inference pipeline
│   ├── loss.py                        # Utility module for special loss criterias
│   ├── schedulers.py                  # Utility module for training schedulers (LR & Aux tasks)
│   ├── utils.py                       # Utility functions for the package (plots + penalties + masks)
│   └── diagnose.py                    # Debug reports on trained model health
├── data/                              # External data folder (for synthetic or real EMR)
│   ├── generate_synthetic_data.ipynb  # A notebook that generates synthetic data similar in structure to mediator's output (for tests)
│   ├── source/                        # Notebook will point here and auto-generate the train-test splits
│   ├── train/
│   └── test/
├── unittests/                         # Unit and integration tests (dataset / model / utils)
├── evaluation.ipynb                   # Main research and experiments notebook
├── README.md                         
├── .gitignore
├── requirements.txt
├── LICENCE
├── CITATION.cff
├── setup.py
└── pyproject.toml
```

As noted, this model feeds of the output of the [Mediator](https://github.com/shaharoded/Mediator) temporal abstraction engine. It can work with any temporal-interval dataset, but note that the embedding has knowledge-base component, so a `tak-repo-portable.json` like object is mandatory.

---

## 🛠️ Installation

Install the project as an editable package from the **root** directory:

```bash
pip install -e .

# Ensure your working directory is properly set to the root repo of this project
# Be sure to set the path in your local env properly.
```

---

## 🚀 Usage

### 1. Prepare Dataset and Update Config

```python
import pandas as pd
from transform_emr.dataset import EMRDataset
from transform_emr.config.dataset_config import *
from transform_emr.config.model_config import *

# Load data (verify you paths are properly defined)
temporal_df = pd.read_csv(TRAIN_TEMPORAL_DATA_FILE, low_memory=False)
ctx_df = pd.read_csv(TRAIN_CTX_DATA_FILE)

print(f"[Pre-processing]: Building tokenizer...")
processor = DataProcessor(temporal_df, ctx_df, tak_repo_path=TAK_REPO_PATH, scaler=None)
temporal_df, ctx_df = processor.run()

tokenizer = EMRTokenizer.from_processed_df(temporal_df)
train_ds = EMRDataset(train_df, train_ctx, tokenizer=tokenizer)
MODEL_CONFIG['ctx_dim'] = int(train_ds.context_df.shape[1]) # Dinamically updating shape
```

### 2. Train Model

```python
from transform_emr.train import run_training
run_training()
```

Model checkpoints are saved under `checkpoints/phase1/`, `checkpoints/phase2/`, and `checkpoints/phase3/`.
You can also run each phase individually by calling `prepare_data()`, `phase_one()`, `phase_two()`, and
`phase_three()` separately. All three phases use the same DataLoaders. See `train.py` for reference.

### 3. Inference and Complication Risk Prediction

The primary inference task is **complication risk prediction**: for each patient, generate a single
free-running trajectory and read the outcome head at every step to produce a probability curve per
complication over time. Use `generate_risk_curves` for this purpose.

```python
import joblib
from pathlib import Path
from transform_emr.embedder import EMREmbedding
from transform_emr.transformer import GPT
from transform_emr.dataset import DataProcessor, EMRTokenizer, EMRDataset
from transform_emr.inference import generate_risk_curves
from transform_emr.config.model_config import *

# Load tokenizer and scaler
tokenizer = EMRTokenizer.load(Path(CHECKPOINT_PATH) / "tokenizer.pt")
scaler = joblib.load(Path(CHECKPOINT_PATH) / "scaler.pkl")

# Preprocess test data, truncated to the same input window used during Phase-3 alignment
processor = DataProcessor(df, ctx_df, scaler=scaler, tak_repo_path=TAK_REPO_PATH, max_input_days=5)
df, ctx_df = processor.run()
dataset_input = EMRDataset(df, ctx_df, tokenizer=tokenizer)

# Load the best available checkpoint (Phase-3 if available, otherwise Phase-2)
embedder_model, *_ = EMREmbedding.load(PHASE1_CHECKPOINT, tokenizer=tokenizer)
p3_ckpt = Path(PHASE3_CHECKPOINT)
p2_ckpt = Path(PHASE2_CHECKPOINT)
ckpt_path = p3_ckpt if p3_ckpt.exists() else p2_ckpt
model, *_ = GPT.load(ckpt_path, embedder=embedder_model)
model.eval()

# Generate risk curves — one row per generated step, P_<outcome> columns per complication
risk_df = generate_risk_curves(model, dataset_input, max_len=500, temperature=1.0, rep_decay=0.6)
```

The returned `risk_df` has columns `{PatientId, Step, Token, IsInput, TimePoint, P_<outcome_name>, ...}`.
Rows with `IsInput == 0` are generated steps; the `P_*` columns hold sigmoid outcome-head probabilities
at that step. Evaluate using time-stratified AUC (see `evaluation.ipynb`).

`infer_event_stream` is also available for raw event generation (returns token stream without risk scores).


### 4. Using as a module

You can perform local tests (not unit-tests) by activating the `.py` files, using the module as a package, as long as the file you are activating has __main__ section.

For example, run this from the root:
```bash
python -m transform_emr.train

# Or

python -m transform_emr.inference

# Both modules have a __main__ activation to train / infer on a trained model
```
---

## 🧪 Running Unit-Tests

Run all tests:

Without validation prints:
```bash
python -m pytest unittests/
```

With validation prints:
```bash
python -m pytest -q -s unittests/
```

---

## 📦 Packaging Notes

To package without data/checkpoints:

```powershell
# Clean up any existing temp folder
Remove-Item -Recurse -Force .\transform_emr_temp -ErrorAction SilentlyContinue

# Recreate the temp folder
New-Item -ItemType Directory -Path .\transform_emr_temp | Out-Null

# Copy only what's needed
Copy-Item -Path .\transform_emr -Destination .\transform_emr_temp -Recurse
Copy-Item -Path .\setup.py, .\evaluation.ipynb, .\README.md, .\requirements.txt -Destination .\transform_emr_temp

# Remove __pycache__ folders (platform-specific bytecode, not for distribution)
Get-ChildItem -Path .\transform_emr_temp -Filter __pycache__ -Recurse -Directory | Remove-Item -Recurse -Force

# Zip it
Compress-Archive -Path .\transform_emr_temp\* -DestinationPath .\emr_model.zip -Force

# Clean up
Remove-Item -Recurse -Force .\transform_emr_temp
```

---

## 📌 Notes

- This project uses synthetic EMR data (`data/train/` and `data/test/`).
- For best results, ensure consistent preprocessing when saving/loading models.

---

## 🔄 End-to-End Workflow

Raw EMR Tables
│
▼
Per-patient Event Tokenization (with normalized absolute timestamps)
│
▼
🧠 Phase 1 – Train EMREmbedding (token + time + patient context)
│
▼
📚 Phase 2 – Pretrain a Transformer decoder over learned embeddings (next-token prediction + outcome auxiliary task).
│
▼
🎯 Phase 3 – Outcome Head Fine-tuning: freeze backbone, fine-tune only the outcome head on
             teacher-forced data (same DataLoaders as Phase 2), analogous to BERT head fine-tuning.
│
▼
→ Predict next medical events (token + time) and read complication risk curves from the outcome head (in `evaluation.ipynb`)

---

## 📦 Module Overview

### 1. **`dataset.py`** – Temporal EMR Preprocessing

| Component            | Role                                                                                             |
|---------------------|--------------------------------------------------------------------------------------------------|
| `DataProcessor`        | Performs all necessary data processing, from input data to tokens_df.  |
| `EMRTokenizer`        | Transforming a processed temporal_df into a tokenizer that can be saved and passed between objects for compatability.                     |
| `EMRDataset`        | Converts raw EMR tables into per-patient token sequences with relative time.                     |

| `collate_emr()`     | Pads sequences and returns tensors|

📌 **Why it matters:**  
Medical data varies in density and structure across patients. This dynamic preprocessing handles irregularity while preserving medically-relevant sequencing via `START/END` logic and relative timing.

>> This modules assumes the existance of prepared tak-repo-portable.json file, outputed from the Mediator as a hierarchy mapper of the different concepts.
---

### 2. **`embedder.py`** – EMR Representation Learning

| Component           | Role                                                                                              |
|--------------------|---------------------------------------------------------------------------------------------------|
| `Time2Vec`          | Learns periodic + trend encoding from inter-event durations.                                      |
| `EMREmbedding`      | Combines token, time, and patient context embeddings to create token representation.  |
| `train_embedder()`  | Trains the embedding model with teacher-forced next-token prediction (temporal BCE), with MSE on time prediction and MLM task as auxilary goal.                            |

⚙️ **Phase 1: Learning Events Representation**  
Phase 1 learns a robust, patient-aware representation of their event sequences. It isolates the core structure of patient timelines without being confounded by the autoregressive depth of Transformers.
The embedder uses:
- 4 levels of tokens - The event token is seperated to 4 hierarichal components to impose similarity between tokens of the same domain: `GLUCOSE` -> `GLUCOSE_TREND` -> `GLUCOSE_TREND_Inc` -> `GLUCOSE_TREND_Inc_START`
- 1 level of time - ABS T from ADMISSION, to understand global patterns and relationships between non sequential events.

This architecture constructs event representations by concatenating five hierarchical levels: Raw Concept, Concept, Value, Position, and Absolute Time. This creates a dense vector that captures the intrinsic hierarchy of medical concepts (e.g., Glucose_High is a child of Glucose) while explicitly binding them to their timestamp.

We choose concatenation (Early Fusion) for the temporal component-unlike the standard additive approach to preserve the integrity of the medical signal. By keeping the time dimensions separate from the concept dimensions in the input, the model can clearly distinguish the "what" from the "when". This ensures that the core identity of a pathology (e.g., Hyperglycemia) remains stable and recognizable ("Hyperglycemia is Hyperglycemia") regardless of its timing, while allowing the projection layer to learn how time modifies its clinical significance (e.g., Morning vs. Evening).

Context Handling To condition these embeddings on static patient attributes (e.g., Age, Sex), we project the patient context vector and **add** it to the event sequence. This acts as a global bias, shifting the entire event manifold into a patient-specific subspace. This ensures that even before the Transformer layers, the event representations are already calibrated to the patient's demographic risk profile. Since the inference output the context projection and event embedding separately, we use **context dropout** (passing p% of the trajectories with no context) so that the embedder will learn to work with / without it, while still pushing the context projection layer towards the shared latent space. 

The training uses next token prediction loss (temporal-window BCE) + time prediction MSE (Δt) + MLM prediction loss.
MLM will avoid masking tokens which will damage the broader meaning like ADMISSION, TERMINAL_OUTCOMES...

---

### 3. **`transformer.py`** – Causal Language Model over EMR Timelines

| Component           | Role                                                                                              |
|--------------------|---------------------------------------------------------------------------------------------------|
| `GPT`               | Transformer decoder stack over learned embeddings for next token prediction, with an additional head for delta_t prediction. Model inputs a trained embedder.                                               |
| `CausalSelfAttention` | Multi-head attention using causal mask to enforce chronology. Uses temporal RoPE to inject absolute time into attention scores.|
| `MLP` | SwiGLU MLP (SiLU Gating), based on common LLM optimizations.                                 |
| `AdaLNBlock` | Transformer block with AdaLN-Zero conditioning (adaptive layer norm), to bias prediction based on the patient context.                                 |
| `pretrain_transformer()` | Complete Phase-2 training logic using legality-masked temporal multi-hot BCE (focal), masked set-CE, Δt loss, and outcome BCE auxiliary losses.                         |
| `finetune_transformer()` | Phase-3 outcome head fine-tuning: freezes the backbone and fine-tunes only the outcome head on teacher-forced data (same DataLoaders as Phase 2), analogous to fine-tuning a BERT classification head. Uses the same soft-label targets as Phase 2 but with gradient isolation on the head only. Saves full-model checkpoints loadable with `GPT.load()`. |

⚙️ **Phase 2: Learning Sequence Dependencies**  
Once the EMR structure is captured, the transformer learns to model sequential dependencies in event progression:  
- What tends to follow a certain event?  
- How does timing affect outcomes?  
- How does patient context modulate the trajectory?

The training uses next token prediction loss (temporal-window masked BCE Focal Loss + masked CE loss) + time prediction MSE (Δt) + outcome prediction BCE auxillary task.
The training is guided by teacher's forcing, showing the model the correct context at every step (exposing [0, t-1] at step t from T where T is block_size), while also masking logits for illegal predictions based on the true trajectory. As training progress, the model's input ([0, t-1]) is partially masked (CBM) to teach the model to handle inaccuracies in generation, while avoiding masking same tokens as the EMREmbedding + MEAL + _START + _END tokens, to not clash with the legal set of next tokens to model can use.

The training flow uses warmup/curriculum scheduling (LR warmup, BCE-only phase, and staged auxiliary losses). The embedder is trainable during Phase 2, but updated with a lower learning rate than the transformer blocks.

---

### 4. **`inference.py`** – Generating output from the model

| Component           | Role                                                                                              |
|--------------------|---------------------------------------------------------------------------------------------------|
| `generate_risk_curves()` | **Primary inference function.** Generates one autoregressive trajectory per patient and reads the outcome head at every step, returning a DataFrame of per-step complication probabilities. |
| `infer_event_stream()` | Generates a predicted stream of tokens without risk scores. Useful for inspecting which events the model predicts, independent of outcome probabilities. |
| `get_token_embedding()` | Returns the embedding vector of a specific token from a trained embedder. |
| `_build_illegal_mask()` | Builds a Boolean `[V]` mask of token ids that are structurally illegal to generate next, given the current interval and meal-order state. |
| `_update_legality_state()` | Mutates interval open-counts and advances meal-order rank after each generated token. |
| `_decode_token_components()` | Decodes a position-token string into `(concept_id, value_id)` for feeding back into the model. |

NOTE: Inference is step-by-step (not batched), so it is significantly slower than training.
---

### 5. **`evaluation.ipynb`** – Risk-based complication prediction evaluation.

Runs the full end-to-end evaluation pipeline: data loading, three-phase training, risk-curve generation,
and statistical analysis. The primary evaluation metric is time-stratified AUC.

| Component           | Role                                                                                              |
|--------------------|---------------------------------------------------------------------------------------------------|
| `extract_ground_truth()` | Builds a `{patient_id → {outcome → first_occurrence_hours}}` dict from the full (untruncated) test dataset. |
| `time_stratified_auc()` | At each 24 h window: score = max outcome-head probability in window, label = complication occurred in same window. Computes AUROC and AUPRC per complication, averaged across windows. |
| `time_accuracy()` | For patients where a complication occurred: MAE between the generated step with peak probability and actual onset time. |
| `calibrate_temperature()` | Learns a per-outcome temperature scalar via LBFGS (NLL minimisation). Does not affect rank order (AUC unchanged); improves probability calibration for direct interpretation. |
| `reliability_diagram()` | Plots calibration curves before and after temperature scaling. |

---

## ✅ Model Capabilities

- ✔️ **Handles irregular time-series data** using relative deltas and Time2Vec.
- ✔️ **Captures both short- and long-range dependencies** with deep transformer blocks.
- ✔️ **Supports variable-length patient histories** using custom collate and attention masks.
- ✔️ **Imputes and predicts** events in structured EMR timelines.

---

## 📚 Citation & Acknowledgments

This work builds on and adapts ideas from the following sources:

- **Time2Vec** (Kazemi et al., 2019):  
  The temporal embedding design is adapted from the Time2Vec formulation.  
  📄 *A. Kazemi, S. Ghamizi, A.-H. Karimi. "Time2Vec: Learning a Vector Representation of Time." NeurIPS 2019 Time Series Workshop.*  
  [arXiv:1907.05321](https://arxiv.org/abs/1907.05321)

- **nanoGPT** (Karpathy, 2023):  
  The training loop and transformer backbone are adapted from [nanoGPT](https://github.com/karpathy/nanoGPT),  
  with modifications for multi-stream EMR inputs, multiple embeddings, and a k-step prediction loss.

- **RoPE / RoFormer** (Su et al., 2021):  
  The attention module uses rotary position embeddings adapted to continuous/absolute timestamps (temporal RoPE) to inject time into Q/K rotations.  
  📄 *J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen. "RoFormer: Enhanced Transformer with Rotary Position Embedding." arXiv:2104.09864.*  
  [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)

- **AdaLN-Zero** (Peebles, W., & Xie, S., 2023):  
  Inspired by the paper "Scalable diffusion models with transformers", I added a customized block to the transformer designed to allow static context influence all generation steps. The [paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.pdf) uses this method to inform the diffusion model of the label of the image it should generate.

And more...
