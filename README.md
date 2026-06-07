# INTERVenE-ar — Autoregressive EMR Trajectory Generation with Outcome Risk Curves

**INTERVenE** (**INTERV**al-based **E**MR transformer) is a model family for outcome prediction from temporal abstractions. The family has two variants — **INTERVenE-ar** (autoregressive trajectory generation) and **INTERVenE-enc** (bidirectional encoder for joint risk + time-to-event prediction) — sharing a common backbone of hierarchical interval tokens, temporal RoPE, and AdaLN-Zero patient conditioning. This repository hosts the **INTERVenE-ar** variant.

This repository implements a three-phase deep learning pipeline for modeling longitudinal Electronic Medical Records (EMRs). The architecture combines temporal embeddings, patient context, and Transformer-based sequence modeling to predict or impute patient events over time, and to read complication-risk curves from a dedicated outcome head.

<img src="images\Model Sceme.png" width="100%">

This repo is part of an unpublished thesis and will be finalized post-submission. **Please do not reuse without permission**.

The results shown here (in `evaluation.ipynb`) are on random data, as my research dataset is private. This model will be used on actual EMR data, stored in a closed environment. For that, it is organized as a package that can be installed:

```bash
root/
│
├── intervene_ar/                     # Core Python package
│   ├── config/                        # Configuration modules
│   │   ├── __init__.py
│   │   ├── tak-repo-portable.json     # TAKRepository object from Mediator (see related project)
│   │   ├── dataset_config.py
│   │   └── model_config.py
│   ├── __init__.py                    
│   ├── dataset.py                     # Dataset, DataPreprocess and Tokenizer
│   ├── embedder.py                    # Embedding model (EMREmbedding) + training
│   ├── transformer.py                 # Transformer architecture (InterveneGPT) + training
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
├── evaluation.ipynb                   # Self-contained eval notebook — patient-level AUC/F1, peak MAE, length-of-stay, calibration & trajectory plots
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
from intervene_ar.dataset import EMRDataset
from intervene_ar.config.dataset_config import *
from intervene_ar.config.model_config import *

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
from intervene_ar.train import run_training
embedder, model_p2, model_p3, test_raw = run_training()
```

`run_training()` follows a strict train / val / test contract:

- **Three-way patient split**: train / val / test. The test split is held out and never seen during training or early-stop selection — it is consumed only by `evaluation.ipynb` for headline metrics.
- **Scaler is fit on train** (saved to `checkpoints/scaler.pkl`) and reused on val/test.
- **Tokenizer** is built once from train and cached at `checkpoints/tokenizer.pt`.
- **Phase 1 caching**: when `(embed_dim, time2vec_dim, ctx_dim)` match the cached Phase-1 checkpoint, the embedder is reused and Phase 1 is skipped — Phase 2/3 are always retrained on each call.
- **DataLoaders**: Phase 1 + Phase 3 use bucket-batched natural distribution; Phase 2 uses a weighted bucket sampler so rare outcomes get balanced exposure (`pos_weight` is omitted there because the sampler already rebalances).

Model checkpoints are saved under `checkpoints/phase1/`, `checkpoints/phase2/`, and `checkpoints/phase3/`. You can also call `prepare_data()` directly and run individual phases; see `train.py` for reference.

### 3. Inference and Complication Risk Prediction

The primary inference task is **complication risk prediction**: for each patient, generate a single
free-running trajectory and read the outcome head at every step to produce a probability curve per
complication over time. Use `generate` with `collect_risk_scores=True` for this purpose.

```python
import joblib
from pathlib import Path
from intervene_ar.embedder import EMREmbedding
from intervene_ar.transformer import InterveneGPT
from intervene_ar.dataset import DataProcessor, EMRTokenizer, EMRDataset
from intervene_ar.inference import generate
from intervene_ar.config.model_config import *

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
model, *_ = InterveneGPT.load(ckpt_path, embedder=embedder_model)
model.eval()

# Generate risk curves — one row per generated step, P_<outcome> columns per complication.
# Default `max_duration_hours=336.0` caps each trajectory at 14 days; `collect_risk_scores`
# defaults to False (faster) so pass True to read the outcome head per step.
risk_df = generate(model, dataset_input, max_duration_hours=336.0, max_len=500,
                   temperature=1.0, rep_decay=0.6, collect_risk_scores=True)

# Raw event stream only (no risk scores, faster)
event_df = generate(model, dataset_input, max_duration_hours=336.0, max_len=500,
                    temperature=1.0, rep_decay=0.6)
```

The returned `risk_df` has columns `{PatientId, Step, Token, IsInput, IsOutcome, IsTerminal, TimePoint, P_<outcome_name>, ...}`.
Rows with `IsInput == 0` are generated steps; the `P_*` columns hold sigmoid outcome-head probabilities
at that step. Evaluate using time-stratified AUC (see `evaluation.ipynb`).

Patients that exhaust `max_len` without generating a terminal token receive a forced DEATH or RELEASE
token (chosen by highest logit), clamped to <= 336 h. The fallback rate is printed after generation.


### 4. Using as a module

You can perform local tests (not unit-tests) by activating the `.py` files, using the module as a package, as long as the file you are activating has __main__ section.

For example, run this from the root:
```bash
python -m intervene_ar.train

# Or

python -m intervene_ar.inference

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
Remove-Item -Recurse -Force .\intervene_ar_temp -ErrorAction SilentlyContinue

# Recreate the temp folder
New-Item -ItemType Directory -Path .\intervene_ar_temp | Out-Null

# Copy only what's needed
Copy-Item -Path .\intervene_ar -Destination .\intervene_ar_temp -Recurse
Copy-Item -Path .\setup.py, .\evaluation.ipynb, .\README.md, .\requirements.txt -Destination .\intervene_ar_temp

# Remove __pycache__ folders (platform-specific bytecode, not for distribution)
Get-ChildItem -Path .\intervene_ar_temp -Filter __pycache__ -Recurse -Directory | Remove-Item -Recurse -Force

# Zip it
Compress-Archive -Path .\intervene_ar_temp\* -DestinationPath .\intervene_ar.zip -Force

# Clean up
Remove-Item -Recurse -Force .\intervene_ar_temp
```

---

## 📌 Notes

- This project uses synthetic EMR data (`data/train/` and `data/test/`).
- For best results, ensure consistent preprocessing when saving/loading models.

---

## 🔄 End-to-End Workflow

```text
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
             natural-distribution batches (oversample=False + pos_weight), analogous to BERT head fine-tuning.
│
▼
→ Predict next medical events (token + time) and read complication risk curves from the outcome head (in `evaluation.ipynb`)
```

---

## 📦 Module Overview

### 1. **`dataset.py`** – Temporal EMR Preprocessing

| Component            | Role                                                                                             |
|---------------------|--------------------------------------------------------------------------------------------------|
| `DataProcessor`        | Performs all necessary data processing, from input data to tokens_df.  |
| `EMRTokenizer`        | Builds vocabulary and per-outcome prevalence ratios from a processed temporal_df; filters outcomes below `OUTCOME_RARE_THRESHOLD_PCT`; saves/loads with `BucketBatchSampler` / `WeightedBucketBatchSampler` support. |
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
| `train_embedder()`  | Phase-1 training. Loss = temporal next-token BCE (multi-hot over a future window) + Δt MSE auxiliary (joined once a single-stage scheduler lifts it after a BCE-only warmup). MLM has been removed in favour of a cleaner BCE+Δt curriculum. |

⚙️ **Phase 1: Learning Events Representation**  
Phase 1 learns a robust, patient-aware representation of their event sequences. It isolates the core structure of patient timelines without being confounded by the autoregressive depth of Transformers.
The embedder uses:
- 4 levels of token components - The event token is split into 4 hierarchical components to impose similarity between tokens of the same domain: `GLUCOSE` -> `GLUCOSE_TREND` -> `GLUCOSE_TREND_Inc` -> `GLUCOSE_TREND_Inc_START`
- 1 level of time - ABS T from ADMISSION, to understand global patterns and relationships between non sequential events.

This architecture constructs event representations by concatenating these five hierarchical levels: Raw Concept, Concept, Value, Position, and Absolute Time. This creates a dense vector that captures the intrinsic hierarchy of medical concepts (e.g., Glucose_High is a child of Glucose) while explicitly binding them to their timestamp.

We choose concatenation (Early Fusion) for the temporal component-unlike the standard additive approach to preserve the integrity of the medical signal. By keeping the time dimensions separate from the concept dimensions in the input, the model can clearly distinguish the "what" from the "when". This ensures that the core identity of a pathology (e.g., Hyperglycemia) remains stable and recognizable ("Hyperglycemia is Hyperglycemia") regardless of its timing, while allowing the projection layer to learn how time modifies its clinical significance (e.g., Morning vs. Evening).

Context Handling To condition these embeddings on static patient attributes (e.g., Age, Sex), we project the patient context vector and **add** it to the event sequence. This acts as a global bias, shifting the entire event manifold into a patient-specific subspace. This ensures that even before the Transformer layers, the event representations are already calibrated to the patient's demographic risk profile. Since the inference output the context projection and event embedding separately, we use **context dropout** (passing p% of the trajectories with no context) so that the embedder will learn to work with / without it, while still pushing the context projection layer towards the shared latent space. 

The training uses next-token prediction loss (temporal-window BCE) + time-delta MSE (Δt) auxiliary. Δt is held back behind a BCE-only warmup, then unlocked once Phase-1 has a meaningful main signal; its λ is calibrated once from the loss ratio at unlock and capped at a fraction of BCE so it never dominates. The legacy MLM auxiliary was removed — CBM (curriculum-by-masking, applied during Phase 2 over interval-atomic pairs) covers the same robustness need without adding a separate head.

---

### 3. **`transformer.py`** – Causal Language Model over EMR Timelines

| Component           | Role                                                                                              |
|--------------------|---------------------------------------------------------------------------------------------------|
| `InterveneGPT`               | Transformer decoder stack over learned embeddings for next token prediction, with an additional head for delta_t prediction. Model inputs a trained embedder.                                               |
| `CausalSelfAttention` | Multi-head attention using causal mask to enforce chronology. Uses temporal RoPE to inject absolute time into attention scores.|
| `MLP` | SwiGLU MLP (SiLU Gating), based on common LLM optimizations.                                 |
| `AdaLNBlock` | Transformer block with AdaLN-Zero conditioning (adaptive layer norm), to bias prediction based on the patient context.                                 |
| `pretrain_transformer()` | Phase-2 training loop. Main loss: legality-masked temporal multi-hot BCE (next-token) with a **soft-kernel terminal extension** — terminal-token BCE uses a learnable per-class decay constant (`model.log_tau_lm`) over a wider hard horizon (`phase2_terminal_bce_window_hours`), giving DEATH/RELEASE positive gradient pre-event instead of only the immediate next step. Auxiliaries scheduled in two stages: stage-0 = `ce` (masked set-CE), `dt` (Δt MSE), `ttt` (time-to-terminal MSE on `log1p` hours, direction C); stage-1 unlocks `ranking` (pairwise AUROC-proxy on the outcome head) once stage-0 plateaus. Each λ is calibrated once from the loss ratio at activation and capped at its `aux_fraction_caps` share of BCE. Outcome head uses **time-decayed soft labels** with a per-outcome learnable τ (`outcome_log_tau`) and a hard horizon `outcome_horizon_hours`. |
| `finetune_transformer()` | Phase-3 outcome head + pool head fine-tune. The backbone is held at a tiny LR (`phase3_backbone_lr_factor`, default 0.01) so head gradients can still nudge it; the outcome head trains at full `phase3_learning_rate`. Natural-distribution batches (`oversample=False`) so `pos_weight` in `BCEWithLogitsLoss` correctly compensates for class imbalance without double-counting. A **patient-level attention pool head** (per-outcome learnable queries cross-attending over backbone hidden states) is added as an auxiliary: BCE against patient-level "outcome k appears anywhere in the trajectory" labels, λ calibrated once at the end of epoch 1 and capped at `phase3_pool_fraction_cap` of raw outcome BCE. Saves full-model checkpoints loadable with `InterveneGPT.load()`. |

⚙️ **Phase 2: Learning Sequence Dependencies**  
Once the EMR structure is captured, the transformer learns to model sequential dependencies in event progression:  
- What tends to follow a certain event?  
- How does timing affect outcomes?  
- How does patient context modulate the trajectory?

Phase-2 loss breakdown:

- **Next-token BCE** (main): temporal-window multi-hot with legality masking. Terminal tokens use a soft-kernel extension with learnable per-class τ over a wider hard horizon — gives DEATH/RELEASE training signal pre-event, not only on the immediate-next position.
- **Δt MSE** (aux, stage 0): time-delta regression on the Time2Vec head.
- **Time-to-terminal MSE** (`ttt`, aux, stage 0): regress `log1p(hours-to-next-terminal)` at every non-terminal, non-pad query position. Independent length-prediction signal — the model learns admission horizon irrespective of which terminal lands.
- **Masked set-CE** (`ce`, aux, stage 0): light next-token CE nudge over the legal set.
- **Outcome BCE** (always on): time-decayed soft labels over the per-outcome head; per-outcome learnable τ; hard horizon `outcome_horizon_hours`.
- **Pairwise ranking** (aux, stage 1): AUROC-proxy ranking loss on the outcome head, unlocked only after stage-0 plateaus so the head has meaningful logits to rank.

Training is teacher-forced ([0, t-1] visible at step t), with illegal-prediction logits masked from the true trajectory. CBM (curriculum-by-masking) atomically masks interval-pair tokens (START + matching END together) on the input side to teach the model to handle generation-time corruption without breaking interval legality. The embedder remains trainable during Phase 2 at a lower LR than the transformer blocks. A OneCycleLR schedule drives LR warmup; auxiliary λs follow the staged scheduler above.

---

⚙️ **Phase 3: Outcome Head Alignment**
Phase 2 trains the outcome head under teacher-forced input — but at inference the head is read off free-running generated trajectories, creating a teacher-forcing → autoregressive distribution gap. Phase 3 closes that gap:

- Backbone held at `phase3_lr × phase3_backbone_lr_factor` (default 1e-6) — head gradients still nudge it, but it never overfits to the outcome objective at the cost of LM quality.
- Outcome head trains at full `phase3_learning_rate`.
- Natural-distribution batches (`oversample=False`) so `pos_weight` handles imbalance without double counting.
- A **patient-level attention pool head** (P4): per-outcome learnable queries cross-attend over the backbone's final hidden states to produce one pooled feature per (patient, outcome); scalar projection turns it into a patient-level logit; BCE against patient_label[b, k] = "outcome k appears anywhere in the GT trajectory". λ_pool is calibrated once at the end of epoch 1 and capped at `phase3_pool_fraction_cap` of raw outcome BCE — protects the per-step outcome head from patient-level coarseness.

---

### 4. **`inference.py`** – Generating output from the model

| Component           | Role                                                                                              |
|--------------------|---------------------------------------------------------------------------------------------------|
| `generate()` | **Primary inference function.** Generates one autoregressive trajectory per patient. With `collect_risk_scores=True`, reads the outcome head at every step and returns per-step complication probabilities (`P_*` columns). Patients that reach `max_len` receive a forced terminal token (DEATH or RELEASE by highest logit), clamped to <= 336 h. |
| `get_token_embedding()` | Returns the embedding vector of a specific token from a trained embedder. |

NOTE: Inference is step-by-step (autoregressive), so it is significantly slower than training. With that being said, model uses batch inference (multiple patients at the same time), KV cache (reduces per-step work from O(T·d²) to O(d²)) and FP16 quantization, all together significantly helps the inference speed.
---

### 5. **`evaluation.ipynb`** – Self-contained complication-risk evaluation.

End-to-end evaluation on the held-out test split: re-process raw test data with the fitted scaler, build a 2-day truncated seed dataset, generate one autoregressive trajectory per patient (with `collect_risk_scores=True`), then score.

Headline framing is **patient-level peak-detector AUC + F1**: each (patient, outcome) contributes a single (max_P over generated positions, label = did the outcome ever occur in GT) pair. Stable on rare outcomes — no per-window count-noise amplification. RELEASE_EVENT is excluded from the AUC headline (it is the negation of DEATH in this cohort, so reporting both double-counts the same ranking task) and reported separately via length-of-stay regression.

| Component           | Role                                                                                              |
|--------------------|---------------------------------------------------------------------------------------------------|
| `extract_ground_truth()` / `extract_ground_truth_episodes()` | First-occurrence and all-episode GT extracted from the untruncated test set. |
| `extract_patient_horizons()` | Per-patient eval horizon = min(last GT event, 336 h cap). |
| `per_patient_max_auc()` | **Headline**. Per-(patient, outcome) (max_P, label) pair; AUROC, AUPRC, max-F1 (sweep PR curve), F1@0.5. |
| `weighted_mean_auc()` | Support-weighted (by `n_pos`) mean across outcomes — rare outcomes contribute less. |
| `time_accuracy_nearest()` | Peak-time MAE to the **nearest** GT occurrence (fair when complications recur). |
| `length_of_stay_mae()` | Trajectory-length regression: admission → end of (input+generated) vs GT release time. Replaces the RELEASE peak-MAE. |
| `compute_gen_stats()` | Trajectory-collapse diagnostics: gen vs GT span ratios, terminal-first-24h fraction, length MAE. |
| `calibrate_temperature()` | Per-outcome temperature scalar via LBFGS — does not change rank order; improves probability calibration. |
| `reliability_diagram()` | Before/after calibration curves per outcome. |
| Sample-patient plot | Per-patient risk trajectory (one line per outcome) with input/generation boundary and GT-event markers. |

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
