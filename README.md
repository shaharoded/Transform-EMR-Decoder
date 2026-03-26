# Event Prediction in EMRs

This repository implements a two-phase deep learning pipeline for modeling longitudinal Electronic Medical Records (EMRs). The architecture combines temporal embeddings, patient context, and Transformer-based sequence modeling to predict or impute patient events over time.

<img src="images\Model Sceme.png" width="100%">

This repo is part of an unpublished thesis and will be finalized post-submission. **Please do not reuse without permission**.

The results shown here (in `evaluation.ipynb`) are on random data, as my research dataset is private. This model will be used on actual EMR data, stored in a closed environment. For that, it is organized as a package that can be installed:

```bash
event-prediction-in-diabetes-care/
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
│   ├── train.py                       # Full training pipeline (2-phase)
│   ├── inference.py                   # Inference pipeline
│   ├── loss.py                        # Utility module for special loss criterias
│   ├── schedulers.py                  # Utility module for auxillary lambda schedulers
│   ├── utils.py                       # Utility functions for the package (plots + penalties)
│   └── debug_tools.py                 # Debug loop for epochs (logits)
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

As noted, this model feeds of the output of the [Mediator](https://github.com/shaharoded/Mediator) temporal abstraction engine.

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
MODEL_CONFIG['ctx_dim'] = train_ds.context_df.shape[1] # Dinamically updating shape
```

### 2. Train Model

```python
from transform_emr.train import run_two_phase_training
run_two_phase_training()
```

Model checkpoints and scaler are saved under `checkpoints/phase1/` and `checkpoints/phase2/`.
You can also split this part to it's components, running the prepare_data(), phase_one(), phase_two() seperatly,
but you'll need to adjust the imports. Use `train.py` structure for that.

### 3. Inference from the Model

```python
    import random
    import joblib
    from pathlib import Path
    from transform_emr.embedder import EMREmbedding
    from transform_emr.transformer import GPT
    from transform_emr.dataset import DataProcessor, EMRTokenizer, EMRDataset
    from transform_emr.config.model_config import *

    # Load test data
    df = pd.read_csv(TEST_TEMPORAL_DATA_FILE, low_memory=False)
    ctx_df = pd.read_csv(TEST_CTX_DATA_FILE)

    # Load tokenizer and scaler
    tokenizer = EMRTokenizer.load(Path(CHECKPOINT_PATH) / "tokenizer.pt")
    scaler = joblib.load(Path(CHECKPOINT_PATH) / "scaler.pkl")

    # Run preprocessing
    processor = DataProcessor(df, ctx_df, scaler=scaler, tak_repo_path=TAK_REPO_PATH, max_input_days=5)
    df, ctx_df = processor.run()

    patient_ids = df["PatientID"].unique()
    df_subset = df[df["PatientID"].isin(patient_ids)].copy()
    ctx_subset = ctx_df.loc[patient_ids].copy()

    # Create dataset
    dataset = EMRDataset(df_subset, ctx_subset, tokenizer=tokenizer)

    # Load models
    embedder, _, _, _, _ = EMREmbedding.load(EMBEDDER_CHECKPOINT, tokenizer=tokenizer)
    model, _, _, _, _ = GPT.load(TRANSFORMER_CHECKPOINT, embedder=embedder)
    model.eval()

    # Run inference
    result_df = infer_event_stream(model, dataset, temperature=1.0)  # optional: adjust temperature
```

This results_df will include both input events and generated events and will have these columns:
{"PatientID", "Step", "Token", "IsInput", "IsOutcome", "IsTerminal", "TimePoint"}

You can analize the model's performance by comparing the input (`dataset.tokens_df`) to the output:
 - Were all complications generated?
 - Were all complications generated on time? (Set a forgiving boundry like 24h window)


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
pytest unittests/
```

With validation prints:
```bash
pytest -q -s unittests/
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
- `model_config.py: MODEL_CONFIG.ctx_dim` should only be updated **after** dataset initialization to avoid embedding size mismatches. You should update this value with your full context dimention (without PatientID idx).

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
📚 Phase 2 – Pre-train a Transformer decoder over learned embeddings, as a next-token-prediction task.
│
▼
→ Predict next medical events (token + time) and deduce outcome predictions from them (in `evaluation.ipynb`)

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

>> This modules assumes the existance of prepared tak_repo.pkl file, outputed from the Mediator as a hierarchy mapper of the different concepts.
---

### 2. **`embedder.py`** – EMR Representation Learning

| Component           | Role                                                                                              |
|--------------------|---------------------------------------------------------------------------------------------------|
| `Time2Vec`          | Learns periodic + trend encoding from inter-event durations.                                      |
| `EMREmbedding`      | Combines token, time, and patient context embeddings. Adds `[CTX]` token for global patient info. |
| `train_embedder()`  | Trains the embedding model with teacher-forced next-token prediction.                            |

⚙️ **Phase 1: Learning Events Representation**  
Phase 1 learns a robust, patient-aware representation of their event sequences. It isolates the core structure of patient timelines without being confounded by the autoregressive depth of Transformers.
The embedder uses:
- 4 levels of tokens - The event token is seperated to 4 hierarichal components to impose similarity between tokens of the same domain: `GLUCOSE` -> `GLUCOSE_TREND` -> `GLUCOSE_TREND_Inc` -> `GLUCOSE_TREND_Inc_START`
- 1 level of time - ABS T from ADMISSION, to understand global patterns and relationships between non sequential events.

This architecture constructs event representations by concatenating five hierarchical levels: Raw Concept, Concept, Value, Position, and Absolute Time. This creates a dense vector that captures the intrinsic hierarchy of medical concepts (e.g., Glucose_High is a child of Glucose) while explicitly binding them to their timestamp.

We choose concatenation (Early Fusion) for the temporal component-unlike the standard additive approach to preserve the integrity of the medical signal. By keeping the time dimensions separate from the concept dimensions in the input, the model can clearly distinguish the "what" from the "when". This ensures that the core identity of a pathology (e.g., Hyperglycemia) remains stable and recognizable ("Hyperglycemia is Hyperglycemia") regardless of its timing, while allowing the projection layer to learn how time modifies its clinical significance (e.g., Morning vs. Evening).

Context Handling To condition these embeddings on static patient attributes (e.g., Age, Sex), we project the patient context vector and **add** it to the event sequence. This acts as a global bias, shifting the entire event manifold into a patient-specific subspace. This ensures that even before the Transformer layers, the event representations are already calibrated to the patient's demographic risk profile. Since the inference output the context projection and event embedding separately, we use **context dropout** (passing p% of the trajectories with no context) so that the embedder will learn to work with / without it, while still pushing the context projection layer towards the shared latent space. 

The training uses next token prediction loss (k-window BCE) + time prediction MSE (Δt) + MLM prediction loss.
MLM will avoid masking tokens which will damage the broader meaning like ADMISSION, TERMINAL_OUTCOMES...

---

### 3. **`transformer.py`** – Causal Language Model over EMR Timelines

| Component           | Role                                                                                              |
|--------------------|---------------------------------------------------------------------------------------------------|
| `GPT`               | Transformer decoder stack over learned embeddings for next token prediction, with an additional head for delta_t prediction. Model inputs a trained embedder.                                               |
| `CausalSelfAttention` | Multi-head attention using causal mask to enforce chronology.                                 |
| `MLP` | SwiGLU MLP (SiLU Gating), based on common LLM optimizations.                                 |
| `AdaLNBlock` | Transformer block with AdaLN-Zero conditioning (adaptive layer norm), to bias prediction based on the patient context.                                 |
| `train_transformer()` | Complete training logic for the model using a BCE with multi-hot targets to account for EMR irregularities.                         |

⚙️ **Phase 2: Learning Sequence Dependencies**  
Once the EMR structure is captured, the transformer learns to model sequential dependencies in event progression:  
- What tends to follow a certain event?  
- How does timing affect outcomes?  
- How does patient context modulate the trajectory?

The training uses next token prediction loss (k-window masked BCE Focal Loss) + time prediction MSE (Δt) + structural penalties + outcome prediction BCE auxillary task.
The training is guided by teacher's forcing, showing the model the correct context at every step (exposing [0, t-1] at step t from T where T is block_size), while also masking logits for illegal predictions based on the true trajectory. As training progress, the model's input ([0, t-1]) is partially masked (CBM) to teach the model to handle inaccuracies in generation, while avoiding masking same tokens as the EMREmbedding + MEAL + _START + _END tokens, to not clash with the penalties the model recieves.

The training flow uses a warmup period where the model is to learn patterns using a frozen embedder (so that the sharp gradients won't cause forgetting to the embedder's weights).

---

### 4. **`inference.py`** – Generating output from the model

| Component           | Role                                                                                              |
|--------------------|---------------------------------------------------------------------------------------------------|
| `get_token_embedding()` | Select a token and get it's embeddings based on an input embedder.                                 |
| `infer_event_stream()` | Generate predicted stream of events on an input dataset (Test), using a masking process to block prediction of illegal tokens in relation to the predictions so far.                         |


NOTE: Unlike the parallel batching in the training process, inference on the transformer is step-by-step, hence slow (especially with the updating of illegal tokens on the fly).
---

### 5. **`evaluation.ipynb`** – Evaluation of the model's performance based on dynamic activations of `inference.py`.

| Component           | Role                                                                                              |
|--------------------|---------------------------------------------------------------------------------------------------|
| `evaluate_events` | Calculates full classification evaluation methods given gold-standard DataFrame and generated DataFrame.                                 |
| `evaluate_across_k` | Handles Inference + Evaluation from pre-trained model across all K.                 |
| `plot_metrics_trend` | Plots global evaluation over K. |
| `build_3x_matrix` | Was the model able to predict a future RELEASE / COMPLICATION EATH?.                                 |
| `build_full_outcome_matrix` | Was the model able to predict a future **specific** OUTCOME (from `dataset.config`).                         |
| `build_timeaware_matrix` | Was the model able to predict a future **specific** OUTCOME (from `dataset.config`) at the correct time?                         |

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

- **AdaLN-Zero** (Peebles, W., & Xie, S., 2023):  
  Inspired by the paper "Scalable diffusion models with transformers", I added a customized block to the transformer designed to allow static context influence all generation steps. The [paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.pdf) uses this method to inform the diffusion model of the label of the image it should generate.

And more...
