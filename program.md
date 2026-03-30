# autoresearch — EMR Event Prediction

Autonomous hyperparameter and architecture search on an EMR next-event prediction model.

---

## Background

The model learns to generate the future stream of medical events for a hospital patient given their history. It uses MIMIC-III derived data containing diabetes patients' longitudinal events (lab results, vitals, diagnoses, medications, meals, outcomes such as complications and death).

The architecture is a two-phase pipeline:

- **Phase 1 — EMREmbedding**: learns a compact, time-aware representation of each clinical event using hierarchical token embeddings, Time2Vec, and patient context.
- **Phase 2 — GPT Transformer**: a causal decoder trained over the Phase-1 embeddings to predict the next clinical event in a patient's timeline.

The key clinical targets are *complications* (15 outcome types like `KIDNEY_COMPLICATION_EVENT`, `CARDIO-VASCULAR_DISORDER_EVENT`, etc.). The model must predict both *what* will happen and *when*.

---

## Setup

To set up a new experiment run:

1. **Agree on a run tag** with the user (e.g. `apr1`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>`.
3. **Read all in-scope files** for full context:
   - `README.md` — repository overview.
   - `prepare.py` — fixed: data loading, evaluation metric. Do NOT modify.
   - `train.py` — the file you will modify. Model config, training settings, training loop.
   - `emr_model/transform_emr/embedder.py` — Phase-1 embedding model.
   - `emr_model/transform_emr/transformer.py` — Phase-2 GPT model.
   - `emr_model/transform_emr/loss.py` — loss functions.
   - `emr_model/transform_emr/schedulers.py` — auxiliary loss scheduling.
4. **Verify data exists**: `emr_model/data/source/temporal_events.csv` and `context_data.csv` must exist.
5. **Initialize results.tsv**: create it with only the header row (see format below). The baseline will be recorded after the first run.
6. **Confirm and go**.

---

## Experimentation

Each experiment:
1. Edits `train.py` (and optionally `emr_model/transform_emr/*.py`).
2. Commits the change.
3. Runs training.
4. Records the result.
5. Keeps or reverts.

**What you CAN modify:**
- `train.py` — the primary edit target. Change `MODEL_CONFIG`, `TRAINING_SETTINGS`, loss schedules, or any training logic.
- `emr_model/transform_emr/*.py` — architecture changes (attention mechanism, MLP design, normalization, etc.).

**What you CANNOT modify:**
- `prepare.py` — this is the fixed ground truth: data loading and evaluation metric.
- `emr_model/data/` — the training data is fixed.

**The goal**: minimize `val_bce_loss` (validation cross-entropy on next-event prediction). Lower is better.

**Memory**: keep peak VRAM reasonable. A large increase in memory for a small improvement is not worth it.

**Simplicity criterion**: all else being equal, simpler is better. A small gain with lots of new code is suspect. Removing code while maintaining performance is always a win.

**The first run** must always be the unmodified baseline — run `train.py` as-is first.

---

## Running an experiment

```bash
python train.py > run.log 2>&1
```

Each run trains both phases from scratch (Phase-1 + Phase-2). Typical run time depends on hardware. With the default config on a modern GPU, expect roughly 20–60 minutes per experiment (epoch budgets enforce a ceiling).

**Extract the result:**

```bash
grep "^val_bce_loss:\|^peak_vram_mb:" run.log
```

If this returns nothing, the run crashed. Inspect the error:

```bash
tail -n 50 run.log
```

**Timeout**: if a run has not printed the `---` summary after 90 minutes, kill it and treat as a crash.

---

## Output format

A successful run ends with a block like:

```
---
val_bce_loss:      3.124500
phase2_best_val:  2.980100
phase2_epochs:    23
total_seconds:    2847.3
peak_vram_mb:     4312.0
embed_dim:        64
n_layer:          4
n_head:           4
block_size:       512
num_params:       1,245,632
```

The primary metric is `val_bce_loss` (lower is better).

---

## Logging results

Log every completed experiment (crash, keep, or discard) to `results.tsv`.
This file is **not committed** (it is gitignored); it accumulates across the whole session.

The TSV has a header row and 5 columns (tab-separated — no commas in descriptions):

```
commit	val_bce_loss	memory_gb	status	description
```

Columns:
1. Short git commit hash (7 chars)
2. `val_bce_loss` value (e.g. `3.124500`) — use `0.000000` for crashes
3. Peak VRAM in GB, 1 decimal (e.g. `4.2` — divide peak_vram_mb by 1024) — use `0.0` for crashes
4. Status: `KEEP`, `DISCARD`, or `CRASH`
5. Short description (what did this experiment try?)

Example:

```
commit	val_bce_loss	memory_gb	status	description
a1b2c3d	3.124500	4.2	KEEP	baseline
b2c3d4e	3.098200	4.3	KEEP	increase embed_dim to 128
c3d4e5f	3.200000	4.1	DISCARD	remove dropout (overfit)
d4e5f6g	0.000000	0.0	CRASH	n_layer=12 OOM on this GPU
```

---

## The experiment loop

**LOOP FOREVER — do NOT stop to ask the user for permission to continue.**

The user may be asleep or away. They expect you to keep working until manually interrupted.

```
LOOP:
1. Inspect git state (current branch, last commit)
2. Propose an experiment idea. Think about:
   - Which hyperparameters haven't been tried?
   - Did previous failed experiments point to anything?
   - What architectural insights might help?
3. Edit train.py (and/or emr_model/transform_emr/*.py)
4. git commit
5. python train.py > run.log 2>&1
6. grep "^val_bce_loss:\|^peak_vram_mb:" run.log
7. If empty: crash. Read tail -n 50 run.log, attempt fix (1-2 tries max),
   then log as CRASH and move on.
8. Record in results.tsv
9. If val_bce_loss improved (lower): KEEP - advance the branch
10. If equal or worse: DISCARD - git reset --hard HEAD~1
```

**Crashes**: distinguish between fixable bugs (typo, import error: fix and retry) and fundamental failures (OOM, NaN loss: log CRASH and move on with a different idea).

**Stuck?** Re-read the architecture files, look at the last few kept experiments, try combining ideas, or try more radical changes (different attention variant, different normalization, different optimizer settings).

---

## Research directions to explore

Start with hyperparameter sweeps, then move to architectural changes:

**Phase-1 (embedder) hyperparameters:**
- `time2vec_dim` (controls temporal embedding richness)
- `phase1_learning_rate`
- `phase1_n_epochs` and early-stop patience

**Phase-2 (transformer) hyperparameters:**
- `embed_dim`, `n_layer`, `n_head` (model capacity)
- `block_size` (context window)
- `phase2_learning_rate`, `weight_decay`
- `bce_k_window` (how many future tokens are soft targets)
- Auxiliary loss fractions (`aux_fraction_caps`)
- Scheduler curriculum settings

**Architectural changes (requires editing emr_model files):**
- Rotary position encodings instead of absolute position embeddings
- GQA / MQA (grouped-query attention) in the GPT
- SwiGLU vs GELU vs ReLU squared in the MLP
- RMSNorm vs LayerNorm
- Different Time2Vec configurations
- Depth vs width trade-offs (more layers + smaller dim, or vice versa)

**Training improvements:**
- Learning rate schedule (cosine decay, warmup)
- Gradient clipping magnitude
- Optimizer choice for embedder (AdamW vs SGD with momentum)
