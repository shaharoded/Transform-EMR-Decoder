# autoresearch — EMR Event Prediction

Autonomous hyperparameter and architecture search for a clinical EMR sequence model.

## What this is

An AI agent autonomously experiments on an EMR (Electronic Medical Record) next-event prediction model overnight. Each experiment modifies `train.py`, trains for a fixed epoch budget, checks if the validation metric improved, keeps or discards, and repeats — logging every result in `results.tsv`.

The architecture is a two-phase pipeline (`emr_model/`):

- **Phase 1 — EMREmbedding** — learns compact, time-aware representations of clinical events using hierarchical token embeddings, Time2Vec, and static patient context.
- **Phase 2 — GPT Transformer** — a causal decoder over Phase-1 embeddings, predicting the next clinical event in a patient's timeline.

The training data is derived from MIMIC-III and contains diabetes patients' longitudinal event sequences including lab results, vitals, diagnoses, medications, meals, and outcome events (complications, death, release).

## Files that matter

| File | Role |
|------|------|
| `prepare.py` | **Fixed.** Data loading and fixed evaluation metric. Do NOT modify. |
| `train.py` | **Agent edits this.** Model config, training settings, training loop. |
| `program.md` | **Human edits this.** Instructions for the autonomous agent. |
| `emr_model/transform_emr/` | Model source code (agent may modify for architecture changes). |
| `emr_model/data/source/` | Training data (fixed). |
| `results.tsv` | Experiment log (untracked by git). |

## Quick start

**Requirements:** Python 3.10+, a CUDA GPU (recommended), [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies
uv sync

# 2. Run a single training experiment (both phases)
python train.py
```

The first run builds and caches the tokenizer, which takes a few minutes.
Subsequent runs skip this step.

## Running the agent

Start a Claude Code session in this directory and point it at `program.md`:

```
Hi, have a look at program.md and let's kick off autoresearch — let's do the setup first.
```

The agent will create a branch, run the baseline, and iterate autonomously.

## Metric

`val_ce_loss` — cross-entropy loss on next-event prediction (validation set, padding excluded).
Lower is better. Defined in `prepare.py::evaluate_val_ce` — never modified.

The metric intentionally uses plain CE (not the multi-hot BCE from training) so that changing auxiliary loss weights or k-window size does not affect the comparison. Only actual model quality does.

## Design choices

- **Two-file split.** `prepare.py` is fixed ground truth; `train.py` is fully editable. Clear boundary between what changes and what doesn't.
- **Fresh start per experiment.** Phase checkpoints are cleared before each run so experiments are independent. The tokenizer is cached across runs (it doesn't depend on hyperparameters).
- **Epoch budget with early stopping.** Training terminates when validation stops improving (patience = 10 epochs), bounded by a maximum epoch count. This is fair across model sizes.
- **Single GPU, single file.** No distributed training, no complex configs.

## Project structure

```
prepare.py              fixed data utilities + evaluation metric
train.py                model config + training (agent modifies)
program.md              agent instructions (human modifies)
pyproject.toml          dependencies
results.tsv             experiment log (gitignored, untracked)
run.log                 last experiment output (gitignored)
emr_model/
  transform_emr/        core model source (agent may modify)
    embedder.py         Phase-1 EMREmbedding
    transformer.py      Phase-2 GPT
    dataset.py          tokenizer and dataloader
    loss.py             loss functions
    schedulers.py       auxiliary loss scheduling
    utils.py            utilities
  data/
    source/
      temporal_events.csv    patient event sequences (~130 MB)
      context_data.csv       patient context features
  checkpoints/          saved model weights (gitignored)
```
