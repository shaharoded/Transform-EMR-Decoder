# autoresearch — EMR Event Prediction: Trajectory-Generation Fix

Fix the **generation-collapse failure mode** discovered on the deployed M-256
model: the model emits a terminal token (DEATH / RELEASE) within ~1 hour of
the 2-day seed end on 100 % of patients, median generation length 3 tokens.
Under the original truncated eval this looked great (AUROC 0.918 / AUPRC 0.630).
Under the honest **horizon-extended eval** (windows extend to each patient's
true admission horizon, score = 0 for windows past generation end) AUROC
collapses to ~0.452, AUPRC to ~0.107 — the model has essentially no multi-day
discriminative power because it never generates a multi-day trajectory.

`status.md` Sections 1 and 1b carry the full diagnosis. Don't repeat the
architecture sweep — M-256 stays.

---

## Goal

Train (or post-process) a model that, evaluated under the horizon-extended
contract, improves on:

| Metric (current baseline) | Direction | Why |
|---------------------------|-----------|-----|
| `outcome_auroc` 0.452     | ↑         | Multi-day ranking |
| `outcome_auprc` 0.107     | ↑         | Multi-day precision-recall over prevalence |
| `onset_mae_hrs` ~85       | ↓         | Outcome timing |
| `mae_release_hrs`, `mae_death_hrs` | ↓ | Terminal timing — direct generation-length signal |
| `gen_median_hours` ~1     | ↑         | Trajectory length match per-patient horizon (~152 h median) |
| `gen_frac_terminal_first24h` 1.0 | ↓ | Premature terminal emission |
| Truncated `outcome_auroc` 0.918 | report; large drop flagged | Don't trade the near-term signal away |

Phase 1 / 2 / 3 training losses must remain descending; no metric may regress
past the noise floor. Both eval views (horizon-extended primary, truncated
secondary) reported for every experiment.

---

## What's locked vs in scope

**Locked.** `api.py`, `evaluation.py`, `emr_model/data/`, model architecture
(M-256: `embed_dim=256, n_layer=4, n_head=4, time2vec_dim=32, dropout=0.10`),
VRAM ≤ 24 GB.

**In scope** (ranked by leverage for this failure):

1. **Architecture & losses** — `transformer.py`, `embedder.py`, `loss.py`,
   `schedulers.py`, `utils.py`. Add losses that directly penalise the
   failures we observe (premature terminal, short trajectory, miscalibrated
   time-to-terminal).
2. **Training procedure** — Phase 2 / 3 loops (scheduled sampling, etc.).
3. **Inference** — `inference.py`. Learned decoding only (beam search,
   length-normalised scoring, hazard-driven terminal sampling, model-output-
   driven temperature). **No hard rules** — no terminal masking, no
   min-length floor, no hand-coded "must generate at least N steps" gate.
4. **Config** — `model_config.py`. Welcome alongside a structural change;
   config-only edits are not the primary lever.

---

## Headline metrics (auto-emitted)

The summary block of every run already prints these lines — just grep them.
**No instrumentation work to do.** `evaluation.py::compute_gen_stats` (called
inside `evaluate_on_test_set`) derives all `gen_*` values from `risk_df`; the
contract `inference.generate` must keep is its returned DataFrame's columns:
`PatientId`, `TimePoint`, `IsInput`, `IsTerminal`, `P_<outcome>`.

```
outcome_auroc:               <horizon-extended primary>
outcome_auprc:               <horizon-extended secondary>
onset_mae_hrs:               <mean across outcomes>
gen_median_steps:            <currently 3, target rises toward GT>
gen_median_hours:            <currently ~1, target → median patient horizon ~152>
gen_p90_hours:               <upper-tail length>
gen_n_with_terminal:         <patient count that emitted a terminal>
gen_frac_terminal_first24h:  <currently 1.0, target ↓>
gen_length_mae_hrs:          <|gen_span − GT_horizon| mean>
phase{1,2,3}_best_val
per_outcome <TSV table>
```

Log a row in `results/results-trajectory-fix.tsv` after each full run with
these columns + commit hash + description.

---

## Research directions

Starting points — combine, replace, or invent alternatives. Every change
needs a **falsifiable hypothesis** about why it should extend generations
without hand-coded rules. **Inference-first experiments are encouraged
because the deployed checkpoints are already on disk** — you can iterate on
decoding without paying the Phase 1 / 2 / 3 training cost.

**Inference-side (cheap — no retraining):**

- **F1. Beam search with length-normalised scoring.** Multi-candidate
  decoding, score / length^α. *Falsifiable*: if generation extends without
  any constraint, the single-trajectory sampler was the bottleneck; if not,
  the model itself strongly prefers terminal regardless of beam.
- **F2. Sampling-temperature schedule.** Higher temperature in the first N
  steps to escape the immediate-terminal local minimum, anneal as
  generation proceeds. *Falsifiable*: `gen_median_hours` rises monotonically
  with starting temperature; AUROC may drop slightly if signal is
  temperature-sensitive.
- **F3. Hazard-driven terminal sampling at inference.** Use the existing
  outcome head's terminal logits to draw the terminal time from a smoothed
  distribution instead of emitting on first peak. No retraining needed.

**Training-side (requires retraining; reuse Phase-1 cache where possible):**

- **A. Scheduled sampling.** Gradually replace teacher-forced tokens with
  the model's own predictions in Phase 2 (anneal `p` from 0 → ~0.3). The
  model trains under its own distribution and stops compounding errors at
  inference. *Falsifiable*: median generation length rises as `p` increases.
- **B. Trajectory-length loss.** Phase-2 sequence-level loss penalising
  cumulative-Δt mismatch vs GT patient horizon. *Falsifiable*:
  `gen_length_mae_hrs` drops below ~48 h.
- **C. Time-to-terminal regression head.** Auxiliary regression on
  `log1p(t_terminal − t_now)` at every non-terminal position. *Falsifiable*:
  head R² > 0.3; terminal MAE drops substantially.
- **D. Discrete-time hazard for terminals.** Replace BCE on DEATH/RELEASE
  with hazard bins (1 h, 6 h, 24 h, 72 h, 168 h); inference samples
  terminal time from the hazard distribution. *Falsifiable*: terminal-MAE
  drops, generation no longer collapses to 0 h.
- **E. Narrow terminal `tau_lm`.** The 168 h soft-kernel window for
  terminals taught the model "predict terminal soon" minimises BCE
  everywhere. Narrow to 12–24 h, and/or down-weight terminal in
  `pos_weight`. *Falsifiable*: `gen_frac_terminal_first24h` drops without
  hurting complication-class AUROC.

The agent is encouraged to **prototype F1–F3 first** on the existing
checkpoints (no retraining) — that yields a fast read on whether
generation-length is fixable purely at inference time before paying any
training cost.

---

## Process

1. **Re-read this file** at iteration start.
2. **Check state**: `git status`, `git log --oneline -5`, last few rows of
   `results/results-trajectory-fix.tsv`.
3. **Diagnose**: run `diagnose.py` + read recent `gen_*` lines from the last
   `run.log` to confirm the failure mode being targeted is actually present.
4. **One change per experiment, with a falsifiable hypothesis.**
5. **Smoke test** (`sample=50, phase{1,2,3}_n_epochs=1`) — confirm the
   summary block prints sensibly including `gen_*` lines.
6. **Commit** with a 3-part message: change / diagnostic / expectation.
7. **Full run**: `python api.py > run.log 2>&1` (or just the eval-only path
   when the experiment is inference-side — see "Checkpoints" below).
8. **Log** the row to `results/results-trajectory-fix.tsv`.
9. **KEEP / DISCARD** (rules below). On DISCARD, `git reset --hard
   <last_keep_commit>`.
10. **Update `status.md`**.

### KEEP / DISCARD

**KEEP** iff **all**:

- Peak VRAM ≤ 24 GB; training losses descending.
- At least one headline metric (`outcome_auroc`, `outcome_auprc`,
  `onset_mae_hrs`) improves past the noise floor
  (AUROC ≥ +0.005, AUPRC ≥ +0.005, MAE ≥ −5 h).
- No headline metric regresses past the same noise floor. Truncated
  `outcome_auroc` doesn't drop more than 0.02 below 0.918.
- `gen_median_hours` strictly above previous best (or already ≥ 50 % of
  median patient horizon).
- `gen_frac_terminal_first24h` strictly below previous best (or already < 0.10).

Otherwise **DISCARD** → `git reset --hard <last_keep_commit>`.

---

## Training and evaluation run in separate processes

`api.py` now trains Phase 1/2/3 in one process and then **re-launches itself**
with `--eval-only` for the held-out test evaluation. This is a robustness fix:
in the previous architecture sweep, cumulative training RAM (DataLoader
workers, optimiser state, persistent caches) caused several `SIGKILL`s during
or right after Phase-3 validation (`ab7aae1` XL-512, the `L-384` between-phase
crash). Running eval in a fresh subprocess means it starts with a clean memory
slate; autoregressive generation on 8,562 patients can never be killed by
training residue.

Practical implications:

- `python api.py > run.log 2>&1` still produces one continuous log — the
  subprocess inherits stdout/stderr.
- The parent writes `emr_model/checkpoints/train_summary.json` after Phase 3
  with `phase2/3_best_val`, epoch counts, total training seconds. The eval
  subprocess reads that file and folds those numbers into the summary block.
- `python api.py --eval-only` runs eval directly on whatever checkpoints are
  in `emr_model/checkpoints/`, without any training. Useful for inference-
  side experiments (Directions F1–F3) — change `inference.py`, run
  `python api.py --eval-only`, get a full summary in seconds. No need to
  rerun Phase 1/2/3.
- The eval subprocess hits `processed_datasets.pt` cache (built by the
  parent) so it skips the 1.4 GB CSV re-read and is ready in <1 minute.

---

## Checkpoints — preserve across DISCARDs

The deployed M-256 checkpoints (`phase1/`, `phase2/`, `phase3/ckpt_best.pt`,
`tokenizer.pt`, `scaler.pkl`) are shipped with this branch under
`emr_model/checkpoints/` (gitignored locally; copy them into place after
clone). Avoid retraining when you don't need to:

- **Phase 1 (embedder) cache.** `api.py` auto-reuses
  `phase1/ckpt_best.pt` when `(embed_dim, time2vec_dim, ctx_dim)` are
  unchanged. Don't touch Phase 1 unless your hypothesis actually requires
  re-embedding. Most directions on this branch don't.
- **Phase 2 checkpoint reuse.** When you change *only* Phase-3 or inference
  behaviour, you can also reuse `phase2/ckpt_best.pt`. `api.py` always
  re-trains Phase 2 from scratch by default — when an experiment is
  Phase-3-or-inference-only, skip the Phase-2 retrain by manually pointing
  Phase 3 at the cached Phase 2 checkpoint (see `transformer.py::GPT.load`).
- **Inference-only experiments** (Directions F1–F3) don't retrain anything
  — they just call `evaluate_on_test_set` with a modified `generate()`.

**Retain best weights across DISCARDs.** Before any experiment that will
retrain Phase 2 or Phase 3, **back up the current best** with:

```bash
cp -r emr_model/checkpoints emr_model/checkpoints.bak_<short_commit>
```

If the run is a DISCARD and you `git reset --hard`, the working-tree
checkpoint files survive the git reset (they're gitignored), but the
training itself will have overwritten them. Restoring the backup gets you
back to the last-KEEP weights without having to retrain Phase 1/2 from
scratch:

```bash
rm -rf emr_model/checkpoints && mv emr_model/checkpoints.bak_<commit> emr_model/checkpoints
```

Same trick before any structural change to Phase 1 (which forces a Phase-1
retrain via the embedder-config-changed branch in `api.py`): keep a backup
of the deployed Phase-1 so you can restore the baseline embeddings
instantly when the new Phase-1 doesn't pan out.

---

## When to stop

Stop when you have a **publishable multi-day event-prediction result** under
the horizon-extended eval:

- AUROC clearly above chance and meaningfully above 0.452.
- AUPRC clearly above the per-outcome prevalence baselines (lifts ≥ 2× for
  most outcomes).
- `gen_median_hours` a meaningful fraction of median patient horizon (~150 h).
- Terminal MAE small enough to be clinically informative.
- Training and diagnostics confirm no collapse elsewhere.

No hard AUC threshold; the agent uses judgement on whether the combined
picture would survive peer review. If after honest structural attempts
across multiple directions no configuration achieves the above, document
the trade-off in `status.md` and pause — the truncated-eval baseline
(Section 1) remains publishable under the "next-48 h event-window
predictor" framing.

---

## Reproducibility

- Branch: `autoresearch-trajectory`. Code commits go here; no force-push to
  `main`.
- Ledger: `results/results-trajectory-fix.tsv` (header on first row).
- Checkpoints: `emr_model/checkpoints/` (gitignored). Back up before
  retraining experiments per the section above.
- Journal: `status.md` at repo root, sectioned by experiment. Sections 1
  and 1b from `autoresearch-optimization` stay intact as the "before"
  reference for every experiment on this branch.
