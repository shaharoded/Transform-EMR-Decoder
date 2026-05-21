# autoresearch — EMR Event Prediction: Trajectory-Generation Fix

Autonomous research loop to fix the generation-collapse failure mode discovered
on the deployed M-256 model: the model emits a terminal token (DEATH / RELEASE)
within ~1 hour of the 2-day seed end on 100 % of patients, median generation
length 3 tokens. Under the original truncated evaluation this looked great
(AUROC 0.918 / AUPRC 0.630). Under an honest **horizon-extended evaluation**
(windows that span each patient's true admission horizon, score = 0 for windows
past generation end) AUROC collapses to ~0.5 — the model has essentially no
multi-day discriminative power because it never generates a multi-day
trajectory in the first place.

The architecture sweep (S-128 / M-256 / M-256-deep / L-384) is **closed**;
M-256 is the locked size. This loop is about fixing what the model *does* with
that capacity, not its size.

---

## Background — what's wrong

`status.md` Sections 1 and 1b carry the full diagnosis. The short version:

- Median generated trajectory length: **3 tokens**, mean 4.4, p90 8, max 19.
- **100 % of patients reach a natural terminal** (DEATH or RELEASE) within
  ~1 hour of seed end.
- Each patient contributes ~1 evaluation window per outcome (vs an expected
  ~5–10 for a multi-day patient horizon).
- Under horizon-extended eval (`extract_patient_horizons` + score=0 for missing
  windows): mean AUROC drops 0.918 → 0.452, mean AUPRC drops 0.630 → 0.107,
  per-outcome lifts collapse to ~1.0–1.5×.
- Per-patient horizons in the test set: median 152 h (~6.3 days), mean 166 h.

The Phase-2 / Phase-3 training optimised next-48h outcome-head scoring (the
truncated eval target) and the LM head learned that "predict terminal soon"
minimises trajectory-level BCE under the soft-kernel three-tier `log_tau_lm`
init (terminals 168 h, outcome-class 48 h, default 12 h). The model behaves
exactly as that loss tells it to — but the resulting generations don't span
the multi-day horizon the autoregressive framing claims.

---

## Goal

Train a model that:

1. **Generates trajectories of plausible length** matching each patient's
   actual hospitalisation horizon (median ~6 days, max 14 days).
2. **Improves AUROC / AUPRC on the horizon-extended eval** vs the deployed
   M-256 baseline (current mean AUROC 0.452 / AUPRC 0.107 — both should rise
   well above chance).
3. **Reduces onset-MAE** on terminal events (DEATH, RELEASE), since matching
   the actual hospitalisation length is the direct test that generation
   timing is calibrated.
4. **Keeps Phase 1 / 2 / 3 losses descending** — no honest gain may come from
   a training collapse elsewhere (e.g. driving BCE up to suppress terminals).

The Section 1 truncated eval (next-48h scoring, AUROC 0.918) is the
*upper bound* of the current model; if a fix improves the horizon-extended eval
but tanks the truncated eval, that is a trade-off worth measuring and probably
not worth keeping. Both numbers must be reported for every experiment.

VRAM stays under 24 GB. The architecture stays at M-256
(`embed_dim=256, n_layer=4, n_head=4, time2vec_dim=32, dropout=0.10`). Any
architecture change has to be justified in writing.

---

## Hard constraints

- **No hard-coded minimum trajectory length.** Do not mask terminal tokens for
  the first N generation steps. Do not enforce a floor on generation length.
  The fix must come from training, not from sampling-time hacks.
- **Do not modify `api.py`'s data-split logic** (70/15/15 by PatientId,
  `random_state=42`).
- **Do not modify `evaluation.py`'s scoring rules.** The horizon-extended
  metric introduced in commit `af3cf53` (Section 1b of `status.md`) is the
  fixed contract — `pooled_episode_auc` with `patient_horizons` is the
  headline for every experiment in this branch.
- **Generation length must be learned**, not bounded. The model has to learn
  *when* to emit a terminal, not just *whether*.

---

## What's in scope to modify

Ranked by where the leverage is for this failure mode:

- **Architecture & losses** (`transformer.py`, `embedder.py`, `loss.py`,
  `schedulers.py`, `utils.py`) — the primary lever. Add losses that directly
  penalise the failures we observe: a too-short generation, an early-terminal
  bias, a mismatched cumulative-Δt vs ground-truth horizon, an outcome head
  that only sees the seed-end position.
- **Training procedure** (`transformer.py` Phase 2 / Phase 3 loops) — adding
  scheduled sampling (gradually replacing teacher-forced tokens with the
  model's own predictions) is high on the list; it directly attacks the
  exposure-bias hypothesis.
- **Inference** (`inference.py`) — fair game for *learned* decoding changes
  (beam search, length-normalised scoring, hazard-driven terminal sampling,
  probability-temperature adjustments tied to a model-output signal), and
  for diagnostic instrumentation (generation length / terminal-time
  distribution, returned via a `gen_stats` side-channel). **No hard rules**:
  no terminal masking, no min-length floor, no hand-coded "must generate at
  least N steps" gate.
- **Config** (`model_config.py`) — welcome if a structural change needs a
  paired hyperparameter, but config-only edits are not the primary lever
  in this branch.

**Out of scope:** `api.py`, `evaluation.py`, `emr_model/data/`, the
architecture size sweep (closed; M-256 stays).

---

## Headline metrics for every experiment

Each full run must record, and the agent's `status.md` row must include:

| Metric | Source | Direction |
|--------|--------|-----------|
| `outcome_auroc` (horizon-extended) | `evaluation.py` (horizon-extended contract) | ↑ |
| `outcome_auprc` (horizon-extended) | `evaluation.py` (horizon-extended contract) | ↑ |
| `onset_mae_hrs` (mean across outcomes) | `evaluation.py::time_accuracy`         | ↓ |
| `mae_release_hrs`                  | `evaluation.py::time_accuracy`              | ↓ |
| `mae_death_hrs`                    | `evaluation.py::time_accuracy`              | ↓ |
| `outcome_auroc_truncated`          | second pass with `patient_horizons=None`    | report; large regression flagged |
| `gen_median_steps`                 | `inference.py::generate` (`gen_stats`)      | match GT — currently 3 |
| `gen_median_hours`                 | `inference.py::generate` (`gen_stats`)      | match per-patient horizons (~150 h) |
| `gen_frac_terminal_first24h`       | `inference.py::generate` (`gen_stats`)      | ↓ from current 100 % |
| `gen_length_mae_hrs`               | `inference.py::generate` (`gen_stats`)      | ↓ ; mean abs error between generated trajectory span and per-patient GT horizon |
| `phase{1,2,3}_best_val`            | training logs                               | descending across the experiment |

MAE — both per-terminal (`mae_release_hrs`, `mae_death_hrs`) and aggregated
(`onset_mae_hrs`) — is a co-equal headline with AUROC / AUPRC. A trajectory
that ranks outcomes well but emits them at wildly wrong times is not a
publishable result.

The agent must add generation-instrumentation columns to `inference.py::generate`
as a returned `gen_stats` dict (median/mean/p90 steps and hours, fraction
terminating in first 24 h, length-MAE vs per-patient horizon), and surface
them in the `api.py` summary block alongside the existing `outcome_*` lines.

The agent must add generation-instrumentation columns (median steps, median
hours, fraction terminating early, length MAE) to `inference.py::generate` as
a return-value side-channel (`gen_stats` dict), and surface them in the
`api.py` summary block alongside the existing `outcome_*` lines.

---

## Diagnostic loop (required, every experiment)

Before declaring KEEP, run on the freshly-trained checkpoint:

1. **Generation-length distribution** on the held-out test split:
   - Count of generated tokens per patient (median / mean / p90 / max).
   - Generated trajectory span in hours per patient.
   - Per-patient fraction completed = `gen_hours / patient_horizon_hours`.
   - Fraction of patients emitting terminal in the first 24 h post-seed.
2. **Terminal-timing accuracy**: for patients whose GT contains a
   DEATH/RELEASE, compare the generated terminal time vs the GT terminal time.
   MAE in hours per outcome.
3. **Per-phase loss curves** still descending (no instability, no aux flat).
4. **Outcome head still differentiates** — `diagnose.py` Reports 2 (logit
   separation) and 5 (token gradient utility) should not have regressed.

## Research directions

Starting points. The agent is free to combine, replace, or invent
alternatives — but **every proposed change must come with a falsifiable
hypothesis** about why it should extend generations without hand-coded rules.

### A — Scheduled sampling (reduce teacher-forcing exposure bias)

The model trained purely teacher-forced but inferences autoregressively. In
Phase 2 (and optionally Phase 3), gradually replace a growing fraction of the
teacher-forced inputs with the model's own previous-step predictions.
Anneal the replacement probability `p` from 0 at start to ~0.3 by end of
Phase 2. The model is forced to recover from its own predictions during
training and stops compounding errors at inference time.

Falsifiable: median generation length should rise monotonically as `p`
increases on an in-training probe. Phase-2 BCE may rise slightly (harder
task) but should still descend within each `p` regime.

### B — Trajectory-length loss

Phase 1's Δt MSE supervises *per-step* gap prediction but nothing supervises
that the **cumulative sum of generated Δt** matches the GT patient horizon.
Add a Phase-2 sequence-level loss: sample a teacher-forced prefix, generate
the remainder autoregressively (no-grad or with small grad), compare
`sum(Δt_generated)` to `t_GT_terminal − t_prefix_end`, MSE in log1p hours.
This directly penalises trajectories that don't span the right length.

Falsifiable: `gen_length_mae_hrs` should drop below ~48 h on smoke test;
`gen_median_hours` should track per-patient horizon medians within ±25 %.

### C — Time-to-terminal regression auxiliary

Add an auxiliary head predicting `log1p(t_terminal − t_now)` at every
non-terminal position. MSE loss against GT distance-to-terminal. The
backbone is pulled toward a representation that *knows* how far away
discharge / death is — currently it only knows "imminent vs not".

Falsifiable: head's R² on the regression target should rise above 0.3 in
Phase 2 validation; generation should extend because terminal-prediction is
no longer the model's only outlet for terminal-timing signal.

### D — Discrete-time hazard for terminals

Replace the soft-kernel BCE on terminal classes (DEATH / RELEASE) with a
hazard head: at each step the model predicts `P(terminal in [t, t+Δ])` for
log-spaced Δ bins (1 h, 6 h, 24 h, 72 h, 168 h). At inference, the terminal
time is drawn from the hazard distribution rather than from greedy
token-level prediction. Separates "what event" from "when this patient
leaves" and gives a calibrated time-to-terminal signal.

Falsifiable: median terminal time should align with the per-patient hazard
expectation, not collapse to 0 h. `mae_release_hrs` and `mae_death_hrs`
should drop substantially.

### E — Re-weight terminal tokens in the LM head

The Phase-2 soft-kernel BCE with `log_tau_lm[terminal]=log(168/336)` gives
terminals a 168-h supervision window — every position within 168 h of a
terminal event has a positive target for that terminal class. The model has
likely learned that "predict terminal soon" minimises the BCE almost
everywhere. Try narrowing `tau_lm[terminal]` to e.g. 12–24 h and/or
down-weighting the terminal class in the LM-head `pos_weight`.

Falsifiable: narrower terminal tau should reduce
`gen_frac_terminal_first24h` without harming Phase-2 outcome AUROC; complic-
ation-class targets (CARDIO / KIDNEY) should not be affected.

### F — Inference: beam search / length-normalised decoding

The current sampler is single-trajectory with `temperature=1.0` and a
repetition penalty. Beam search with a length-normalised score
(score / length^α) explores multiple candidates and avoids the local
single-step "emit terminal now" trap. Alternatively, expose a learned
"early-terminal penalty" computed from a model-output (e.g. the
time-to-terminal head's expectation) rather than a hand-coded length floor.

Falsifiable: beam search with width 4 and α∈[0.5, 1.0] should extend
generations without any hand-coded constraint; if it doesn't, the
underlying ranking already strongly prefers terminal regardless of beam.

### G — Re-balance Phase-3 oversampling

Phase 3 fine-tunes the outcome head on natural-distribution data, but
Phase 2 oversamples for rare outcomes. The LM head sees an event
distribution skewed toward outcome-rich patients (who often have shorter
horizons / earlier terminals). Try training Phase 2 on natural-distribution
data, or reduce the oversampling weight on patients with very early
terminals.

Falsifiable: oversampling stats should show the bias toward early-terminal
patients; reducing it should shift `gen_median_hours` upward.

---

## Process

1. **Re-read this `program.md`** at the start of every iteration.
2. **Inspect git state**: `git status`, `git log --oneline -5`,
   `cat results/results-trajectory-fix.tsv` (new ledger for this branch).
3. **Run `diagnose.py`** on the current checkpoint to confirm the failure mode
   you intend to target is the one that's actually broken.
4. **Run the generation-length diagnostic** (Directions A–F all need to move
   this number; without it, you don't know if your fix worked).
5. **Propose ONE structural experiment** with a falsifiable hypothesis. No
   hyperparameter-only changes.
6. **Smoke test** (sample=50, 1 epoch per phase) → confirm summary block
   prints both truncated and horizon-extended metrics, plus the
   generation-length stats.
7. **git commit** with a 3-part message: change / diagnostic-that-motivated /
   what you expected.
8. **Full run** → `python api.py > run.log 2>&1`.
9. **Log row** to `results/results-trajectory-fix.tsv` with the columns from
   the headline-metrics table above.
10. **KEEP / DISCARD** per the rules below.
11. **Update `status.md`** at the repo root.

### KEEP / DISCARD rules

The bar is "honest improvement, no regression in any metric beyond noise".
The first KEEP doesn't need to fix everything — it just has to move the
needle without giving back ground elsewhere.

**KEEP** iff **all** of:

- Peak VRAM ≤ 24 GB.
- At least one of the horizon-extended headline metrics
  (`outcome_auroc`, `outcome_auprc`, `onset_mae_hrs`) improves vs the
  current best by more than the per-metric noise floor (AUROC ≥ +0.005,
  AUPRC ≥ +0.005, MAE ≥ −5 h).
- No headline metric regresses by more than the same noise floor
  (i.e. no metric goes ≥ 0.005 AUROC / 0.005 AUPRC / 5 h MAE in the wrong
  direction). The truncated-eval AUROC counts as a metric: do not let
  it drop more than 0.02 below the deployed baseline (0.918 → don't go
  below 0.898).
- `gen_median_hours` strictly above the previous best (any movement in the
  right direction is progress), or already at ≥ 50 % of median patient
  horizon (then this constraint is satisfied).
- `gen_frac_terminal_first24h` strictly below the previous best, or
  already below 10 %.
- Phase 1 / 2 / 3 training losses descending across their full schedule
  (no flat aux, no diverging val).

**DISCARD** otherwise → `git reset --hard <last_keep_commit>`.

Generation-length and AUROC/AUPRC/MAE may improve in different experiments;
the running best of each is the bar. An experiment that gives +0.01 AUROC
but loses 5 % `gen_median_hours` is a DISCARD because it regressed a key
metric. An experiment that gains 30 h on `gen_median_hours` with AUROC/AUPRC
flat is a KEEP — useful progress on generation collapse without trading
predictive quality.

---

## When to stop

Stop when the model produces a result that is **publishable as a multi-day
event-prediction model** under the honest horizon-extended evaluation. The
shape of a publishable result:

- Horizon-extended `outcome_auroc` clearly above chance (well past 0.5) and
  meaningfully above the current 0.452 baseline.
- AUPRC clearly above the prevalence baselines per outcome — the per-outcome
  lift table tells a coherent positive story (most outcomes ≥ 2× lift).
- `gen_median_hours` is a meaningful fraction of the median patient horizon —
  the trajectory is no longer collapsed.
- Terminal MAE (DEATH / RELEASE) is small enough that the generated terminal
  timing is clinically informative, not noise.
- Training and diagnostics confirm losses are descending and the model isn't
  collapsing in some other way.

No hard threshold on AUROC or any single metric — the agent uses judgement
on whether the combined picture would survive peer review. When the loop
hits an iteration where further experiments are unlikely to add anything
without harming an already-decent metric, write the final summary and
pause.

If, after a fair set of honest structural attempts across multiple
directions, no configuration achieves the above (the deployed baseline plus
the trajectory-collapse fix are mutually exclusive), declare it honestly:
document the trade-off in `status.md` and pause. The truncated-eval
baseline (Section 1) remains a publishable result under the "next-48h event
window predictor" framing.

---

## Reproducibility

- Branch: `autoresearch-trajectory` (this branch). Code changes commit here;
  no force-push to `main`.
- Ledger: `results/results-trajectory-fix.tsv` (header on first row, schema
  matches the headline-metrics table above).
- Checkpoints: `emr_model/checkpoints/` (gitignored, re-built per experiment).
- `status.md`: this branch's progress journal, sectioned by experiment. Keep
  the existing Section 1 and Section 1b from `autoresearch-optimization`
  intact at the top — those are the "before" reference for every experiment
  in this branch.
