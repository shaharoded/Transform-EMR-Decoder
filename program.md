# autoresearch — EMR Event Prediction

Autonomous hyperparameter and architecture search on an EMR complication prediction model.

---

## Background

The model learns to predict the future stream of medical events for a hospital patient given their history, with a focus on identifying complications before they occur. Data is derived from MIMIC-III: diabetes patients' longitudinal event sequences — lab results, vitals, diagnoses, medications, meals, and outcome events (complications, death, release).

There are 15 clinical complication targets (e.g. `KIDNEY_COMPLICATION_EVENT`, `CARDIO-VASCULAR_DISORDER_EVENT`). The model must predict both *what* will happen and *when*. These events are rare and clinically critical.

### Three-phase training pipeline

**Phase 1 — EMREmbedding** (`embedder.py`):
- Learns a compact, time-aware representation of each clinical event.
- Components: hierarchical token embeddings (raw concept → concept → concept+value → concept+value+position), Time2Vec for inter-event duration, and a static patient context vector.
- Loss: teacher-forced BCE + time MSE + MLM auxiliary.
- Checkpoint is cached and reused when `(embed_dim, time2vec_dim, ctx_dim)` are unchanged.

**Phase 2 — GPT Transformer** (`transformer.py`):
- Causal decoder over Phase-1 embeddings.
- `AdaLNBlock`: AdaLN-Zero injects patient context (shift/scale/gate per block).
- `CausalSelfAttention`: temporal RoPE uses actual `abs_ts` deltas instead of token-index differences.
- Loss curriculum: Focal BCE → CE (ranking) → outcome auxiliary, controlled by `schedulers.py`.
- Uses an oversampled DataLoader to balance rare positive outcomes.
- Phase-2 checkpoint is cleared before every experiment — runs are independent.

**Phase 3 — Outcome Head Fine-tuning** (`transformer.py::finetune_transformer`):
- Backbone fully frozen; only the outcome head is trained.
- Uses natural-distribution DataLoader (no oversampling) — important for `pos_weight` correctness.
- Loss: outcome BCE only, with time-decayed soft labels.
- This is the final checkpoint used for evaluation.

### Evaluation

Evaluation runs after Phase 3 via `evaluation.py::evaluate_on_test_set`. It uses **autoregressive generation**, not teacher-forced logits:

1. Load held-out validation patients (raw, never seen during training).
2. Truncate each patient's history to 2 days (generation seed).
3. Generate an autoregressive trajectory up to 500 steps at temperature 1.0 with repetition penalty.
4. Divide each trajectory into 24-hour non-overlapping windows.
5. Label each window 1 if any ground-truth episode of that complication falls within ±24h.
6. Pool all (patient, window) pairs → AUROC and AUPRC per complication.
7. Mean across complications with ≥3 positive windows.

This mirrors real clinical deployment: the model generates a future trajectory and the outcome-head risk scores are compared against what actually happened.

---

## Setup

1. **Agree on a run tag** with the user (e.g. `may1`). Branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>`.
3. **Read all in-scope files** (in this order):
   - `api.py` — fixed: data loading, training orchestration, summary print format. Do NOT modify.
   - `evaluation.py` — fixed: evaluation protocol and metric definitions. Do NOT modify.
   - `emr_model/transform_emr/config/model_config.py` — `MODEL_CONFIG` and `TRAINING_SETTINGS`. This is your primary edit target for hyperparameter changes.
   - `emr_model/transform_emr/embedder.py` — Phase-1 embedding model.
   - `emr_model/transform_emr/transformer.py` — Phase-2/3 GPT and training loops.
   - `emr_model/transform_emr/loss.py` — `FocalBCELoss`, `MaskedFocalBCE`, `MaskedSetCE`.
   - `emr_model/transform_emr/schedulers.py` — auxiliary loss curriculum scheduling.
   - `emr_model/transform_emr/utils.py` — masking, temporal targets, repetition penalties.
   - `emr_model/transform_emr/inference.py` — autoregressive generation (used by evaluation).
   - `emr_model/transform_emr/diagnose.py` — model health diagnostics. Run before proposing any experiment.
4. **Verify data exists**: `emr_model/data/source/temporal_data.csv` and `context_data.csv`.
5. **Check results.tsv**: if it contains only a header row, the first run establishes the baseline. Append to it — never reinitialise.
6. **Confirm and go.**

---

## Experimentation

**What you CAN modify:**
- `emr_model/transform_emr/config/model_config.py` — `MODEL_CONFIG` (architecture dims) and `TRAINING_SETTINGS` (hyperparameters, scheduler config). This is always the first place to try.
- `emr_model/transform_emr/*.py` — architecture changes (embedder, transformer, loss, schedulers, utils, inference).

**What you CANNOT modify:**
- `api.py` — fixed training orchestration.
- `evaluation.py` — fixed evaluation protocol.
- `emr_model/data/` — fixed training data.

**Simplicity criterion:** a small gain with lots of new code is suspect. Removing code while maintaining performance is always a win.

---

## The goal

Maximise `outcome_auroc` on the held-out validation set (primary metric, higher is better, 0.5 = random, 1.0 = perfect).

`outcome_auprc` and `onset_mae_hrs` are secondary — improve them when possible but do not sacrifice AUROC.

---

## Running an experiment

### Step 1 — Smoke test first (always)

Before every full training run, verify the pipeline end-to-end with a small subset:

```python
# In emr_model/transform_emr/config/model_config.py — set temporarily:
"sample": 50,
"phase1_n_epochs": 1,
"phase2_n_epochs": 1,
"phase3_n_epochs": 1,
```

```bash
python api.py > smoke.log 2>&1
grep "^outcome_auroc:\|^---" smoke.log
```

If the summary block appears without a crash — pipeline is wired correctly. Restore `sample: None` and the original epoch counts before the real run. Do **not** log smoke test results to `results.tsv`.

If the smoke test crashes — fix the bug before running full training. A crash on a full run wastes GPU hours.

### Step 2 — Full run

```bash
python api.py > run.log 2>&1
```

**Extract the result:**
```bash
grep "^outcome_auroc:\|^outcome_auprc:\|^onset_mae_hrs:\|^phase3_best_val:\|^peak_vram_mb:" run.log
```

If empty — crash. Inspect with `tail -n 50 run.log`.

**Timeout**: treat as crash if no `---` summary after 90 minutes.

---

## Output format

```
---
outcome_auroc:    0.000000
outcome_auprc:    0.000000
onset_mae_hrs:    0.00
phase2_best_val:  ...
phase2_epochs:    ...
phase3_best_val:  ...
phase3_epochs:    ...
total_seconds:    ...
peak_vram_mb:     ...
embed_dim:        256
n_layer:          4
n_head:           4
num_params:       ...
```

---

## Logging results

Append every completed experiment to `results.tsv` (gitignored — do not commit it).

```
commit	outcome_auroc	outcome_auprc	onset_mae_hrs	peak_vram_gb	status	description
```

- `commit`: 7-char git hash
- `peak_vram_gb`: `peak_vram_mb / 1024`, 1 decimal place
- `status`: `KEEP`, `DISCARD`, or `CRASH`
- Use `0.000000` / `0.00` / `0.0` for crashes
- `description`: one-line summary of what changed

---

## The experiment loop

**LOOP FOREVER — do NOT stop to ask for permission. The user is away.**

**Before every experiment: re-read this `program.md`.** The task list and rules are updated between sessions. If you operate from memory you will drift back toward hyperparameter sweeps. Re-read the *Research directions* section in full at the start of every iteration.

**Before every full run: run a smoke test** (sample=50, 1 epoch per phase). If the smoke test crashes, fix the bug before burning GPU hours on a 90-minute run.

**Before every experiment: run `diagnose.py` on the current best checkpoint** to inspect what is actually broken in the model you are trying to improve. Propose your experiment from the diagnostic output, not from speculation.

```
LOOP:
1. RE-READ program.md (especially "Research directions" and "Rules of the road").
2. Inspect git state (branch, last commit, results.tsv tail).
3. cd emr_model && python -m transform_emr.diagnose > ../diag.log 2>&1 && cd ..
4. Read diag.log. Gate on these questions before proposing anything:
   a. Report 4 — what are the actual lambda_max values for ce, outcome, hazard?
      Any lambda_max < 0.001 is near-silent — that aux loss is not learning.
   b. Report 5 — where do outcome tokens rank by grad/occ?
      Bottom half of vocab = the loss is not reaching them.
   c. Report 2 — what is sigmoid[pos] - sigmoid[neg] (logit separation)?
      < 0.05 means the model barely distinguishes outcome-positive from negative positions.
   d. PROBE Δt HEAD — what is Pearson r and R²?
      r < 0.1 or pred_std < 0.05h means the time head has collapsed to a constant.
   e. PROBE OUTCOME HEAD LABEL ALIGNMENT — any flip=True rows? Fix sign error first.
   Inspect the Phase-1 training log too: is `tr_mlm` and `tr_dt` actually decreasing
   across epochs? If a loss is flat across all of Phase-1, the corresponding head is
   not learning — that is the failure to fix, not lambda calibration.
5. Propose ONE experiment that targets the highest-priority broken aux task. No
   hyperparameter-only changes. If your only idea is "tune X up/down", stop and
   re-read Research directions for a structural alternative.
6. SMOKE TEST: set sample=50, epochs=1, run `python api.py > smoke.log 2>&1`.
   Confirm the summary block appears. Restore the full config.
7. git commit (description: what changed + diagnostic that motivated it + what you expect)
8. python api.py > run.log 2>&1
9. grep "^outcome_auroc:\|^outcome_auprc:\|^onset_mae_hrs:\|^peak_vram_mb:" run.log
10. If empty: crash — tail -n 50 run.log, fix once if it's a bug, else log CRASH and move on.
11. Append to results.tsv with a 3-part description (change / diagnostic / observation).
12. Decide KEEP vs DISCARD using the rules below.
```

### Rules of the road (KEEP / DISCARD)

These rules supersede a simple AUROC comparison.

1. **A fixed auxiliary task cannot be un-fixed.** Once a Phase-1 or Phase-2 aux loss
   genuinely meets the *learning bar* below and a probe confirms the head has learned
   something non-trivial, that fix is locked in. You cannot roll the codebase back to a
   version where that task is broken, even if AUROC dips slightly. AUROC variance from
   random init alone is ~±0.01 — that is not a reason to undo a real structural fix.

2. **The learning bar (this is what "locked" means).** An aux is "learning" only if
   **both** of these hold:
   - **(a) Honest raw-loss drop.** The *raw* (un-weighted) loss decreases by ≥30%
     between its calibration epoch and the end of training, measured at ≥6-decimal
     precision. λ is fixed after calibration, so a weighted-loss curve that is flat
     at 4-decimal display can still hide a 5% raw drop or a 0% raw drop — log raw
     values to disambiguate. A weighted-loss "decrease" smaller than the 4-decimal
     display step is not evidence of anything.
   - **(b) Non-trivial ablation cost.** Disabling the aux (set its λ to 0 at config
     time, fully retrain through the same phases) costs ≥0.005 AUROC vs the
     ablated-otherwise-identical run. An aux whose ablation lies within the ±0.01
     random-init noise floor is decorative — it cannot be called "locked" and should
     be removed unless it is a *mandatory aux* per Rule 6.

3. **A fixed aux task can be replaced or removed.** Replacement (different loss /
   different target) and removal (delete the head and its loss term) are both honest
   changes. What is forbidden is silently regressing to a known-broken version while
   pretending it is a new experiment.

4. **The agent must distinguish AUROC gains from architecture vs from data shape.**
   If a KEEP run made multiple changes at once (e.g. added an aux *and* widened a BCE
   window *and* added a head), attribute the gain by ablating each component
   individually before claiming the architectural change is responsible. Data-shape
   changes (BCE window per token class, oversampling, masking strategy) frequently
   account for ≥+0.05 AUPRC while the headline architecture change does nothing —
   examples seen on this project. Log honest attribution in the description.

5. **AUROC is still the primary KEEP signal** once aux tasks are healthy. Among
   experiments where the aux tasks are all in their fixed state, KEEP the one with
   higher AUROC. If two are tied within ±0.005, prefer the simpler one (less code).
   A run that "wins" by +0.003 on a single seed is **not a real win** — re-run with
   a fresh Phase-1 (delete `checkpoints/phase1/`) before logging KEEP if the margin
   is below 0.005.

6. **Mandatory auxiliaries that must be solved properly (not removed).**
   The following are **required** in the codebase and must meet the learning bar
   (Rule 2). They are not eligible for the "fail/remove" option. If they appear
   broken, fix them — do not delete them.
   - **Phase 1**: one primary loss (multi-hot temporal BCE) + one time loss
     (Δt regression / Time2Vec supervision). Phase 1 has no other purpose;
     removing the time loss leaves no incentive to learn the time encoding.
   - **Phase 2**: primary BCE + next-token CE (confirmed learning across exps) +
     time loss (Δt gate + magnitude). These are the three signals that make the
     LM able to generate plausible future trajectories — without all three, the
     autoregressive trajectory used by `evaluate_on_test_set` degrades regardless
     of any outcome aux.
   - **Phase 2 outcome-direction signal**: **at least one** of
     {outcome soft-BCE, pairwise ranking loss, discrete-time hazard} must remain
     active and meet the learning bar. The agent may choose which one and may
     drop the others, but it may not run with all three silenced. The purpose is
     a head that explicitly pushes the model toward outcome-aware predictions in
     the natural-distribution Phase-3 fine-tune.

   For each mandatory aux that fails the learning bar today: the next experiments
   on that aux must focus on **fixing** it (better target, better head design,
   calibration of λ, different loss family) — not on patching the AUROC by
   piling on new heads elsewhere. A flat raw loss is the signal to dig in, not
   ignore.

7. **CRASH** = log the row with NaNs and `DISCARD`, then `git reset --hard HEAD~1`.

8. **DISCARD** = `git reset --hard HEAD~1` so the next experiment starts from the
   current best, not from the failed one.

**Embedder caching**: Phase 1 is skipped automatically when the checkpoint matches `(embed_dim, time2vec_dim, ctx_dim)`. Verify "Config unchanged — loading cached embedder" appears in run.log to confirm the cache was hit.

**Crashes**: fix typos/import errors and retry once. OOM or NaN loss — log as CRASH and move on.

---

## Reading diagnose.py output

Run from `emr_model/` as `python -m transform_emr.diagnose`. Loads Phase-3 checkpoint if available, otherwise Phase-2. Outputs to stdout.

### Report 1 — Per-outcome AUROC (teacher-forced LM logits)

Per-complication AUROC computed from LM-head logits under teacher forcing (correct input at every step). **These numbers are systematically higher than `evaluation.py`'s generation-based AUROC** because the model gets perfect context. Use this report to compare outcomes against each other and to track trends within a run, not to predict the final evaluation score.

- `Sep` = mean logit at positive positions minus mean logit at negative positions.
- Sep < 0.05 → logits are barely separated; the LM head is not learning outcome timing.
- `<<<` flag → AUROC < 0.55 for that outcome (near random). `>>>` → AUROC > 0.75 (strong signal).
- **LM head vs Outcome head table**: if outcome head consistently loses to LM head, Phase-3 fine-tuning is not contributing. If outcome head wins (marked `HEAD <<<`), the dedicated head is adding value.

### Report 2 — Logit calibration

All outcome logits pooled. Focus on:
- `Separation` and `Sigmoid[pos] - Sigmoid[neg]`: **healthy ≥ 0.1**, concerning < 0.05, bad < 0.02.
- `Logit[pos] mean` and `Logit[neg] mean`: if both are large negative numbers (e.g. −5 vs −7), the model has suppressed all outcome logits. The relative gap matters, but extremely negative logits indicate the outcome tokens are being pushed down by BCE training on frequent non-outcome tokens.

### Report 3 — Temporal coverage

How many positions have at least one positive target in the BCE window vs the eval window.
- BCE window too sparse (e.g. < 5% positions with ≥1 positive): the loss is nearly always zero → weak gradient. Consider widening `phase2_bce_window_hours`.
- BCE window too dense (> 50%): every position looks positive → calibration signal is noisy. The two numbers (BCE% and Eval%) should both be meaningful but not saturated.

### Report 4 — Lambda calibration (actual trained values)

Shows the real `lambda_max` computed during training from `lambda_max = cap × (anchor_bce / anchor_aux)`.

- `lambda_max` < 0.001 → **gradient-starved**: that loss term contributes almost nothing. Increase its `aux_fraction_caps` entry in `phase2_scheduler`.
- `anchor_bce` is the BCE loss at calibration epoch; `anchor_aux` is the raw aux loss. A very small `bce/aux` ratio (e.g. 0.0001) means BCE was tiny when calibration ran → multiply the cap to compensate.
- If the checkpoint is missing (training not yet run), falls back to showing the configured caps.

### Report 5 — Token gradient utility

Gradient² per occurrence for each token in the vocabulary. Outcome tokens should rank in the **top 30–40%** of vocabulary. If they're in the bottom half, the loss is not reaching them.

- `grad/occ` should be at least 1e-6 for meaningful learning. Below 1e-8 is near-zero.
- `<< LOW SIGNAL` flag → that outcome token is in the bottom half.
- Compare top-10 and bottom-10 tokens to understand which parts of the vocabulary dominate gradient flow.

### Report 6 — Context vector influence

Compares BCE loss with normal, zeroed, and shuffled patient context vectors.
- `delta (zeroed)` ≈ 0 → context is not being used. Check AdaLN conditioning and `ctx_dim`.
- `delta (shuffled)` ≈ `delta (zeroed)` → the model isn't distinguishing patients. Expected if context has low variance in the batch.
- Large negative delta (shuffled/zeroed gives higher loss) → context is genuinely helpful.

### Report 7 — Embedder linear probe

Cross-validated AUROC from frozen Phase-1 embeddings alone (logistic regression).
- > 0.65 → Phase-1 already captures useful outcome-predictive structure. Good foundation.
- ≈ 0.50 → Phase-1 embeddings carry no outcome signal. Phase-2 is doing all the work (or not).
- This measures the *embedding quality*, not the downstream model.

### Report 8 — Vocab health

Flags two pathological categories:
- **Frequent-noisy**: high-frequency tokens where the model has low confidence and the next-token distribution is very uncertain. These tokens may be adding noise to the BCE gradient.
- **Rare-unlearned**: low-frequency tokens where the model has never learned to predict them. These may include outcome tokens — check if they appear here.

### PROBE — Δt HEAD

Pearson r and R² between predicted and actual inter-event time gaps (in hours).
- r < 0.1 or pred_std < 0.05h → **Δt head has collapsed**: predicts the same gap for every event regardless of context. The time head is not contributing to temporal reasoning.
- Healthy: r > 0.3, pred_std comparable to true_std.

### PROBE — Outcome head label alignment

For each outcome, compares mean head logit at positive vs negative positions.
- `flip = True` → the head predicts **higher logits when the outcome is absent**. This is a sign error in the label construction or loss polarity — fix it before any other change.
- `gap` > 0 is correct direction. `gap` close to 0 means the head has learned nothing.
- `auroc` from the outcome head directly (not the LM head): < 0.5 = inverted, ≈ 0.5 = random, > 0.6 = useful.

### PROBE — Outcome head logit distribution

Mean, std, p50, p99, abs-max of each outcome head's raw logits across all non-pad positions.
- Very high `std` or `abs_max` (e.g. > 10) → logits are exploding. Gradient clipping or lower learning rate for Phase 3.
- Very low `std` (< 0.01 for all outcomes) → head is outputting near-constant values; it has not learned to differentiate timing.

---

## Code quality and GPU performance

### Code quality

- Every function must have a docstring following the project standard (Purpose / Method / Args / Returns). Do not skip this.
- Prefer small, focused changes. One architectural idea per commit.
- Do not leave dead code, commented-out blocks, or half-finished experiments in the codebase. If you try something and discard it, `git reset --hard HEAD~1`.
- Added code should be GPU-friendly - optimize for performance whereever poossible.

### GPU performance — do not break these

The following optimisations are already active. Do not accidentally remove them:

- **Mixed precision (BF16 AMP)**: `torch.autocast(device_type=..., dtype=torch.bfloat16)` wraps the forward *and* the backward in `pretrain_transformer`. Removing it roughly doubles memory use and slows training.
- **Gradient checkpointing**: each `AdaLNBlock` forward is wrapped in `torch.utils.checkpoint.checkpoint(...)`. Removing it increases peak VRAM by ~30–40% and will OOM on a 48 GB card at this model size. The `_ckpt` closure uses default-arg block capture to avoid the closure bug (all blocks recomputing using the last block's weights).
- **Bucket batching**: `get_dataloader(..., bucket_batching=True)` groups sequences by length to minimise padding waste within each batch. Removing it cuts effective GPU utilisation.
- **Grad accumulation**: `grad_accumulation_steps=4` in `TRAINING_SETTINGS` simulates a larger effective batch without the VRAM cost. If you change batch size, adjust this to keep the effective batch constant.

### GPU performance — things worth trying

- **`torch.compile(model)`**: if the PyTorch version on the pod supports it (`torch.__version__ >= 2.0`), wrapping the model with `torch.compile` can give 10–30% throughput improvement with no code changes. Add it after Phase-1 embedding load, before Phase-2 training.
- **Profile before optimising**: if a run seems slower than expected, check `peak_vram_mb` in run.log and whether the GPU is actually saturated (`nvidia-smi dmon`). Do not optimise blindly.

## Architecture notes (what is already implemented)

These are baked into the current codebase — do not re-implement:

- **Temporal BCE**: loss window is in real hours (`phase1_bce_window_hours`, `phase2_bce_window_hours`), not token steps. Step-based BCE created contradictory gradients for outcome tokens.
- **AdaLN-Zero**: patient context injected at every block via AdaLN. Do not swap to RMSNorm — the mean subtraction in LayerNorm is load-bearing for AdaLN-Zero's gate initialisation.
- **Temporal RoPE**: Q and K rotated by actual `abs_ts` deltas, not token index. Index-based RoPE is meaningless for irregular time series.
- **SwiGLU MLP**: standard in current GPT blocks.
- **Weight-tied LM head**: LM head shares weights with token embedding.
- **Phase-3 outcome fine-tuning**: backbone frozen, outcome head trained on natural-distribution data with time-decayed soft labels.
- **Curriculum scheduling**: auxiliary losses (ce, dt, outcome) activated in stages after BCE warm-up, with lambda calibration relative to BCE magnitude.

---

## Research directions

### How to approach every task

The goal is to **fix broken architecture and make learning meaningful**, not tune hyperparameters on a broken one. If something is architecturally wrong, no cap or LR adjustment will fix it. Run `diagnose.py` before and after every experiment and confirm in the output that the specific failure mode you targeted has changed.

The tasks below are a prioritised starting point, not an exhaustive list. You are free — and encouraged — to draw on any architectural ideas from similar deep learning research (clinical NLP, time-series transformers, event prediction, survival models, etc.) if they address a diagnosed failure mode. The bar is: does it make the gradient signal more meaningful, does it give the model a better structural inductive bias for this problem, or does it fix a known gap between how the model is trained and how it is evaluated? If yes, try it. You do not need permission for individual experiments — that is the point of the loop.

Examples of the kind of lateral thinking that is in scope:
- Replacing a loss that produces near-zero gradient with one that is better calibrated to this data distribution
- Adding supervision signal from a different angle (e.g. contrastive, ranking, or survival-style losses) if the current BCE/CE is provably not reaching the outcome tokens
- Redesigning how the dataset is built or how sequences are batched if there is evidence the current approach creates misleading targets
- Borrowing positional encoding or attention designs from time-series or irregularly-sampled sequence models

**Logging discipline**: write a `description` that captures three things on one line:
1. What you changed
2. What diagnostic observation motivated it
3. What you expected / observed

Example: `"wrap backward in autocast; diag Report-4 showed lambda_outcome near-silent due to AMP checkpoint mismatch; phase2 grad stable"`
Not just: `"fix checkpoint bug"`.

This allows the experiment log to be read as a research journal, not just a list of commits.

Tasks are ordered by priority. Do not start Task N+1 until Task N is resolved.

---

### Experiment history and settled findings (~54 experiments to date)

Read this before proposing anything — it records what has been tried and what conclusions were drawn. Re-read it every session.

**Current best — exp49** (`672695b`): AUROC = 0.804, AUPRC = 0.282, MAE = 84.9.
**Current baseline (post-Task-A fix) — exp52** (`d4a94ec`): AUROC = 0.788, AUPRC = 0.239, MAE = 87.6. Baseline AUROC moved down from exp49 because exp52 retrained Phase-1 with a locked-in Δt fix; the regression is dominated by fresh-Phase-1 noise. Continue building from exp52.

**Confirmed locked in (do not undo or roll back):**
- `outcome_cap = 9–10` in `phase2_scheduler` — values <6 starve the gradient, >10 destabilise.
- `bce_only_epochs = 4` (exp28) — stronger LM base before curriculum unlocks helps net.
- `outcome ramp_epochs = 3` (exp32 reverted) — ramp=1 is a zero-sum tradeoff across outcomes.
- `early-stop-patience = 10` (exp49) — longer P3 lets the outcome head converge through transient val plateaus.
- Phase-3 differential LR (`backbone 1e-6, head 1e-4`) — matches best AUROC across configs. **Costs ~13 GB of VRAM** (exp18 ~6 GB → exp21 ~19 GB); see Task D.
- AMP/checkpoint fix (`loss.backward()` inside `torch.autocast`) — gradient stability confirmed (Task 1).
- Temporal attention bias (Task 4B, exp40) — kept in the baseline. **Costs ~6 GB of VRAM** (exp39 ~19 GB → exp40 ~25 GB); see Task D.
- Shared hazard head + per-bin bias (exp46) — `hazard_logit[k,b] = outcome_logit[k] + bias[k,b]`.
- **Time2Vec log-spaced freq init** (exp52, Task A fix) — frequencies span 12–25k rad / normalized-unit with alternating signs. Δt R² 0.024 → 0.083 (≥ 0.05 bar), `tr_dt` decreases monotonically. **Task A is fixed and locked.**

**Confirmed failing — do not repeat:**
- `n_layer 4→6`: regression both times.
- `bce_window 12→6h` and `12→24h`: both worse.
- `outcome_cap > 10` / `< 6`: regression.
- `outcome_ramp_epochs=0`: destabilises.
- `time2vec_dim 32→64`: fresh P1 weaker, net regression.
- Outcome→LM coupling (exp19, exp20, exp33–36, **exp51 even with shared hazard + patience=10**): coupling shifts the AR-generation trajectory distribution away from training and the outcome head ends up miscalibrated on the shifted trajectories. **Structurally incompatible with the generation-based evaluation. Do not retry.**
- Wider outcome head (exp24, 2D hidden) / deeper outcome head (exp50, extra hidden layer): both delayed Stage 1 curriculum activation and hurt net AUROC. Outcome-head *capacity* is not the bottleneck.
- Token-type flag embeddings (exp39): hurt or noise — abandoned.
- Hazard cap >5 (exp43) and hazard bins=6 (exp45): both worse than exp42.
- Phase-3 weight_decay=0 (exp26) and ReduceLROnPlateau in P3 (exp27): both hurt.
- **Phase-1 MLM**: three honest variants all failed.
  - exp37 (disabled): AUROC -0.015.
  - exp38 (hierarchy-masked, all four token IDs masked): AUROC -0.035.
  - exp53 (span-MLM, span=4): AUROC +0.005 but **broke locked Task A** — Δt R² regressed from 0.083 to -0.118, violating Rules #1.
  - exp54 (running on the pod as of this writing): clean **removal** — `mlm_head`, `forward_with_mlm`, `build_mlm`, loss term, and scheduler entry all deleted. If exp54 lands within ±0.01 of exp52 AUROC, Task B is officially fail-removed. If it drops further, restore the simplest MLM variant (exp37 baseline, cap=1.5, no hierarchy masking) and proceed without trying to "fix" it.

**Active open problems.**
- **Two-tier (terminals vs. rest) BCE window is the live research lead — but pause before generalising it.** exp59 widened the BCE window to 168 h for terminal tokens (RELEASE / death) and gained +0.10 AUPRC and dropped `max_len` 89%→12%. This is a data-shape change, not an architectural one — and it dwarfs every aux-loss gain the project has logged. **However**: hand-picking windows per individual event family (e.g. one window for vitals, another for labs, another for treatments) is dangerous. Within MIMIC-III, lab measurement timescales vary enormously (glucose minutes; CBC hours; lipid panels days), as do vital cadences across ICU settings. Fitting per-family windows to observed correlations on this dataset is implicit hyperparameter tuning on the eval signal and **will not transfer**. The right way to push this lead:
  1. First confirm via Task 0 step 4 that the +0.10 gain attributes specifically to the terminal-token widening, not to the aux losses present in exp59.
  2. Then keep the two-tier split (terminals vs. everything else) as the only hand-coded shape change. Justification: terminals are structurally different events — they end the sequence — so a different supervision window is principled, not dataset-specific.
  3. For lab/vital/treatment heterogeneity, do **not** hand-pick windows. Instead let the model learn the window: e.g. a learned per-token-class log-Δt weighting on the BCE loss, or a soft attention over a small log-spaced grid of windows. The window becomes a parameter, not a hyperparameter.
  4. Validate any further shape change with a fresh-Phase-1 re-run before logging KEEP. Data-shape gains are unusually seductive and unusually prone to overfitting to MIMIC-III's specific event distribution.
- **The mandatory aux losses still need to be solved properly, not propped up.** Phase-1 Δt is "locked" at R²=0.083 — better than nothing but small. Phase-2 outcome soft-BCE / hazard / ranking are flat or near-floor. At least one of the three Phase-2 outcome-direction signals must be brought to a genuine learning state per Rule 6. The agent is required to find which works — but is not required to keep all three.

---

## Research directions

Open tasks below. Work them **in the listed order**. Do not skip ahead. **No hyperparameter sweeps**; if your only proposed change is a number, you have not understood the task. Re-read this section before every experiment.

### Task 0 — Honest audit (DO THIS FIRST, BEFORE ANY NEW EXPERIMENT)

The session through exp60 racked up several "locked" aux tasks (ranking, hazard, outcome, dt) whose weighted-loss curves are flat at 4-decimal precision and whose AUROC gains are inside the ±0.01 random-init noise floor. Before launching any new experiment, run the following audit against the current best codebase and update results.tsv with the findings:

1. **Raw-loss probe at 6+ decimals.** Re-log every aux (phase-1 Δt, phase-2 ce / dt / outcome / ranking / hazard) at ≥6 decimal places across all curriculum epochs. Compute the raw drop fraction `(raw_first_active - raw_final) / raw_first_active`. Any aux with < 30% raw drop fails the Rule-2(a) bar.

2. **Per-aux ablation.** Take the current best (exp59 or exp60 if it KEEPs) as the codebase. For each aux that is *not* mandatory under Rule 6, run a single experiment with that aux's λ forced to 0 from the start (no curriculum unlock; equivalent to deleting it). Compare AUROC/AUPRC vs the un-ablated run on the same fresh Phase-1.
   - Ablation cost ≥ 0.005 AUROC → aux is real, keep it.
   - Ablation cost within ±0.005 AUROC → aux is decorative, remove it.
   - Order: ablate hazard first (flat at 4-decimal display is the strongest suspect), then ranking, then outcome soft-BCE.
   - Rule 6 floor: at least one of {outcome soft-BCE, ranking, hazard} must remain. If the first two ablations both come back "decorative", do not ablate the third — it stays by mandate.

3. **Re-run any ±0.005 KEEP on a fresh Phase-1** to confirm the gain is not single-seed noise. Specifically: exp56 (Task C lock, +0.003 AUROC) needs re-confirmation. If it does not survive a fresh Phase-1, downgrade its status from "locked" to "neutral".

4. **Attribute exp59's +0.10 AUPRC gain.** exp59 widened the BCE window for terminal tokens to 168 h and gained AUPRC 0.282→0.386 alongside max_len 89→12 %. Ablate that change in isolation (keep all aux as in exp59, revert BCE window to exp58's setting) to confirm the data-shape change — not the aux losses — drove the AUPRC win. This is the live research lead; understanding *why* it worked unlocks the next direction.

5. **Update results.tsv with the audit outcome.** For each aux re-classified as "decorative" by ablation, add a row noting: `<commit>  <auroc_with>  <auroc_without>  <delta>  AUDIT  removed <aux> per Rule 2(b)`.

Only after Task 0 is complete may the agent launch a new experimental direction. The point of this audit is to stop calling near-zero deltas "learning" and stop attributing data-shape AUPRC gains to architecture.

---

### Task 1 — COMPLETE
Gradient stability. Done.

### Task A — Phase-1 Δt head — LOCKED (exp52)
Time2Vec log-spaced frequency init (12–25k rad/normalized-unit, alternating signs). Δt probe R² 0.024 → 0.083, `tr_dt` monotonically decreasing. Fixed. Locked. Do not undo.

### Task B — Phase-1 MLM — REMOVE (no learning across three honest attempts)

exp37 (disabled), exp38 (hierarchy-masked), exp53 (span-MLM) all failed to make MLM learn: `tr_mlm` flat across Phase-1 in every variant, embedder linear probe (Report 7) unchanged in every variant, and exp53 additionally broke the locked Task A by destabilising Δt training. **A loss term that does not learn is not a learning signal — it is noise dressed as supervision, and it stays in the codebase only out of inertia.** Three attempts is enough; remove it.

The MLM fail/remove path was committed as exp54 but the pod was shut down before evaluation, so the AUROC was never measured. The removal commit has been **reverted** so the codebase is back at the exp52 baseline. **Task D will run on exp52** for clean attribution. After Task D is locked, redo exp54 (delete `mlm_head`, `forward_with_mlm`, `build_mlm`, the loss term, and the `phase1_scheduler` MLM entry) on top of D-fixed code. Log the resulting AUROC.

**The removal is not contingent on AUROC outcome.** Per Rule #1 (a broken aux is not a learning signal), MLM stays removed even if AUROC drops — that drop is then a free signal that *Phase-1 needs a different self-supervised task*, which becomes the new follow-up question, not a reason to put a flat-loss MLM back. If AUROC drops more than ~0.015 below the post-D exp52-equivalent, log it and proceed to Task C anyway; if Task C still falls short of the exp49 mark, you may then propose a *different* Phase-1 self-supervised task (next-event prediction, contrastive over event types, or similar) — not a re-introduction of MLM.

---

### Task D — Memory and time efficiency (NEXT — DO BEFORE TASK C)

**Why this is now the priority:** peak VRAM jumped from ~6 GB (exp1–18) to ~19 GB at exp21 (Phase-3 differential LR), then to ~25 GB at exp40 (temporal attention bias). The model is currently using almost all of a 48 GB A40. Task C will add a ranking-loss tensor of shape `[B, T, K, pairs]` or similar — without efficiency work first, that experiment will OOM. The same applies to any future architectural addition. Equally, fixing this unlocks running on smaller cards (24 GB, etc.) for the rest of the project.

The two memory pressure points are well-localised. Investigate them in this order:

#### D1 — Temporal attention bias kernel fallback

`CausalSelfAttention.forward` adds a learned bias `g(Δt_ij)` to attention. This likely forces `F.scaled_dot_product_attention` (SDPA) off its memory-efficient or flash-attention backend onto the math fallback, which materialises the full `[B, n_head, T, T]` attention-weight matrix in fp32 and saves it for backward. At `B=16, n_head=4, T~500, n_layer=4`, that matches the observed ~6 GB jump from exp39→exp40.

**Steps:**
1. Audit how the bias is applied — is it added manually to `q @ k.T` before a hand-written softmax, or passed via `attn_mask=` to SDPA?
2. Construct the bias as a `[1, n_head, T, T]` (or broadcastable) **bf16** tensor and pass it via `attn_mask=` to `F.scaled_dot_product_attention`. Check whether SDPA selects the memory-efficient backend (use `torch.nn.attention.sdpa_kernel(...)` context manager or set `enable_math=False, enable_flash=True, enable_mem_efficient=True` and verify no fallback warning).
3. If the bias depends only on per-position values (Δt) and not on i,j independently, factor it as a low-rank approximation that flash-attention will accept — or precompute once per batch and broadcast.
4. Measure VRAM and step time before and after.

**Pass criterion:** peak VRAM drops by ≥3 GB **and** Phase-2 step time does not regress more than 10%, **and** AUROC stays within ±0.01 of exp52 baseline. Lock the fix.

#### D2 — Phase-3 differential LR activation cost

Storing backbone activations for backward through 4 transformer layers is the dominant cost in Phase 3, accounting for the ~13 GB jump from exp18→exp21. The improvement from differential LR is genuine (matched best AUROC across configs) and remains locked — but the activation-memory cost can be cut without changing the optimisation.

**Steps:**
1. Apply **gradient checkpointing** to the backbone *in Phase 3* (not just Phase 2). Each `AdaLNBlock` recomputes its forward during backward — trades ~30% compute for ~50% activation memory.
2. Reduce Phase-3 `batch_size` to half (e.g. 16 → 8) and double `grad_accumulation_steps` so the effective batch stays the same. P3 only has 21 epochs in exp49 — the wall-time cost is small.
3. Use AMP/bf16 in Phase 3's backward pass too (verify `loss.backward()` is inside `torch.autocast` for P3, not just P2).

**Pass criterion:** peak Phase-3 VRAM drops by ≥5 GB **and** AUROC stays within ±0.01 of exp52 baseline. Lock the fix.

#### D3 — Phase-2 step time

If D1/D2 free up VRAM headroom, also try:
1. **`torch.compile(model)`** after Phase-1 load, before Phase-2 training. 10–30% throughput on supported PyTorch versions, no accuracy cost. Verify with a smoke test first — `torch.compile` occasionally hits graph-break issues with custom attention.
2. Check whether `bucket_batching=True` is actually clustering by length effectively at the current sample distribution. If most batches are mostly-padded, switch to a tighter bucket size.

#### D4 — Probe / smoke-test the OOM bound

After D1/D2, attempt one experiment that *adds* a small dummy `[B, T, K]` tensor to the Phase-2 forward (size-matched to what Task C will introduce). Confirm no OOM at the current batch size. This validates that Task C has headroom.

**Stop criterion for Task D:** peak VRAM is ≤ 18 GB across all phases (giving ~30 GB headroom on a 48 GB card and fitting comfortably on 24 GB cards), **and** AUROC is within ±0.01 of exp52, **and** total runtime has not regressed more than 15%. Move on.

---

### Task C — Phase-2 outcome loss (survival / ranking loss)

**Failure mode**: outcome head soft-BCE loss is near-flat during Phase-2 training even at correct lambda. AUPRC has not recovered above 0.28. The hazard auxiliary (exp42, exp46, exp49) added structure and helped AUROC, but the *outcome-head soft-BCE* itself is still not optimising what evaluation measures.

**This is the only learning-task the agent has NOT yet honestly attempted.** Coupling (exp33–exp51) was repeatedly tried instead — coupling is rejected. **The survival / pairwise ranking loss has not been implemented yet.** Do it after Task D unblocks the VRAM budget.

**The mechanism:**
For each outcome k and each patient, sample positive positions (within the eval window of the outcome) and negative positions (outside the window, or any position from a patient where k never occurs). Apply a pairwise margin loss on `outcome_head[k]` logits:

```
L_rank_k = mean over (pos, neg) pairs of  softplus( logit_neg - logit_pos )
```

This is a direct AUROC proxy — minimising it directly raises the probability that a positive position is ranked above a negative one, which is exactly what `evaluate_on_test_set` measures (pooled window AUROC).

**Implementation steps:**
1. Add a `pairwise_ranking_loss` in `loss.py` operating on `outcome_logits` of shape `[B, T, K]` with the same positive/negative position mask used by the current soft-BCE.
2. **Additive first**: keep the existing soft-BCE outcome loss, add the ranking loss with its own cap entry in `phase2_scheduler.aux_fraction_caps` (e.g. `ranking: 0.2`), schedule it in the same stage as `outcome`.
3. Watch `tr_ranking` and `tr_outcome` both during Phase-2 training. If `tr_ranking` decreases visibly and AUROC + AUPRC improve, the task is alive.
4. **Replacement second (only if additive works)**: drop soft-BCE entirely and run with ranking-only outcome loss. Cleaner gradient, no `tau` / `cap` calibration coupling.

**Pass criterion (fixed task):** `tr_ranking` decreases across Phase-2 **AND** AUROC ≥ 0.804 (the exp49 mark) **AND** AUPRC ≥ 0.282. Once you hit this, the ranking loss is locked in.

**Fail option:** if both additive and replacement variants fail to improve AUROC after honest attempts, document the failure precisely (what `tr_ranking` did, what AUROC did) and remove the ranking loss. **Do not** fall back to coupling — that is rejected.

---

### Order and rules

1. Work the open tasks **in order: finish B (exp54) → D (efficiency) → C (ranking loss)**. Do not move to the next until the current one is either fixed and locked or honestly removed. Task A is locked.
2. Once a task is fixed (loss decreases + probe passes the bar above), it is **locked**. The codebase moves forward with the fix in place. You may not silently regress to a pre-fix version even if it gives a higher AUROC — variance from random init is ±0.01 and does not justify rolling back a real structural fix. You may *replace* a fixed task with a different method, or *remove* it entirely (and report the consequence) — both are fine.
3. **No hyperparameter sweeps.** No "tune LR / cap / ramp / patience / batch size" experiments. Every experiment must change *what is being learned*, *how it is being learned*, or *how efficiently it is being computed* (Task D) — not just what number is in the config.
4. **Re-read this `program.md` and `diagnose.py` output between every experiment.** If you skip this, you will drift.
5. **Smoke test (sample=50, 1 epoch per phase) before every full run.** Confirm the summary block appears. Pipeline crashes on full runs are wasted hours.
6. **Memory and time efficiency are first-class research targets.** Adding code that genuinely improves either, without hurting AUROC, is a KEEP — same status as an AUROC improvement. Removing code while maintaining performance is always a win.

### When to stop

Stop and report to the user when:
- Tasks B, D, and C are each either fixed and locked, or have been honestly attempted and removed/concluded, **and**
- The only remaining levers are hyperparameter tuning.

Write a final summary: the state of each task (fixed / removed / failed), the current best AUROC/AUPRC/MAE, peak VRAM, and what is still open.

