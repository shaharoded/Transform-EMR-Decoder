# autoresearch — Patient-Level Eval Reframing

## The reframing

Prior loop used per-(patient, window) AUC. It was dragged by rare
outcomes and double-punished partial wins. New eval is per-patient
peak-detector: for each (patient, outcome), score = `max P_outcome` over
the generated portion, label = outcome occurred in GT. MAE = distance
between `argmax_t P` and the **nearest** GT occurrence.

### Headline keys (in `api.py` summary block)

| Key | Meaning |
|---|---|
| `patient_auroc_weighted:` | Support-weighted mean per-outcome AUROC. **Primary**. |
| `patient_auprc_weighted:` | Support-weighted mean AUPRC. |
| `patient_auroc_simple:`   | Unweighted mean (sanity). |
| `n_outcomes_used:`        | Outcomes passing the 1 % prevalence threshold. |
| `patient_per_outcome\t…`  | Per-outcome AUROC/AUPRC/n_pos/n_neg/prevalence. |
| `peak_mae_hrs\t…`         | Per-outcome MAE to nearest GT occurrence. |

Legacy keys (`outcome_auroc`, `multi_horizon\t…`, etc.) still emit for
back-compat. Outcomes below 1 % test-set prevalence get `auroc=nan` and
are excluded from the weighted mean.

VRAM ≤ 24 GB. Data and `bak_originals` are read-only.

## What's already on this branch

You inherit a clean project. **Do not touch `api.py` or `evaluation.py`.**
Edit only `emr_model/transform_emr/**` and its `config/**`.

- `evaluation.py` already has `per_patient_max_auc`,
  `weighted_mean_auc`, `time_accuracy_nearest`, and the legacy
  per-window functions.
- `api.py` summary block emits all keys above.
- Z architecture (narrow + frozen `log_tau_lm[terminal]`) is the
  starting point — code on HEAD, no checkpoint on disk (pod is fresh).
- Ledger: `results/results-trajectory-fix.tsv`.

## The loop

10k-sample is the primary workspace (~25–30 min per training run).
Full-data confirm only at end of a block or when running best is stable.

```
1. Read program.md. Check git log + last rows of
   results/results-trajectory-fix.tsv.
2. Propose ONE change with a falsifiable hypothesis.
3. SMOKE (sample=50, phase{1,2,3}_n_epochs=1):
      python api.py > smoke.log 2>&1
   Gate-A: no NaN/inf in any tr_* loss term.
   Gate-B: every aux's raw magnitude within ~1–2 OOM of BCE.
   Gate-C: calibrated λ in [1e-3, 10].
   Gate-D: summary block prints, all headline keys present.
   (P3-specific gates listed in P3 section.)
4. git add <files> && git commit -m "<tag>: change / why / expected" && git push
5. EXPERIMENT (sample=10000):
      python api.py > run.log 2>&1
   POST-TRAIN:
   T1: every aux's raw loss decreases across its active phase.
   T2: early stop didn't fire before auxes finished ramping.
   T3: diagnose.py shows real discrimination on key probes.
6. Append row to results/results-trajectory-fix.tsv (new headline keys).
7. Write `### <tag>` block in status.md → `Verdict: KEEP|DISCARD — …`.
8. Journal commit + push.
9. DISCARD → git revert --no-edit <CODE_SHA> && git push.
10. KEEP → cp -r emr_model/checkpoints emr_model/checkpoints.bak_keep_<tag>.
11. After each KEEP, re-eval running best at 10k (--eval-only) to refresh.
12. FULL-DATA CONFIRM (sample=None) when running best stable across
    2–3 DISCARDs, OR a block ends, OR user asks.
```

### KEEP rule (vs running best at 10k)

- All smoke gates A–D + post-train T1–T3 passed.
- ≥ 1 headline lifts past noise: AUROC ≥ +0.010, AUPRC ≥ +0.010, MAE ≤ −5 h.
- No headline regresses by the same threshold.
- `gen_to_gt_ratio_median` doesn't drop below 0.4.

Otherwise DISCARD → revert.

## Research directions (in order)

### P0 — Baselines under the new headline

Two baselines must exist before any new direction is judged. Both are
straight 10k runs; the better one becomes the running best for P1.

**B0-Z** — Z is already on HEAD. No code change required, just run.
Z = direction E (narrow + frozen terminal `log_tau_lm`). Marker block
is in `emr_model/transform_emr/transformer.py` ~line 463 — search for
the comment `# Direction E: freeze the terminal entries of log_tau_lm`.
Terminal-init value is set in
`emr_model/transform_emr/config/model_config.py` (`_log_tau_terminal`,
currently `math.log(12.0 / 336.0)`).

**B0-C-ttt** — re-apply commit **`dd3fc1b`** ("C-ttt-head: time-to-
terminal regression aux (direction C) on Z") on top of HEAD. Adds an
auxiliary head predicting `log1p(t_terminal − t_now)` per non-terminal
position with MSE loss; shares the backbone. Touches four files:
```
emr_model/transform_emr/config/model_config.py
emr_model/transform_emr/diagnose.py
emr_model/transform_emr/inference.py
emr_model/transform_emr/transformer.py
```
Recipe:
```
git show dd3fc1b --stat                 # inspect scope
git cherry-pick --no-commit dd3fc1b     # apply diff, keep staged
# resolve conflicts if any (HEAD is post-Z, dd3fc1b was on top of Z)
# then commit as your own B0-C-ttt tag
```
Was DISCARDed under the old per-window eval (rare-7 flipped), but
produced DEATH window-AUC 0.79 — expected to dominate on patient-level
DEATH AUC.

### P1 — MIL patient-level max-BCE aux loss (Phase 3)

```
score_patient = softmax_t(logit_outcome(t) / T) · logit_outcome(t)
loss_mil = BCE(σ(score_patient), patient_binary_label)
```

Soft max (temperature ~1.0, learnable per-outcome). Schedule via
`LambdaScheduleController` with `aux_fraction_cap` ~ 0.20. Existing
per-position BCE stays as the 48-h calibration anchor.

**Falsifiable**: patient AUROC ≥ +0.03 vs best of {B0-Z, B0-C-ttt};
per-window AUROC drop < 0.10.

### P2 — Soft-argmax time loss, positives only (Phase 3)

```
weights = softmax(logit_outcome(t) / T_time)
predicted_t = sum_t(weights · t)
loss_time = smooth_l1(predicted_t, nearest_t_in_gt(outcome, patient))
```

Per-outcome learnable `T_time` (~13 scalars). Only for patients with the
outcome.

**Falsifiable**: `peak_mae_hrs` for {DEATH, RELEASE} drops ≥ 5 h;
patient AUROC doesn't regress.

### P3 — Risk-aware LM head (architectural coupling)

Currently the LM and outcome heads only share a backbone. P3 makes the
outcome-head's prediction influence which tokens the LM emits.

**The change.** Linear projection `bias_proj: n_outcomes → vocab_size`.
Per position:
```python
# B=batch, T=seq, D=hidden, V=vocab, K=n_outcomes
h           # (B,T,D)
lm_logits = LM_head(h)                  # (B,T,V)
o_logits  = outcome_head(h)             # (B,T,K)  ← must be 3D
P         = torch.sigmoid(o_logits)     # (B,T,K)

assert P.shape == (B,T,K)
assert lm_logits.shape == (B,T,V)

bias = bias_proj(P)                     # (B,T,V)
assert bias.shape == lm_logits.shape

combined_logits = lm_logits + bias      # (B,T,V)
```
No shift — `P[t]` and `lm_logits[t]` both predict from `h_t`. (Prior
attempts died on: silent broadcast from missing T axis on P; off-by-one
shift; bias_proj weight not zero-initialised so step-0 disrupted CE.)

**Init must be no-op**: `nn.init.zeros_(bias_proj.weight)`. Step 0
combined_logits == lm_logits exactly.

**Phase 3 unfreezing**: LM head must have `requires_grad=True`, LR
multiplier ~0.1×base. `bias_proj` and outcome head get full base LR.

**Smoke gates** (additions on top of A–D):
- **P3a**: zero-init no-op — CE on first batch matches non-coupled
  baseline to 1e-6.
- **P3b**: after first backward, all three grads non-zero:
  `bias_proj.weight.grad.norm()`, `outcome_head[-1].weight.grad.norm()`,
  `LM_head.weight.grad.norm()`, each > 1e-8.
- **P3c**: shape asserts pass.

**Per-epoch print** (Phase 3 and `diagnose.py`):
```
P3 coupling stats epoch <e>:
  ||bias|| / ||lm_logits||  mean: <r>  max: <r>
  bias_proj.weight row norms (per outcome): [DEATH=.., RELEASE=.., ...]
  per-outcome contribution to terminal logits: [DEATH→TERM=<>, ...]
```
Healthy ratio band: **0.05–0.3**. Below → no coupling formed.
Above → bias dominates, LM atrophies.

**Behavioural probe** (`diagnose.py`, held-out batch):
```
Pearson(P_DEATH[t], terminal_token_logit[t]):  <ρ>  (> 0.3 healthy)
gen_to_terminal_hrs on positives vs negatives, Δ in hours (>0 healthy)
```
Δ ≤ 0 → coupling didn't form behaviourally → DISCARD even if AUC moved.

**Falsifiable**: patient DEATH/RELEASE AUROC ≥ +0.03 vs P1+P2;
behavioural Δ > 12 h; coupling ratio in [0.05, 0.3].

The `bias_proj` row weights — which outcomes bias which tokens — are
publication-worthy figure material.

### P4 — Patient-level pooling head

Learned attention pool over generated hidden states, queried by
per-outcome embeddings → per-patient score replaces "max P_outcome" in
eval. Per-position outcome head stays for 48-h calibration. ~150 LOC.

Defer unless P1+P2+P3 plateau.

**Falsifiable**: patient AUROC ≥ +0.05 vs P1+P2+P3.

### P5 — Architecture scale-up

**Trigger**: P0–P4 honestly attempted, running best clearly beats
`bak_originals` (≥ +0.03 patient AUROC, trajectory honesty preserved),
recent 10k experiments DISCARDing.

Lift M-256 lock. Scan grid with running-best loss recipe:

| Tag | embed_dim | n_layer | n_head | Approx params |
|---|---|---|---|---|
| M-128 | 128 | 4 | 4 | ~2 M |
| M-256 | 256 | 4 | 4 | ~6 M (baseline) |
| M-384 | 384 | 6 | 6 | ~15 M |
| M-512 | 512 | 6 | 8 | ~25 M |
| M-768 | 768 | 8 | 12 | ~55 M |

OOM at full-data confirm → halve batch + double grad-accum; if still
OOM, that's the size ceiling.

**Decision**: smallest variant within ~0.005 of best (prefer smaller).
Confirm at full data.

### P6 — M0 ablation: per-position outcome BCE redundancy

After P5, down-weight / disable per-position BCE (`aux_fraction_cap`
→ 0.02 or off). Structural diagnostic, not a KEEP/DISCARD candidate:
- Patient AUROC holds + cap=48h doesn't collapse → per-position BCE
  was redundant for ranking; keep small for calibration only.
- cap=48h collapses → per-position BCE is the calibration anchor; keep.

### P7 — Final: toggle `USE_QA_DATA` on the very best model

**Trigger**: P5 chose architecture, P6 settled the loss recipe,
running best is the genuine end-of-loop candidate.

Toggle `USE_QA_DATA = True` in
`emr_model/transform_emr/config/dataset_config.py`. This adds context
features AND new tokens — the cached vocab/scaler/datasets are stale.

**Pre-flight (mandatory before `python api.py`)**:
```bash
rm -f emr_model/checkpoints/tokenizer.pt
rm -f emr_model/checkpoints/scaler.pkl
rm -f emr_model/checkpoints/processed_datasets.pt
rm -rf emr_model/checkpoints/phase1
```

Phase 1 retrains from scratch (only experiment in the whole loop that
does). Verify after rebuild: `len(tokenizer.token2id)` must be strictly
greater than the non-QA value — if equal, QA tokens weren't picked up.

Smoke (sample=50) before full data; this final run is at sample=None
since the result is publishable, not a 10k probe.

**Falsifiable**: full-data `patient_auroc_weighted` lifts ≥ +0.005 OR
QA-introduced tokens visibly emitted in generated trajectories. If
neither, the non-QA running best stands as the final result.

## Inference-side directions (no retraining)

`python api.py --eval-only`:
- **F1**. Beam search with length-normalised scoring.
- **F2**. Temperature schedule to escape immediate-terminal local minimum.

Try after backbone work plateaus.

## Stop criterion

No quality target — push as high as the model honestly allows. Stop when:
- All P0–P7 honestly attempted,
- Last 2–3 10k experiments DISCARDed,
- Full-data confirm of running best done.

Write final report in `status.md`: running-best numbers (weighted
AUROC, per-outcome AUROC for DEATH/RELEASE/each complication, peak-MAE,
trajectory honesty stats), and what was tried.

## Reproducibility

- Branch `autoresearch-trajectory`; no force-push to `main`.
- Ledger: `results/results-trajectory-fix.tsv`.
- Canonical baseline: `emr_model/checkpoints.bak_originals/` (read-only).
- Running-best backups: `emr_model/checkpoints.bak_keep_<tag>/`.
- Journal: `status.md` (Sections 1 / 1b stay intact).

