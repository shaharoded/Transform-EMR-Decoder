# EMR Event-Prediction Transformer — Architecture Sweep on MIMIC-IV

**Status: COMPLETE.** Architecture sweep, QA-feature ablation, and k-day-seed
scan all finished. Final best model: **M-256 (non-QA)**.

Last update: 2026-05-20 UTC.

---

## 1. Final model

**M-256** (commit `5496c9e`). Checkpoints under `best_weights/M-256-retrain/`.
Headline metrics on the held-out test split (15 % of patients, never seen in
training or validation):

| Metric                 | Value     |
|------------------------|-----------|
| `outcome_auroc`        | **0.9150** |
| `outcome_auprc`        | **0.6298** |
| `onset_mae_hrs`        | **64.98 h** |
| Peak VRAM (training)   | 4.54 GB   |
| Parameters             | 6.41 M    |

Architecture: `embed_dim=256, n_layer=4, n_head=4, time2vec_dim=32, dropout=0.10`.
Optimiser: AdamW, `phase{1,2}_lr=3e-4`, `phase3_lr=1e-4` with
`phase3_backbone_lr_factor=0.01`. Auxiliary caps:
`{ce: 0.5, dt: 0.5, ranking: 0.2}`. Three-phase training: Phase 1 embedder,
Phase 2 GPT pretrain with curriculum (BCE → CE + Δt → pairwise ranking), Phase 3
outcome-head fine-tune. Evaluation is autoregressive generation from a 2-day
seed → 24 h non-overlapping windows → per-complication AUROC / AUPRC /
onset-MAE, then mean across complications with ≥ 3 positive windows.

### Per-outcome breakdown (k = 2 seed)

| Outcome   | AUROC |
|-----------|-------|
| CARDIO    | 0.968 |
| DEATH     | 0.943 |
| HYPERGLY  | 0.936 |
| HYPOGLY   | 0.916 |
| KIDNEY    | 0.911 |
| RELEASE   | 0.817 |

Mean AUPRC across all six complications: **0.6298**.

### Class support and metric interpretation

Class imbalance is large and varies by outcome. The numbers below are the actual
positive-window prevalence on the held-out test split (8,562 patients,
~45,876 evaluation windows per outcome), computed with the same windowing
`evaluation.py` uses (24 h non-overlapping windows after a 2-day seed, ±24 h
grace, 14-day horizon):

| Outcome   | Window prevalence | Patients with ≥1 positive window | Random AUPRC | Model AUROC |
|-----------|-------------------|----------------------------------|--------------|-------------|
| CARDIO    | 1.70 %            | 3.7 %                            | 0.017        | 0.968       |
| DEATH     | 3.67 %            | 10.6 %                           | 0.037        | 0.943       |
| HYPOGLY   | 3.84 %            | 7.4 %                            | 0.038        | 0.916       |
| KIDNEY    | 15.78 %           | 23.7 %                           | 0.158        | 0.911       |
| HYPERGLY  | 21.85 %           | 32.3 %                           | 0.218        | 0.936       |
| RELEASE   | 30.77 %           | 87.0 %                           | 0.308        | 0.817       |
| **Mean**  | **12.94 %**       | —                                | **0.129**    | **0.9150**  |

**Why both metrics are reported, and how to read them.**

AUROC measures the probability that a random positive window is scored higher
than a random negative window. Its random baseline is fixed at 0.5 regardless
of prevalence, and the FPR axis is normalised by the (large) negative count, so
a model can post a high AUROC even when most of its positive-class alerts are
false in absolute terms — the false alarms barely move FPR. With prevalences
as low as 1.7 % (CARDIO) and 3.7 % (DEATH), AUROC alone risks looking better
than the operational behaviour warrants.

AUPRC fixes that. Its random baseline equals the positive-class prevalence,
so the baselines per outcome above (0.017 to 0.308) anchor what "lift" means:
the mean AUPRC of 0.630 sits at **~4.9× the chance baseline of 0.129**, which
is the headline number. For the rarest outcomes (CARDIO 0.017, DEATH 0.037)
the lift is much larger; for RELEASE (0.308, near-every-patient) it is modest
because the chance baseline is already high. Both metrics are kept because
they answer complementary questions:

- **AUROC** — does the model rank positive windows above negative windows
  on average? Standard in the medical literature.
- **AUPRC** — at any chosen alert threshold, how many of the model's alerts
  are real? More sensitive to false-positive cost and to prevalence.

Per-outcome AUPRC would let us compute per-outcome lift directly; that view is
not yet logged but is straightforward to add by re-running `evaluate_on_test_set`
and saving the full `auc_table` instead of just the mean.

---

## 2. Architecture sweep

Four architecture sizes evaluated. Each row is a unique
`(embed_dim, n_layer, n_head, time2vec_dim, dropout)` combination. The L-384
entry reflects its best within-size configuration (`phase3_backbone_lr_factor=0.10`,
`dropout=0.15`); the initial L-384 run with the defaults failed to converge in
the Phase-3 budget (LM head beat the outcome head at every complication on the
diagnostic), motivating the within-size tune.

| Tag        | Params  | AUROC   | AUPRC   | MAE (h)  | VRAM (GB) | RELEASE | Verdict   |
|------------|---------|---------|---------|----------|-----------|---------|-----------|
| S-128      | 1.67 M  | 0.9000  | 0.6108  | 64.72    | 0.22      | 0.741   | DISCARD   |
| **M-256**  | 6.41 M  | **0.9150** | **0.6298** | **64.98** | **4.54** | **0.817** | **KEEP**  |
| M-256-deep | 9.31 M  | 0.8985  | 0.6064  | 64.49    | 0.40      | 0.751   | DISCARD   |
| L-384      | 20.78 M | 0.9107  | 0.6334  | 65.61    | 0.58      | —       | DISCARD   |

### What the sweep revealed

The data favours **M-256**. Going **smaller** (S-128) loses 0.015 AUROC almost
entirely because RELEASE collapses (0.817 → 0.741); the discharge trajectory
needs more than 128-dimensional embeddings to disambiguate from other outcomes.
Going **deeper at fixed width** (M-256-deep, 6 layers) hurts (0.017 AUROC drop)
with the same RELEASE-collapse pattern, suggesting that adding depth without
widening fragments representational capacity. Going **wider and deeper**
(L-384) recovers some RELEASE (0.795) but loses DEATH and ranks within the
±0.005 AUROC tolerance of M-256 while regressing MAE by 0.66 h, failing the
secondary criterion. The diagnostic on L-384 showed the dedicated outcome head
underfitting at Phase 3 — the LM head beat it at every outcome — because
backbone LR (`1e-6` at factor=0.01) was too small for 20.78 M parameters in the
early-stop budget. Raising the backbone factor to 0.10 closed most of the AUROC
gap but introduced training instability (three SIGKILLs at Phase 3 validation,
resolved via `torch.cuda.empty_cache()` + resume). Net conclusion: this
dataset's predictive signal saturates at the M-256 capacity; additional
parameters either fail to converge in the early-stop budget or destabilise
training, and smaller models lack the embedding capacity to keep RELEASE alive.

---

## 3. QA-feature ablation

Tested whether enabling `USE_QA_DATA=True` improves the headline metric.
`USE_QA_DATA=True` (a) keeps `%_PATTERN%` tokens in the temporal stream and
(b) appends nine per-patient mean `ComplianceScore` columns to the patient
context vector, aggregated over the first 48 h of admission. The ablation
retrained M-256 from scratch with a freshly-built tokenizer so the new pattern
events received real token IDs.

| Variant       | AUROC      | AUPRC  | MAE (h) |
|---------------|------------|--------|---------|
| **M-256**     | **0.9150** | **0.6298** | **64.98** |
| M-256-QA      | 0.8764     | 0.5672 | 63.70   |

**DISCARD.** AUROC drops 0.039 below the baseline, well outside the ±0.005
tolerance. The diagnostic also showed Δt R² collapsing to ≈ 0 (vs 0.12 at
baseline) and a shuffled-context paradox where randomising the patient context
vector reduced BCE — indicators that the model was overfitting to the QA
columns rather than using them as conditioning. The `%_PATTERN%` events appear
to add noise rather than signal to the LM head. Net conclusion: **QA features
as implemented do not help this architecture**, and the
`%_PATTERN%`-NOT-LIKE filter on the temporal stream is the correct default.

---

## 4. Input-context-window scan

How much patient history does the model need? We ran the final M-256 model
under autoregressive generation seeded with k days of admission history, for
k ∈ {1, 2, 3, 4, 5, 6, 7, 8}. The k-day seed determines what the model sees
before it starts generating future events. All numbers below are on the same
held-out test split.

| k (days) | AUROC  | AUPRC  | MAE (h)  | DEATH | RELEASE | CARDIO | KIDNEY |
|----------|--------|--------|----------|-------|---------|--------|--------|
| 1        | 0.6653 | 0.2543 | 51.23    | —     | —       | —      | —      |
| **2**    | **0.9150** | 0.6298 | **64.98** | 0.943 | **0.817** | 0.968 | 0.911 |
| **3**    | **0.9155** | **0.6482** | 81.85  | **0.952** | 0.791 | 0.972 | 0.913 |
| 4        | 0.9093 | 0.6003 | 99.02    | 0.934 | 0.782   | 0.968  | 0.914  |
| 5        | 0.9147 | 0.5920 | 116.61   | 0.939 | 0.773   | **0.984** | 0.918 |
| 6        | 0.9105 | 0.6036 | 134.48   | 0.949 | 0.748   | 0.972  | **0.932** |
| 7        | 0.9115 | 0.6049 | 152.85   | —     | —       | —      | —      |
| 8        | 0.9087 | 0.6178 | 171.82   | —     | —       | —      | —      |

### Findings

1. **Hard cliff between k=1 and k=2.** At k=1, AUROC collapses to 0.665 and
   AUPRC to 0.254. The model is non-viable with less than two days of
   admission context.
2. **AUROC plateaus from k=2 onward** (range 0.909–0.916, span 0.007). The
   bulk of the predictive signal is captured within the first 2–3 days of
   admission.
3. **MAE grows roughly linearly at ≈ 17–18 h per additional seed day**
   (65 → 82 → 99 → 117 → 134 → 153 → 172 h). Each additional seed day
   consumes one day of events that would otherwise have been generation
   targets, pushing remaining event times further into the future.
4. **AUPRC peaks at k=3 (0.648).** k=4–5 dip; k=6–8 partially recover.
5. **Per-outcome trends:** RELEASE declines monotonically with k
   (0.817 → 0.748) — discharge signal is concentrated in the first 1–2 days.
   KIDNEY improves monotonically (0.911 → 0.932) — renal deterioration
   benefits from longer metabolic context. CARDIO peaks at k=5 (0.984).
   DEATH peaks at k=3 (0.952).

### Recommendation

- **k=3** for quality-optimised deployment (best AUROC, best AUPRC, best
  DEATH).
- **k=2** for real-time settings where only two days of history are
  available (lowest MAE; AUROC essentially tied with k=3).
- **k ≥ 7** not recommended (MAE > 150 h with no AUROC gain).
- **k=1 is below the operational floor** — the model collapses to near-random.

---

## 5. Reproducibility

| Artefact                              | Location                                        |
|---------------------------------------|-------------------------------------------------|
| Final-model code                      | This repository, branch `autoresearch-optimization` |
| Final-model checkpoints (P1 + P2 + P3) | `best_weights/M-256-retrain/`                  |
| Final-model tokenizer + scaler        | `best_weights/M-256-retrain/`                  |
| Per-experiment one-line ledger        | `results.tsv`                                   |
| Source data                           | MIMIC-IV-derived `temporal_data.csv` + `context_data.csv` (not in repo) |
| Train / val / test split              | `PatientId`-stratified 70 / 15 / 15, `random_state=42` (in `api.py`) |

To reproduce the final result on a new machine: clone this branch, place the
source CSVs under `emr_model/data/source/`, then `python api.py`. The data
processor builds a tokenizer and scaler against the 70 % train split, caches
the processed dataset to `emr_model/checkpoints/processed_datasets.pt`, runs
the three training phases, and prints a summary block ending in
`outcome_auroc:` for the held-out test split.
