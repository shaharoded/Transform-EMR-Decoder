# EMR Event-Prediction Transformer — Benchmarking Phase

Fresh journal for the benchmarking phase. The iteration-loop journal is in
prior git commits if anyone needs the history.

## Inherited from the iteration loop

Locked architecture and recipe (full spec in `program.md`):

- **4-layer transformer** with `time2vec_dim=32, dropout=0.10, head_dim=64` (heads scale with `embed_dim`). Size selected by P6 sweep — start small (M-128) and grow.
- **AdaLN-Zero** patient-context conditioning + temporal RoPE + Time2Vec
- **Per-token learnable `log_tau_lm`**, terminal entries frozen at `log(12/336)`
- **C-ttt aux head** (time-to-terminal regression, MSE on `log1p(t_terminal − t_now)`)
- **Δt two-head** (gate + softplus magnitude)
- **P4 pool aux** at `aux_fraction_cap=0.05` (Phase 3)
- **I2b inference gate** (ttt-driven terminal-logit bias when `hrs-to-terminal < 48h`)
- **CBM** input masking p=0.25 (Phase 2 only)

Three-phase training: embedder → LM curriculum → outcome head fine-tune. Phase 3
backbone with `lr_factor=0.01`.

## Outcome configuration

Head-targeted outcomes are set in
`emr_model/transform_emr/config/dataset_config.py`:

- 7 complications (in `OUTCOMES`) (but in data only 5 will pass the min. support threshold)
- 2 terminals (in `TERMINAL_OUTCOMES`): DEATH, RELEASE
- `AUC_EXCLUDE = ("RELEASE_EVENT",)` in `evaluation.py` — RELEASE stays in head
  training (so the model emits it correctly) but is excluded from the AUROC /
  AUPRC / F1 headline (it's `¬DEATH` in this cohort, redundant ranking task).
  Reported as length-of-stay MAE instead.

**6 evaluated outcomes for the headline**: 5 complications + DEATH.

LM vocab is built from the training data, so any token present in the dataset
appears in input sequences regardless of `OUTCOMES` — `OUTCOMES` controls only
the head-training targets, sampler weights, and CBM forbid list.

## Benchmarking journal

Agent appends `### <tag>` blocks below as experiments run. Each block records:

- Tag, what changed (1–2 lines), commit SHA
- Smoke gate results (A–D)
- Post-train gate results (T1–T3)
- Headline numbers (`patient_auroc_weighted`, `patient_auprc_weighted`, `patient_max_f1_weighted`,
  per-outcome AUROC/AUPRC/max-F1/F1@0.5, peak-MAE per complication, length-of-stay MAE)
- Trajectory honesty (`gen_to_gt_ratio_median`, `gen_frac_terminal_first24h`)
- **Per-aux training trace table** (unlock epoch, λ_max, anchor raw_aux, final raw_aux, Δ%) — mandatory
- Verdict (KEEP / DISCARD) with reason

### P6-M-128-10k — RECIPE-TRANSFER SMOKE (Step 1)

**What:** Validate the locked I2b recipe transfers to the FRESH dataset. M-128
(embed_dim=128, n_head=2, n_layer=4; 1.75M params), `sample=10000` (7000 train /
1500 val / 1500 test patients), full epochs (100 cap each phase, early-stopped).
Tokenizer/scaler/embedder rebuilt from this sample (smoke-built tiny checkpoints
wiped first). Config commit `d05589b`.

**Smoke gates (sample=50, epochs=1 — commit ee4ce76→d05589b path):**
- A (no NaN/inf): PASS — all phase losses finite.
- B (aux raw within 1–2 OOM of BCE): PASS — ttt highest at ~1.6 OOM, within bound.
- C (calibrated λ in [1e-3,10]): PASS — only P3 λ calibrate in 1 epoch (ranking 1.49,
  pool 0.29); P2 auxes pending (bce_only_epochs=4 > 1) as expected.
- D (summary + all headline keys): PASS — `n_outcomes_used=6` (5 complications +
  DEATH; KETOACIDOSIS+ACIDOSIS auto-filtered <1%, RELEASE AUC-excluded).

**Post-train gates (10k full run):**
- T1 (every aux descends across active phase): PASS — see trace table.
- T2 (early-stop after auxes ramped): PASS — P2 ranking unlocked ep54, warmup ep57,
  P2 early-stopped ep70 (57 < 70); P3 λ calibrated ep1, ran 48 ep.
- T3 (real discrimination): PASS — all 6 outcomes AUROC 0.704–0.931.

**Headline (held-out test, 1500 patients):**
- `patient_auroc_weighted` **0.813**, `patient_auprc_weighted` 0.683,
  `patient_max_f1_weighted` 0.639, `patient_f1_at_0_5_weighted` 0.410,
  simple AUROC 0.812. n_outcomes=6.
- Per-outcome AUROC / AUPRC / maxF1 / F1@0.5 / peak-MAE(h):
  - CARDIO-VASCULAR 0.931 / 0.667 / 0.637 / 0.609 / 39.0
  - DISGLYCEMIA_Hyper 0.848 / 0.793 / 0.729 / 0.636 / 25.0
  - KIDNEY 0.817 / 0.743 / 0.650 / 0.592 / 27.5
  - HYPEROSMOLALITY 0.808 / 0.743 / 0.704 / 0.205 / 32.0
  - DISGLYCEMIA_Hypo 0.765 / 0.282 / 0.378 / 0.076 / 47.4
  - DEATH 0.704 / 0.329 / 0.343 / 0.116 / 153.9
- Length-of-stay MAE 63.8h (median 53.1, p90 134.7, n=1307).
- Multi-horizon AUROC: cap48 0.642, cap168 0.657, cap336 0.605.

**Trajectory honesty (near-perfect):** `gen_to_gt_ratio_median` 1.023,
`gen_frac_terminal_first24h` 0.075, gen_median 106.3h vs gt_median 103.9h,
1499/1500 natural terminals (no forced-terminal over-generation).

**Per-aux training trace table (mandatory):**

| Phase | Aux | Unlock/calib ep | λ_max | anchor raw | final raw | Δ% |
|---|---|---|---|---|---|---|
| 1 | dt | calib ep3 (active ep4) | 0.0415 | 1.918 (ep1) | 0.752 (ep40) | −60.8% |
| 2 | ce | calib ep3 (active ep4) | 0.1167 | 1.476 | 0.0041 | −99.7% |
| 2 | dt | calib ep3 (active ep4) | 0.2151 | 0.801 | 0.103 | −87.2% |
| 2 | ttt | calib ep3 (active ep4) | 0.0048 | 21.430 | 0.125 | −99.4% |
| 2 | ranking | unlock ep54 (calib ep53) | 0.0259 | 0.245 (ep53) | 0.107 (ep69) | −56.4% |
| 3 | outcome BCE | ep1 | — | 2.829 | 1.867 (ep48) | −34.0% |
| 3 | ranking | calib ep1 | 0.9643 | 0.587 | 0.347 (ep48) | −40.9% |
| 3 | pool | calib ep1 | 0.1504 | 0.941 | 0.060 (ep48) | −93.7% |

All |Δ| ≥ 34% — every aux is learning (none flagged <5%).

**Verdict: RECIPE-TRANSFER CONFIRMED.** The locked recipe transfers cleanly to the
fresh dataset — all smoke gates A–D and post-train T1–T3 pass, all auxes descend
strongly, no degenerate outputs (honest trajectories), AUROC headline 0.813 with
all 6 outcomes well above chance. This is a verification probe, not a KEEP/DISCARD.
phase2_val 0.191 (70 ep), phase3_val 2.117 (48 ep), 1.75M params, peak_vram 196MB.
**Proceed to Step 2 — P6 full-data architecture sweep (M-128 → M-256 → M-384 → M-512 → M-768).**

### P6-M-128-full — full-data sweep point 1/5

**What:** M-128 (embed_dim=128, n_head=2, n_layer=4; 1.75M params) at FULL data
(39954 train / 8562 val / 8562 test), locked I2b recipe, full-data
tokenizer/scaler/embedder. Config commit `b8c7095`. **early-stop-patience=5**
(this run launched before the user raised patience→15 for M-256+).

**Gates:** Smoke A–D pass (live-confirmed). Post-train T1 (all auxes descend),
T2 (ranking unlock ep17 < warmup ep20 < P2 stop ep26; P3 ran 49 ep), T3 (all 6
outcomes 0.785–0.982) — all PASS.

**Headline (held-out 8562 test patients):**
- `patient_auroc_weighted` **0.883** (+0.070 vs 10k probe 0.813),
  `patient_auprc_weighted` 0.798, `patient_max_f1_weighted` 0.721,
  `patient_f1_at_0_5_weighted` 0.536, simple AUROC 0.887. n_outcomes=6.
- Per-outcome AUROC / AUPRC / maxF1 / F1@0.5 / peak-MAE(h):
  - CARDIO-VASCULAR 0.982 / 0.909 / 0.853 / 0.786 / 34.3
  - DISGLYCEMIA_Hyper 0.908 / 0.881 / 0.774 / 0.442 / 23.2
  - KIDNEY 0.905 / 0.877 / 0.773 / 0.708 / 29.2
  - DISGLYCEMIA_Hypo 0.884 / 0.634 / 0.626 / 0.563 / 43.4
  - HYPEROSMOLALITY 0.859 / 0.831 / 0.733 / 0.577 / 29.6
  - DEATH 0.785 / 0.329 / 0.413 / 0.115 / 169.7
- Length-of-stay MAE 59.1h (median 41.4, p90 142.7, n=7446).
- Multi-horizon AUROC: cap48 0.533, cap168 0.610, cap336 0.572.

**Trajectory honesty:** `gen_to_gt_ratio_median` 0.606 (honest, >0.4 floor;
under-generates a touch more than the 10k probe's 1.02), `gen_frac_terminal_first24h`
0.052, gen_median 61.8h vs gt_median 102.1h, 8561/8562 natural terminals.

**Per-aux training trace table:**

| Phase | Aux | Unlock/calib ep | λ_max | anchor raw | final raw | Δ% |
|---|---|---|---|---|---|---|
| 1 | dt | calib ep3 | 0.0261 | 1.354 (ep1) | 0.748 (ep23) | −44.8% |
| 2 | ce | calib ep3 | 0.0724 | 1.283 | 0.0023 | −99.8% |
| 2 | dt | calib ep3 | 0.1152 | 0.807 | 0.053 | −93.4% |
| 2 | ttt | calib ep3 | 0.0026 | 21.254 | 0.057 | −99.7% |
| 2 | ranking | unlock ep17 (calib ep16) | 0.0266 | 0.110 (ep16) | 0.066 (ep26) | −39.8% |
| 3 | outcome BCE | ep1 | — | 2.484 | 1.567 (ep49) | −36.9% |
| 3 | ranking | calib ep1 | 0.8077 | 0.615 | 0.275 (ep49) | −55.2% |
| 3 | pool | calib ep1 | 0.1318 | 0.942 | 0.0001 (ep49) | −99.99% |

All |Δ| ≥ 37% — every aux learning.

**Verdict: P6-SWEEP BASELINE (sweep point 1/5).** Strong full-data headline 0.883;
all 6 outcomes well above chance, honest trajectories. Sets the bar for
M-256/384/512/768. NOTE: used patience 5; if M-128 ends competitive for the winner,
re-run at patience 15 for clean comparison. Proceed to M-256.

### P6-M-256-full — full-data sweep point 2/5

**What:** M-256 (embed_dim=256, n_head=4, n_layer=4; 6.71M params) full data,
locked recipe, **patience=15** (first run with the raised patience). Config commit `17366ab`.

**Gates:** Smoke A–D pass. T1 (all auxes descend), T2 (ranking unlock ep12 <
warmup ep15 < P2 stop ep50; P3 ran 100), T3 (all 6 outcomes 0.794–0.975) — PASS.

**Headline (held-out 8562 test):**
- `patient_auroc_weighted` **0.891** (+0.008 vs M-128 0.883),
  `patient_auprc_weighted` 0.801, `patient_max_f1_weighted` 0.722,
  `patient_f1_at_0_5_weighted` 0.492, simple AUROC 0.891.
- Per-outcome AUROC / AUPRC / peak-MAE(h):
  - CARDIO 0.975 / 0.788 / 33.4  (−0.007 vs M-128)
  - DISGLYCEMIA_Hyper 0.909 / 0.875 / 24.2  (flat)
  - KIDNEY 0.900 / 0.864 / 26.9  (−0.005)
  - HYPEROSMOLALITY 0.890 / 0.873 / 25.8  (+0.030)
  - DISGLYCEMIA_Hypo 0.877 / 0.534 / 45.4  (−0.007)
  - DEATH 0.794 / 0.408 / 176.3  (+0.010)
- Length-of-stay MAE 74.9h (worse than M-128's 59.1h).
- Multi-horizon AUROC: cap48 0.516, cap168 0.562, cap336 0.535.

**Trajectory honesty — REGRESSED vs M-128:** `gen_to_gt_ratio_median` **0.407**
(M-128 0.606; now at the 0.4 honesty floor), `gen_frac_terminal_first24h` **0.198**
(M-128 0.052), gen_median 41.9h vs gt_median 102.8h. The model under-generates and
emits terminals early ~4× more than M-128. Classic AUROC↔calibration Pareto tension;
the longer Phase-2 LM training (50 ep vs M-128's 26, from patience=15) is the likely driver.

**Two flags for the sweep decision:**
1. **Phase-3 hit the 100-epoch CAP** (ran all 100, still marginally improving — did
   NOT plateau-stop). Per plan: if M-256 ends up the winner, it's a candidate for an
   extended-epoch re-run before final reporting.
2. The +0.008 AUROC over M-128 came **with** a real honesty regression — capacity is
   buying ranking at the cost of trajectory calibration, exactly the documented Pareto.

**Per-aux training trace table:**

| Phase | Aux | Unlock/calib ep | λ_max | anchor raw | final raw | Δ% |
|---|---|---|---|---|---|---|
| 1 | dt | calib ep3 | — | 1.534 (ep1) | 0.744 (ep40) | −51.5% |
| 2 | ce | calib ep3 | 0.0826 | 0.930 | 0.0016 | −99.8% |
| 2 | dt | calib ep3 | 0.0935 | 0.821 | 0.034 | −95.9% |
| 2 | ttt | calib ep3 | 0.0021 | 21.628 | 0.038 | −99.8% |
| 2 | ranking | unlock ep12 (calib ep11) | 0.0325 | 0.091 (ep11) | 0.032 (ep50) | −65.1% |
| 3 | outcome BCE | ep1 | — | 2.204 | 1.422 (ep100) | −35.5% |
| 3 | ranking | calib ep1 | ~0.93 | 0.472 | 0.233 (ep100) | −50.6% |
| 3 | pool | calib ep1 | 0.1166 | 0.945 | 0.0003 (ep100) | −99.97% |

All |Δ| ≥ 35% — every aux learning.

**Verdict: P6-SWEEP point 2/5 — running best (AUROC 0.891).** M-128 (0.883) is 0.008
below — outside the 0.005 equivalence window, so M-256 currently leads. BUT the honesty
regression is a strike against M-256 as the deployable model. Sweep continues to
M-384/512/768. Decision deferred until the full grid + honesty are weighed together.

### P6-M-384-full — full-data sweep point 3/5 (AUROC peak passed)

**What:** M-384 (embed_dim=384, n_head=6, n_layer=4; 14.88M params) full data,
patience=15, Phase-3 plateau-stopped ep86. Config commit `3cafcca`.

**Gates:** Smoke A–D pass. T1 (all auxes descend), T2 (ranking unlock ep11 < P2
stop ep49), T3 (all 6 outcomes 0.733–0.926) — PASS.

**Headline (held-out 8562 test):**
- `patient_auroc_weighted` **0.876** — **BELOW both M-256 (0.891) and M-128 (0.883)**.
  The sweep AUROC peak is M-256; M-384 turns down.
- AUPRC_w 0.773, maxF1_w 0.707, F1@0.5_w 0.500, simple AUROC 0.866.
- Per-outcome AUROC (all ≤ M-256): CARDIO 0.926 (−0.049), DISGLYCEMIA_Hyper 0.910 (flat),
  KIDNEY 0.885 (−0.016), HYPEROSMOLALITY 0.883 (−0.007), DISGLYCEMIA_Hypo 0.858 (−0.019),
  DEATH 0.733 (−0.061).
- Length-of-stay MAE 77.5h. Multi-horizon: cap48 0.507, cap168 0.542, cap336 0.520.

**KEY METHODOLOGICAL FINDING — AUROC↔calibration divergence with capacity:**
M-384 posts the **best validation losses of the entire sweep** (phase2 0.144 vs M-256
0.149 vs M-128 0.153; phase3 1.634 vs 1.672 vs 1.774) **yet the worst AUROC**. The
bigger model fits the soft-label BCE (likelihood/calibration) better but *ranks*
worse — and generates least honestly. This is the documented Pareto, now crisp across 3 sizes.

**Trajectory honesty — worst of the sweep:** `gen_to_gt_ratio_median` **0.324**
(M-256 0.407, M-128 0.606), `gen_frac_terminal_first24h` **0.373** (M-256 0.198,
M-128 0.052). Monotone honesty decline with size.

**Per-aux training trace table:**

| Phase | Aux | Unlock/calib ep | λ_max | anchor raw | final raw | Δ% |
|---|---|---|---|---|---|---|
| 1 | dt | calib ep3 | — | 1.589 (ep1) | 0.745 (ep36) | −53.1% |
| 2 | ce | calib ep3 | — | 0.790 | 0.0007 | −99.9% |
| 2 | dt | calib ep3 | — | 0.823 | 0.030 | −96.4% |
| 2 | ttt | calib ep3 | — | 21.249 | 0.031 | −99.9% |
| 2 | ranking | unlock ep11 (calib ep10) | 0.0340 | 0.0625 (ep10) | 0.027 (ep49) | −57.1% |
| 3 | outcome BCE | ep1 | — | 2.176 | 1.362 (ep86) | −37.4% |
| 3 | ranking | calib ep1 | — | 0.451 | 0.213 (ep86) | −52.9% |
| 3 | pool | calib ep1 | 0.1156 | 0.942 | ~0 (ep86) | −100% |

**Sweep curve so far (AUROC_w / gen_to_gt / frac_term24h):**
- M-128 (1.75M): 0.883 / 0.606 / 0.052
- M-256 (6.71M): **0.891** / 0.407 / 0.198  ← AUROC peak
- M-384 (14.88M): 0.876 / 0.324 / 0.373  ← turned down on both axes

**Verdict: P6-SWEEP point 3/5 — AUROC PEAK PASSED at M-256.** M-384 is worse on
AUROC and honesty → capacity past M-256 overfits on this dataset. M-512/768 are now
confirmatory (expect continued decline). Running best remains **M-256 (0.891)**;
**M-128 (0.883) is 0.008 back but by far the most honest** — the eventual winner
decision will weigh AUROC vs honesty, not AUROC alone. Continue to M-512.

## Reproducibility

- Branch `autoresearch-trajectory`.
- Ledger: `results/results-trajectory-fix.tsv` (iteration-loop rows preserved; benchmarking rows appended).
- Canonical baseline: `emr_model/checkpoints.bak_originals/` (read-only).
- Running-best backups: `emr_model/checkpoints.bak_keep_<tag>/`.
- Iteration-loop history: prior git commits (not on disk).
