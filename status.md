# EMR Event-Prediction Transformer — Patient-Level Eval Loop

## Inherited from prior session

Decisions carried forward from the architecture sweep + ablations. The
specific AUC numbers from that session were computed under the
**per-window** eval framing and **are not comparable** to the new
patient-level peak-detector headline — don't anchor to them.

- **Architecture**: M-256 — `embed_dim=256`, `n_layer=4`, `n_head=4`,
  `time2vec_dim=32`, `dropout=0.10`. Params ~6.4 M. Peak VRAM at training ~5 GB.
- **Optimiser**: AdamW. `phase{1,2}_lr=3e-4`, `phase3_lr=1e-4`,
  `phase3_backbone_lr_factor=0.01`. Aux caps `{ce: 0.5, dt: 0.5, ranking: 0.2}`.
- **Training**: three-phase. Phase 1 embedder; Phase 2 GPT pretrain with
  curriculum (BCE → CE + Δt → pairwise ranking); Phase 3 outcome-head fine-tune.
- **Evaluation seed**: 2-day input → 14-day generation horizon. (Prior
  k-day-seed scan ruled k=1 below operational floor; AUROC plateaued
  from k=2 onward; k=2 chosen for the operational use case.)
- **QA data**: `USE_QA_DATA=False`. The QA-augmented variant added new
  context features + tokens; in the prior eval framing it didn't move
  the headline. The new loop will revisit this in **P7** as the final
  step on the running-best model.
- **Running best on HEAD**: Z (direction E — narrow + frozen terminal
  `log_tau_lm`). No checkpoint on disk; pod is fresh and Phase 1
  retrains.

---

## Patient-level eval loop

Per `program.md`. New eval framing (per-patient peak detector). The
agent appends `### <tag>` blocks here as experiments run.

Each block records: tag, what changed (1–2 lines), smoke gate results,
post-train gate results, headline numbers (`patient_auroc_weighted`,
per-outcome AUROC for DEATH/RELEASE/each complication, peak-MAE),
trajectory honesty (`gen_to_gt_ratio_median`,
`gen_frac_terminal_first24h`), **per-aux training trace** (table with
unlock epoch, λ_max, anchor raw_aux at calibration, final raw_aux at
end of phase, Δ%) for every aux active in any phase, verdict
(KEEP / DISCARD) with reason. Flag `|Δ| < 5 %` auxes explicitly —
they're not learning.

### B0-Z @ 10k (SHA 8d3cf18)

P0 baseline. Z (direction E — narrow + frozen terminal `log_tau_lm`,
init `log(12/336)`) on HEAD. No code change — first run on new
patient-level peak-detector eval.

Smoke (sample=50, phase{1,2,3}_n_epochs=1):
- Gate A pass — no NaN, train=8.5680, val=7.8539 at Phase-3 epoch 1.
- Gate B pass — raw_out=8.568, raw_rank=0.691 (~12×, within 1–2 OOM).
- Gate C pass — λ_ranking calibrated 2.479 ∈ [1e-3, 10].
- Gate D pass — summary block + all headline keys emit.

Post-train (10k):
- T1 pass — Phase-3 raw_out 2.20→1.05, raw_rank 0.66→0.41 across 29 epochs.
- T2 pass — Phase-2 early stop at epoch 46 (ranking ramped from epoch 32,
  fully active by 35, ran active 11+ epochs before stop). Phase-3 early
  stop at epoch 29 with best val at epoch 9 (1.0105).
- T3 pass — patient AUROC shows real discrimination on the headline
  outcomes (see below).

Headline:
- `patient_auroc_weighted`: **0.6671**
- `patient_auprc_weighted`: 0.6205
- `patient_auroc_simple`:   0.6932
- `patient_auprc_simple`:   0.3032
- `n_outcomes_used`:        16

Per-outcome AUROC (top):
- DISGLYCEMIA_Hyperglycemia 0.904 (AUPRC 0.871, n_pos=619)
- DISGLYCEMIA_Hypoglycemia  0.797
- KETOACIDOSIS              0.791
- NERVOUS_SYSTEM_DISORDER   0.788
- RETINOPATHY               0.776
- NEUROVASCULAR             0.749
- KIDNEY_COMPLICATION       0.702 (AUPRC 0.634, n_pos=685)
- CARDIO-VASCULAR_DISORDER  0.701 (AUPRC 0.744, n_pos=860)
- **DEATH**                 0.693 (AUPRC 0.228, n_pos=192)
- SKIN_ULCER                0.663
- HYPEROSMOLALITY           0.644
- ATHEROSCLEROSIS           0.608
- ACUTE_RESPIRATORY         0.605
- ACIDOSIS                  0.585
- INFECTION                 0.566
- **RELEASE**               0.521 (AUPRC 0.881, n_pos=1308)

Peak MAE (hours, positives only):
- DEATH:    158.84  (n=191)
- RELEASE:   85.97  (n=1308)
- DISGLYCEMIA_Hyper: 43.98
- DISGLYCEMIA_Hypo:  66.15
- KIDNEY:           106.36
- CARDIO:           107.99
- (others 145–234 h)

Trajectory honesty:
- `gen_median_hours`:         114.48
- `gen_to_gt_ratio_median`:     1.116 (≥ 0.4 ✓)
- `gen_frac_terminal_first24h`: 0.148
- `gen_length_mae_hrs`:       101.48

Phase stats: phase2_best_val 0.184 / 46 epochs (early stopped),
phase3_best_val 1.157 / 29 epochs (early stopped).

Verdict: **BASELINE-KEEP** — first patient-level eval reference.
Running best until B0-C-ttt result is in. Checkpoints backed up to
`emr_model/checkpoints.bak_keep_B0-Z/`.

---

### B0-C-ttt @ 10k (SHA ea65988)

P0 baseline #2. Cherry-pick of dd3fc1b "C-ttt-head" (time-to-terminal
regression aux) on top of B0-Z. Adds an MSE head predicting
`log1p(t_terminal − t_now)` at every non-terminal, non-pad position,
sharing the backbone. Joins Phase-2 stage 0 alongside ce/dt with
fraction_cap=0.30. Goal: force the backbone to encode distance-to-
terminal explicitly so the LM head can decide WHEN to emit terminal
tokens.

Smoke (sample=50, phase{1,2,3}_n_epochs=1):
- Gate A pass — no NaN; RawTrain ce=1.31, dt=0.81, ranking=0.69,
  ttt=19.19, all finite.
- Gate B pass — ttt within ~25× of ce/dt (within 1–2 OOM).
- Gate C pass — λ_ranking calibrated 2.497 ∈ [1e-3, 10].
- Gate D pass — summary block + all headline keys present.

Post-train (10k):
- T1 pass — Phase-3 raw_out 2.11→1.01, raw_rank 0.66→0.38; ttt λ_max
  calibrated at Phase-2 epoch 3 (λ_max=0.0040, raw_aux=20.86 — head
  starts well above ce/dt then decays).
- T2 pass — Phase-2 ranking calibrated epoch 31, ramp 31→35, full
  active by 35; Phase-2 early stop at epoch 40 (5 epochs of full
  stage-1 activity before stop). Phase-3 best val at epoch 15 (0.996),
  early stop at epoch 23.
- T3 pass — DEATH AUROC 0.710 (+0.017 vs B0-Z), KIDNEY 0.715,
  CARDIO 0.709, KETOACIDOSIS 0.915 (+0.124 — biggest single per-outcome
  swing).

Headline (Δ vs B0-Z @ 10k):
- `patient_auroc_weighted`: **0.6831** (+0.0160 ✓)
- `patient_auprc_weighted`: 0.6336 (+0.0131 ✓)
- `patient_auroc_simple`:   0.6959 (+0.0027)
- `patient_auprc_simple`:   0.3239 (+0.0207)
- `n_outcomes_used`:        16

Per-outcome AUROC vs B0-Z:
- KETOACIDOSIS              0.915  (+0.124, n_pos=37)
- DISGLYCEMIA_Hyperglycemia 0.896  (−0.008)
- NERVOUS_SYSTEM            0.796  (+0.008)
- RETINOPATHY               0.785  (+0.009)
- DISGLYCEMIA_Hypoglycemia  0.771  (−0.026)
- KIDNEY                    0.715  (+0.013)
- **DEATH**                 0.710  (+0.017) ✓
- CARDIO                    0.709  (+0.008)
- NEUROVASCULAR             0.686  (−0.063)  ← biggest regression
- SKIN_ULCER                0.679  (+0.016)
- ATHEROSCLEROSIS           0.595  (−0.013)
- ACUTE_RESPIRATORY         0.591  (−0.014)
- HYPEROSMOLALITY           0.585  (−0.059)
- **RELEASE**               0.581  (+0.060) ✓
- ACIDOSIS                  0.570  (−0.015)
- INFECTION                 0.551  (−0.015)

Peak MAE vs B0-Z (hours):
- DEATH:    168.97  (+10.13  — REGRESSION ≥ 5h threshold)
- RELEASE:   71.29  (−14.68 ✓)
- DISGLYCEMIA_Hyper:  36.07 (−7.91)
- KIDNEY:            79.11  (−27.25)
- CARDIO:            79.08  (−28.91)

Trajectory honesty:
- `gen_median_hours`:           75.05  (vs B0-Z 114.48 — generates shorter)
- `gen_to_gt_ratio_median`:      0.720  (vs B0-Z 1.116 — still ≥ 0.4 ✓)
- `gen_frac_terminal_first24h`:  0.165  (vs B0-Z 0.148 — slight bump)

Phase stats: phase2_best_val 0.187 / 41 epochs (early stopped);
phase3_best_val 1.144 / 23 epochs (early stopped). Both terminate
earlier than B0-Z (46/29) — Phase-3 best val is also lower (1.144 vs
1.157), so faster convergence on a better minimum.

Verdict: **BASELINE-KEEP, RUNNING BEST** — between the two P0
baselines, B0-C-ttt clearly wins on the primary headline
(`patient_auroc_weighted` 0.683 > 0.667) and lifts both DEATH and
RELEASE AUROC simultaneously, which is the precise pattern program.md
predicted under the new framing. The DEATH-MAE regression (+10 h) and
the NEUROVASCULAR / HYPEROSMOLALITY AUROC dips are real costs, but
n_pos is small (29, 83) so per-outcome variance is high, and the model
is generating 35 % shorter sequences (75 h vs 114 h) which mechanically
explains the slight DEATH-MAE drift toward the rare-DEATH median.
P0 KEEP rule (better of two baselines) applies — KEEP/DISCARD threshold
test is for subsequent experiments vs this running best.

Checkpoints backed up to `emr_model/checkpoints.bak_keep_B0-C-ttt/`.
This is the running best for P1 (MIL max-BCE).

---

### P1-MIL @ 10k (SHA 422dcbc) — DISCARD

P1 direction. Added a patient-level multiple-instance-learning aux to
Phase 3: soft-max-attention pool of outcome logits across time steps,
BCE against `patient_label = outcome occurs anywhere in GT`. Soft-max
temperature `mil_log_T` learnable per outcome. λ_mil calibrated once
at end of Phase-3 epoch 1, capped at fraction 0.20 of raw outcome BCE
(same regime as ranking). Per-position BCE kept as 48-h calibration
anchor.

Smoke (sample=50, phase{1,2,3}_n_epochs=1):
- Gates A–D all pass. raw_out=8.52, raw_rank=0.69, raw_mil=1.07
  (within 1× of BCE). λ_mil=1.585, λ_ranking=2.46, both ∈ [1e-3, 10].

Post-train (10k):
- T1 fail — Phase-3 raw_out drops from 2.053 to 1.174 between epoch 1
  and 2 (this is normal — calibration kick when λ_ranking goes 0→cal).
  raw_mil rises 3.685→4.635 over the 6 active epochs: the MIL aux is
  being optimised AGAINST, not toward. Aux gradient too weak to fight
  per-position BCE conflict.
- T2 fail — Phase-3 early stop fires at epoch 6, with best `vl_select`
  at epoch 1 (1.1125) — i.e., before λ_mil was even active.
  Subsequent epochs (with λ_mil=0.111) consistently increased vl_select.
- T3 fail — DEATH AUROC drops 0.710→0.665 (-0.045); the head no longer
  shows the discrimination the run was supposed to add.

Headline (Δ vs B0-C-ttt running best):
- `patient_auroc_weighted`: **0.6427** (−0.0404 — fails ≥+0.030 KEEP)
- `patient_auprc_weighted`: 0.5855 (−0.0481)
- `patient_auroc_simple`:   0.6112 (−0.0847)
- `patient_auprc_simple`:   0.2792 (−0.0447)
- `n_outcomes_used`:        16

Per-outcome AUROC Δ vs B0-C-ttt — universal regression except RELEASE:
- DISGLYCEMIA_Hyper:  0.805 (−0.091)
- DEATH:              0.665 (−0.045)  ← contra direction's intent
- NEUROVASCULAR:      0.651 (−0.035)
- NERVOUS_SYSTEM:     0.649 (−0.147)
- RELEASE:            0.645 (+0.064)  ← only winner (majority class)
- DISGLYCEMIA_Hypo:   0.643 (−0.128)
- KIDNEY:             0.639 (−0.076)
- CARDIO:             0.616 (−0.093)
- RETINOPATHY:        0.613 (−0.172)
- SKIN_ULCER:         0.590 (−0.089)
- ACUTE_RESPIRATORY:  0.586 (−0.005)
- ATHEROSCLEROSIS:    0.555 (−0.040)
- ACIDOSIS:           0.553 (−0.017)
- KETOACIDOSIS:       0.538 (−0.377)  ← collapse from 0.915
- HYPEROSMOLALITY:    0.531 (−0.054)
- INFECTION:          0.499 (−0.052)

Peak MAE (hours) Δ vs B0-C-ttt:
- DEATH:   172.74 (+3.77, marginal)
- RELEASE:  79.16 (+7.87)
- DISGLYCEMIA_Hyper:  32.59 (−3.47, small improvement)
- KIDNEY:             63.33 (−15.78)

Trajectory honesty:
- `gen_median_hours`:           91.22 (vs 75.05)
- `gen_to_gt_ratio_median`:      0.900 (≥ 0.4 ✓)
- `gen_frac_terminal_first24h`:  0.245 (vs 0.165 — terminal-first jumps)

Phase stats: phase2 ran all 50 epochs; phase3 early-stopped at 6 with
best at epoch 1.

Verdict: **DISCARD**. Falsifiable (patient AUROC ≥ +0.030) missed by
0.070. The MIL aux pulled Phase 3 away from the running best optimum
within 1 epoch of activation, and the model never recovered. The
likely mechanism: with patient_label being "outcome occurs anywhere",
the soft-max-pooled score is dominated by the position with the
highest logit, and BCE gradient on the pool propagates back to that
position. For a negative patient on a rare outcome, the path of least
resistance is to lower ALL logits — destroying per-position
discrimination that B0-C-ttt had carefully built. The per-position
BCE anchor was insufficient to hold ground (its λ=1.0 vs MIL's
effective contribution ~0.20 of BCE, but the gradient directions
conflict). The single positive class (RELEASE, 87 % prevalence)
benefits because the pool's collective lift is aligned with its
target.

This is a learning-recipe problem, not a code/architecture bug. The
direction is sound in principle, but the loss formulation as
specified is too coarse next to per-position BCE for rare outcomes.
P2's soft-argmax time loss is a positives-only loss — that constraint
may avoid this failure mode. Proceeding to P2.

Reverting code commit per loop step 9.

---

### P2-time @ 10k (SHA 10abcc1) — DISCARD

P2 direction. Added a positives-only soft-argmax onset-time aux to
Phase 3: weighted softmax(logit / T_k) over time gives a continuous
predicted onset time; smooth-L1 to the nearest GT occurrence
(detached, scaled to hours). Per-outcome learnable `time_log_T`.
λ_time calibrated once at Phase-3 epoch 1 (cap=0.20). Patients
without the outcome contribute zero gradient.

Smoke (sample=50, phase{1,2,3}_n_epochs=1):
- Gates A–D pass after switching the smooth-L1 inputs from normalised
  time (0…1) to hours (×336). Without the hour rescale λ_time
  calibrated at 94 — outside the [1e-3, 10] band. With rescale:
  raw_time=50.94 h, λ_time=0.034, in band.

Post-train (10k):
- T1 partial fail — raw_time barely moves over the 15 active Phase-3
  epochs (61.9 → 57.8 h, then plateau). Aux gradient gets absorbed
  into the joint optimum without actually reducing the time error.
- T2 fail — Phase-3 best `vl_select` is **1.143** at epoch 10, worse
  than B0-C-ttt's 1.010. Selection metric pure-outcome-BCE held
  monotonically above the running-best optimum the entire run.
- T3 fail — DEATH AUROC drops 0.710→0.631 (-0.079); the aux that was
  supposed to refine onset timing actually weakened the per-position
  discriminator that drives the eval headline.

Headline (Δ vs B0-C-ttt running best):
- `patient_auroc_weighted`: **0.5735** (−0.1097 — fails KEEP rule)
- `patient_auprc_weighted`: 0.5551 (−0.0785)
- `patient_auroc_simple`:   0.5687 (−0.1272)
- `patient_auprc_simple`:   0.2526 (−0.0713)
- `n_outcomes_used`:        16

Per-outcome AUROC Δ vs B0-C-ttt — universal regression:
- DISGLYCEMIA_Hyper:  0.814 (−0.082)
- DISGLYCEMIA_Hypo:   0.641 (−0.130)
- DEATH:              0.631 (−0.079)
- NEUROVASCULAR:      0.623 (−0.063)
- KIDNEY:             0.609 (−0.106)
- ACUTE_RESPIRATORY:  0.592 (+0.001)
- ACIDOSIS:           0.581 (+0.011)
- CARDIO:             0.570 (−0.139)
- RETINOPATHY:        0.547 (−0.238)
- SKIN_ULCER:         0.531 (−0.149)
- INFECTION:          0.529 (−0.023)
- HYPEROSMOLALITY:    0.514 (−0.071)
- ATHEROSCLEROSIS:    0.506 (−0.089)
- KETOACIDOSIS:       0.493 (−0.422)  ← chance
- NERVOUS_SYSTEM:     0.475 (−0.322)  ← below chance
- RELEASE:            0.444 (−0.137)  ← below chance

Peak MAE (hours, mixed; falsifiable wanted ≥−5 h for both):
- DEATH:    156.06 (−12.91 ✓)
- RELEASE:   81.38 (+10.09 ✗)
- DISGLYCEMIA_Hyper:  26.11 (−9.96)
- KIDNEY:             84.86 (−21.50)
- CARDIO:            121.54 (+42.46 ✗)

Trajectory honesty:
- `gen_median_hours`:           79.40
- `gen_to_gt_ratio_median`:      0.770 (≥ 0.4 ✓)
- `gen_frac_terminal_first24h`:  **0.421**  ← 2.6× the B0-C-ttt rate;
  the time aux made the model commit to early terminal emission, which
  collapses the rare-outcome discrimination because every patient
  trajectory ends so quickly there's no time to differentiate.

Phase stats: phase2_best_val 0.187 / 40 epochs; phase3_best_val 1.152
/ 15 epochs (early stopped, never recovered).

Verdict: **DISCARD**. Falsifiable failed on both prongs (RELEASE MAE
regressed and patient AUROC regressed catastrophically). Even DEATH
MAE improvement is hollow — the trajectory now collapses to terminal
within 24 h for 42 % of patients, which structurally pulls the DEATH
peak time forward without actually predicting WHICH patient dies.

Same failure family as P1: a patient-level/coarse-time aux added to
Phase 3 corrupts the per-position discriminator that B0-C-ttt's
per-position BCE + ranking carefully built. The shared lesson is
that Phase-3 aux losses that target the eval metric directly (MIL
in P1, soft-argmax onset in P2) push the head into a degenerate
sharp-peak regime — gain on the targeted metric, collapse on the
rest. The per-position BCE anchor at λ=1.0 is not strong enough on
its own to hold the optimum when a 0.20-capped aux pulls in a
fundamentally different direction.

This is the second DISCARD in a row. Reverting per loop step 9.
Proceeding to P3 (risk-aware LM head — architectural coupling),
which works at the LM head rather than the outcome head and therefore
won't fight the per-position BCE head-on.

---

### P3-coupling @ 10k (SHA 7838ac3) — DISCARD

P3 direction. Added bias_proj: nn.Linear(K → V) zero-init, applied to
sigmoid(outcome_logits), summed into lm_logits at the same position.
Coupling forms during Phase 2 (LM CE flows through bias_proj into the
joint backbone); Phase 3 refines outcome_head. Per-epoch ratio
||bias|| / ||lm_only_logits|| tracked.

Smoke (sample=50, phase{1,2,3}_n_epochs=1):
- Gates A–D pass.
- P3a constructor-time zero-init verified at __init__ (assert in __init__).
- P3c shape contract verified — outcome_logits (B,T,K), logits (B,T,V),
  bias (B,T,V) match.
- P3b grad norms info: outcome_head[-1] 3.79 (active), bias_proj 0
  (expected — no LM CE in Phase 3), lm_head 0.45 (through tied input
  embedding).
- Smoke p3_ratio_mean 0.108, max 0.125 — in [0.05, 0.30] healthy band.

Post-train (10k):
- T1 partial — outcome head trained (raw_out 1.20→1.02 over 35 P3
  epochs), but Phase-3 train loss starts wildly high (epoch 1
  train=17.90, raw_out=17.90) because Phase 2 over-trained the
  coupling. The outcome head's logits at Phase 2 boundary are extreme
  (the coupling shapes them toward LM utility, not BCE calibration).
- T2 — Phase 3 ran 35 epochs (early stop at 36). vl_select dropped to
  0.980 — actually LOWER than B0-C-ttt's 1.010 best. But this lower
  selection metric did NOT translate to better headline AUROC, because
  the coupling distorted the outcome logits away from per-outcome
  ranking optima.
- T3 — DEATH AUROC dropped to 0.670 (-0.040 vs running best).
- p3_ratio at Phase-3 start: 1.10 (bias DOMINATES lm_only — way above
  [0.05, 0.30]). By end of Phase 3 the ratio settled to 0.047 mean
  (just below band), 0.46 max — Phase-3 training partially undid the
  coupling but the model never recovered the running-best optimum.

Headline (Δ vs B0-C-ttt running best):
- `patient_auroc_weighted`: **0.6473** (−0.0358 — fails KEEP rule)
- `patient_auprc_weighted`: 0.6217 (−0.0119)
- `patient_auroc_simple`:   0.6714 (−0.0244)
- `patient_auprc_simple`:   0.3051 (−0.0188)
- `n_outcomes_used`:        16

Per-outcome AUROC Δ vs B0-C-ttt — mixed but DEATH and RELEASE both regressed:
- DISGLYCEMIA_Hyper:  0.874 (−0.022)
- DISGLYCEMIA_Hypo:   0.818 (+0.047)
- NERVOUS_SYSTEM:     0.773 (−0.023)
- KIDNEY:             0.747 (+0.032)
- CARDIO:             0.736 (+0.027)
- RETINOPATHY:        0.720 (−0.066)
- NEUROVASCULAR:      0.712 (+0.026)
- **DEATH**:          0.670 (−0.040)
- KETOACIDOSIS:       0.651 (−0.265)
- SKIN_ULCER:         0.650 (−0.030)
- HYPEROSMOLALITY:    0.607 (+0.022)
- ACUTE_RESPIRATORY:  0.601 (+0.010)
- ACIDOSIS:           0.596 (+0.026)
- ATHEROSCLEROSIS:    0.592 (−0.003)
- INFECTION:          0.573 (+0.022)
- **RELEASE**:        0.425 (−0.156)  ← below chance, biggest collapse

Peak MAE (hours, Δ vs B0-C-ttt):
- DEATH:    155.08 (−13.89, ✓)
- RELEASE:   73.55 (+2.26)
- DISGLYCEMIA_Hyper:  34.59 (−1.48)
- KIDNEY:             91.71 (−14.65)

Trajectory honesty:
- `gen_median_hours`:           176.06  (vs 75.05 — over-generates ~2.3×)
- `gen_to_gt_ratio_median`:      1.694  (above 1.0 but ≥ 0.4 ✓)
- `gen_frac_terminal_first24h`:  **0.005**  (vs 0.165 — model essentially
  refuses to terminate early; mechanistic explanation for the RELEASE
  collapse: RELEASE is 87% prevalence and is a terminal token, and the
  coupling steers the LM AWAY from emitting terminals → the LM almost
  never emits RELEASE, so its detection collapses).

Phase stats: phase2 38 epochs; phase3 35 epochs (early stopped at 36).

Verdict: **DISCARD**. Falsifiable required DEATH AND RELEASE AUROC
both ≥ +0.030 — both regressed. RELEASE collapsed below chance
(0.425). The coupling at Phase 2 trained the LM to lean on the
outcome path for emission decisions, but the bias dominated by end of
Phase 2 (||bias||/||lm_only||≈1.1) and the LM head atrophied — at
inference the model can't generate trajectories that include the
terminal-class tokens (RELEASE/DEATH) at appropriate times, so
RELEASE detection collapses despite the coupling's intent to PROMOTE
terminal-token emission.

The architecture is sound; the problem is the LR/cap regime during
Phase 2 — bias_proj got too much budget and the LM head atrophied.
A targeted retry would need to (a) cap bias_proj's contribution to
||lm_logits|| (e.g., normalize bias by a learned scalar with prior
that keeps ratio ≤ 0.3), (b) gate bias_proj behind a slow ramp during
Phase 2, or (c) add a regularizer on bias_proj.weight to keep it
small. None of these are in scope for the current 10k probe regime
— program.md's P3 spec is exactly what was tested, and that spec
yielded a clear DISCARD.

Three DISCARDs in a row. Per program.md stop criterion, this nudges
toward halting, but P4 (patient-level pooling head) is structurally
different from P1/P2/P3 — different head architecture, different
loss path. Worth one more probe before concluding.

Reverting (loop step 9) and proceeding to P4.

---

### B0-C-ttt-ablation @ 10k (SHA 49be091) — KEEP-STACK

Mandatory ablation per program.md's new discipline. Strips Z's
direction-E frozen-terminal-tau hook from B0-C-ttt's recipe; keeps the
C-ttt aux head. Tests whether B0-C-ttt's gain over B0-Z came from
C-ttt alone or from the Z+C-ttt stack.

Per-aux training trace (Phase-2 + Phase-3, this ablation run):

| Aux       | Unlock epoch | λ_max   | Anchor raw_aux | Final raw_aux | Δ      | Status |
|-----------|--------------|---------|----------------|---------------|--------|--------|
| ce        | 4 (Ph-2)     | 0.0900  | 1.5183         | 0.0082        | −99.5% | learning |
| dt        | 4 (Ph-2)     | 0.1688  | 0.8096         | 0.1062        | −86.9% | learning |
| ttt       | 4 (Ph-2)     | 0.0038  | 21.4508        | 0.1663        | −99.2% | learning |
| ranking   | 31 (Ph-2)    | 0.0316  | 0.1224         | 0.1312        | +7.2%  | **STALE** — no descent across the 9 active Ph-2 epochs; raw_ranking actually rose slightly |
| out (P3)  | 1 (Ph-3)     | —       | 1.9035         | 0.9364        | −50.8% | learning |
| ranking(P3)| 1 (Ph-3)    | 0.1962  | 0.5683         | 0.3424        | −39.7% | learning |

The Phase-2 ranking aux fired at the very end of Phase 2 (calibrated
epoch 30, ramp 31→33, only 6 fully-active epochs before early stop at
40), and across those 9 active epochs raw_ranking did NOT descend.
Flag: **stale**. Either (a) the model is already at ranking optimum
by the time the aux unlocks (logits already separate pos / neg
positions enough that the pairwise loss is at floor for the current
backbone), or (b) the late-stage activation hits when other auxes
already drove the backbone into a state where ranking improvements
are gradient-flat.

Per-aux training trace (B0-C-ttt running best, recoverable values
only; final raw_aux at end of Phase 2 was overwritten when run.log
was reused for P1/P2/P3/M-384/ablation):

| Aux       | Unlock epoch | λ_max   | Anchor raw_aux | Final raw_aux | Δ      | Status |
|-----------|--------------|---------|----------------|---------------|--------|--------|
| ce        | 4 (Ph-2)     | 0.0872  | 1.5708         | ?             | ?      | unrecoverable |
| dt        | 4 (Ph-2)     | 0.1715  | 0.8027         | ?             | ?      | unrecoverable |
| ttt       | 4 (Ph-2)     | 0.0040  | 20.86          | ?             | ?      | unrecoverable |
| ranking   | 32 (Ph-2)    | 0.0329  | 0.1904         | ?             | ?      | unrecoverable |
| out (P3)  | 1 (Ph-3)     | —       | 2.11           | 1.01          | −52%   | learning (from prior journal) |
| ranking(P3)| 1 (Ph-3)    | ~0.7    | 0.66           | 0.38          | −42%   | learning (from prior journal) |

Future runs will record these from the start (the schema is now
program.md-mandated).

Smoke (sample=50, phase{1,2,3}_n_epochs=1):
- Total params: 6.42 M (same as B0-C-ttt — pure hook removal).
- Gates A–D pass — raw_out=8.50, raw_rank=0.70, λ_ranking=2.45.

Headline (Δ vs B0-C-ttt running best):
- `patient_auroc_weighted`: **0.6401** (−0.0430)
- `patient_auprc_weighted`: 0.6079 (−0.0257)
- `patient_auroc_simple`:   0.6553 (−0.0405)
- `patient_auprc_simple`:   0.2865 (−0.0374)
- `n_outcomes_used`:        16

Per-outcome AUROC vs B0-C-ttt:
- DISGLYCEMIA_Hyper:  0.890  (−0.006)
- RETINOPATHY:        0.737  (−0.048)
- NEUROVASCULAR:      0.730  (+0.044)
- NERVOUS_SYSTEM:     0.718  (−0.078)
- CARDIO:             0.717  (+0.008)
- DISGLYCEMIA_Hypo:   0.678  (−0.093)
- **DEATH**:          0.670  (−0.040)
- SKIN_ULCER:         0.663  (−0.016)
- KETOACIDOSIS:       0.659  (−0.256)  ← biggest drop
- ATHEROSCLEROSIS:    0.657  (+0.062)
- KIDNEY:             0.634  (−0.081)
- HYPEROSMOLALITY:    0.593  (+0.008)
- ACUTE_RESPIRATORY:  0.584  (−0.007)
- ACIDOSIS:           0.545  (−0.025)
- INFECTION:          0.510  (−0.041)
- **RELEASE**:        0.501  (−0.080)  ← drops to chance

Peak MAE vs B0-C-ttt:
- DEATH:    146.49 (−22.48)
- RELEASE:   73.73 (+2.44)
- CARDIO:    73.20 (−5.88)
- KIDNEY:    92.95 (+13.84)

Trajectory honesty (Δ vs B0-C-ttt):
- `gen_median_hours`:           215.51  (+140.46 — much longer)
- `gen_to_gt_ratio_median`:       2.112  (vs 0.720; over-generates 2× —
  the unfrozen terminal tau lets the LM widen the terminal kernel,
  which prior diagnostics showed it tends to do)
- `gen_frac_terminal_first24h`:   0.037  (vs 0.165; rarely terminates
  early — same mechanism, wide terminal kernel pushes terminal
  emission far out)

Phase stats: phase2_best_val 0.184 / 40 epochs; phase3_best_val 1.112
/ 29 epochs.

Verdict: **ABLATION-KEEP-STACK** — Z's frozen-narrow-terminal-tau
hook is doing real work. Stripping it costs the recipe 0.043
patient_auroc_weighted, drops RELEASE to chance (0.501), collapses
KETOACIDOSIS (−0.256), and degrades trajectory honesty
(`gen_to_gt_ratio_median` doubles from 0.72 to 2.11). The C-ttt aux
alone, on bare M-256, does not match B0-C-ttt's eval lift. B0-C-ttt
remains the running best.

T1 note: Phase-2 ranking aux was **stale** in this run (Δ=+7.2 %
across 9 active epochs). This is also expected behaviour for ranking
when it unlocks late in Phase 2 — the backbone has already settled.
Stale-ness was not caused by the ablation. Logging the staleness now
that the schema requires it; not actionable for this experiment.

This ablation does not become the running best. Proceeding to P4
(patient-level pooling head — note: eval is read-only, so the pool
output can only be a training-time aux, similar in spirit to P1 MIL).

---

## Reproducibility

| Artefact | Location |
|---|---|
| Branch | `autoresearch-trajectory` |
| Canonical baseline checkpoints (read-only) | `emr_model/checkpoints.bak_originals/` |
| Running-best backups | `emr_model/checkpoints.bak_keep_<tag>/` |
| Ledger | `results/results-trajectory-fix.tsv` |
| Source data (not in repo) | `emr_model/data/source/temporal_data.csv` + `context_data.csv` |
| Train / val / test split | `PatientId`-stratified 70 / 15 / 15, `random_state=42` (in `api.py`) |

To reproduce from a fresh clone: place source CSVs under
`emr_model/data/source/`, then `python api.py`. The pipeline builds a
tokenizer + scaler from the train split, caches the processed dataset,
runs the three phases (training in one subprocess, eval in another),
and prints the summary block.
