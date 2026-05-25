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
`gen_frac_terminal_first24h`), verdict (KEEP / DISCARD) with reason.

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
