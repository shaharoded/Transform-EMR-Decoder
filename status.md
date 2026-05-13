# autoresearch — loop status (UTC 2026-05-13 ~04:30)

## TL;DR — **NEW BEST exp69** (`ebe9618`)

```
AUROC 0.870   (+0.020 vs exp66; +0.037 vs exp63)
AUPRC 0.512   (+0.060)
MAE   82.6h
max_len 0.0%  (-11.6pp vs exp66 — PERFECT termination, no fallback)
DEATH    0.950  (-0.033)
CARDIO   0.933  (+0.034)  KIDNEY 0.844 (+0.042)
HYPERGLY 0.852  (+0.016)  HYPOGLY 0.811 (-0.024)
RELEASE  0.832  (+0.105 !!!)              peak VRAM 9.4 GB
```

## Session summary (this loop)

7 experiments + 1 diagnose pass + 2 bug fixes, two big KEEPs.

| Exp | Commit | AUROC | AUPRC | RELEASE | max_len% | Status |
|---|---|---|---|---|---|---|
| exp63 | `033e019` | 0.833 | 0.434 | 0.694 | 8.5 | (pre-session best) |
| exp64 | `2c60c2a` | 0.797 | 0.364 | 0.688 | 14.9 | DISCARD (skip-P3 cost +0.036) |
| exp65 | `12ce6fe` | 0.829 | 0.409 | 0.732 | 12.6 | DISCARD (selector bug — methodology) |
| exp66 | `82387ca` | 0.850 | 0.452 | 0.727 | 11.6 | KEEP (P3 ranking + selector fix) |
| exp67 | `d854e7d` | 0.819 | 0.397 | 0.594 | 3.7 | DISCARD (ranking-only P3) |
| exp68 | `260d0dc` | 0.802 | 0.397 | 0.681 | 21.1 | DISCARD (oversampled P3) |
| **exp69** | **`ebe9618`** | **0.870** | **0.512** | **0.832** | **0.0** | **KEEP — current best** |
| exp70 | (smoke crash) | — | — | — | — | CRASH (P1 soft-kernel → NaN tstamps) |

### Two architectural wins

1. **exp66** (`82387ca`): added P2's pairwise ranking loss to Phase 3
   + early-stop on `val_outcome_raw` (stable across the λ=0→λ_cal
   transition). Fixed exp65's bug where `val_total` jump locked the
   selector onto epoch 1. +0.017 AUROC.

2. **exp69** (`ebe9618`): replaced the hard 12h/168h two-tier LM-head
   BCE window with a learnable per-token-class exponential soft
   kernel. `log_tau_lm` Parameter[V] learns the decay scale per
   class — terminals end at ~340h (2× exp59's hand-picked), default
   tokens at ~7.5h (tighter). +0.020 AUROC, +0.060 AUPRC, +0.105
   RELEASE, max_len 11.6 → 0.0pp.

### Direction A in P3 fully explored

- **sub-1** (joint BCE + ranking): KEPT (exp66).
- **sub-2** (ranking-only): DISCARD (exp67) — BCE calibrates RELEASE.
- **sub-3** (oversampled DL in P3): DISCARD (exp68) — `pos_weight`
  calibration mismatched on rare positives.

Both losses are needed in P3; the joint regime from exp66 is locked.

### Direction C (LM-head soft kernel) outcome

- **P2 soft kernel** (exp69): KEEP, massive win.
- **P1 soft kernel** (exp70): CRASH on smoke. 1 epoch of P1 with the
  new kernel destabilises the time2vec/time_head → NaN abs_ts in AR
  generation → IntCastingNaNError in eval pooler. Per program.md
  Direction C "Risk. If the learned-kernel implementation is wrong it
  could break exp59's signal" — confirmed. Reverted cleanly.

## Open directions (entering next session)

- **Direction B** — patient-trajectory contrastive aux for HYPOGLY /
  RELEASE. RELEASE is no longer the weakness (0.832); **HYPOGLY 0.811
  is now the weakest** active outcome. Biggest surface change in queue.
- **P1 soft kernel — retry with mitigations** — initialise log_tau_emb
  at a value that keeps the soft kernel ≈ binary at epoch 0; freeze
  log_tau for the first N epochs while time2vec stabilises; add tau
  upper bound. Optional follow-up.
- **Investigate Δt regression**: exp69 diag showed Δt R² collapsed
  from 0.18 to -1.22 (pred mean 3.14h vs true 0.91h — a 3× bias).
  Suspect fresh-P1 randomness, but Task A is supposed to be locked.
  Worth a careful look — does the locked Task A really survive a
  fresh P1, or is it only fixed when the cache is preserved?
- **Outcome-head capacity question**: diag shows outcome head loses
  to LM head teacher-forced (all 4 active outcomes). Eval is rescued
  by the one-hot override at emit time. Worth exploring whether
  outcome head could be simpler / replaced with LM-head readout.

## What changed in the codebase

- `transformer.py`: `log_tau_lm` Parameter[V] (init log(12/336) /
  log(168/336) at terminals), P2 BCE routes to soft kernel when
  `phase2_use_soft_kernel=True`, finetune_transformer (P3) gains
  pairwise ranking term + stable `val_outcome_raw` selector.
- `utils.py`: new `get_temporal_soft_targets` (scatter_add along V,
  differentiable w.r.t. tau). `@torch.no_grad()` removed from
  `get_temporal_multi_hot_targets` (no-op for binary path).
- `inference.py`: 5-tuple unpack from `forward_with_cache` (hazard
  removal stale-unpack fix).
- `diagnose.py`: 4-tuple unpack from `model()` (hazard fix) +
  accept Phase 3 ckpt when Phase 2 best is missing.
- `config/model_config.py`: `phase2_use_soft_kernel: True`.

## Process discipline

- `results.tsv`: 77 data rows + header. Untracked.
- All exp65–70 git operations: clean DISCARD via `git reset --hard
  HEAD~1` (or HEAD~2 when the experiment included a status commit).
  Bug fixes kept as standalone commits.
- 7-decimal raw aux logging unchanged.
- Committing locally only.
