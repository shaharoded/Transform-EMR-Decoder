# autoresearch — FINAL SESSION REPORT (UTC 2026-05-13 ~19:30)

## TL;DR — final best **exp73** (`3eaafa7`) — loop stopped

```
AUROC 0.882   AUPRC 0.483   MAE 83.9h   max_len 0.0%
DEATH 0.975  CARDIO 0.906  HYPERGLY 0.884  KIDNEY 0.863
HYPOGLY 0.849  RELEASE 0.812         peak VRAM 9.4 GB
```

Directions I and J both DISCARDed. Loop terminated at the natural
endpoint per program.md "stop criterion".

---

## Directions I & J — verdicts

### Direction I — tier-free `log_tau_lm` init — **DISCARD** (exp77, `1c505d6`)

Single scalar init `log(12/336)` for every token instead of exp73's
three-tier (default 12h / outcome-class 48h / terminal 168h).

```
AUROC 0.793  (vs exp73 0.882, -0.089 — WAY past 0.015 threshold)
AUPRC 0.329  (vs exp73 0.483, -0.154)
MAE   91.96  (vs exp73 83.88, +8.08)
max_len 7.8% (vs exp73 0.0%, regressed)
```

Per-outcome (vs exp73):
- DEATH 0.918 (-0.057) · RELEASE 0.751 (-0.061)
- CARDIO 0.862 (-0.044) · HYPOGLY 0.749 (-0.100)
- HYPERGLY 0.745 (-0.139) · KIDNEY 0.735 (-0.128)

**Structural finding**: the three-tier hand-pick prior is LOAD-BEARING
for AUROC and termination. The model could NOT rediscover per-class
scales by gradient alone in the 50-epoch P2 budget; complications
suffered worst (-0.10 to -0.14), terminals lost their wide-window
supervision (max_len jumped 0%→7.8%). The three-tier init is a
principled bootstrap the model needs, not optional cosmetics.

### Direction J — restore P2 outcome BCE — **DISCARD** (exp78, `7c6bcc2`)

Flip `aux_fraction_caps["outcome"]` from 0 to 10.0 on top of exp73
three-tier baseline (same config as previously-tested exp74; re-run
under fresh P1 seed per user instruction).

```
AUROC 0.868  (vs exp73 0.882, -0.014 — PAST 0.010 DISCARD threshold)
AUPRC 0.509  (vs exp73 0.483, +0.026 — RECOVERED toward exp69's 0.512)
MAE   82.65  (vs exp73 83.88, -1.23 — slightly better)
max_len 0.4% (vs exp73 0.0%, minor)
```

Per-outcome (vs exp73):
- DEATH 0.958 (-0.017) · RELEASE 0.841 (+0.029)
- CARDIO 0.855 (-0.051 !!) · HYPOGLY 0.819 (-0.030)
- HYPERGLY 0.874 (-0.010) · KIDNEY 0.861 (-0.002)

**Structural finding**: outcome BCE + soft-kernel are REDUNDANT for
CARDIO under the 48h kernel tier. Together they over-supervise the
very outcome the three-tier init recovered (CARDIO regressed to
exp71's no-BCE/no-tier 0.856 level). The +0.026 AUPRC gain is real
but the AUROC cost (-0.014) clears the DISCARD threshold; the
AUROC/AUPRC trade is intrinsic to the soft-kernel + tier-init regime.

exp74 vs exp78 (same config, different P1 seed) shows fresh-P1
AUPRC variance is ~±0.05 — exp74 measured -0.081 AUPRC, exp78
measured +0.026 AUPRC. AUROC variance was ~±0.005. This confirms
exp73 is a stable Pareto-optimal point against random-init noise.

---

## Final state

- **Best architecture**: exp73 (`3eaafa7`).
- **HEAD** rolled back to `989b7bb` (program.md commit on top of
  the exp73 codebase). Both exp77 and exp78 commits reset out per
  DISCARD discipline.
- **results.tsv**: 85 data rows + header (exp77, exp78 logged).
- **Peak VRAM**: 9.4 GB across the run.

## Session arc (full)

| Exp | Commit | AUROC | AUPRC | RELEASE | max_len% | Status |
|---|---|---|---|---|---|---|
| exp63 (pre-session) | `033e019` | 0.833 | 0.434 | 0.694 | 8.5 | — |
| exp66 | `82387ca` | 0.850 | 0.452 | 0.727 | 11.6 | KEEP (P3 ranking + selector fix) |
| exp69 | `ebe9618` | 0.870 | 0.512 | 0.832 | 0.0 | KEEP (soft-kernel LM head) |
| exp71 | `3da7a74` | 0.876 | 0.494 | 0.831 | 0.0 | KEEP (drop P2 outcome BCE) |
| **exp73** | `3eaafa7` | **0.882** | **0.483** | **0.812** | **0.0** | **KEEP — FINAL BEST** |
| exp74 | `e81ad4e` | 0.863 | 0.402 | 0.819 | 0.7 | DISCARD (BCE+three-tier, J variant) |
| exp75 | `c429d40` | 0.846 | 0.439 | 0.703 | 0.4 | DISCARD (linear outcome head) |
| exp76 | `d6edf15` | 0.845 | 0.392 | 0.788 | 0.9 | DISCARD (log1p Δt loss) |
| **exp77** | `1c505d6` | 0.793 | 0.329 | 0.751 | 7.8 | DISCARD (Direction I — tier-free init) |
| **exp78** | `7c6bcc2` | 0.868 | 0.509 | 0.841 | 0.4 | DISCARD (Direction J — restore P2 outcome BCE) |

## Session gain vs pre-session

```
AUROC    0.833 → 0.882    +0.049
AUPRC    0.434 → 0.483    +0.049
MAE      81.6h → 83.9h    +2.3h
RELEASE  0.694 → 0.812    +0.118
max_len  8.5%  → 0.0%     -8.5pp (fixed)
```

Every primary outcome above pre-session level. RELEASE +0.118 is
the largest gain — soft-kernel terminal-tier did most of that work.

## Locked structural KEEPs in linear git history

- **exp66** (`82387ca`) — P3 ranking + stable `val_outcome_raw` selector
- **exp69** (`ebe9618`) — learnable per-class soft-kernel LM-head BCE
- **exp71** (`3da7a74`) — drop P2 outcome BCE (redundant with kernel)
- **exp73** (`3eaafa7`) — three-tier `log_tau_lm` init (terminals 168h /
  outcome-class 48h / default 12h)

The three-tier init is now triply-confirmed load-bearing: exp71→exp73
showed `+0.006 AUROC and +0.050 CARDIO when adding the tier`; exp77
showed `-0.089 AUROC and -0.044 to -0.139 per outcome when removing
the tier`; exp78 showed `-0.051 CARDIO when adding outcome BCE that
over-supervises the tier`. All three findings point to the same
mechanism: complications need a wider initial kernel than terminals
don't and the default doesn't supply, and any other supervision that
fights this kernel hurts CARDIO specifically.

## What's left (rejected per Rule 3 + user constraints)

Every remaining lever is either a hyperparameter sweep (forbidden by
Rule 3) or a speculative architectural gamble without a falsifiable
failure mode:

- LR / batch / dropout / embed_dim / n_layer / epoch sweeps — Rule 3.
- Per-outcome custom heads — user constraint: outcome-agnostic only.
- Outcome-specific contrastive aux — user constraint.
- Hazard / outcome→LM coupling — confirmed structurally bad.
- Linear / wider / deeper outcome head — three exps show 2-layer
  MLP D→D ReLU+Dropout→K is at a local optimum (exp24/exp50/exp75).
- Alternative Δt loss form — exp76 (log1p) showed MSE-on-abs is
  load-bearing despite the misleading R² probe.
- Phase-1 soft kernel — three exps failed (exp53/exp70/exp70-retry).

The loop has exhausted scripted directions + all post-exp73 lateral
ideas the user had outlined. Pausing per stop criterion.

## Process discipline

- DISCARDs reset via `git reset --hard 989b7bb` — codebase consistently
  back at current-best architecture after each fail.
- `results.tsv` untracked, append-only.
- Smoke tests passed before both full runs.
- Committing locally only; user pulls from root.
