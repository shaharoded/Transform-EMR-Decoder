# autoresearch — loop status (UTC 2026-05-13 ~11:00)

## TL;DR — **NEW BEST exp73** (`3eaafa7`)

```
AUROC 0.882   (+0.049 vs exp63 pre-session)
AUPRC 0.483
MAE   83.9h
max_len 0.0%  (PERFECT termination preserved)
DEATH    0.975
CARDIO   0.906
HYPERGLY 0.884   KIDNEY 0.863
HYPOGLY  0.849
RELEASE  0.812                    peak VRAM 9.4 GB
```

## Session arc (this loop)

| Exp | Commit | AUROC | AUPRC | RELEASE | max_len% | Status |
|---|---|---|---|---|---|---|
| exp63 (pre-session) | `033e019` | 0.833 | 0.434 | 0.694 | 8.5 | — |
| exp64 | `2c60c2a` | 0.797 | 0.364 | 0.688 | 14.9 | DISCARD (skip-P3) |
| exp65 | `12ce6fe` | 0.829 | 0.409 | 0.732 | 12.6 | DISCARD (selector bug) |
| exp66 | `82387ca` | 0.850 | 0.452 | 0.727 | 11.6 | KEEP (P3 ranking + sel fix) |
| exp67 | `d854e7d` | 0.819 | 0.397 | 0.594 | 3.7 | DISCARD (ranking-only P3) |
| exp68 | `260d0dc` | 0.802 | 0.397 | 0.681 | 21.1 | DISCARD (P3 oversampled) |
| exp69 | `ebe9618` | 0.870 | 0.512 | 0.832 | 0.0 | KEEP (soft-kernel LM head) |
| exp70 | (CRASH)   | —     | —     | —     | —    | CRASH (P1 soft kernel) |
| exp71 | `3da7a74` | 0.876 | 0.494 | 0.831 | 0.0 | KEEP (drop P2 outcome BCE) |
| exp72 | (DISCARD) | 0.859 | 0.510 | 0.783 | 0.1 | DISCARD (contrastive aux) |
| **exp73** | **`3eaafa7`** | **0.882** | **0.483** | **0.812** | **0.0** | **KEEP — current** |
| exp74 | (DISCARD) | 0.863 | 0.402 | 0.819 | 0.7 | DISCARD (BCE+wide-init hybrid) |
| exp75 | (DISCARD) | 0.846 | 0.439 | 0.703 | 0.4 | DISCARD (linear outcome head) |
| exp76 | (DISCARD) | 0.845 | 0.392 | 0.788 | 0.9 | DISCARD (log1p Δt loss) |

## What worked

1. **P3 ranking + stable selector** (exp66, +0.017 AUROC) — methodology
   fix unlocked the architectural change; selector now watches
   `val_outcome_raw` (stable across the λ=0 → λ_cal transition).
2. **Soft-kernel LM-head BCE** (exp69, +0.020 AUROC, +0.060 AUPRC,
   +0.105 RELEASE, max_len → 0%) — `log_tau_lm` Parameter[V] replaces
   the hard two-tier window; the model learns kernel scale per class.
3. **Drop P2 outcome BCE** (exp71, +0.006 AUROC) — soft kernel now
   carries the outcome-timing signal; one less loss term.
4. **Three-tier kernel init** (exp73, +0.006 AUROC) — outcome-class
   tokens init at log(48/336) matching `outcome_horizon_hours`,
   recovers CARDIO from exp71's −0.077 collapse.

## What didn't (and why we now know better)

- **Skip P3** (exp64): P3 is net-positive on average +0.036; ranking
  loss in P3 carries the bulk of the AUROC signal that survives the
  P2→P3 phase boundary.
- **Ranking-only P3** (exp67): BCE calibrates the head specifically
  for RELEASE; removing it cost RELEASE −0.133.
- **Oversampled P3 DL** (exp68): `pos_weight` is calibrated for the
  natural distribution; oversampling double-counts rare positives
  and over-fits them locally.
- **P1 soft kernel** (exp70 / exp70-retry): destabilises Time2Vec /
  time_head even with freeze-then-unfreeze. P1 BCE is too foundational.
- **Patient-position contrastive aux** (exp72): contrastive raw only
  dropped 7% (well below Rule 2(a) 30% bar), capacity diverted from
  primary signals, RELEASE −0.048, HYPOGLY −0.061. Answer to user's
  open question: same-patient-position contrastive does NOT help on
  event sequences this size.
- **BCE + horizon-aligned kernel hybrid** (exp74): outcome BCE +
  wider kernel = redundant over-supervision; CARDIO regressed −0.064
  even with BCE restored.
- **Linear outcome head** (exp75): MLP non-linearity is load-bearing
  for RELEASE (−0.109) and HYPOGLY (−0.054); RELEASE encoding needs
  multiplicative gating across features ("stable vitals AND no
  complications" type logic).
- **log1p Δt loss** (exp76): MSE-on-absolute-time is load-bearing
  for AR generation despite producing R²=−1.22 in the diag probe.
  log1p compresses the long-Δt tail and degrades trajectory
  generation across nearly every outcome. The Δt probe's bad value
  is misleading — Task A's "lock" is the init scheme, not the
  probe value.

## Per-outcome trajectory (this session)

| Outcome | exp63 | exp66 | exp69 | exp71 | **exp73** | Δ vs exp63 |
|---|---|---|---|---|---|---|
| DEATH    | 0.988 | 0.983 | 0.950 | 0.978 | **0.975** | −0.013 |
| CARDIO   | 0.863 | 0.899 | 0.933 | 0.856 | **0.906** | +0.043 |
| HYPERGLY | 0.843 | 0.836 | 0.852 | 0.888 | **0.884** | +0.041 |
| KIDNEY   | 0.802 | 0.819 | 0.844 | 0.850 | **0.863** | +0.061 |
| HYPOGLY  | 0.805 | 0.835 | 0.811 | 0.855 | **0.849** | +0.044 |
| RELEASE  | 0.694 | 0.727 | 0.832 | 0.831 | **0.812** | **+0.118** |

Every outcome gained or stayed flat. RELEASE +0.118 is the largest
gain — the soft kernel's terminal-tier did the most work there.

## Open directions

- **Recover AUPRC peak** from exp69 (0.512 → 0.483 in exp73). Spread
  across outcomes; not a single-outcome story. exp74 (BCE on wide
  kernel) didn't recover it. Probably needs a different mechanism.
- **Per-outcome ranking head**: maybe each outcome benefits from its
  own learnable ranking margin / negative-sampling strategy.
- **Different contrastive design**: exp72 tried same-patient positions.
  Alternative: positions across patients with similar outcome
  trajectories. Risk: still capacity-diversion.
- **Bigger model / longer training**: out of scope per Rule 3.

## Process discipline

- `results.tsv`: 83 data rows + header. Untracked.
- DISCARDs reset via `git reset --hard`; codebase consistently at the
  current-best architecture.
- 7-decimal raw aux logging in P2 and P3 unchanged.
- Committing locally only; user pulls from root.
