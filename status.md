# autoresearch — loop status (UTC 2026-05-13 ~07:30)

## TL;DR — **NEW BEST exp73** (`3eaafa7`)

```
AUROC 0.882   (+0.012 vs exp69; +0.049 vs exp63 pre-session)
AUPRC 0.483
MAE   83.9h
max_len 0.0%   (perfect termination preserved)
DEATH    0.975
CARDIO   0.906  (+0.050 vs exp71 recovery; -0.027 vs exp69 0.933 peak)
HYPERGLY 0.884
KIDNEY   0.863  HYPOGLY 0.849
RELEASE  0.812                    peak VRAM 9.4 GB
```

The headline is the **horizon-aligned three-tier `log_tau_lm` init**:
default tokens at log(12/336), outcome-class tokens at log(48/336)
(matches `outcome_horizon_hours`), terminals at log(168/336). exp71
had dropped P2 outcome BCE and CARDIO collapsed −0.077 because its
soft kernel was init'd at the default 12h window. Aligning the
init with the eval window's actual horizon recovered CARDIO +0.050
(0.856 → 0.906), most of the way back from the collapse, while
keeping the simpler-codebase win from exp71.

## Session arc

| Exp | Commit | AUROC | AUPRC | RELEASE | max_len% | Status |
|---|---|---|---|---|---|---|
| exp63 | `033e019` | 0.833 | 0.434 | 0.694 | 8.5 | (pre-session) |
| exp64 | `2c60c2a` | 0.797 | 0.364 | 0.688 | 14.9 | DISCARD (skip-P3) |
| exp65 | `12ce6fe` | 0.829 | 0.409 | 0.732 | 12.6 | DISCARD (selector bug) |
| exp66 | `82387ca` | 0.850 | 0.452 | 0.727 | 11.6 | KEEP (P3 ranking + sel fix) |
| exp67 | `d854e7d` | 0.819 | 0.397 | 0.594 | 3.7 | DISCARD (P3 ranking-only) |
| exp68 | `260d0dc` | 0.802 | 0.397 | 0.681 | 21.1 | DISCARD (P3 oversampled) |
| exp69 | `ebe9618` | 0.870 | 0.512 | 0.832 | 0.0 | KEEP (soft-kernel LM head) |
| exp70 | (n/a)     | —     | —     | —     | —    | CRASH (P1 soft kernel) |
| exp71 | `3da7a74` | 0.876 | 0.494 | 0.831 | 0.0 | KEEP (drop P2 outcome BCE) |
| exp72 | (rolled-back) | 0.859 | 0.510 | 0.783 | 0.1 | DISCARD (contrastive aux) |
| **exp73** | **`3eaafa7`** | **0.882** | **0.483** | **0.812** | **0.0** | **KEEP — current** |

## What worked vs what didn't

**Worked:**
1. **P3 ranking + stable val_outcome_raw selector** (exp66) — fixing the
   selector bug let the ranking gradient train for 27 epochs; +0.017.
2. **Soft-kernel LM-head BCE** (exp69) — learnable per-token-class
   exponential decay replaced exp59's hard two-tier window; +0.020 AUROC,
   +0.060 AUPRC, +0.105 RELEASE, max_len 11.6 → 0.0pp.
3. **Drop P2 outcome BCE** (exp71) — soft-kernel LM head carries the
   timing signal; +0.006 AUROC, one less loss term.
4. **Horizon-aligned three-tier kernel init** (exp73) — CARDIO recovered;
   AUROC +0.006 vs exp71.

**Didn't work:**
- Skip P3 (exp64): P3 is net-positive +0.036.
- Ranking-only P3 (exp67): BCE calibrates RELEASE; −0.133 RELEASE.
- Oversampled P3 DL (exp68): `pos_weight` mismatched; broad regression.
- P1 soft kernel (exp70, exp70-retry): destabilises Time2Vec/time_head
  even with freeze-then-unfreeze. Phase 1 BCE is too foundational.
- Same-patient contrastive aux (exp72): raw 3.31 → 3.0 only (7% drop,
  well below Rule 2(a) 30% bar), backbone capacity diverted, AUROC −0.017.
  Answers the open question — same-patient-position contrastive does
  NOT help on event sequences this size.

## Per-outcome trajectory (this session)

| Outcome | exp63 | exp66 | exp69 | exp71 | **exp73** |
|---|---|---|---|---|---|
| DEATH    | 0.988 | 0.983 | 0.950 | 0.978 | **0.975** |
| CARDIO   | 0.863 | 0.899 | 0.933 | 0.856 | **0.906** |
| HYPERGLY | 0.843 | 0.836 | 0.852 | 0.888 | **0.884** |
| KIDNEY   | 0.802 | 0.819 | 0.844 | 0.850 | **0.863** |
| HYPOGLY  | 0.805 | 0.835 | 0.811 | 0.855 | **0.849** |
| RELEASE  | 0.694 | 0.727 | 0.832 | 0.831 | **0.812** |

Every outcome has gained net since session start, with RELEASE
(+0.118), CARDIO (+0.043), KIDNEY (+0.061), HYPOGLY (+0.044),
HYPERGLY (+0.041) up substantially. DEATH (−0.013) is the only
outcome marginally below pre-session — and 0.975 is still excellent.

## Open directions

- **Δt regression investigation**: exp69 / exp71 / exp73 all have a
  Δt head R² ≈ -1.22 (bias issue — pred mean ~3h vs true ~1h)
  despite Task A being "locked". Worth a careful look at whether
  the lock holds across fresh-P1 retrains.
- **Outcome-head simplification**: diag (on exp69) showed LM head
  beats outcome head teacher-forced for all four active outcomes.
  The outcome head is largely a wrapper around what the LM head
  already encodes plus the one-hot override at emit time. Could
  simplify by replacing the outcome head with a small linear
  classifier on top of LM logits.
- **Per-outcome P3 BCE**: exp71 showed P2 outcome BCE was helping
  CARDIO. exp73 partially recovered CARDIO via the kernel init. A
  CARDIO-targeted aux is still on the table if exp73's recovery
  proves insufficient against fresh seeds.
- **P1 contrastive** — different from exp72: positives are NOT
  same-patient positions but two random patients with very similar
  early-window features (matched at the embedder level). Untested.

## Process discipline

- `results.tsv`: 80 data rows + header. Untracked.
- exp73 retrained P1 fresh (cleared cache before run). Per Rule 5,
  +0.006 margin over exp71 just clears the 0.005 noise floor, plus
  the CARDIO recovery is well outside any fresh-P1 noise band, so
  the lock is real.
- 7-decimal raw aux logging unchanged.
- Committing locally only; user pulls from root.
