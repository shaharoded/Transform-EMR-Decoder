# autoresearch — loop status (UTC 2026-05-13 ~04:00)

## TL;DR — **NEW BEST exp69** (`ebe9618`)

```
AUROC 0.870   (+0.020 vs exp66; +0.037 vs exp63)
AUPRC 0.512   (+0.060)
MAE   82.6h
max_len 0.0%   (-11.6pp from exp66 — PERFECT termination,
                no patient hit the 500-token fallback)
DEATH    0.950  (-0.033)
CARDIO   0.933  (+0.034)  KIDNEY 0.844 (+0.042)
HYPERGLY 0.852  (+0.016)  HYPOGLY 0.811 (-0.024)
RELEASE  0.832  (+0.105 !!!)              peak VRAM 9.4 GB
```

The headline: **RELEASE +0.105 in a single experiment**, the biggest
RELEASE jump in the project's history. AUPRC +0.060 is the biggest
single-experiment AUPRC gain since exp59's data-shape change.

## What exp69 did

Direction C — replaced the hard 12h-default / 168h-terminal two-tier
LM-head BCE window with a learnable per-token-class soft kernel:

    target[b, t, v] = clamp( Σ_{s : 0<dt<=horizon}
                              exp(-dt(t,s) / tau[v])
                              · 1[target_ids[b,s]==v], 0, 1 )

`log_tau_lm` is a `Parameter[V=346]`. Init: log(12/336) default,
log(168/336) at terminal ids. New utils.py function
`get_temporal_soft_targets` uses scatter_add along V to avoid
materialising a [B,T,V] one-hot intermediate (differentiable
w.r.t. tau).

Phase 2 only (Phase 1 deferred per program.md). Implementation
preserves exp66's joint P3 regime (ranking + BCE + stable selector).

## What `log_tau_lm` learned

After 50 P2 epochs:

- Default tokens (init 12h): **median 7.5h**, p10 6.0h, p90 19.3h.
  The model wants *tighter* neighborhood signal than the hard 12h
  window gave.
- Terminal tokens (init 168h): **end at 326h, 351h**.
  The model wants ~**2× more** pre-terminal positive signal than
  exp59 hand-picked. That explains the RELEASE +0.105.
- 291/346 tokens moved by > 22% from init — broad learning.

## Status — picking next

In flight: nothing. Next likely: **diagnose exp69** to identify
remaining failure modes, then pick the structural attack.

Candidates after diagnose:
- **Phase 1 soft kernel** (Direction C extension per program.md):
  if P2 wins cleanly, P1 follow-up is sanctioned. Risk: P1 3h
  window already narrow; could disturb embedder.
- **Direction B** — patient-trajectory contrastive aux. RELEASE
  is now 0.832; HYPOGLY 0.811 is the new weakest. Contrastive
  could lift either, but it's the biggest surface area / highest
  risk experiment in the queue.
- **HYPOGLY-targeted analysis**: it's the one outcome that didn't
  benefit. DEATH also lost slightly. Both have one-hot-override
  dynamics from `_outcome_ids` not present in the soft kernel —
  worth a look at whether the override needs to come back for
  outcome tokens.

## Last completed (this session)

| Exp | Commit | AUROC | AUPRC | RELEASE | max_len% | Status |
|---|---|---|---|---|---|---|
| exp63 | `033e019` | 0.833 | 0.434 | 0.694 | 8.5 | KEEP |
| exp64 | `2c60c2a` | 0.797 | 0.364 | 0.688 | 14.9 | DISCARD (skip-P3) |
| exp65 | `12ce6fe` | 0.829 | 0.409 | 0.732 | 12.6 | DISCARD (selector bug) |
| exp66 | `82387ca` | 0.850 | 0.452 | 0.727 | 11.6 | KEEP (P3 ranking + selector fix) |
| exp67 | `d854e7d` | 0.819 | 0.397 | 0.594 | 3.7 | DISCARD (ranking-only P3) |
| exp68 | `260d0dc` | 0.802 | 0.397 | 0.681 | 21.1 | DISCARD (oversampled P3) |
| **exp69** | **`ebe9618`** | **0.870** | **0.512** | **0.832** | **0.0** | **KEEP — current** |

## Process discipline

- `results.tsv`: 76 data rows + header. Untracked.
- exp69 retrained Phase 1 fresh (smoke deleted the cache). Rule 5
  considerations: fresh-P1 historical noise floor is ±0.01-0.02
  AUROC; exp69's +0.020 just clears it on AUROC alone, but +0.060
  AUPRC and +0.105 RELEASE are well outside any fresh-P1 noise band.
- 7-decimal raw aux logging unchanged.
- Committing locally only.
