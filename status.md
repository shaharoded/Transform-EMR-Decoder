# autoresearch — loop status (UTC 2026-05-13 ~02:30)

## TL;DR — current best **exp66** (`82387ca`)

```
AUROC 0.850   AUPRC 0.452   MAE 81.5h   max_len 11.6%
DEATH 0.983   CARDIO 0.899   HYPERGLY 0.836   HYPOGLY 0.835
KIDNEY 0.802  RELEASE 0.727                  peak VRAM 8.4 GB
```

## Status — in flight

**exp69 — Direction C (soft-kernel LM-head BCE)** (`ebe9618`).

Replaces exp59's two-tier window (12h default / 168h terminals) with a
learnable per-token-class exponential decay kernel. `log_tau_lm` is a
Parameter[V] initialised at log(12/336) by default and log(168/336)
at terminal ids, so the kernel starts at exp(-1)=0.37 weight where the
hard tier used to give 1.0. Phase 2 only (Phase 1 deferred per
program.md).

This run will retrain Phase 1 from scratch — the smoke wiped the
cache and ran a 1-epoch 50-patient phase1. ~62 min ETA total.

Sub-questions exp69 answers:
- AUROC within ±0.01 of exp66 + AUPRC ≥ 0.430 + max_len <15% → KEEP.
- log_tau_lm[terminal] moves > 0.01 from init → kernel actively learns.
- AUROC drops > 0.01 → fall back to "outcome-class tokens (terminals +
  complications) use 48h, everything else 12h" per program.md
  Direction C fallback rule.

## Last completed (this session)

| Exp | Commit | AUROC | AUPRC | RELEASE | max_len% | Status |
|---|---|---|---|---|---|---|
| exp63 | `033e019` | 0.833 | 0.434 | 0.694 | 8.5 | KEEP (was best before exp66) |
| exp64 | `2c60c2a` | 0.797 | 0.364 | 0.688 | 14.9 | DISCARD (skip-P3 cost +0.036) |
| exp65 | `12ce6fe` | 0.829 | 0.409 | 0.732 | 12.6 | DISCARD (selector bug) |
| **exp66** | **`82387ca`** | **0.850** | **0.452** | **0.727** | **11.6** | **KEEP — current** |
| exp67 | `d854e7d` | 0.819 | 0.397 | 0.594 | 3.7 | DISCARD (ranking-only P3: RELEASE -0.133) |
| exp68 | `260d0dc` | 0.802 | 0.397 | 0.681 | 21.1 | DISCARD (oversampled P3: HYPOGLY -0.133) |

## What we now know

- Direction A in P3 fully explored: sub-1 (joint loss) KEEP, sub-2
  (ranking-only) DISCARD, sub-3 (oversampled) DISCARD. exp66's
  joint regime is locked.
- Both outcome BCE and ranking are required in P3 — for different
  reasons. BCE calibrates the head for RELEASE (collapsed −0.133 when
  removed). Ranking carries most of the AUROC signal (+0.044 cost
  when ablated in P2; similar in P3).
- Data-shape changes in P3 (oversampling) hurt rare positives
  catastrophically — `pos_weight` calibration is built for the
  natural distribution.

## Open directions

- **C (in flight as exp69)** — soft-kernel LM-head BCE.
- **G** — remove outcome soft-BCE project-wide. **Less attractive**
  after exp67 (BCE in P3 is necessary for RELEASE); audit_0.2c was
  P2-only.
- **B** — patient-trajectory contrastive aux for RELEASE. Biggest
  surface. Defer.
- **E** — dead (hazard head fully removed from codebase).
- **C fallback** — if exp69 fails guardrails, single principled
  two-tier: outcome-class tokens use 48h, everything else 12h.

## Notable observations

- exp66 → current best path: it took the right *methodology* fix
  (selector on val_outcome_raw) on top of a single architectural
  change (P3 ranking loss) to gain +0.017 AUROC.
- HYPOGLY's behaviour: collapsed in exp67 (BCE-off in P3) AND in
  exp68 (oversampled P3). It's the outcome most sensitive to P3
  configuration — anything that perturbs P3's loss surface hurts it
  more than other outcomes. Worth watching closely in exp69.

## Process discipline

- `results.tsv`: 75 data rows + header. Untracked.
- exp69 retrains Phase 1 from scratch (smoke deleted the cache).
- 7-decimal raw aux logging unchanged.
- Committing locally only.
