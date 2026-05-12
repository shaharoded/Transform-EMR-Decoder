# autoresearch — loop status (UTC 2026-05-12)

## TL;DR

Current best — **exp63** (`033e019`)
```
AUROC 0.833    AUPRC 0.434    MAE 81.6h    max_len 8.5%
DEATH 0.988   CARDIO 0.863   HYPERGLY 0.843   HYPOGLY 0.805
KIDNEY 0.802  RELEASE 0.694                  peak VRAM 8.3 GB
```

Loop resumed after the previous pause. Diagnose ran clean on exp63 (fix:
6 call-sites in `diagnose.py` were still unpacking 5-tuples post hazard
removal). Outcome head is healthy on all 6 active outcomes (no flips,
LM-probe AUROC 0.81–0.99); Δt head R²=0.18, r=0.49.

## Status — in flight

**exp64 — Direction D (skip Phase 3)** — single-config-change ablation.

`phase3_n_epochs: 50 → 0`. Eval falls back to the Phase-2 best checkpoint.
This is the cheapest, most-informative single experiment in the queue per
program.md: it tests whether Phase 3 has been net-negative since the
data-shape gains landed. exp62 (P3 NaN'd → P2 fallback) hit RELEASE=0.813,
which is the *real* upper-bound the audit revealed; skipping P3 cleanly
should reproduce or beat it.

Sub-questions this experiment answers:
- If AUROC ≥ exp63 → P3 is net-negative, remove it.
- If AUROC drops > 0.01 → P3 is contributing on average, then move to
  Direction A and reshape P3's loss.
- If RELEASE jumps but AUROC drops → P3 over-fits the head to outcomes
  other than RELEASE.

## Last completed

| Exp | Commit | AUROC | AUPRC | RELEASE | max_len% | Status |
|---|---|---|---|---|---|---|
| exp60 | `c2f3856` | 0.833 | 0.396 | 0.756 | 8.7 | KEEP (later: per-family caveat) |
| audit_0.4b | `71ddbe9` | 0.825 | 0.427 | 0.681 | 8.9 | KEEP (principled split) |
| audit_0.2a | `083bfdb` | 0.828 | 0.401 | 0.698 | 14.9 | KEEP (hazard removed) |
| audit_0.2c | `b0cabac` | 0.819 | 0.428 | 0.651 | 12.7 | AUDIT (outcome BCE borderline) |
| **exp63** | **`033e019`** | **0.833** | **0.434** | **0.694** | **8.5** | **KEEP (current best)** |

## Open directions (post-exp63)

- **D (in flight)** — skip P3. Cheapest probe. Single config change.
- **A** — reshape P3 loss (add ranking, swap to ranking-only, oversample
  DataLoader in P3, P2-wins checkpoint selection). Run if D shows P3 is
  contributing on average but damaging RELEASE.
- **G** — remove outcome soft-BCE, keep ranking-only (one commit, audit
  0.2c already showed AUPRC +0.027 cost-free). Easy after D.
- **C** (refreshed) — soft-kernel BCE at LM head, learnable per-class tau.
  Replaces the hard 168h-vs-12h two-tier with a model-learned kernel.
  Bigger change, comes after D/A/G land.
- **B** — patient-trajectory contrastive aux for RELEASE. Biggest new
  surface area; defer until simpler RELEASE attacks are exhausted.
- **E** — inference-side hazard boost (now that hazard head is un-trained).
  Cheap; opportunistic.

## Notable observations

- The model's outcome-head logit distribution shows a clean two-cluster
  split: 6 active outcomes have logit means in [-10, -3] with stds 1.6–3.8;
  the 10 never-occurring outcomes are pushed to ~ -29 with std ~ 5.2. The
  outcome head has correctly learnt to silence non-occurring outcomes.
- DEATH gap = 10.8 at LM-probe (auroc 0.991) — the one-hot override
  mechanism is dominant for DEATH, consistent with the Direction-F caveat.
- RELEASE has the smallest "active" gap (3.26) — still the weakness.

## Process discipline

- `results.tsv` is untracked; survives `git reset --hard`. 70 data rows
  + header through exp63.
- 6-decimal raw aux logging is committed (`39c3896`). Any "flat aux"
  claim from here on must check raw values first.
- DISCARD = `git reset --hard HEAD~1`. CRASH = log row with NaNs +
  DISCARD.
- Committing locally only; user pulls from root.
