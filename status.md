# autoresearch — pickup-ready status — 2026-05-12 15:10 UTC

> **Read this first.** Designed to be enough to resume the loop without
> re-reading the conversation log.

## In flight right now

- **Experiment**: `exp63` — freeze `outcome_log_tau` in Phase-3 only
- **Commit**: `033e019` (HEAD)
- **Process**: `python api.py` launched ~14:10 UTC, expected to finish ~15:20 UTC
- **Output**: `/tmp/run_exp63.log` (`grep "^outcome_auroc\|^---" /tmp/run_exp63.log` for summary)
- **Wakeup scheduled**: 16:05 UTC
- **GPU**: 1× A5000 24 GB (peak per-run ~8.3 GB)

## Canonical baseline (current best — exp63 is being measured against this)

`audit_0.2a` (config has hazard removed; the data-shape change for terminals
is still in place). Effective state:

```
AUROC    0.828
AUPRC    0.401
MAE      81.00
max_len  14.9 %
RELEASE  0.698
```

Phase-2 scheduler currently has: `ce`, `dt`, `outcome`, `ranking`. `hazard`
removed (decorative per Rule 2(b) audit).

Phase-2 LM-head BCE uses a two-tier widening: terminals 168 h, others 12 h
(the complications-48 h tier was dropped per the f770850 caveat against
hand-picked per-family windows).

`outcome_log_tau` (K=6 learnable params from exp62b) is in the model and
trained during Phase-2. exp63's only change is freezing it in Phase-3.

## How to resume the loop if you disconnect mid-run

```bash
cd /workspace/autoresearch

# 1. See the most recent runs
ls -lt /tmp/run_*.log | head -3
grep -E "^outcome_auroc:|^outcome_auprc:|^onset_mae_hrs:|^---|max_len=500" /tmp/run_exp63.log | tail -15

# 2. Verify process state
ps aux | grep "python api" | grep -v grep
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# 3. Check current canonical commit
git log --oneline -5

# 4. If the run finished:
#    - Append a row to results.tsv following the format below
#    - KEEP if AUROC ≥ baseline-0.005 AND no Rule-6 mandatory aux regresses
#    - DISCARD if AUROC < baseline-0.005 OR Phase-3 NaN'd
#    - On DISCARD: git reset --hard HEAD~1 (results.tsv is NOT tracked so rows survive)

# 5. Smoke before any new full run
#    Set sample=50 + 1/1/1 epochs in config, run, check summary appears, revert config.
```

## audit / experiment chain in this session

| Commit | Name | Effect | Status |
|---|---|---|---|
| `c2f3856` | exp60 | terminals 168h + complications 48h | superseded |
| `c093e86` | exp61 | + P1 wide windows | DISCARD |
| `18a3caa` | audit_0.4 | revert BOTH tiers (12h uniform) | AUDIT (showed data-shape drove the gain) |
| `71ddbe9` | audit_0.4b | terminals-only 168h | KEEP |
| `083bfdb` | audit_0.2a | + hazard removed | KEEP (canonical baseline) |
| `27b8809` | audit_0.2b | + ranking removed | AUDIT (-0.044 ⇒ ranking is real) |
| `b0cabac` | audit_0.2c | + outcome removed | AUDIT (-0.009 ⇒ outcome real, kept) |
| `c56108c` | exp62 | learnable tau, P3 NaN'd | DISCARD (P3 crash; eval used P2 ckpt) |
| `aa267eb` | exp62b | NaN fix in get_future_outcome_targets | DISCARD (+0.003 within noise; RELEASE -0.024) |
| `033e019` | **exp63** | freeze outcome_log_tau in P3 | **in flight** |

## Rule 2(a) raw-drop audit (from audit_0.4b's fully-trained run)

Every aux passes the 30 % bar with margin. The earlier "weighted-display flat"
worry was a logging artefact. All are genuinely learning:

| Aux | Raw drop | Rule 2(b) cost | Decision |
|---|---|---|---|
| Δt (P1) | 71 % | — (mandatory) | LOCKED |
| ce | 84 % | — (mandatory) | LOCKED |
| dt (P2) | 77 % | — (mandatory) | LOCKED |
| outcome | 94 % | −0.009 AUROC | KEPT |
| ranking | strong | **−0.044 AUROC** | LOCKED |
| hazard | 97.6 % | +0.003 (decorative) | REMOVED |

## Key learnings from this session

1. **Data-shape wide BCE window for terminals is the dominant gain** in the
   whole project — +0.031 AUROC, +0.185 AUPRC, −73 pp max_len. Anything that
   tries to undo it (or per-family generalise it) needs strong evidence per
   the `f770850` caveat.
2. **Hazard learns its sub-task fine but doesn't translate to AUROC** —
   removed per Rule 2(b).
3. **Ranking is the highest-impact aux** (cost −0.044 if removed). Don't
   touch.
4. **Learnable per-outcome tau (exp62/62b/63) is the live direction.** P2
   alone gave RELEASE 0.813 (vs baseline 0.698) when the eval accidentally
   skipped P3. Whether exp63 (freeze tau in P3) recovers that is being
   measured right now.
5. **6-decimal raw aux loss logging is committed** (`39c3896`) — any future
   "flat loss" claim must be verified against the raw log first.
6. **results.tsv is now untracked from git** (since `8360804`). Prior
   `git reset --hard` for DISCARDs no longer destroys rows.

## What to try next if exp63 KEEPs

- Investigate the trained `outcome_log_tau` values: what tau did each outcome
  converge to? Surprising values are research signal. Add a one-line print at
  end of Phase-2 (`model.outcome_log_tau.exp() * 336` in hours).
- The learned-window concept could be applied at the LM-head BCE level too,
  but that touches the locked-in terminals-vs-everything split — careful.

## What to try next if exp63 DISCARDs

- Read `outcome_log_tau.requires_grad_(False)` in `finetune_transformer` —
  maybe also freeze the outcome_head's first hidden layer? P3 may be
  over-fitting independent of tau. Try Phase-3 with smaller `phase3_learning_rate`.
- Or accept that P2-only with learnable tau is the win and adjust the
  pipeline so P3 saves the P2-with-tau checkpoint when P3 doesn't improve
  val_loss. That's an `api.py` change but `api.py` is locked — would have to
  modify the P3 fall-through logic in `finetune_transformer`.
- Open-ended idea: contrastive aux at the patient-trajectory level (pull
  healthy together, push apart from complications). Targets RELEASE
  specifically.

## results.tsv pointer

`/workspace/autoresearch/results.tsv` — 70 rows, untracked, all session
experiments logged. Newest 3 rows are the exp62 / exp62b / exp63-pending
sequence. Use `tail -n 10 results.tsv | awk -F'\t' '{print $1, $5, $6}'` for
a quick scan.

## File ownership and what NOT to modify

- `api.py`, `evaluation.py`, `emr_model/data/` — FROZEN. Do not edit.
- Everything under `emr_model/transform_emr/*.py` and
  `emr_model/transform_emr/config/model_config.py` — fair game.
- `program.md` — the user updates this; treat it as instruction. Re-read
  before every experiment.

## Process discipline reminders

- Re-read `program.md` (especially "Rules of the road" and "Research
  directions") before launching a new experiment.
- Smoke test (sample=50, 1 ep per phase) BEFORE every full run.
- 6-decimal raw aux logging line should appear in every Phase-2 epoch log:
  `--> RawTrain ce=… dt=… outcome=… hazard=… ranking=…`
- Append to `results.tsv` immediately after every full run.
- Rule 5: a +0.003 AUROC on single seed is **not** a win — re-run with fresh
  P1 (delete `checkpoints/phase1/`) before logging KEEP if below ±0.005.
