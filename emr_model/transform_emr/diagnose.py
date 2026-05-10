"""Standalone diagnostics for model research.

Run:
    python -m transform_emr.diagnose
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from transform_emr.config.dataset_config import (
    OUTCOMES,
    TAK_REPO_PATH,
    TERMINAL_OUTCOMES,
    TRAIN_CTX_DATA_FILE,
    TRAIN_TEMPORAL_DATA_FILE,
)
from transform_emr.config.model_config import (  
    CHECKPOINT_PATH,
    PHASE1_CHECKPOINT,
    PHASE2_CHECKPOINT,
    PHASE3_CHECKPOINT,
    TRAINING_SETTINGS,
)
from transform_emr.dataset import DataProcessor, EMRDataset, EMRTokenizer, collate_emr, get_dataloader  
from transform_emr.embedder import EMREmbedding  
from transform_emr.loss import MaskedFocalBCE  
from transform_emr.transformer import GPT  
from transform_emr.utils import (  
    apply_masks_to_logits,
    build_luts,
    compute_legality_masks_tf,
    masked_softmax,
)


def _get_patient_id_col(df: pd.DataFrame) -> str:
    if "PatientID" in df.columns:
        return "PatientID"
    if "PatientId" in df.columns:
        return "PatientId"
    raise KeyError("No patient id column found. Expected 'PatientID' or 'PatientId'.")


def _get_outcome_token_ids(tokenizer: EMRTokenizer) -> List[Tuple[str, int]]:
    return [(name, tokenizer.token2id[name]) for name in OUTCOMES if name in tokenizer.token2id]


def _load_validation_data(sample: int, batch_size: int):
    temporal_path = Path(TRAIN_TEMPORAL_DATA_FILE)
    ctx_path = Path(TRAIN_CTX_DATA_FILE)
    if not temporal_path.exists() or not ctx_path.exists():
        project_root = Path(CHECKPOINT_PATH).parent
        source_temporal = project_root / "data" / "source" / "temporal_data.csv"
        source_ctx = project_root / "data" / "source" / "context_data.csv"
        temporal_path = source_temporal if source_temporal.exists() else temporal_path
        ctx_path = source_ctx if source_ctx.exists() else ctx_path

    temporal_df = pd.read_csv(temporal_path, low_memory=False)
    ctx_df = pd.read_csv(ctx_path)

    pid_col_temporal = _get_patient_id_col(temporal_df)
    pid_col_ctx = _get_patient_id_col(ctx_df)

    if sample:
        pids = temporal_df[pid_col_temporal].dropna().unique()
        sample_n = min(int(sample), len(pids))
        chosen = np.random.RandomState(42).choice(pids, size=sample_n, replace=False)
        temporal_df = temporal_df[temporal_df[pid_col_temporal].isin(chosen)].copy()
        ctx_df = ctx_df[ctx_df[pid_col_ctx].isin(chosen)].copy()

    tokenizer_path = Path(CHECKPOINT_PATH) / "tokenizer.pt"
    if tokenizer_path.exists():
        tokenizer = EMRTokenizer.load(str(tokenizer_path))
        processor = DataProcessor(temporal_df, ctx_df, scaler=None, tak_repo_path=TAK_REPO_PATH)
        temporal_df, ctx_df = processor.run()
    else:
        processor = DataProcessor(temporal_df, ctx_df, scaler=None, tak_repo_path=TAK_REPO_PATH)
        temporal_df, ctx_df = processor.run()
        tokenizer = EMRTokenizer.from_processed_df(temporal_df)
        tokenizer.save(str(tokenizer_path))

    pids = temporal_df[pid_col_temporal].dropna().unique()
    train_ids, val_ids = train_test_split(pids, test_size=0.2, random_state=42)

    val_df = temporal_df[temporal_df[pid_col_temporal].isin(val_ids)].copy()
    if pid_col_ctx in ctx_df.columns:
        val_ctx = ctx_df[ctx_df[pid_col_ctx].isin(val_ids)].copy()
    else:
        val_ctx = ctx_df.loc[ctx_df.index.isin(val_ids)].copy()

    val_ds = EMRDataset(val_df, val_ctx, tokenizer=tokenizer)
    val_dl = get_dataloader(val_ds, batch_size=batch_size, collate_fn=collate_emr, oversample=False)
    return val_dl, tokenizer


def _entropy(counter: Counter) -> float:
    n = sum(counter.values())
    if n == 0:
        return 0.0
    return -sum((c / n) * math.log(max(c / n, 1e-12)) for c in counter.values())


def _vocab_health_report(model, val_dl, tokenizer, pad_idx, device, max_batches: int = 3, k_window: int = 12) -> None:
    print("\n" + "=" * 90)
    print("REPORT 8 - VOCAB HEALTH (frequent-noisy / rare-unlearned)")
    print("=" * 90)

    V = len(tokenizer.token2id)
    occ = torch.zeros(V, dtype=torch.long, device=device)
    conf_sum = torch.zeros(V, dtype=torch.float32, device=device)
    top1_hit = torch.zeros(V, dtype=torch.long, device=device)
    next_counts = [Counter() for _ in range(V)]

    counts_obj = getattr(tokenizer, "token_counts", None)
    if torch.is_tensor(counts_obj):
        freq = counts_obj.detach().cpu().numpy().astype(int)
    else:
        freq = np.zeros(V, dtype=int)

    luts = build_luts(tokenizer)
    for key, value in list(luts.items()):
        if torch.is_tensor(value):
            luts[key] = value.to(device)

    with torch.no_grad():
        for b_idx, batch in enumerate(val_dl):
            if b_idx >= max_batches:
                break
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            logits, _, _, _ = model(
                parent_raw_ids=batch["parent_raw_ids"],
                concept_ids=batch["concept_ids"],
                value_ids=batch["value_ids"],
                position_ids=batch["position_ids"],
                abs_ts=batch["abs_ts"],
                context_vec=batch["context_vec"],
            )

            pred_logits = logits[:, :-1, :]
            target_ids = batch["targets"][:, 1:]
            nonpad = target_ids != pad_idx

            illegal = compute_legality_masks_tf(
                target_ids,
                luts["is_start"],
                luts["is_end"],
                luts["base_id"],
                luts["start_ids_per_base"],
                luts["end_ids_per_base"],
                luts["meal_rank"],
                luts["meal_pred_rank"],
                luts["K_meals"],
                luts["conflict_mat"],
                luts["predict_block"],
            )
            pred_logits = apply_masks_to_logits(pred_logits, illegal)
            allowed = (~illegal) & nonpad.unsqueeze(-1)
            probs = masked_softmax(pred_logits, allowed)

            gt = target_ids.unsqueeze(-1)
            p_gt = probs.gather(-1, gt).squeeze(-1)
            top1 = pred_logits.argmax(-1)
            is_top1 = (top1 == target_ids) & nonpad

            for b in range(target_ids.size(0)):
                valid_t = nonpad[b].nonzero(as_tuple=False).squeeze(-1)
                if valid_t.numel() == 0:
                    continue
                ids = target_ids[b, valid_t]
                occ.index_add_(0, ids, torch.ones_like(ids, dtype=occ.dtype))
                conf_sum.index_add_(0, ids, p_gt[b, valid_t])
                top1_hit.index_add_(0, ids, is_top1[b, valid_t].to(torch.long))

                for t in valid_t.tolist():
                    v = int(target_ids[b, t].item())
                    for u in target_ids[b, t + 1 : t + 1 + k_window].tolist():
                        if u == pad_idx:
                            break
                        next_counts[v][int(u)] += 1

    occ_np = occ.cpu().numpy()
    conf_np = (conf_sum / occ.clamp(min=1)).cpu().numpy()
    top1_np = (top1_hit / occ.clamp(min=1)).cpu().numpy()

    rows = []
    for vid in range(V):
        support = max(len(next_counts[vid]), 1)
        h = _entropy(next_counts[vid])
        h_norm = h / math.log(support) if support > 1 else 0.0
        rows.append(
            {
                "vid": vid,
                "token": tokenizer.id2token[vid],
                "freq": int(freq[vid]) if vid < len(freq) else 0,
                "occ": int(occ_np[vid]),
                "avg_conf": float(conf_np[vid]),
                "top1": float(top1_np[vid]),
                "next_entropy": float(h_norm),
            }
        )

    frequent_noisy = [
        r
        for r in sorted(rows, key=lambda x: (-x["freq"], -x["occ"]))
        if r["freq"] > 500 and r["avg_conf"] < 0.15 and r["next_entropy"] > 0.70
    ][:20]

    rare_unlearned = [
        r for r in sorted(rows, key=lambda x: (x["freq"], x["avg_conf"])) if r["freq"] < 50 and r["avg_conf"] < 0.08
    ][:20]

    print("\n[Frequent but noisy]")
    for r in frequent_noisy:
        print(
            f"{r['token']:<45} occ={r['occ']:<6} conf={r['avg_conf']:.3f} "
            f"top1={r['top1']:.3f} nextH={r['next_entropy']:.2f} freq={r['freq']}"
        )

    print("\n[Rare and under-learned]")
    for r in rare_unlearned:
        print(
            f"{r['token']:<45} occ={r['occ']:<6} conf={r['avg_conf']:.3f} "
            f"top1={r['top1']:.3f} nextH={r['next_entropy']:.2f} freq={r['freq']}"
        )


def run_diagnostics(sample: int = 2000, batch_size: int = 32) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    training_settings = dict(TRAINING_SETTINGS)

    print("[Diag] Loading validation data...")
    val_dl, tokenizer = _load_validation_data(sample=sample, batch_size=batch_size)

    print("[Diag] Loading checkpoints...")
    missing = []
    if not Path(PHASE1_CHECKPOINT).exists():
        missing.append(PHASE1_CHECKPOINT)
    if not Path(PHASE2_CHECKPOINT).exists():
        missing.append(PHASE2_CHECKPOINT)
    if missing:
        print("[Diag] Missing checkpoint(s). Run training first to generate them:")
        for ckpt in missing:
            print(f"  - {ckpt}")
        print("[Diag] Aborting diagnostics.")
        return

    embedder, *_ = EMREmbedding.load(PHASE1_CHECKPOINT, tokenizer=tokenizer)
    # Prefer Phase-3 (outcome head fine-tuned) — it matches the model evaluated in evaluation.py.
    # Fall back to Phase-2 if Phase-3 has not been trained yet.
    if Path(PHASE3_CHECKPOINT).exists():
        model, *_ = GPT.load(PHASE3_CHECKPOINT, embedder=embedder)
        print("[Diag] Using Phase-3 checkpoint (outcome head fine-tuned — matches evaluation.py).")
    else:
        model, *_ = GPT.load(PHASE2_CHECKPOINT, embedder=embedder)
        print("[Diag] Phase-3 not found — falling back to Phase-2 checkpoint.")
    model = model.to(device).eval()

    outcome_list = _get_outcome_token_ids(tokenizer)
    pad_idx = model.embedder.padding_idx

    scores_by_outcome = {tid: [] for _, tid in outcome_list}
    labels_by_outcome = {tid: [] for _, tid in outcome_list}
    head_scores_by_outcome = {tid: [] for _, tid in outcome_list}

    bce_win_h = float(training_settings.get("phase2_bce_window_hours", 12.0))
    eval_win_h = float(training_settings.get("outcome_window_hi_hours", 48.0))
    scale = 336.0
    bce_norm = bce_win_h / scale
    eval_norm = eval_win_h / scale

    bce_pos_counts = []
    eval_pos_counts = []
    probe_x, probe_y = [], []
    ctx_bce_normal = 0.0
    ctx_bce_zero = 0.0
    ctx_bce_shuffle = 0.0
    ctx_batches = 0

    with torch.no_grad():
        for n_batches, batch in enumerate(val_dl):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            logits, _, outcome_pred, _ = model(
                parent_raw_ids=batch["parent_raw_ids"],
                concept_ids=batch["concept_ids"],
                value_ids=batch["value_ids"],
                position_ids=batch["position_ids"],
                abs_ts=batch["abs_ts"],
                context_vec=batch["context_vec"],
            )

            pred_logits = logits[:, :-1, :].float()
            target_ids = batch["targets"][:, 1:]
            nonpad = target_ids != pad_idx

            cur_ts = batch["abs_ts"][:, :-1]
            fut_ts = batch["abs_ts"][:, 1:]
            dt_mat = fut_ts.unsqueeze(1) - cur_ts.unsqueeze(2)
            time_in_window = (dt_mat > 0) & (dt_mat <= eval_norm)

            full_tgt = batch["targets"]
            all_ts = batch["abs_ts"]
            dt_full = all_ts.unsqueeze(1) - cur_ts.unsqueeze(2)
            bce_in = (dt_full > 0) & (dt_full <= bce_norm)
            eval_in = (dt_full > 0) & (dt_full <= eval_norm)
            non_pad_mask = (full_tgt != pad_idx).unsqueeze(1).expand_as(bce_in)
            bce_pos_counts.append((bce_in & non_pad_mask).sum(dim=2)[nonpad].cpu().float())
            eval_pos_counts.append((eval_in & non_pad_mask).sum(dim=2)[nonpad].cpu().float())

            for i, (name, tid) in enumerate(outcome_list):
                fut_is_o = target_ids == tid
                label = (time_in_window & fut_is_o.unsqueeze(1)).any(dim=2)
                scores_by_outcome[tid].append(pred_logits[:, :, tid][nonpad].cpu().numpy())
                labels_by_outcome[tid].append(label[nonpad].cpu().numpy())
                if outcome_pred is not None and outcome_pred.dim() == 3 and outcome_pred.shape[-1] > i:
                    head_scores_by_outcome[tid].append(outcome_pred[:, :-1, i][nonpad].cpu().numpy())

            if n_batches < 2:
                emb_out, _ = model.embedder(
                    parent_raw_ids=batch["parent_raw_ids"],
                    concept_ids=batch["concept_ids"],
                    value_ids=batch["value_ids"],
                    position_ids=batch["position_ids"],
                    abs_ts=batch["abs_ts"],
                    patient_contexts=batch["context_vec"],
                    return_mask=False,
                )
                emb_flat = emb_out[:, :-1, :][nonpad].cpu().numpy()
                any_outcome = torch.zeros_like(nonpad, dtype=torch.bool)
                for _, tid in outcome_list:
                    fut_is_o = target_ids == tid
                    any_outcome |= (time_in_window & fut_is_o.unsqueeze(1)).any(dim=2)
                probe_x.append(emb_flat)
                probe_y.append(any_outcome[nonpad].cpu().numpy())

            if ctx_batches < 3:
                def _fwd_bce(ctx):
                    lg, _, _, _ = model(
                        parent_raw_ids=batch["parent_raw_ids"],
                        concept_ids=batch["concept_ids"],
                        value_ids=batch["value_ids"],
                        position_ids=batch["position_ids"],
                        abs_ts=batch["abs_ts"],
                        context_vec=ctx,
                    )
                    pl = lg[:, :-1, :].float()
                    mh = torch.zeros_like(pl)
                    for _, tid in outcome_list:
                        mh[:, :, tid] = (time_in_window & (target_ids == tid).unsqueeze(1)).any(dim=2).float()
                    valid = nonpad.unsqueeze(-1).float()
                    return (F.binary_cross_entropy_with_logits(pl, mh, reduction="none") * valid).sum() / valid.sum().clamp(min=1.0)

                ctx_vec = batch["context_vec"]
                ctx_bce_normal += _fwd_bce(ctx_vec).item()
                ctx_bce_zero += _fwd_bce(torch.zeros_like(ctx_vec)).item()
                idx = torch.randperm(ctx_vec.size(0), device=device)
                ctx_bce_shuffle += _fwd_bce(ctx_vec[idx]).item()
                ctx_batches += 1

    print("\n" + "=" * 90)
    print("REPORT 1 - PER-OUTCOME AUROC BREAKDOWN (48h eval window, teacher-forced LM logits)")
    print("=" * 90)
    print("  NOTE: Uses teacher-forced LM logits — scores are systematically higher than")
    print("  evaluation.py's generation-based AUROC. Use for within-run ranking and trend")
    print("  analysis, not as a direct comparison to the summary outcome_auroc.")
    print(f"{'Outcome':<38} {'AUROC':>6}  {'PosRate%':>8}  {'nPos':>6}  {'Sep':>7}")
    print("-" * 75)

    aurocs = []
    for name, tid in outcome_list:
        sc = np.concatenate(scores_by_outcome[tid]) if scores_by_outcome[tid] else np.array([])
        lb = np.concatenate(labels_by_outcome[tid]).astype(bool) if labels_by_outcome[tid] else np.array([], dtype=bool)
        if lb.size == 0:
            print(f"{name:<38} {'NO POS':>6}  {0.0:>7.3f}%")
            continue
        n_pos = int(lb.sum())
        pos_rate = 100.0 * n_pos / max(len(lb), 1)
        if n_pos == 0:
            print(f"{name:<38} {'NO POS':>6}  {pos_rate:>7.3f}%")
            continue
        auc = roc_auc_score(lb, sc)
        sep = float(sc[lb].mean() - sc[~lb].mean())
        aurocs.append(auc)
        flag = " <<<" if auc < 0.55 else (" >>>" if auc > 0.75 else "")
        print(f"{name:<38} {auc:>6.4f}  {pos_rate:>7.3f}%  {n_pos:>6}  {sep:>7.4f}{flag}")

    print("-" * 75)
    if aurocs:
        print(f"{'MEAN OUTCOME AUROC':<38} {float(np.mean(aurocs)):>6.4f}")

    head_valid = any(len(v) > 0 for v in head_scores_by_outcome.values())
    if head_valid:
        print("\nLM head vs Outcome head")
        print(f"{'Outcome':<38} {'LM':>6}  {'Head':>6}  {'Winner'}")
        print("-" * 60)
        for name, tid in outcome_list:
            lb = np.concatenate(labels_by_outcome[tid]).astype(bool)
            if lb.sum() == 0 or len(head_scores_by_outcome[tid]) == 0:
                continue
            lm_auc = roc_auc_score(lb, np.concatenate(scores_by_outcome[tid]))
            head_auc = roc_auc_score(lb, np.concatenate(head_scores_by_outcome[tid]))
            winner = "HEAD <<<" if head_auc > lm_auc + 0.02 else ("lm >>>" if lm_auc > head_auc + 0.02 else "~same")
            print(f"{name:<38} {lm_auc:>6.4f}  {head_auc:>6.4f}  {winner}")

    print("\n" + "=" * 90)
    print("REPORT 2 - LOGIT CALIBRATION (all outcomes combined)")
    print("=" * 90)
    all_sc = np.concatenate([np.concatenate(scores_by_outcome[tid]) for _, tid in outcome_list if scores_by_outcome[tid]])
    all_lb = np.concatenate([np.concatenate(labels_by_outcome[tid]) for _, tid in outcome_list if labels_by_outcome[tid]]).astype(bool)
    if all_lb.size > 0 and all_lb.any() and (~all_lb).any():
        print(f"Total positions : {len(all_lb):,}")
        print(f"Positive rate   : {all_lb.mean() * 100:.3f}%")
        print(f"Overall AUROC   : {roc_auc_score(all_lb, all_sc):.4f}")
        print(f"Logit[pos] mean : {all_sc[all_lb].mean():.4f}  std: {all_sc[all_lb].std():.4f}")
        print(f"Logit[neg] mean : {all_sc[~all_lb].mean():.4f}  std: {all_sc[~all_lb].std():.4f}")
        sep = float(all_sc[all_lb].mean() - all_sc[~all_lb].mean())
        sig_pos = 1.0 / (1.0 + np.exp(-all_sc[all_lb].mean()))
        sig_neg = 1.0 / (1.0 + np.exp(-all_sc[~all_lb].mean()))
        print(f"Separation      : {sep:.4f}")
        print(f"Sigmoid[pos]    : {sig_pos:.4f}")
        print(f"Sigmoid[neg]    : {sig_neg:.4f}")

    print("\n" + "=" * 90)
    print(f"REPORT 3 - TEMPORAL COVERAGE  (BCE={bce_win_h:.0f}h  EVAL={eval_win_h:.0f}h)")
    print("=" * 90)
    if bce_pos_counts and eval_pos_counts:
        bce_all = torch.cat(bce_pos_counts).numpy()
        eval_all = torch.cat(eval_pos_counts).numpy()
        bce_pct = float((bce_all > 0).mean() * 100)
        eval_pct = float((eval_all > 0).mean() * 100)
        bce_mean = float(bce_all.mean())
        eval_mean = float(eval_all.mean())
        print(f"BCE  window ({bce_win_h:.0f}h): {bce_pct:5.1f}% with >=1 positive (mean {bce_mean:.2f})")
        print(f"Eval window ({eval_win_h:.0f}h): {eval_pct:5.1f}% with >=1 positive (mean {eval_mean:.2f})")

    print("\n" + "=" * 90)
    print("REPORT 4 - AUX LOSS LAMBDA CALIBRATION (actual trained values from checkpoint)")
    print("=" * 90)
    sched = training_settings.get("phase2_scheduler", {})
    caps = sched.get("aux_fraction_caps", {})

    # lambda_schedule_state is stored in the Phase-2 ckpt_last (most recent training state).
    # lambda_max = cap × (anchor_main_loss / anchor_aux_loss), computed once at calibration epoch.
    _lambda_state = None
    for _ckpt_candidate in [
        str(Path(PHASE2_CHECKPOINT).parent / "ckpt_last.pt"),
        PHASE2_CHECKPOINT,
    ]:
        if Path(_ckpt_candidate).exists():
            try:
                _raw = torch.load(_ckpt_candidate, map_location="cpu", weights_only=True)
                _lambda_state = _raw.get("lambda_schedule_state")
                if _lambda_state:
                    break
            except Exception as _e:
                print(f"  [Could not read lambda state from {_ckpt_candidate}: {_e}]")

    if _lambda_state and "auxiliaries" in _lambda_state:
        print(f"  {'aux':<12}  {'lambda_max':>12}  {'anchor_bce':>12}  {'anchor_aux':>12}  {'bce/aux':>8}  {'cap':>6}")
        print("  " + "-" * 72)
        for name, spec in _lambda_state["auxiliaries"].items():
            lm  = spec.get("lambda_max")
            am  = spec.get("anchor_main_loss")
            aa  = spec.get("anchor_aux_loss")
            cap = caps.get(name, "?")
            ratio_str = f"{am / aa:.4f}" if (am and aa and aa > 1e-9) else "N/A"
            lm_str = f"{lm:.6f}" if lm is not None else "pending"
            am_str = f"{am:.6f}" if am is not None else "N/A"
            aa_str = f"{aa:.6f}" if aa is not None else "N/A"
            print(f"  {name:<12}  {lm_str:>12}  {am_str:>12}  {aa_str:>12}  {ratio_str:>8}  {str(cap):>6}")
            if lm is not None and lm < 1e-3:
                print(f"  *** WARNING: {name} lambda_max={lm:.6f} — near-silent (<0.001).")
                print(f"      Gradient from {name} loss is negligible. Increase its cap in phase2_scheduler.")
        warmup = _lambda_state.get("warmup_complete_epoch")
        if warmup is not None:
            print(f"\n  Warmup completed at epoch: {warmup}")
        stage = _lambda_state.get("current_stage")
        if stage is not None:
            print(f"  Current curriculum stage: {stage}")
    else:
        print("  No calibrated lambda state in checkpoint — training not yet run or checkpoint missing.")
        print("  Showing configured caps from model_config.py (formula: lambda_max = cap × bce / aux):")
        for key, value in caps.items():
            print(f"    {key:<12}: cap={value}")
        if caps.get("outcome", 0) < 2.0:
            print("  WARNING: outcome cap looks low for temporal BCE scale; consider 10-20.")
        if caps.get("ce", 0) < 0.5:
            print("  WARNING: CE cap looks low for ranking signal; consider >= 1.")

    print("\n" + "=" * 90)
    print("REPORT 5 - TOKEN GRADIENT UTILITY (outcome + top/bottom tokens)")
    print("=" * 90)

    model.train()
    luts = build_luts(tokenizer)
    for key, value in list(luts.items()):
        if torch.is_tensor(value):
            luts[key] = value.to(device)

    crit = MaskedFocalBCE.from_counts(
        counts=tokenizer.token_counts,
        token_weights=getattr(tokenizer, "token_weights", None),
        beta=0.999,
        min_count=5,
        clip_max=8.0,
        gamma=1.3,
        tau=0.85,
        neg_bounds=(0.03, 0.30),
    ).to(device)

    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    V = len(tokenizer.token2id)
    grad_sq = torch.zeros(V, device=device)
    occ = torch.zeros(V, device=device)

    for i, batch in enumerate(val_dl):
        if i >= 3:
            break
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        logits, _, _, _ = model(
            parent_raw_ids=batch["parent_raw_ids"],
            concept_ids=batch["concept_ids"],
            value_ids=batch["value_ids"],
            position_ids=batch["position_ids"],
            abs_ts=batch["abs_ts"],
            context_vec=batch["context_vec"],
        )
        pred_logits = logits[:, :-1, :]
        target_ids = batch["targets"][:, 1:]
        illegal = compute_legality_masks_tf(
            target_ids,
            luts["is_start"],
            luts["is_end"],
            luts["base_id"],
            luts["start_ids_per_base"],
            luts["end_ids_per_base"],
            luts["meal_rank"],
            luts["meal_pred_rank"],
            luts["K_meals"],
            luts["conflict_mat"],
            luts["predict_block"],
        )
        pred_logits = apply_masks_to_logits(pred_logits, illegal)
        nonpad = target_ids != pad_idx
        allowed = (~illegal) & nonpad.unsqueeze(-1)

        B, T = target_ids.shape
        all_ts = batch["abs_ts"]
        cur_ts = all_ts[:, :-1]
        dt = all_ts.unsqueeze(1) - cur_ts.unsqueeze(2)
        in_win = (dt > 0) & (dt <= bce_norm)
        tgt_exp = batch["targets"].unsqueeze(1).expand(B, T, all_ts.size(1)).masked_fill(~in_win, pad_idx)
        multi_hot = torch.zeros(B, T, V, device=device)
        multi_hot.scatter_(2, tgt_exp, 1.0)
        multi_hot[..., pad_idx] = 0.0
        multi_hot = multi_hot.masked_fill(illegal, 0.0)

        loss, _ = crit(pred_logits, multi_hot, allowed)
        loss.backward()

        W = model.embedder.position_embed.weight
        if W.grad is not None:
            grad_sq += (W.grad.detach() ** 2).sum(dim=1)
        ids = target_ids[nonpad]
        if ids.numel() > 0:
            occ.index_add_(0, ids, torch.ones_like(ids, dtype=occ.dtype))
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

    model.eval()
    score = (grad_sq / occ.clamp(min=1)).cpu().numpy()
    sorted_ids = np.argsort(-score)
    rank_map = {int(vid): r + 1 for r, vid in enumerate(sorted_ids)}

    print(f"{'Token':<40} {'occ':>6}  {'grad/occ':>10}  {'rank':>6}")
    print("-" * 65)
    for name, tid in outcome_list:
        r = rank_map.get(tid, -1)
        o = int(occ[tid].item())
        g = float(score[tid])
        flag = " << LOW SIGNAL" if r > V // 2 else ""
        print(f"{name:<40} {o:>6}  {g:>10.4e}  {r:>6}{flag}")

    print("\nTop 10 tokens by grad/occ:")
    for vid in sorted_ids[:10]:
        print(f"  {tokenizer.id2token[int(vid)]:<45} {float(score[int(vid)]):.4e}")
    print("Bottom 10 tokens by grad/occ:")
    for vid in sorted_ids[-10:]:
        print(f"  {tokenizer.id2token[int(vid)]:<45} {float(score[int(vid)]):.4e}")

    print("\n" + "=" * 90)
    print("REPORT 6 - CONTEXT VECTOR INFLUENCE")
    print("=" * 90)
    if ctx_batches > 0:
        n = ctx_bce_normal / ctx_batches
        z = ctx_bce_zero / ctx_batches
        s = ctx_bce_shuffle / ctx_batches
        print(f"BCE (normal ctx)  : {n:.6f}")
        print(f"BCE (zeroed ctx)  : {z:.6f}   delta={z - n:+.6f}")
        print(f"BCE (shuffled ctx): {s:.6f}   delta={s - n:+.6f}")

    print("\n" + "=" * 90)
    print("REPORT 7 - EMBEDDER LINEAR PROBE (frozen Phase-1)")
    print("=" * 90)
    if probe_x:
        x = np.concatenate(probe_x)
        y = np.concatenate(probe_y)
        pos_frac = float(y.mean())
        if 0.0 < pos_frac < 1.0:
            if len(x) > 50000:
                idx = np.random.RandomState(42).choice(len(x), 50000, replace=False)
                x, y = x[idx], y[idx]
            clf = LogisticRegression(max_iter=200, C=0.1, solver="lbfgs")
            cv_auc = cross_val_score(clf, x, y, cv=3, scoring="roc_auc", n_jobs=1).mean()
            print(f"3-fold CV ROC-AUC from embedder alone: {cv_auc:.4f}")
        else:
            print(f"Skipped: degenerate labels (pos_frac={pos_frac:.3f})")

    _vocab_health_report(model, val_dl, tokenizer, pad_idx, device=device, max_batches=3, k_window=12)

    probe_dt_head(model, val_dl, n_batches=2)
    probe_outcome_label_alignment(model, val_dl, tokenizer, n_batches=2)
    probe_outcome_logit_distribution(model, val_dl, n_batches=2)
    probe_outcome_lm_coupling(model, val_dl, n_batches=2)

    print("\n[Diag] Done.")


def diagnose_generation_time(df: pd.DataFrame, max_duration_hours: float = 336.0) -> dict:
    """
    Purpose: Audit the temporal behaviour of a generated event stream to detect
        Δt mean-collapse and verify the generator respected the training horizon.
    Method: Restrict to generated rows (IsInput == 0), compute per-patient Δt,
        and compare across-patient variance of mean-Δt to within-patient Δt
        variance. A small ratio means every patient ticks at the same rate
        regardless of context — i.e. the time head has collapsed to the dataset
        mean. Also reports the fraction of patients that ran past the horizon.

    Args:
        df (pd.DataFrame): Output of `transform_emr.inference.generate()`.
        max_duration_hours (float): Training horizon (default 336 h = 14 days).

    Returns:
        dict: Summary stats — global Δt percentiles, per-patient mean-Δt spread,
            horizon-overrun fraction, and a `mean_regression_warning` flag.
    """
    gen = df[df["IsInput"] == 0].copy()
    if len(gen) == 0:
        print("[diagnose-time] No generated rows — nothing to analyse.")
        return {}

    gen = gen.sort_values(["PatientId", "Step"])
    gen["dt"] = gen.groupby("PatientId")["TimePoint"].diff()
    dts = gen["dt"].dropna().to_numpy()

    per_pid_mean_dt = gen.groupby("PatientId")["dt"].mean().dropna().to_numpy()
    per_pid_max_t = gen.groupby("PatientId")["TimePoint"].max()

    within_std = float(np.std(dts)) if len(dts) else float("nan")
    across_std = float(np.std(per_pid_mean_dt)) if len(per_pid_mean_dt) else float("nan")
    ratio = (across_std / within_std) if within_std > 1e-6 else float("nan")

    stats = {
        "n_patients":                  int(gen["PatientId"].nunique()),
        "n_gen_rows":                  int(len(gen)),
        "dt_mean":                     float(np.mean(dts)) if len(dts) else float("nan"),
        "dt_std":                      within_std,
        "dt_p50":                      float(np.percentile(dts, 50)) if len(dts) else float("nan"),
        "dt_p90":                      float(np.percentile(dts, 90)) if len(dts) else float("nan"),
        "dt_p99":                      float(np.percentile(dts, 99)) if len(dts) else float("nan"),
        "per_patient_mean_dt_std":     across_std,
        "across_over_within_std_ratio": ratio,
        "frac_patients_past_horizon":  float((per_pid_max_t > max_duration_hours).mean()),
        "max_timepoint":               float(per_pid_max_t.max()),
        "mean_regression_warning":     bool(ratio < 0.1) if ratio == ratio else False,
    }

    print("\n" + "=" * 90)
    print("GENERATION TIME DIAGNOSTICS")
    print("=" * 90)
    print(f"patients={stats['n_patients']}  gen_rows={stats['n_gen_rows']}")
    print(
        f"  Δt (h):  mean={stats['dt_mean']:.2f}  std={stats['dt_std']:.2f}  "
        f"p50={stats['dt_p50']:.2f}  p90={stats['dt_p90']:.2f}  p99={stats['dt_p99']:.2f}"
    )
    print(
        f"  Per-patient mean-Δt: across-std={stats['per_patient_mean_dt_std']:.3f}  "
        f"(across/within ratio={stats['across_over_within_std_ratio']:.3f})"
    )
    print(
        f"  Max TimePoint observed: {stats['max_timepoint']:.1f} h  "
        f"(horizon={max_duration_hours}); "
        f"{100 * stats['frac_patients_past_horizon']:.1f}% of patients ran past horizon."
    )
    if stats["mean_regression_warning"]:
        print("  ⚠ Mean-regression suspected: per-patient mean Δt barely varies vs within-patient Δt.")
    return stats

def probe_dt_components(model, val_dl, n_batches: int = 1):
    """Inspect dt_gate sigmoid distribution and dt_magnitude separately."""
    device = next(model.parameters()).device
    model.eval()
    gates, mags = [], []
    with torch.no_grad():
        for i, batch in enumerate(val_dl):
            if i >= n_batches: break
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            _, _, _, gate_logit = model(
                parent_raw_ids=batch["parent_raw_ids"], concept_ids=batch["concept_ids"],
                value_ids=batch["value_ids"], position_ids=batch["position_ids"],
                abs_ts=batch["abs_ts"], context_vec=batch["context_vec"],
            )
            gp = torch.sigmoid(gate_logit).cpu().numpy().ravel()
            gates.append(gp)
    gp = np.concatenate(gates)
    print(f"gate_prob: mean={gp.mean():.3f} std={gp.std():.3f} "
          f"frac>0.99={(gp>0.99).mean():.3f} frac<0.01={(gp<0.01).mean():.3f}")
    
def _collect_val_batches(model, val_dl, n_batches: int = 1):
    """
    Purpose: Cache a small validation slice so probes share one forward pass.
    Method: Run the model in eval mode (no autocast), keep float32 outputs on CPU.

    Args:
        model: GPT, already on device, in eval mode.
        val_dl (DataLoader): Validation dataloader.
        n_batches (int): Batches to consume.

    Returns:
        dict: logits, outcome_logits, abs_t_pred, abs_ts, targets, pad_idx.
    """
    device = next(model.parameters()).device
    pad_idx = model.embedder.padding_idx
    model.eval()

    logits_list, oh_list, dt_pred_list, abs_ts_list, tgt_list = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(val_dl):
            if i >= n_batches:
                break
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            logits, abs_t_pred, outcome_logits, _ = model(
                parent_raw_ids=batch["parent_raw_ids"],
                concept_ids=batch["concept_ids"],
                value_ids=batch["value_ids"],
                position_ids=batch["position_ids"],
                abs_ts=batch["abs_ts"],
                context_vec=batch["context_vec"],
            )
            logits_list.append(logits.float().cpu())
            oh_list.append(outcome_logits.float().cpu())
            dt_pred_list.append(abs_t_pred.float().cpu())
            abs_ts_list.append(batch["abs_ts"].float().cpu())
            tgt_list.append(batch["targets"].cpu())

    # Different batches may have different T_max — pad each batch to a common
    # T before concat. Pad with `pad_idx` for targets and 0.0 elsewhere so the
    # `nonpad = (targets != pad_idx)` mask filters them out in every probe.
    T_max = max(x.size(1) for x in tgt_list)

    def _pad_T(t, fill):
        if t.size(1) == T_max:
            return t
        pad_shape = list(t.shape)
        pad_shape[1] = T_max - t.size(1)
        pad = torch.full(pad_shape, fill, dtype=t.dtype)
        return torch.cat([t, pad], dim=1)

    return {
        "logits":         torch.cat([_pad_T(x, 0.0)     for x in logits_list],  dim=0),
        "outcome_logits": torch.cat([_pad_T(x, 0.0)     for x in oh_list],      dim=0),
        "abs_t_pred":     torch.cat([_pad_T(x, 0.0)     for x in dt_pred_list], dim=0),
        "abs_ts":         torch.cat([_pad_T(x, 0.0)     for x in abs_ts_list],  dim=0),
        "targets":        torch.cat([_pad_T(x, pad_idx) for x in tgt_list],     dim=0),
        "pad_idx":        pad_idx,
    }


def probe_dt_head(model, val_dl, n_batches: int = 1) -> dict:
    """
    Purpose: Detect whether the Δt head has collapsed to a constant or roughly
        tracks ground-truth deltas.
    Method: Compute predicted Δt = abs_t_pred[t] - abs_ts[t] vs true Δt =
        abs_ts[t+1] - abs_ts[t] over non-pad positions. Report Pearson r, R²,
        and per-patient std of predicted Δt.

    Args:
        model: GPT in eval mode.
        val_dl (DataLoader): Validation dataloader.
        n_batches (int): Batches to consume.

    Returns:
        dict: pearson_r, r2, pred_std_h, true_std_h, pred_mean_h, true_mean_h.
    """
    cache = _collect_val_batches(model, val_dl, n_batches=n_batches)
    abs_ts     = cache["abs_ts"]
    abs_t_pred = cache["abs_t_pred"]
    pad_idx    = cache["pad_idx"]
    tgt        = cache["targets"]

    true_dt = (abs_ts[:, 1:] - abs_ts[:, :-1])
    pred_dt = (abs_t_pred[:, :-1] - abs_ts[:, :-1])
    valid   = (tgt[:, 1:] != pad_idx) & (tgt[:, :-1] != pad_idx)

    p = pred_dt[valid].numpy() * 336.0
    t = true_dt[valid].numpy() * 336.0

    if len(p) < 5:
        print("[probe-dt] not enough valid positions")
        return {}

    pm, tm = float(p.mean()), float(t.mean())
    cov = float(((p - pm) * (t - tm)).mean())
    r   = cov / (p.std() * t.std() + 1e-12)
    r2  = 1.0 - ((p - t) ** 2).sum() / (((t - tm) ** 2).sum() + 1e-12)

    B = abs_ts.size(0)
    per_b_std = []
    for b in range(B):
        v = (tgt[b, 1:] != pad_idx) & (tgt[b, :-1] != pad_idx)
        if v.sum() >= 3:
            per_b_std.append(float((pred_dt[b, :v.size(0)][v].numpy() * 336.0).std()))

    print("\n" + "=" * 90)
    print("PROBE - Δt HEAD")
    print("=" * 90)
    print(f"  positions   : {len(p):,}")
    print(f"  pred Δt (h) : mean={pm:.3f}  std={p.std():.3f}")
    print(f"  true Δt (h) : mean={tm:.3f}  std={t.std():.3f}")
    print(f"  Pearson r   : {r:.4f}")
    print(f"  R²          : {r2:.4f}")
    if per_b_std:
        print(
            f"  per-patient pred Δt std (h): "
            f"mean={np.mean(per_b_std):.3f}  p10={np.percentile(per_b_std, 10):.3f}"
        )
    if abs(p.std()) < 0.05 or r < 0.1:
        print("  ⚠ Δt head looks collapsed (low std and/or near-zero correlation).")

    return {
        "pearson_r": float(r),
        "r2":        float(r2),
        "pred_std_h":  float(p.std()),
        "true_std_h":  float(t.std()),
        "pred_mean_h": pm,
        "true_mean_h": tm,
    }


def probe_terminal_logits(model, val_dl, tokenizer, n_batches: int = 1, top_k_show: int = 5) -> dict:
    """
    Purpose: Diagnose why generation never emits DEATH/RELEASE — inspect LM-head
        rank, probability, and logit gap of terminal tokens at non-pad positions.
    Method: Softmax over full vocab (no legality mask). Per terminal: median
        rank, p10 rank, mean prob, and (logit − mean non-terminal logit). Also
        print top-k logits at the last non-pad position per patient.

    Args:
        model: GPT in eval mode.
        val_dl (DataLoader): Validation dataloader.
        tokenizer (EMRTokenizer): For token2id lookup.
        n_batches (int): Batches to consume.
        top_k_show (int): Top-k tokens to print at end-of-sequence positions.

    Returns:
        dict: per-terminal {median_rank, p10_rank, mean_prob, mean_logit_gap}.
    """
    cache = _collect_val_batches(model, val_dl, n_batches=n_batches)
    logits  = cache["logits"]
    tgt     = cache["targets"]
    pad_idx = cache["pad_idx"]
    nonpad  = (tgt != pad_idx)

    term_ids = [tokenizer.token2id[t] for t in TERMINAL_OUTCOMES if t in tokenizer.token2id]
    if not term_ids:
        print("[probe-terminal] No terminal tokens in vocab.")
        return {}

    probs = torch.softmax(logits, dim=-1)
    sorted_idx = torch.argsort(logits, dim=-1, descending=True)
    V = logits.size(-1)
    ranks = torch.empty_like(sorted_idx)
    ranks.scatter_(2, sorted_idx, torch.arange(V).expand_as(sorted_idx))

    print("\n" + "=" * 90)
    print("PROBE - TERMINAL TOKEN LM-HEAD HEALTH")
    print("=" * 90)
    print(f"{'token':<20} {'median_rank':>12} {'p10_rank':>10} {'mean_prob':>12} {'logit_gap':>12}")

    out = {}
    mask_terms = torch.zeros(V, dtype=torch.bool)
    mask_terms[term_ids] = True
    nonterm_logit_mean = logits[..., ~mask_terms].mean(dim=-1)
    for tid in term_ids:
        tok_name = tokenizer.id2token[tid]
        gap = (logits[..., tid] - nonterm_logit_mean)[nonpad].numpy()
        rk  = ranks[..., tid][nonpad].float().numpy()
        pr  = probs[..., tid][nonpad].numpy()
        print(
            f"{tok_name:<20} {np.median(rk):>12.1f} {np.percentile(rk, 10):>10.1f} "
            f"{pr.mean():>12.6f} {gap.mean():>12.4f}"
        )
        out[tok_name] = {
            "median_rank":    float(np.median(rk)),
            "p10_rank":       float(np.percentile(rk, 10)),
            "mean_prob":      float(pr.mean()),
            "mean_logit_gap": float(gap.mean()),
        }

    last_idx = nonpad.sum(dim=1) - 1
    B = logits.size(0)
    eos_logits = logits[torch.arange(B), last_idx, :]
    print(f"\nEnd-of-sequence top-{top_k_show} predictions (first 3 patients):")
    for b in range(min(3, B)):
        top_v, top_i = torch.topk(eos_logits[b], top_k_show)
        toks = [tokenizer.id2token[int(i)] for i in top_i]
        print(f"  pid#{b}: " + ", ".join(f"{t}={v:.2f}" for t, v in zip(toks, top_v.tolist())))
    return out


def probe_outcome_label_alignment(model, val_dl, tokenizer, n_batches: int = 1,
                                  eval_window_h: float = 48.0) -> pd.DataFrame:
    """
    Purpose: Detect outcome-head sign/label flips. Compare head logits where the
        outcome occurs within eval_window_h vs where it does not.
    Method: Build positives = "outcome token appears in (now, now+eval_window_h]".
        Compute mean-logit gap (pos − neg) and AUROC. Flip = gap < 0 or AUROC < 0.5.

    Args:
        model: GPT in eval mode.
        val_dl (DataLoader): Validation dataloader.
        tokenizer (EMRTokenizer): For token2id lookup.
        n_batches (int): Batches to consume.
        eval_window_h (float): Future window in hours.

    Returns:
        pd.DataFrame: per-outcome rows with mean_logit_pos, mean_logit_neg, gap,
            n_pos, n_neg, auroc, flip flag.
    """
    cache = _collect_val_batches(model, val_dl, n_batches=n_batches)
    oh    = cache["outcome_logits"]
    abs_ts = cache["abs_ts"]
    tgt    = cache["targets"]
    pad    = cache["pad_idx"]

    eval_norm = float(eval_window_h) / 336.0
    cur = abs_ts.unsqueeze(2)
    fut = abs_ts.unsqueeze(1)
    dt  = fut - cur
    in_win = (dt > 0) & (dt <= eval_norm)
    nonpad_cur = (tgt != pad)

    rows = []
    for k, name in enumerate(model.outcome_names):
        if name not in tokenizer.token2id:
            continue
        tid = tokenizer.token2id[name]
        fut_is_o = (tgt == tid).unsqueeze(1)
        label = (in_win & fut_is_o).any(dim=2)

        head_logit = oh[..., k]
        pos = head_logit[label & nonpad_cur].numpy()
        neg = head_logit[(~label) & nonpad_cur].numpy()

        if len(pos) == 0 or len(neg) == 0:
            rows.append({
                "outcome": name, "n_pos": int(len(pos)), "n_neg": int(len(neg)),
                "mean_logit_pos": float("nan"), "mean_logit_neg": float("nan"),
                "gap": float("nan"), "auroc": float("nan"), "flip": False,
            })
            continue

        gap = float(pos.mean() - neg.mean())
        try:
            auc = float(roc_auc_score(
                np.r_[np.ones_like(pos), np.zeros_like(neg)],
                np.r_[pos, neg],
            ))
        except ValueError:
            auc = float("nan")
        rows.append({
            "outcome": name,
            "n_pos":   int(len(pos)),
            "n_neg":   int(len(neg)),
            "mean_logit_pos": float(pos.mean()),
            "mean_logit_neg": float(neg.mean()),
            "gap":   gap,
            "auroc": auc,
            "flip":  bool(gap < 0 or (auc == auc and auc < 0.5)),
        })

    df = pd.DataFrame(rows).sort_values("gap")
    print("\n" + "=" * 90)
    print(f"PROBE - OUTCOME HEAD LABEL ALIGNMENT (eval window={eval_window_h}h)")
    print("=" * 90)
    print(df.round(4).to_string(index=False))
    flipped = df[df["flip"]]["outcome"].tolist()
    if flipped:
        print(f"\n⚠ Possible label flips: {flipped}")
    return df


def probe_calibration_by_abs_time(model, val_dl, tokenizer, n_batches: int = 1,
                                  bins_h=(0, 24, 48, 96, 168, 336, 672)) -> pd.DataFrame:
    """
    Purpose: Detect Time2Vec OOD behaviour. AUROC drop at high absolute times
        (especially past 336h) means time encoding extrapolates poorly.
    Method: Bin positions by hours-from-admission, compute per-bin mean outcome
        logit and AUROC averaged across all outcomes (48h forward window).

    Args:
        model: GPT in eval mode.
        val_dl (DataLoader): Validation dataloader.
        tokenizer (EMRTokenizer): For outcome token ids.
        n_batches (int): Batches to consume.
        bins_h (tuple): Bin edges in hours.

    Returns:
        pd.DataFrame: per-bin {n_positions, mean_logit, mean_auroc}.
    """
    cache = _collect_val_batches(model, val_dl, n_batches=n_batches)
    oh    = cache["outcome_logits"]
    abs_ts = cache["abs_ts"]
    tgt    = cache["targets"]
    pad    = cache["pad_idx"]
    nonpad = (tgt != pad)

    eval_norm = 48.0 / 336.0
    cur = abs_ts.unsqueeze(2)
    fut = abs_ts.unsqueeze(1)
    dt  = fut - cur
    in_win = (dt > 0) & (dt <= eval_norm)

    abs_h = (abs_ts * 336.0).numpy()
    rows = []
    for lo, hi in zip(bins_h[:-1], bins_h[1:]):
        bin_mask = (abs_h >= lo) & (abs_h < hi) & nonpad.numpy()
        if bin_mask.sum() == 0:
            continue
        aurocs, mean_logits = [], []
        for k, name in enumerate(model.outcome_names):
            if name not in tokenizer.token2id:
                continue
            tid = tokenizer.token2id[name]
            label = (in_win & (tgt == tid).unsqueeze(1)).any(dim=2).numpy()
            sc = oh[..., k].numpy()
            lb_b = label[bin_mask].astype(bool)
            sc_b = sc[bin_mask]
            mean_logits.append(float(sc_b.mean()))
            if lb_b.sum() > 0 and (~lb_b).sum() > 0:
                try:
                    aurocs.append(float(roc_auc_score(lb_b, sc_b)))
                except ValueError:
                    pass
        rows.append({
            "bin": f"[{lo}, {hi})h",
            "n_positions": int(bin_mask.sum()),
            "mean_logit":  float(np.mean(mean_logits)) if mean_logits else float("nan"),
            "mean_auroc":  float(np.mean(aurocs))      if aurocs       else float("nan"),
        })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("PROBE - CALIBRATION BY ABSOLUTE TIME (mean over outcomes)")
    print("=" * 90)
    print(df.round(4).to_string(index=False))
    return df


def probe_legality_starvation(model, val_dl, tokenizer, n_batches: int = 1) -> dict:
    """
    Purpose: Confirm DEATH/RELEASE are not blocked by the legality mask at the
        positions where they are GT next token. A non-zero blocked fraction is
        a bug in the legality LUT.
    Method: For each non-pad position with terminal target, query
        compute_legality_masks_tf and check whether the terminal id is illegal.

    Args:
        model: GPT (uses its tokenizer/luts/device).
        val_dl (DataLoader): Validation dataloader.
        tokenizer (EMRTokenizer): For token ids.
        n_batches (int): Batches to consume.

    Returns:
        dict: per-terminal {n_target_positions, n_marked_illegal, frac_blocked}.
    """
    device = next(model.parameters()).device
    pad_idx = model.embedder.padding_idx
    luts = build_luts(tokenizer)
    for k, v in list(luts.items()):
        if torch.is_tensor(v):
            luts[k] = v.to(device)

    term_ids = [tokenizer.token2id[t] for t in TERMINAL_OUTCOMES if t in tokenizer.token2id]
    counts = {tokenizer.id2token[tid]: {"target": 0, "blocked": 0} for tid in term_ids}

    with torch.no_grad():
        for i, batch in enumerate(val_dl):
            if i >= n_batches:
                break
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            tgt = batch["targets"]
            target_next = tgt[:, 1:]
            illegal = compute_legality_masks_tf(
                target_next,
                luts["is_start"], luts["is_end"],
                luts["base_id"],
                luts["start_ids_per_base"], luts["end_ids_per_base"],
                luts["meal_rank"], luts["meal_pred_rank"], luts["K_meals"],
                luts["conflict_mat"], luts["predict_block"],
            )
            nonpad = (target_next != pad_idx)
            for tid in term_ids:
                tok_name = tokenizer.id2token[tid]
                is_target = (target_next == tid) & nonpad
                if is_target.sum() == 0:
                    continue
                blocked = illegal[..., tid] & is_target
                counts[tok_name]["target"]  += int(is_target.sum().item())
                counts[tok_name]["blocked"] += int(blocked.sum().item())

    print("\n" + "=" * 90)
    print("PROBE - LEGALITY MASK STARVATION (terminal targets blocked)")
    print("=" * 90)
    out = {}
    for name, c in counts.items():
        frac = (c["blocked"] / c["target"]) if c["target"] else float("nan")
        print(
            f"  {name:<20} target_positions={c['target']:>6}  blocked={c['blocked']:>6}  "
            f"frac_blocked={frac:.4f}"
        )
        out[name] = {
            "n_target_positions": c["target"],
            "n_marked_illegal":   c["blocked"],
            "frac_blocked":       float(frac) if frac == frac else float("nan"),
        }
    return out


def probe_outcome_lm_coupling(model, val_dl, n_batches: int = 1) -> dict:
    """
    Purpose: Verify that outcome_to_lm has activated and that it produces
        meaningful additive corrections to LM logits at outcome vocab positions.
    Method: Forward a val slice. For each outcome k: compute Pearson r between
        the outcome head logit (raw) and the lm_bias it produces via outcome_to_lm.
        lm_bias = outcome_to_lm(outcome_logits.detach()), computed directly from the
        collected outcome_logits without a second forward pass.
        Also report mean absolute bias and the weight norm of outcome_to_lm.

    Args:
        model: GPT in eval mode (must have outcome_to_lm and _outcome_lm_ids).
        val_dl (DataLoader): Validation dataloader.
        n_batches (int): Batches to consume.

    Returns:
        dict: weight_norm, per-outcome pearson_r and mean_bias.
    """
    if not hasattr(model, "outcome_to_lm") or not hasattr(model, "_outcome_lm_ids"):
        print("\n[probe_outcome_lm_coupling] outcome_to_lm not present in model — skipping.")
        return {}

    device = next(model.parameters()).device
    pad_idx = model.embedder.padding_idx
    model.eval()

    w_norm = model.outcome_to_lm.weight.detach().cpu().norm().item()

    cache = _collect_val_batches(model, val_dl, n_batches=n_batches)
    oh    = cache["outcome_logits"]   # [N, T, K]
    tgt   = cache["targets"]
    nonpad = (tgt != pad_idx)

    # Compute lm_bias from the collected outcome logits (no second forward needed)
    with torch.no_grad():
        lm_bias = model.outcome_to_lm(oh.to(device)).cpu()  # [N, T, K]

    oids = model._outcome_lm_ids.cpu()

    print("\n" + "=" * 90)
    print("PROBE - OUTCOME→LM COUPLING")
    print("=" * 90)
    print(f"  outcome_to_lm weight norm : {w_norm:.6f}")
    if w_norm < 1e-6:
        print("  ⚠ Weight norm near zero — coupling has not activated (expected if Phase-3 not run yet).")

    rows = []
    for k, name in enumerate(model.outcome_names):
        head_logit = oh[..., k][nonpad].numpy()
        lm_bias_k  = lm_bias[..., k][nonpad].numpy()
        if len(head_logit) < 10:
            continue
        hm, bm = head_logit.mean(), lm_bias_k.mean()
        cov = float(((head_logit - hm) * (lm_bias_k - bm)).mean())
        r = cov / (head_logit.std() * lm_bias_k.std() + 1e-12)
        rows.append({
            "outcome":     name,
            "head→bias_r": round(float(r), 4),
            "mean_bias":   round(float(lm_bias_k.mean()), 6),
            "bias_std":    round(float(lm_bias_k.std()), 6),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        print(df.to_string(index=False))
    return {"weight_norm": w_norm, "details": rows}


def probe_outcome_logit_distribution(model, val_dl, n_batches: int = 1) -> pd.DataFrame:
    """
    Purpose: Inspect per-outcome head logit spread to corroborate extreme
        calibration temperatures (Hyperglycemia T=74, RELEASE T=86 ⇒ logits
        ~2 OOM too sharp).
    Method: Forward a val slice, collect outcome head logits at non-pad
        positions, report mean / std / p99 / abs-max per outcome.

    Args:
        model: GPT in eval mode.
        val_dl (DataLoader): Validation dataloader.
        n_batches (int): Batches to consume.

    Returns:
        pd.DataFrame: per-outcome distributional stats sorted by std desc.
    """
    cache = _collect_val_batches(model, val_dl, n_batches=n_batches)
    oh   = cache["outcome_logits"]
    tgt  = cache["targets"]
    nonpad = (tgt != cache["pad_idx"])

    rows = []
    for k, name in enumerate(model.outcome_names):
        v = oh[..., k][nonpad].numpy()
        rows.append({
            "outcome": name,
            "mean":    float(v.mean()),
            "std":     float(v.std()),
            "p50":     float(np.percentile(v, 50)),
            "p99":     float(np.percentile(v, 99)),
            "abs_max": float(np.abs(v).max()),
        })
    df = pd.DataFrame(rows).sort_values("std", ascending=False)
    print("\n" + "=" * 90)
    print("PROBE - OUTCOME HEAD LOGIT DISTRIBUTION")
    print("=" * 90)
    print(df.round(3).to_string(index=False))
    return df


def main() -> None:
    run_diagnostics(sample=2000, batch_size=32)


if __name__ == "__main__":
    main()
