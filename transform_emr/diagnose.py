"""Standalone diagnostics for EMR autoresearch.

Run:
    python -m transform_emr.diagnose
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from transform_emr.config.dataset_config import (  # noqa: E402
    OUTCOMES,
    TAK_REPO_PATH,
    TRAIN_CTX_DATA_FILE,
    TRAIN_TEMPORAL_DATA_FILE,
)
from transform_emr.config.model_config import (  # noqa: E402
    CHECKPOINT_PATH,
    EMBEDDER_CHECKPOINT,
    TRAINING_SETTINGS,
    TRANSFORMER_CHECKPOINT,
)
from transform_emr.dataset import DataProcessor, EMRDataset, EMRTokenizer, collate_emr, get_dataloader  # noqa: E402
from transform_emr.embedder import EMREmbedding  # noqa: E402
from transform_emr.loss import MaskedFocalBCE  # noqa: E402
from transform_emr.transformer import GPT  # noqa: E402
from transform_emr.utils import (  # noqa: E402
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
    if not Path(EMBEDDER_CHECKPOINT).exists():
        missing.append(EMBEDDER_CHECKPOINT)
    if not Path(TRANSFORMER_CHECKPOINT).exists():
        missing.append(TRANSFORMER_CHECKPOINT)
    if missing:
        print("[Diag] Missing checkpoint(s). Run training first to generate them:")
        for ckpt in missing:
            print(f"  - {ckpt}")
        print("[Diag] Aborting diagnostics.")
        return

    embedder, *_ = EMREmbedding.load(EMBEDDER_CHECKPOINT, tokenizer=tokenizer)
    model, *_ = GPT.load(TRANSFORMER_CHECKPOINT, embedder=embedder)
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
    print("REPORT 1 - PER-OUTCOME AUROC BREAKDOWN (48h eval window)")
    print("=" * 90)
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
    print("REPORT 4 - AUX LOSS LAMBDA CALIBRATION GUIDE")
    print("=" * 90)
    sched = training_settings.get("phase2_scheduler", {})
    caps = sched.get("aux_fraction_caps", {})
    if caps:
        print("aux_fraction_caps from train.py:")
        for key, value in caps.items():
            print(f"  {key:<12}: cap={value}")
        print("Interpretation: lambda = cap x (BCE_at_calibration / aux_raw_at_calibration)")
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

    print("\n[Diag] Done.")


def main() -> None:
    run_diagnostics(sample=2000, batch_size=32)


if __name__ == "__main__":
    main()
