from __future__ import annotations
import math
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F

from sklearn.metrics import roc_curve, precision_recall_curve, auc
from scipy.stats import spearmanr


from transform_emr.utils import (
    build_luts, get_multi_hot_targets, compute_legality_masks_tf,
    apply_masks_to_logits, masked_softmax, soft_interval_penalty,
    soft_meal_order_penalty, _gather_valid_ids
)
from transform_emr.loss import MaskedFocalBCE


# ---------- small helpers ----------

def _embedder_seq(model, batch):
    """
    Returns the per-step event embeddings from the embedder: [B, T+1, D].
    Aligns with your forward that prepends [CTX].
    """
    emb = model.embedder.forward(   # your embedder returns embeddings here
        raw_concept_ids=batch["raw_concept_ids"],
        concept_ids=batch["concept_ids"],
        value_ids=batch["value_ids"],
        position_ids=batch["position_ids"],
        abs_ts=batch["abs_ts"],
        patient_contexts=batch["context_vec"],
        return_mask=False,
    )
    return emb  # [B, T+1, D]


# ======================================================================
# 1) VOCAB CLEANUP / CONFUSION REPORT
# ======================================================================

@torch.no_grad()
def vocab_cleanup_report(
    model,
    data_loader,
    k_window: int = 12,
    max_batches: int = 3,
    topn: int = 40,
    device: Optional[torch.device] = None,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    For every token v:
      • freq: tokenizer count
      • avg_conf: P_model(v | context) when v is the GT
      • top1_rate: #correct-top1 / #occurrences
      • next-entropy: H( next<=k | token=v ) using dataset statistics
    Ranks tokens that are:
      • frequent but low-conf + high entropy  => 'frequent_noisy'
      • rare with low-conf                    => 'rare_unlearned'
    Prints a compact table and returns dict of candidate lists.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    tk = model.embedder.tokenizer
    V = len(tk.token2id)

    # running counters
    occ = torch.zeros(V, dtype=torch.long, device=device)
    correct_top1 = torch.zeros(V, dtype=torch.long, device=device)
    conf_sum = torch.zeros(V, dtype=torch.float32, device=device)
    next_counts = [Counter() for _ in range(V)]  # on CPU for entropy

    # legality LUTs once
    luts = build_luts(tk)
    for k, v in list(luts.items()):
        if torch.is_tensor(v):
            luts[k] = v.to(device)

    batches_done = 0
    for batch in data_loader:
        batches_done += 1
        if batches_done > max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}

        logits, _ = model(
            raw_concept_ids=batch["raw_concept_ids"],
            concept_ids=batch["concept_ids"],
            value_ids=batch["value_ids"],
            position_ids=batch["position_ids"],
            abs_ts=batch["abs_ts"],
            context_vec=batch["context_vec"],
        )

        # predict next for steps 1..T (drop [CTX] prediction)
        pred_logits = logits[:, 1:, :]                # [B,T,V]
        target_ids  = batch["targets"]                # [B,T]
        illegal_mask, bonus_mask = compute_legality_masks_tf(
            target_ids, luts["is_start"], luts["is_end"], luts["base_id"],
            luts["start_ids_per_base"], luts["end_ids_per_base"],
            luts["meal_rank"], luts["meal_pred_rank"], luts["K_meals"],
            luts["conflict_mat"], luts["predict_block"]
        )
        pred_logits = apply_masks_to_logits(pred_logits, illegal_mask, bonus_mask)

        allowed = (~illegal_mask) & (target_ids != tk.pad_token_id).unsqueeze(-1)
        P = masked_softmax(pred_logits, allowed)      # [B,T,V]

        # confidence on the GT
        gt = target_ids.unsqueeze(-1)                 # [B,T,1]
        p_gt = P.gather(-1, gt).squeeze(-1)           # [B,T]

        # top1 accuracy per-token
        top1 = pred_logits.argmax(-1)                 # [B,T]
        is_top1 = (top1 == target_ids) & (target_ids != tk.pad_token_id)

        # accumulate
        for b in range(target_ids.size(0)):
            valid_t = (target_ids[b] != tk.pad_token_id).nonzero(as_tuple=False).squeeze(-1)
            if valid_t.numel() == 0:
                continue
            ids_b  = target_ids[b, valid_t]          # [Nv]
            p_b    = p_gt[b, valid_t]                # [Nv]
            top1_b = is_top1[b, valid_t]
            occ.index_add_(0, ids_b, torch.ones_like(ids_b, dtype=torch.long))
            conf_sum.index_add_(0, ids_b, p_b)
            correct_top1.index_add_(0, ids_b, top1_b.to(torch.long))

            # build next<=k distribution for entropy
            T = target_ids.size(1)
            for t in valid_t.tolist():
                v = int(target_ids[b, t].item())
                # collect the next k tokens (stop at PAD)
                for u in target_ids[b, t+1: t+1+k_window].tolist():
                    if u == tk.pad_token_id:
                        break
                    next_counts[v][int(u)] += 1

    # compute stats on CPU
    occ_cpu = occ.cpu().numpy()
    conf_cpu = (conf_sum / (occ.clamp(min=1))).cpu().numpy()
    top1_cpu = (correct_top1 / (occ.clamp(min=1))).cpu().numpy()

    def entropy(cnt: Counter) -> float:
        n = sum(cnt.values())
        if n == 0:
            return 0.0
        return -sum((c/n) * math.log(max(c/n, 1e-12)) for c in cnt.values())

    next_H = [entropy(cnt) for cnt in next_counts]
    # normalize entropy by log(#support) to get 0..1
    norm_H = []
    for v, cnt in enumerate(next_counts):
        s = max(len(cnt), 1)
        h = next_H[v] / math.log(s) if s > 1 else 0.0
        norm_H.append(h)

    rows = []
    for tok, vid in tk.token2id.items():
        rows.append({
            "token": tok,
            "vid": vid,
            "freq": tk.token_counts.get(tok, 0),
            "occ": int(occ_cpu[vid]),
            "avg_conf": float(conf_cpu[vid]),
            "top1": float(top1_cpu[vid]),
            "next_entropy": float(norm_H[vid]),
        })

    # heuristics to propose candidates
    # frequent but noisy: high freq & occ with low conf and high entropy
    rows_sorted = sorted(rows, key=lambda r: (-r["freq"], -r["occ"]))
    frequent_noisy = [r for r in rows_sorted
                      if r["freq"] > 500    # tune threshold to your corpus
                      and r["avg_conf"] < 0.15
                      and r["next_entropy"] > 0.7][:topn]
    # rare unlearned
    rare_unlearned = [r for r in sorted(rows, key=lambda r: r["freq"])
                      if r["freq"] < 50 and r["avg_conf"] < 0.08][:topn]

    # pretty print
    def _print(title, lst):
        print(f"\n[{title}]")
        for r in lst:
            print(f"{r['token']:<45} occ={r['occ']:<7} conf={r['avg_conf']:.3f} "
                  f"top1={r['top1']:.3f}  nextH={r['next_entropy']:.2f}  freq={r['freq']}")

    _print("Frequent but noisy (consider move-to-context or merging)", frequent_noisy)
    _print("Rare & unlearned (consider pruning/merging)", rare_unlearned)

    return {
        "frequent_noisy": [(r["token"], r["avg_conf"]) for r in frequent_noisy],
        "rare_unlearned": [(r["token"], r["avg_conf"]) for r in rare_unlearned],
    }


# ======================================================================
# 2) EMBEDDER QUALITY: PROBE + CLUSTERING
# ======================================================================

def embedder_representation_report(
    model,
    data_loader,
    horizon: int = 12,
    max_batches_probe: int = 4,
    families: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Two diagnostics:

    A) Linear probe (frozen embedder, no Transformer):
       predict whether an OUTCOME token appears within the next `horizon` steps
       (binary label per t). Reports PR-AUC and ROC-AUC.

    B) Family cohesion from position embeddings:
       mean intra-family vs inter-family cosine; prints nearest neighbors.

    Returns a dict with probe metrics and cohesion scores.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    tk = model.embedder.tokenizer
    id2tok = tk.id2token

    # ----- A) short-horizon probe on embedder output -----
    # build a small dataset of (x_t, y_t)
    X, Y = [], []
    batches_done = 0
    for batch in data_loader:
        batches_done += 1
        if batches_done > max_batches_probe:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        emb = _embedder_seq(model, batch)                   # [B, T+1, D]
        ev = emb[:, :-1, :]                                 # align with positions 0..T-1
        tgt = batch["position_ids"]                         # [B, T]
        pad = tk.pad_token_id

        # label=1 iff any OUTCOME appears in (t, t+horizon]
        # outcome tokens: match by name
        outcome_ids = torch.tensor(
            [i for i, s in tk.id2token.items() if s.startswith("DEATH")
             or s.startswith("RELEASE") or s.startswith("COMPLICATION")],
            device=device, dtype=torch.long
        )
        outcome_set = set(outcome_ids.tolist())

        B, T, D = ev.shape
        for b in range(B):
            for t in range(T):
                if tgt[b, t].item() == pad:
                    break
                end = min(T, t + horizon + 1)
                window = tgt[b, t+1:end].tolist()
                y = 1 if any(u in outcome_set for u in window) else 0
                X.append(ev[b, t].detach())
                Y.append(y)

    if len(X) == 0:
        print("[embedder_representation_report] No examples collected.")
        return {}

    X = torch.stack(X, dim=0)           # [N, D]
    Y = torch.tensor(Y, device=device, dtype=torch.float32)  # [N]

    # tiny logistic probe
    W = torch.zeros(X.size(1), 1, device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    optim = torch.optim.Adam([W, b], lr=5e-3, weight_decay=1e-4)
    for _ in range(200):
        z = (X @ W).squeeze(-1) + b
        loss = F.binary_cross_entropy_with_logits(z, Y)
        optim.zero_grad(); loss.backward(); optim.step()

    with torch.no_grad():
        prob = torch.sigmoid((X @ W).squeeze(-1) + b)
        # PR-AUC & ROC-AUC (fast trapezoid approximations)
        def _auc(x, y):
            idx = torch.argsort(x)
            x = x[idx]; y = y[idx]
            return torch.trapz(y, x).item()
        # ROC
        try:
            fpr, tpr, _ = roc_curve(Y.cpu().numpy(), prob.cpu().numpy())
            roc_auc = auc(fpr, tpr)
            p, r, _ = precision_recall_curve(Y.cpu().numpy(), prob.cpu().numpy())
            pr_auc = auc(r, p)
        except Exception:
            # fallback rough
            roc_auc = _auc(torch.linspace(0,1,100), torch.linspace(0,1,100))
            pr_auc  = _auc(torch.linspace(0,1,100), torch.linspace(0,1,100))

    print(f"\n[Probe @h={horizon}]  ROC-AUC={roc_auc:.3f}  PR-AUC={pr_auc:.3f}  (N={len(Y)})")

    # ----- B) family cohesion from position embeddings -----
    E = model.embedder.position_embed.weight.detach()  # [V, D]
    En = F.normalize(E, dim=1)
    tokens = tk.id2token

    # default families if none provided
    if not families:
        families = ["MEAL_", "GLUCOSE_", "ANTIBIOTIC", "BLOOD_PRESSURE", "SODIUM_"]

    def family_ids(prefix: str) -> List[int]:
        return [i for i, t in tokens.items() if t.startswith(prefix)]

    def mean_cos(idsA: List[int], idsB: List[int]) -> float:
        if not idsA or not idsB: return float("nan")
        A = En[torch.tensor(idsA, device=En.device)]
        B = En[torch.tensor(idsB, device=En.device)]
        return (A @ B.t()).mean().item()

    report = {}
    all_ids = [i for i in range(E.size(0))
               if i not in (tk.pad_token_id, tk.mask_token_id, tk.ctx_token_id, tk.null_token_id)]
    inter = mean_cos(all_ids, all_ids)

    for fam in families:
        ids = family_ids(fam)
        if not ids:
            continue
        intra = mean_cos(ids, ids)
        report[f"{fam}intra_cos"] = intra
        report[f"{fam}inter_cos"] = inter
        # nearest neighbors for first few ids
        base = En[ids[: min(5, len(ids))]]
        sims = (base @ En.t())
        nn = torch.topk(sims, k=6, dim=1).indices.tolist()  # include self at rank 0
        print(f"\n[Nearest neighbors] {fam}")
        for i, row in enumerate(nn):
            tok = tokens[ids[i]]
            neigh = [tokens[j] for j in row[1:6]]
            print(f"  {tok:>40} -> {neigh}")

    return {"probe_roc_auc": roc_auc, "probe_pr_auc": pr_auc, **report}


# ======================================================================
# 3) TRAINING-TIME STATS (MaskedFocalBCE + set-mass + time)
# ======================================================================

@torch.no_grad()
def transformer_training_report(
    model,
    data_loader,
    training_settings: Dict,
    max_batches: int = 2,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    What this reports (all batch-averaged, teacher-forced, no grad):
      LOSS-DIRECT DIAGNOSTICS
        • bce_masked / bce_pos / bce_neg / pos_rate
        • focus_pos = mean((1 - σ(logit_pos))^γ)    # if ~0 ⇒ γ too strong; if ~1 ⇒ γ too weak
        • focus_neg = mean(σ(logit_neg)^γ)
        • hard_neg_selected_frac  (approx)          # ≈ how many negs selected per step (if hard_neg_k)
        • hard_neg_loss_share     (approx, focal)   # share of neg loss coming from top-k hard negatives

      MODEL ACCURACY (teacher-forced slice)
        • p_set_mean_nonpad / p_set_p90_nonpad      # probability mass on the K-future “true set”
        • top1_next / top5_next / top10_next        # masked top-n coverage for the immediate next token
        • p_set_head / p_set_mid / p_set_tail       # set-mass by token-frequency bucket
        • eff_k_mean / eff_k_p90                    # effective #positives per non-PAD step

      TIME AND PENALTIES
        • dt_mae / dt_r2 / dt_spearman / dt_viol_rate
        • illegal_mass_pre_mask / illegal_argmax_rate_pre_mask
        • pen_interval / pen_meal
        • pen_end_no_open / pen_dup_start / pen_conflict / pen_unclosed  (interval sub-components)

    Notes:
      • Uses the **same masking** and **MaskedFocalBCE** config you train with (gamma, hard_neg_k, etc.).
      • “Hard-neg” numbers are approximate (computed outside the criterion) but good enough to judge settings.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    tk = model.embedder.tokenizer
    V = len(tk.token2id)

    # robust counts -> tensor[V] on device
    def _counts_as_tensor(tk, V, device):
        obj = getattr(tk, "token_counts", None)
        if obj is None:
            return torch.zeros(V, device=device)
        if isinstance(obj, dict):
            out = torch.zeros(V, device=device, dtype=torch.float32)
            keys_are_str = any(isinstance(k, str) for k in obj.keys())
            if keys_are_str:
                id2tok = getattr(tk, "id2token", None)
                for i in range(V):
                    tok = id2tok[i] if id2tok is not None else str(i)
                    out[i] = float(obj.get(tok, 0))
            else:
                for i in range(V):
                    out[i] = float(obj.get(i, 0))
            return out
        if torch.is_tensor(obj):
            c = obj.to(device).float().flatten()
            return c[:V] if c.numel() >= V else torch.cat([c, torch.zeros(V - c.numel(), device=device)], 0)
        try:
            if isinstance(obj, (list, tuple, np.ndarray)):
                c = torch.as_tensor(obj, device=device, dtype=torch.float32).flatten()
                return c[:V] if c.numel() >= V else torch.cat([c, torch.zeros(V - c.numel(), device=device)], 0)
        except Exception:
            pass
        return torch.zeros(V, device=device)

    # LUTs
    luts = build_luts(tk)
    for k, v in list(luts.items()):
        if torch.is_tensor(v):
            luts[k] = v.to(device)

    # criterion (mirror training)
    criterion = MaskedFocalBCE.from_counts(
        counts=tk.token_counts,
        token_weights=tk.token_weights,
        beta=training_settings.get("beta", 0.999),
        min_count=training_settings.get("min_count", 5),
        clip_max=training_settings.get("clip_max", 8.0),
        gamma=training_settings.get("gamma", 1.2),
        tau=training_settings.get("tau", 0.8),
        neg_bounds=training_settings.get("neg_bounds", (0.05, 0.5)),
        label_smoothing=training_settings.get("label_smoothing", 0.01),
        hard_neg_k=training_settings.get("hard_neg_k", 64),
    ).to(device)
    gamma = float(getattr(getattr(criterion, "focal", criterion), "gamma", 2.0))  # <-- pull γ from focal

    # Accumulators
    bce_masked = bce_pos = bce_neg = 0.0
    Np = Nn = 0.0
    pset_vals, eff_k_list = [], []

    illegal_mass_list = []
    illegal_argmax_cnt = 0
    argmax_total = 0
    top1 = top5 = top10 = 0
    next_total = 0

    counts = _counts_as_tensor(tk, V, device)
    q25, q75 = torch.quantile(counts.float(), torch.tensor([0.25, 0.75], device=counts.device))
    pset_head, pset_mid, pset_tail = [], [], []

    dt_abs_err = []
    dt_true_all, dt_pred_all = [], []
    dt_viol_cnt = dt_viol_den = 0
    pen_int = pen_meal = 0.0
    pen_end_no_open = pen_dup = pen_cnf = pen_unclosed = 0.0

    foc_pos_vals, foc_neg_vals = [], []
    hard_sel_cnt = 0
    hard_neg_loss = 0.0
    total_neg_loss = 0.0
    hard_neg_k = int(getattr(criterion, "hard_neg_k", 0) or 0)

    def softmax_masked(logits, mask):
        l = logits.masked_fill(~mask, float('-inf'))
        logZ = torch.logsumexp(l, dim=-1, keepdim=True)
        P = torch.exp(l - logZ)
        P[~mask] = 0.0
        return P

    batches_done = 0
    for batch in data_loader:
        batches_done += 1
        if batches_done > max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}

        logits_all, abs_t_pred = model(
            raw_concept_ids=batch["raw_concept_ids"],
            concept_ids=batch["concept_ids"],
            value_ids=batch["value_ids"],
            position_ids=batch["position_ids"],
            abs_ts=batch["abs_ts"],
            context_vec=batch["context_vec"],
        )

        logits_pre_mask = logits_all[:, 1:, :]
        target_ids = batch["targets"]

        illegal_mask, bonus_mask = compute_legality_masks_tf(
            target_ids, luts["is_start"], luts["is_end"], luts["base_id"],
            luts["start_ids_per_base"], luts["end_ids_per_base"],
            luts["meal_rank"], luts["meal_pred_rank"], luts["K_meals"],
            luts["conflict_mat"], luts["predict_block"]
        )
        pred_logits = apply_masks_to_logits(logits_pre_mask, illegal_mask, bonus_mask)
        nonpad = (target_ids != tk.pad_token_id)
        allowed = (~illegal_mask) & nonpad.unsqueeze(-1)

        multi_hot = get_multi_hot_targets(
            position_ids=target_ids,
            padding_idx=tk.pad_token_id,
            vocab_size=pred_logits.size(-1),
            k=training_settings["bce_k_window"]
        ).masked_fill_(illegal_mask, 0.0)

        # ---- BCE diagnostics (use your keys)
        loss, info = criterion(pred_logits, multi_hot, allowed)
        lp = float(info["loss_pos"])
        ln = float(info["loss_neg"])
        lam = float(info["lambda_neg"])

        bce_masked += lp + lam * ln
        bce_pos    += lp
        bce_neg    += ln
        Np         += float(info["Np"])
        Nn         += float(info["Nn"])

        # ---- Focal focusing factors
        S = torch.sigmoid(pred_logits)
        pos_mask = allowed & (multi_hot > 0)
        neg_mask = allowed & (multi_hot == 0)
        if pos_mask.any(): foc_pos_vals.append(((1.0 - S[pos_mask]).pow(gamma)).mean().item())
        if neg_mask.any(): foc_neg_vals.append((S[neg_mask].pow(gamma)).mean().item())

        # ---- Hard-neg approx
        if hard_neg_k > 0 and neg_mask.any():
            bce_per_class = F.binary_cross_entropy_with_logits(pred_logits, multi_hot, reduction='none')
            w_neg = S.pow(gamma)
            neg_loss = bce_per_class * neg_mask.float() * w_neg
            total_neg_loss += neg_loss.sum().item()
            S_neg = S.masked_fill(~neg_mask, -1.0)
            k_per = min(hard_neg_k, S_neg.size(-1))
            if k_per > 0:
                topk_vals, topk_idx = torch.topk(S_neg, k=k_per, dim=-1)
                sel_mask_bt = (topk_vals > 0)
                hard_sel_cnt += sel_mask_bt.sum().item()
                gathered = torch.gather(neg_loss, dim=-1, index=topk_idx) * sel_mask_bt.float()
                hard_neg_loss += gathered.sum().item()

        # ---- effective K
        eff_k_list += multi_hot.sum(-1)[nonpad].cpu().tolist()

        # ---- illegal temptation (pre-mask)
        P_pre = torch.softmax(logits_pre_mask, dim=-1)
        illegal_mass_list += (P_pre * illegal_mask).sum(-1)[nonpad].cpu().tolist()
        argmax = logits_pre_mask.argmax(-1)
        illegal_argmax_cnt += (illegal_mask.gather(2, argmax.unsqueeze(-1)).squeeze(-1) & nonpad).sum().item()
        argmax_total       += nonpad.sum().item()

        # ---- set-mass
        P = softmax_masked(pred_logits, allowed)
        set_mass = (P * (multi_hot > 0)).sum(-1)
        pset_vals += set_mass[nonpad].cpu().tolist()

        next_counts = counts.to(device)[target_ids]
        for sm, c, ok in zip(set_mass.flatten().cpu(), next_counts.flatten().cpu(), nonpad.flatten().cpu()):
            if not bool(ok): continue
            if c >= q75:   pset_head.append(float(sm))
            elif c <= q25: pset_tail.append(float(sm))
            else:          pset_mid.append(float(sm))

        # ---- top-n next
        for n in (1,5,10):
            topn = torch.topk(P, k=min(n, V), dim=-1).indices
            hit = (topn == target_ids.unsqueeze(-1)).any(-1) & nonpad
            cnt = hit.sum().item()
            if n == 1: top1 += cnt
            if n == 5: top5 += cnt
            if n == 10: top10 += cnt
        next_total += nonpad.sum().item()

        # ---- time diagnostics
        true_delta = batch["abs_ts"]
        pred_delta = abs_t_pred[:, 1:]
        tmask = nonpad
        dt_abs_err += (pred_delta[tmask] - true_delta[tmask]).abs().cpu().tolist()
        dt_true_all.append(true_delta[tmask].detach().cpu())
        dt_pred_all.append(pred_delta[tmask].detach().cpu())

        if tmask.sum() > 1:
            pdiff = pred_delta[:, 1:] - pred_delta[:, :-1]
            pmask = tmask[:, 1:] & tmask[:, :-1]
            dt_viol_cnt += (pdiff[pmask] < 0).sum().item()
            dt_viol_den += pmask.sum().item()

        # ---- penalties (diagnostic magnitudes)
        _pen = soft_interval_penalty(
            pred_logits, allowed,
            luts["start_ids_per_base"], luts["end_ids_per_base"],
            luts["conflict_mat"], alpha=10.0
        )
        pen_int += float(_pen.item())

        Pcpu = P.detach().cpu()
        s_ids_all, s_mask = _gather_valid_ids(luts["start_ids_per_base"])
        e_ids_all, e_mask = _gather_valid_ids(luts["end_ids_per_base"])
        vm = (s_mask & e_mask)
        if vm.any():
            s_ids = luts["start_ids_per_base"][vm].cpu()
            e_ids = luts["end_ids_per_base"][vm].cpu()
            p_s = Pcpu[:, :, s_ids]; p_e = Pcpu[:, :, e_ids]
            cs_s = torch.cumsum(p_s, dim=1); cs_e = torch.cumsum(p_e, dim=1)
            open_before = torch.relu(torch.cat([torch.zeros_like(cs_s[:, :1]), cs_s[:, :-1]-cs_e[:, :-1]], dim=1))
            pen_end_no_open += float((p_e * torch.exp(-10.0*open_before)).sum().item())
            pen_dup         += float((p_s * (1.0 - torch.exp(-10.0*open_before))).sum().item())
            cm = luts["conflict_mat"][vm][:, vm].float().cpu()
            if cm.numel():
                open_conf = torch.einsum('btn,nm->btm', open_before, cm)
                pen_cnf += float((p_s * (1.0 - torch.exp(-10.0*open_conf))).sum().item())
            open_final = torch.relu(cs_s[:, -1, :] - cs_e[:, -1, :])
            pen_unclosed += float(open_final.sum().item())

        pen_meal += float(soft_meal_order_penalty(
            pred_logits, allowed, luts["meal_rank"], decay=0.9, beta=8.0
        ).item())

    # aggregates
    n_batches = max(1, batches_done)
    bce_masked /= n_batches
    bce_pos    /= n_batches
    bce_neg    /= n_batches
    pos_rate   = Np / max(Np + Nn, 1.0)

    eff_k_list.sort()
    eff_k_mean = float(sum(eff_k_list)/max(len(eff_k_list),1))
    eff_k_p90  = float(eff_k_list[int(0.9*len(eff_k_list))]) if eff_k_list else 0.0

    pset_vals.sort()
    mean_pset = float(sum(pset_vals)/max(len(pset_vals),1))
    p90_pset  = float(pset_vals[int(0.9*len(pset_vals))]) if pset_vals else 0.0

    illegal_mass_mean   = float(sum(illegal_mass_list)/max(len(illegal_mass_list),1))
    illegal_argmax_rate = illegal_argmax_cnt / max(argmax_total, 1)

    top1_acc  = top1  / max(next_total, 1)
    top5_acc  = top5  / max(next_total, 1)
    top10_acc = top10 / max(next_total, 1)

    if dt_true_all:
        y_true = torch.cat(dt_true_all).numpy()
        y_pred = torch.cat(dt_pred_all).numpy()
        mae = float(np.abs(y_pred - y_true).mean())
        r2  = float(1.0 - (np.var(y_true - y_pred) / (np.var(y_true) + 1e-8)))
        rho = float(spearmanr(y_true, y_pred).correlation) if y_true.size > 3 else 0.0
    else:
        mae = r2 = rho = 0.0
    viol_rate = dt_viol_cnt / max(dt_viol_den, 1)

    pen_interval = pen_int / n_batches
    pen_meal     = pen_meal / n_batches
    pen_end_no_open /= n_batches
    pen_dup         /= n_batches
    pen_cnf         /= n_batches
    pen_unclosed    /= n_batches

    focus_pos = float(np.mean(foc_pos_vals)) if foc_pos_vals else 0.0
    focus_neg = float(np.mean(foc_neg_vals)) if foc_neg_vals else 0.0
    hard_neg_selected_frac = (hard_sel_cnt / max(Nn, 1.0)) if hard_neg_k > 0 else 0.0
    hard_neg_loss_share    = (hard_neg_loss / max(total_neg_loss, 1e-8)) if hard_neg_k > 0 else 0.0

    # prints (unchanged formatting)
    print("\n[Loss breakdown]")
    print(f"  bce_masked={bce_masked:.6f}  bce_pos={bce_pos:.6f}  bce_neg={bce_neg:.6f}  pos_rate={pos_rate:.3f}")
    print(f"  focal focus:  pos={focus_pos:.3f}  neg={focus_neg:.3f}  "
          f"hard_neg_frac={hard_neg_selected_frac:.3f}  hard_neg_loss_share={hard_neg_loss_share:.3f}")
    print("[Set-mass] mean={:.3f} p90={:.3f} | eff_k_mean={:.2f} p90={:.1f}".format(mean_pset, p90_pset, eff_k_mean, eff_k_p90))
    print("[Illegal temptation] mass_pre_mask={:.3f}  argmax_illegal_rate={:.3f}".format(illegal_mass_mean, illegal_argmax_rate))
    print("[Top-n next-token] top1={:.3f}  top5={:.3f}  top10={:.3f}".format(top1_acc, top5_acc, top10_acc))
    if pset_head or pset_mid or pset_tail:
        ms = lambda lst: (sum(lst)/len(lst)) if lst else 0.0
        print("[Set-mass by freq] head={:.3f}  mid={:.3f}  tail={:.3f}".format(ms(pset_head), ms(pset_mid), ms(pset_tail)))
    print("[Δt] MAE={:.3f}  R2={:.3f}  Spearman={:.3f}  viol_rate={:.3f}".format(mae, r2, rho, viol_rate))
    print("[Soft penalties] interval={:.4f} (end_no_open={:.3f}, dup={:.3f}, cnf={:.3f}, unclosed={:.3f})  meal={:.4f}"
          .format(pen_interval, pen_end_no_open, pen_dup, pen_cnf, pen_unclosed, pen_meal))

    return {
        "bce_masked": bce_masked, "bce_pos": bce_pos, "bce_neg": bce_neg, "pos_rate": float(pos_rate),
        "focus_pos": focus_pos, "focus_neg": focus_neg,
        "hard_neg_selected_frac": hard_neg_selected_frac, "hard_neg_loss_share": hard_neg_loss_share,
        "p_set_mean_nonpad": mean_pset, "p_set_p90_nonpad": p90_pset,
        "eff_k_mean": eff_k_mean, "eff_k_p90": eff_k_p90,
        "top1_next": top1_acc, "top5_next": top5_acc, "top10_next": top10_acc,
        "p_set_head": float(sum(pset_head)/len(pset_head)) if pset_head else 0.0,
        "p_set_mid":  float(sum(pset_mid)/len(pset_mid))  if pset_mid  else 0.0,
        "p_set_tail": float(sum(pset_tail)/len(pset_tail)) if pset_tail else 0.0,
        "dt_mae": mae, "dt_r2": r2, "dt_spearman": rho, "dt_viol_rate": viol_rate,
        "illegal_mass_pre_mask": illegal_mass_mean,
        "illegal_argmax_rate_pre_mask": illegal_argmax_rate,
        "pen_interval": pen_interval, "pen_meal": pen_meal,
        "pen_end_no_open": pen_end_no_open, "pen_dup_start": pen_dup,
        "pen_conflict": pen_cnf, "pen_unclosed": pen_unclosed,
    }