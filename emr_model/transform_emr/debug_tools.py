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
    apply_masks_to_logits, masked_softmax, soft_illegal_mass_penalty,
    soft_unclosed_interval_penalty, _gather_valid_ids
)
from transform_emr.loss import MaskedFocalBCE


# ---------- small helpers ----------

def _embedder_seq(model, batch):
    """
    Returns the per-step event embeddings from the embedder: [B, T, D].
    """
    emb, _ = model.embedder.forward(   # your embedder returns embeddings here
        raw_concept_ids=batch["raw_concept_ids"],
        concept_ids=batch["concept_ids"],
        value_ids=batch["value_ids"],
        position_ids=batch["position_ids"],
        abs_ts=batch["abs_ts"],
        patient_contexts=batch["context_vec"],
        return_mask=False,
    )
    return emb  # [B, T, D]


def _guess_families(tokenizer, max_groups: int = 8):
    """
    Select up to `max_groups` RawConcept families by corpus prevalence,
    where a family = all PositionTokens that share the same RawConcept.

    We compute family mass by summing token_counts over PositionTokens that
    belong to that RawConcept (no heuristics).
    """
    # Convenience handles from tokenizer
    id2tok    = tokenizer.id2token              # PositionToken string per vid
    counts    = tokenizer.token_counts          # occurrences per PositionToken vid (tensor)
    rawcons   = list(tokenizer.rawconcept2id)   # RawConcept vocabulary (strings)

    # Special tokens to ignore
    specials = {
        tokenizer.pad_token_id,
        getattr(tokenizer, "mask_token_id", -1),
        getattr(tokenizer, "null_token_id", -1),
        getattr(tokenizer, "ctx_token_id", -1),
    }
    special_tokens = {id2tok[vid] for vid in specials if vid in id2tok}

    # Helper: does PositionToken 'tok' belong to RawConcept 'rc'?
    # (PositionToken strings are built from ConceptName/Value with optional _START/_END.)
    def _belongs(tok: str, rc: str) -> bool:
        if tok in special_tokens:
            return False
        # strip trailing START/END tag (if present)
        if tok.endswith("_START"):
            core = tok[:-6]
        elif tok.endswith("_END"):
            core = tok[:-4]
        else:
            core = tok
        # RawConcept is exactly the ConceptName without trailing _STATE/_TREND
        # PositionToken always begins with ConceptName; thus rc must be a prefix of core
        # and aligned at token boundary (rc + "_" or exact match).
        return core == rc or core.startswith(rc + "_")

    # Aggregate counts per RawConcept
    family_mass = []
    for rc in rawcons:
        # Skip specials if they ever appear as raw concepts
        if rc in ("[PAD]", "[MASK]", "[NULL]", "[CTX]"):
            continue
        tot = 0
        # Loop over all PositionTokens; sum counts if they belong to rc
        # (counts is a tensor aligned with id2tok indices)
        for vid, tok in id2tok.items():
            if _belongs(tok, rc):
                tot += int(counts[vid].item())
        if tot > 0:
            family_mass.append((rc, tot))

    # Sort by prevalence and take the top groups
    family_mass.sort(key=lambda x: -x[1])
    families = [rc for rc, _ in family_mass[:max_groups]]
    return families


def _counts_as_tensor(tk, V: int, device: torch.device) -> torch.Tensor:
    """
    Return token_counts as a float tensor of length V on device,
    handling dict/np/list/tensor variants.
    """
    obj = getattr(tk, "token_counts", None)
    if obj is None:
        return torch.zeros(V, device=device)
    if isinstance(obj, dict):
        out = torch.zeros(V, device=device, dtype=torch.float32)
        id2tok = getattr(tk, "id2token", None)
        for i in range(V):
            tok = id2tok[i] if id2tok is not None else str(i)
            out[i] = float(obj.get(tok, 0))
        return out
    if torch.is_tensor(obj):
        c = obj.to(device).float().flatten()
        return c[:V] if c.numel() >= V else torch.cat([c, torch.zeros(V - c.numel(), device=device)], 0)
    if isinstance(obj, (list, tuple, np.ndarray)):
        c = torch.as_tensor(obj, device=device, dtype=torch.float32).flatten()
        return c[:V] if c.numel() >= V else torch.cat([c, torch.zeros(V - c.numel(), device=device)], 0)
    return torch.zeros(V, device=device)

def _count_for_token(tk, vid: int) -> int:
    """
    Return integer count for a single token id `vid`, regardless of
    tokenizer.token_counts' type (dict/tensor/np/list).
    """
    obj = getattr(tk, "token_counts", None)
    if obj is None:
        return 0
    if isinstance(obj, dict):
        tok = tk.id2token[vid]
        return int(obj.get(tok, 0))
    if torch.is_tensor(obj):
        return int(obj.flatten()[vid].item())
    if isinstance(obj, (list, tuple, np.ndarray)):
        arr = np.array(obj).reshape(-1)
        return int(arr[vid]) if vid < arr.size else 0
    return 0


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

        logits, _, _ = model(
            raw_concept_ids=batch["raw_concept_ids"],
            concept_ids=batch["concept_ids"],
            value_ids=batch["value_ids"],
            position_ids=batch["position_ids"],
            abs_ts=batch["abs_ts"],
            context_vec=batch["context_vec"],
        )

        # predict next for steps 1..T
        pred_logits = logits[:, :-1, :]       # [B, T-1, V]
        target_ids  = batch["targets"][:, 1:] # [B, T-1]
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
            "freq": _count_for_token(tk, vid),
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


@torch.enable_grad()
def token_gradient_utility_report(
    model, data_loader, k_window=12, max_batches=3, device=None
):
    """
    Aggregates ||grad||^2 on position_embed.weight[v] under your teacher-forced BCE
    (with the same masking) to score each token's training utility.
    Returns a list of dict rows sorted by 'grad_per_occ'.
    """
    model.train()  # we need autograd, but no optimizer step
    if device is None:
        device = next(model.parameters()).device
    tk = model.embedder.tokenizer
    V  = len(tk.token2id)

    # Accumulators
    grad_sq = torch.zeros(V, device=device)
    occ     = torch.zeros(V, device=device)

    # Build the same LUTs & criterion you use in training
    from transform_emr.utils import (build_luts, get_multi_hot_targets,
                                     compute_legality_masks_tf, apply_masks_to_logits)
    from transform_emr.loss import MaskedFocalBCE

    luts = build_luts(tk)
    for k, v in list(luts.items()):
        if torch.is_tensor(v): luts[k] = v.to(device)

    crit = MaskedFocalBCE.from_counts(
        counts=tk.token_counts, token_weights=tk.token_weights,
        beta=0.999, min_count=5, clip_max=8.0,
        gamma=1.3, tau=0.85, neg_bounds=(0.03, 0.30),
        label_smoothing=0.0, hard_neg_k=0
    ).to(device)

    # Make sure previous grads are clear
    for p in model.parameters(): 
        if p.grad is not None: p.grad = None

    batches = 0
    for batch in data_loader:
        batches += 1
        if batches > max_batches: break
        batch = {k: v.to(device) for k, v in batch.items()}

        logits, _, _ = model(**{
            "raw_concept_ids": batch["raw_concept_ids"],
            "concept_ids":     batch["concept_ids"],
            "value_ids":       batch["value_ids"],
            "position_ids":    batch["position_ids"],
            "abs_ts":          batch["abs_ts"],
            "context_vec":     batch["context_vec"],
        })

        pred_logits = logits[:, :-1, :]
        target_ids  = batch["targets"][:, 1:]
        illegal, bonus = compute_legality_masks_tf(
            target_ids, luts["is_start"], luts["is_end"], luts["base_id"],
            luts["start_ids_per_base"], luts["end_ids_per_base"],
            luts["meal_rank"], luts["meal_pred_rank"], luts["K_meals"],
            luts["conflict_mat"], luts["predict_block"]
        )
        pred_logits = apply_masks_to_logits(pred_logits, illegal, bonus)
        nonpad = (target_ids != tk.pad_token_id)
        allowed = (~illegal) & nonpad.unsqueeze(-1)

        multi_hot = get_multi_hot_targets(
            position_ids=target_ids, padding_idx=tk.pad_token_id,
            vocab_size=pred_logits.size(-1), k=k_window
        ).masked_fill(illegal, 0.0)

        # teacher-forced loss
        loss, _ = crit(pred_logits, multi_hot, allowed)
        loss.backward()

        # accumulate squared grad on position embeddings
        W = model.embedder.position_embed.weight  # [V,D]
        if W.grad is not None:
            grad_sq += (W.grad.detach() ** 2).sum(dim=1)
        # occurrences in batch (targets side)
        ids = target_ids[nonpad]
        if ids.numel(): 
            occ.index_add_(0, ids, torch.ones_like(ids, dtype=occ.dtype))
        # clear grads for next mini-batch
        for p in model.parameters():
            if p.grad is not None: p.grad.zero_()

    # scores
    grad_sq = grad_sq.cpu()
    occ = occ.cpu().clamp(min=1)
    score = (grad_sq / occ).numpy()   # grad per occurrence

    rows = []
    id2tok = tk.id2token
    for vid in range(V):
        rows.append({"token": id2tok[vid], "vid": vid,
                     "occ": int(occ[vid].item()),
                     "grad_sq": float(grad_sq[vid].item()),
                     "grad_per_occ": float(score[vid])})
    rows.sort(key=lambda r: -r["grad_per_occ"])
    print("\n[Token utility] top by grad_per_occ")
    for r in rows[:40]:
        print(f"{r['token']:<45} occ={r['occ']:<6} grad/occ={r['grad_per_occ']:.4e}")
    return rows


# ======================================================================
# 2) EMBEDDER QUALITY: PROBE + CLUSTERING
# ======================================================================

def embedder_representation_report(
    model,
    data_loader,
    horizon: int = 12,
    max_batches_probe: int = 4,
    n_families: int = 8,
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
    families = _guess_families(tk, max_groups=n_families)
    print(f"[embedder] evaluating families (RawConcepts): {', '.join(families)}")

    # ----- A) short-horizon probe on embedder output -----
    # build a small dataset of (x_t, y_t)
    X, Y = [], []
    batches_done = 0
    for batch in data_loader:
        batches_done += 1
        if batches_done > max_batches_probe:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        ev = _embedder_seq(model, batch)                   # [B, T, D]
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

    # --- prevalence baseline for PR-AUC interpretation ---
    prevalence = float(Y.mean().item())
    print(f"[Probe@h={horizon}] prevalence={prevalence:.4f} "
        f"(PR-AUC baseline = {prevalence:.4f})")

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

    # family membership by RawConcept
    special_ids = {
        getattr(tk, "pad_token_id", -1),
        getattr(tk, "mask_token_id", -1),
        getattr(tk, "ctx_token_id", -1),
        getattr(tk, "null_token_id", -1),
    }
    def _belongs_to_rawconcept(tok: str, rc: str) -> bool:
        if tok in {tk.id2token[i] for i in special_ids if i in tk.id2token}:
            return False
        if tok.endswith("_START"): core = tok[:-6]
        elif tok.endswith("_END"): core = tok[:-4]
        else: core = tok
        return (core == rc) or core.startswith(rc + "_")

    def family_ids(rawconcept: str) -> List[int]:
        return [i for i, t in tokens.items() if _belongs_to_rawconcept(t, rawconcept)]

    def mean_cos(idsA: List[int], idsB: List[int]) -> float:
        if not idsA or not idsB: return float("nan")
        A = En[torch.tensor(idsA, device=En.device)]
        B = En[torch.tensor(idsB, device=En.device)]
        return (A @ B.t()).mean().item()

    report = {}
    all_ids = [i for i in range(E.size(0))
               if i not in (tk.pad_token_id, tk.mask_token_id, tk.ctx_token_id, tk.null_token_id)]
    inter = mean_cos(all_ids, all_ids)

    rows = []
    for fam in families:
        ids = family_ids(fam)
        if not ids:
            continue
        intra = mean_cos(ids, ids)

        report[f"{fam}intra_cos"] = intra
        report[f"{fam}inter_cos"] = inter
        if intra is None or inter is None: 
            continue
        rows.append((fam, intra, inter, intra - inter))
        # nearest neighbors for first few ids
        base = En[ids[: min(5, len(ids))]]
        sims = (base @ En.t())
        nn = torch.topk(sims, k=6, dim=1).indices.tolist()  # include self at rank 0
        print(f"\n[Nearest neighbors] {fam}")
        for i, row in enumerate(nn):
            tok = tokens[ids[i]]
            neigh = [tokens[j] for j in row[1:6]]
            print(f"  {tok:>40} -> {neigh}")
    if rows:
        rows.sort(key=lambda r: r[3])  # most worrying (intra<inter) first
        print("\n[Family cohesion Δ = intra - inter]  (lower is worse)")
        print("raw_concept".ljust(32), "intra".rjust(8), "inter".rjust(8), "Δ".rjust(8))
        for fam, intra, inter, d in rows:
            print(f"{fam.ljust(32)} {intra:8.3f} {inter:8.3f} {d:8.3f}")

    return {"probe_roc_auc": roc_auc, "probe_pr_auc": pr_auc, **report}


def embed_norm_vs_freq_plot(model):
    import matplotlib.pyplot as plt
    tk = model.embedder.tokenizer
    W  = model.embedder.position_embed.weight.detach().cpu()
    norms = W.norm(dim=1).numpy()
    freq  = np.array([_count_for_token(tk, i) for i in range(len(tk.token2id))])
    plt.figure()
    plt.scatter(np.log1p(freq), norms, s=6, alpha=0.5)
    plt.xlabel("log(1+freq)"); plt.ylabel("||embedding||2")
    plt.title("Position-embedding norm vs frequency")
    # label a few outliers
    idx = np.argsort(norms)[-20:]
    for i in idx:
        plt.text(np.log1p(freq[i]), norms[i], tk.id2token[i], fontsize=6, alpha=0.7)
    plt.show()


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

    CONTEXT INFLUENCE (always-on ablation)
        Printed:
        • [CTX] ΔBCE zero-normal           # BCE(CTX=zero) - BCE(normal); >0 ⇒ context helps
        • [CTX] ΔBCE shuffle-normal        # BCE(CTX=shuffled) - BCE(normal); > ΔBCE zero ⇒ patient-specific use
        • [CTX] meanKL_zero / meanKL_shuffle
            - Mean KL divergence per step between masked distributions with ablated context vs normal;
            ≳ 0.02-0.05 nats/step indicates meaningful logits shift due to context.
        Returned (keys in the result dict):
        • ctx_bce_normal / ctx_bce_zero / ctx_bce_shuffle
        • ctx_kl_zero / ctx_kl_shuffle

    Notes:
    • Uses the same masking and MaskedFocalBCE config you train with (gamma, hard_neg_k, etc.).
    • “Hard-neg” numbers are approximate (computed outside the criterion) but good enough to judge settings.
    • Context probe adds 2 extra forward passes per batch (zeroed and shuffled context). Keep `max_batches` small if needed.
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
        gamma=training_settings.get("gamma", 1.3),
        tau=training_settings.get("tau", 0.85),
        neg_bounds=training_settings.get("neg_bounds", (0.03, 0.3)),
        label_smoothing=training_settings.get("label_smoothing", 0.0),
        hard_neg_k=training_settings.get("hard_neg_k", 0),
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

    foc_pos_vals, foc_neg_vals = [], []
    hard_sel_cnt = 0
    hard_neg_loss = 0.0
    total_neg_loss = 0.0

    # --- Context probe accumulators ---
    ctx_tot_norm = 0.0
    ctx_tot_zero = 0.0
    ctx_tot_shuf = 0.0
    ctx_kl_zero  = 0.0
    ctx_kl_shuf  = 0.0
    ctx_batches  = 0
    
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
        l = logits.masked_fill(~mask, -1e9)
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

        logits_all, abs_t_pred, _ = model(
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

        # -----------------------
        # Context ablation
        # -----------------------
        # Reference masked distribution under normal context (uses pred_logits already built)
        Pref = softmax_masked(pred_logits, allowed).detach()

        # ZERO context
        logits_all_zero, _, _ = model(
            raw_concept_ids=batch["raw_concept_ids"],
            concept_ids=batch["concept_ids"],
            value_ids=batch["value_ids"],
            position_ids=batch["position_ids"],
            abs_ts=batch["abs_ts"],
            context_vec=torch.zeros_like(batch["context_vec"]),
        )
        logits_zero = apply_masks_to_logits(logits_all_zero[:, 1:, :], illegal_mask, bonus_mask)
        loss_zero, _ = criterion(logits_zero, multi_hot, allowed)

        # SHUFFLED context
        B = batch["context_vec"].size(0)
        perm = torch.randperm(B, device=device)
        logits_all_shuf, _, _ = model(
            raw_concept_ids=batch["raw_concept_ids"],
            concept_ids=batch["concept_ids"],
            value_ids=batch["value_ids"],
            position_ids=batch["position_ids"],
            abs_ts=batch["abs_ts"],
            context_vec=batch["context_vec"][perm],
        )
        logits_shuf = apply_masks_to_logits(logits_all_shuf[:, 1:, :], illegal_mask, bonus_mask)
        loss_shuf, _ = criterion(logits_shuf, multi_hot, allowed)

        # KL(P || Pref) for zero/shuffle (masked softmax)
        P0  = softmax_masked(logits_zero, allowed)
        Psh = softmax_masked(logits_shuf, allowed)

        eps = 1e-12
        def mean_kl(P, Q):
            P = P.clamp_min(eps); Q = Q.clamp_min(eps)
            kl = (P * (P.log() - Q.log())).sum(dim=-1)  # [B,T]
            return float(kl[nonpad].mean().item()) if nonpad.any() else 0.0

        ctx_tot_norm += float(loss)        # baseline masked BCE with normal context
        ctx_tot_zero += float(loss_zero)
        ctx_tot_shuf += float(loss_shuf)
        ctx_kl_zero  += mean_kl(P0,  Pref)
        ctx_kl_shuf  += mean_kl(Psh, Pref)
        ctx_batches  += 1

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
        true_delta = batch["abs_ts"].clamp(0.0, 1.0)          # [B, T]
        pred_full  = abs_t_pred                               # [B, T+1]  (or [B, T+1, 1])

        # squeeze last dim if the head returns [..., 1]
        if pred_full.dim() == 3 and pred_full.size(-1) == 1:
            pred_full = pred_full.squeeze(-1)                 # -> [B, T+1]

        # align to T by dropping CTX if present
        pred_T = pred_full[:, 1:] if pred_full.size(1) == true_delta.size(1) + 1 else pred_full  # [B, T]

        # reuse nonpad from above (same as tmask)
        tmask = nonpad                                        # [B, T] bool
        pred_T = torch.nan_to_num(pred_T, 0.0, 1.0, 0.0).clamp(0.0, 1.0)

        dt_abs_err += (pred_T[tmask] - true_delta[tmask]).abs().cpu().tolist()
        dt_true_all.append(true_delta[tmask].detach().cpu())
        dt_pred_all.append(pred_T[tmask].detach().cpu())

        # monotonic violations (on aligned [B,T])
        pdiff = pred_T[:, 1:] - pred_T[:, :-1]                # [B, T-1]
        pmask = tmask[:, 1:] & tmask[:, :-1]
        dt_viol_cnt += (pdiff[pmask] < 0).sum().item()
        dt_viol_den += pmask.sum().item()

        # ---- penalties (diagnostic magnitudes)
        # 1. Unclosed Penalty (The new Global Constraint)
        # We calculate this on MASKED logits, just like in training
        _pen_unclosed = soft_unclosed_interval_penalty(
            pred_logits, allowed,
            luts["start_ids_per_base"], luts["end_ids_per_base"]
        )
        pen_unclosed_metric += float(_pen_unclosed.item())

        # 2. Illegal Mass Penalty (The new Local Constraint)
        # We calculate this on UNMASKED logits (logits_pre_mask)
        _pen_illegal = soft_illegal_mass_penalty(
            logits_pre_mask, illegal_mask, nonpad
        )
        pen_illegal_metric += float(_pen_illegal.item())

        # 3. Legacy Breakdown (Optional - to verify redundancy)
        # These should be mostly zero now because of the hard mask
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
            
            # These are the "redundant" terms killed by the mask
            pen_end_no_open += float((p_e * torch.exp(-10.0*open_before)).sum().item())
            pen_dup         += float((p_s * (1.0 - torch.exp(-10.0*open_before))).sum().item())

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

    # --- Context probe aggregates ---
    if ctx_batches > 0:
        ctx_bce_normal = ctx_tot_norm / ctx_batches
        ctx_bce_zero   = ctx_tot_zero / ctx_batches
        ctx_bce_shuf   = ctx_tot_shuf / ctx_batches
        ctx_kl_zero_m  = ctx_kl_zero  / ctx_batches
        ctx_kl_shuf_m  = ctx_kl_shuf  / ctx_batches
    else:
        ctx_bce_normal = ctx_bce_zero = ctx_bce_shuf = 0.0
        ctx_kl_zero_m  = ctx_kl_shuf_m = 0.0

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
    print("[Soft penalties] IllegalMass={:.4f}  Unclosed={:.4f} | (Legacy: end_no_open={:.3f}, dup={:.3f})"
            .format(pen_illegal_metric, pen_unclosed_metric, pen_end_no_open, pen_dup))
    
    print("[CTX] ΔBCE zero-normal={:.4f}  ΔBCE shuffle-normal={:.4f}  "
        "meanKL_zero={:.4f}  meanKL_shuffle={:.4f}"
        .format(ctx_bce_zero - ctx_bce_normal,
                ctx_bce_shuf - ctx_bce_normal,
                ctx_kl_zero_m, ctx_kl_shuf_m))

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
        "ctx_bce_normal": ctx_bce_normal,
        "ctx_bce_zero":   ctx_bce_zero,
        "ctx_bce_shuffle":ctx_bce_shuf,
        "ctx_kl_zero":    ctx_kl_zero_m,
        "ctx_kl_shuffle": ctx_kl_shuf_m,
    }



if __name__ == "__main__":
    """
    Diagnostics suite on a validation slice.

    You'll get:
      1) Transformer training report — BCE breakdown, set-mass (mean/p90), top-k,
         illegal temptation, Δt metrics
      2) Embedder representation report — linear probe + cosine neighbors per RawConcept family
      3) Vocab cleanup report — frequent-noisy & rare-unlearned candidates
      4) (Optional) Token gradient-utility & norm-vs-frequency (if present)
    """
    import torch
    from transform_emr.config.model_config import TRAINING_SETTINGS, EMBEDDER_CHECKPOINT, TRANSFORMER_CHECKPOINT
    from transform_emr.dataset import EMRTokenizer, EMRDataset, get_dataloader, collate_emr
    from transform_emr.embedder import EMREmbedding
    from transform_emr.transformer import GPT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- Load tokenizer / models --
    print("\n[0] Loading tokenizer / embedder / GPT …")
    tok = EMRTokenizer.load()
    embedder, *_ = EMREmbedding.load(EMBEDDER_CHECKPOINT, tokenizer=tok, map_location=device)
    gpt, *_ = GPT.load(TRANSFORMER_CHECKPOINT, embedder=embedder, map_location=device)
    gpt.to(device).eval()
    print("    ✓ Loaded")

    # -- Build validation DataLoader (reuse your training split code) --
    # NOTE: Replace the next block with your actual val loader construction:
    # ds_val = EMRDataset(processed_df_val, context_df_val, tok)
    # val_dl = get_dataloader(ds_val, batch_size=TRAINING_SETTINGS["batch_size"], collate_fn=collate_emr, oversample=False)
    raise SystemExit("Fill in your validation dataloader construction and rerun.")

    # 1) Transformer report
    print("\n[1] Transformer training report — look for BCE breakdown, set-mass (mean/p90), top-k,"
          " illegal mass (pre-mask), and Δt quality (MAE, R², violations).")
    transformer_training_report(
        model=gpt,
        data_loader=val_dl,
        training_settings={
            **TRAINING_SETTINGS,
            "gamma": 1.2, "tau": 0.85, "neg_bounds": (0.02, 0.20), "hard_neg_k": 0,
        },
        max_batches=3,
        device=device,
    )

    # 2) Embedder report (families = RawConcepts)
    print("\n[2] Embedder representation report — PR-AUC ≥ ~0.70 @ horizon suggests healthy learning;"
          " neighbors should cluster within each RawConcept family.")
    embedder_representation_report(
        model=gpt,
        data_loader=val_dl,
        horizon=TRAINING_SETTINGS["bce_k_window"],
        max_batches_probe=2,
        n_families=8,
        device=device,
    )

    # 3) Vocab cleanup
    print("\n[3] Vocab cleanup report — focus on 'frequent_noisy' (merge/bucket/move-to-context)"
          " and 'rare_unlearned' (drop/merge).")
    vocab_cleanup_report(
        model=gpt,
        data_loader=val_dl,
        k_window=TRAINING_SETTINGS["bce_k_window"],
        max_batches=3,
        topn=40,
        device=device
    )

    # 4) (Optional) If you added these helpers, uncomment to run:
    print("\n[4] Token gradient-utility — high grad/occ ⇒ impactful; very low grad/occ with high frequency ⇒ likely noise.")
    token_gradient_utility_report(
        model=gpt, data_loader=val_dl,
        k_window=TRAINING_SETTINGS["bce_k_window"], max_batches=3, device=device
    )
    embed_norm_vs_freq_plot(gpt)  # quick outlier scatter (if present)


