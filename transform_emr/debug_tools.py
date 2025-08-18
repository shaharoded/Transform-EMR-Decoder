import torch
import torch.nn.functional as F
from collections import Counter
from typing import Dict, Any

from utils import (
    get_multi_hot_targets,               # multi-hot next-k targets
    compute_legality_masks_tf,           # legality for BCE masking
    apply_masks_to_logits,               # -inf illegal + bonus
    FocalBCELoss,                        # to access alpha weights
)
# build_luts is where start/end/meal/conflict Luts & forbid list come from
from utils import build_luts

@torch.no_grad()
def summarize_token_weights(tokenizer):
    # Show a small table: counts, alpha, and any manual weights
    crit = FocalBCELoss.from_counts(
        tokenizer.token_counts,          # data-driven counts from tokenizer
        token_weights=tokenizer.token_weights,  # your manual boosts for outcomes, etc.
        gamma=1.0, reduction="none"
    )
    alpha = crit.alpha.cpu()
    counts = tokenizer.token_counts.cpu()
    weights = tokenizer.token_weights.cpu()

    # Print top/rare examples (helpful to confirm extremes)
    top = torch.topk(counts, k=min(10, counts.numel())).indices.tolist()
    rare = torch.topk(-counts, k=min(10, counts.numel())).indices.tolist()

    def rows(ixs):
        out = []
        for i in ixs:
            tok = tokenizer.id2token[i]
            out.append((i, tok, int(counts[i]), float(alpha[i]), float(weights[i])))
        return out

    return {
        "top_by_count": rows(top),
        "rare_by_count": rows(rare),
    }

def _bucket_preds(ids, tokenizer, luts):
    """Small histogram over meaningful buckets for quick reading."""
    is_start = luts["is_start"]; is_end = luts["is_end"]; meal_rank = luts["meal_rank"]
    buckets = {
        "PAD": tokenizer.pad_token_id,
        "MASK": tokenizer.mask_token_id,
        "CTX": getattr(tokenizer, "ctx_token_id", None),
        "NULL": getattr(tokenizer, "null_token_id", None),
    }
    counter = Counter()
    flat = ids.view(-1).cpu()
    for v in flat.tolist():
        if v == buckets["PAD"]: counter["PAD"] += 1
        elif v == buckets["MASK"]: counter["MASK"] += 1
        elif v == buckets["CTX"]: counter["CTX"] += 1
        elif v == buckets["NULL"]: counter["NULL"] += 1
        elif meal_rank[v] >= 0: counter["MEAL"] += 1
        elif is_start[v]: counter["START"] += 1
        elif is_end[v]: counter["END"] += 1
        else: counter["OTHER"] += 1
    return counter

@torch.no_grad()
def inspect_minibatch(model, batch, luts, k_window: int) -> Dict[str, Any]:
    device = next(model.parameters()).device
    tok = model.embedder.tokenizer
    pad = tok.pad_token_id

    # ---- forward (same as train loop)
    logits, _ = model(
        raw_concept_ids=batch["raw_concept_ids"].to(device),
        concept_ids=batch["concept_ids"].to(device),
        value_ids=batch["value_ids"].to(device),
        position_ids=batch["position_ids"].to(device),
        abs_ts=batch["abs_ts"].to(device),
        context_vec=batch["context_vec"].to(device)
    )
    pred_logits = logits[:, 1:, :]                   # [B,T,V]
    target_ids  = batch["targets"].to(device)        # [B,T]

    illegal, bonus = compute_legality_masks_tf(
        target_ids, luts["is_start"], luts["is_end"], luts["base_id"],
        luts["start_ids_per_base"], luts["end_ids_per_base"],
        luts["meal_rank"], luts["meal_pred_rank"], luts["K_meals"],
        luts["conflict_mat"]
    )
    pred_logits = apply_masks_to_logits(pred_logits, illegal, bonus)
    pred_ids = pred_logits.argmax(-1)                # [B,T]

    # ---- targets
    multi_hot = get_multi_hot_targets(
        position_ids=target_ids, padding_idx=pad,
        vocab_size=pred_logits.size(-1), k=k_window
    )

    # ---- statistics
    B, T, V = pred_logits.shape
    allowed = (~illegal) & (target_ids != pad).unsqueeze(-1)  # where BCE should apply
    pos_per_step = multi_hot.sum(-1)                          # [B,T]
    allowed_per_step = allowed.sum(-1)                        # [B,T]

    stats = {
        "fraction_pad_steps": float((target_ids == pad).float().mean().cpu()),
        "mean_allowed_vocab": float(allowed_per_step.float().mean().cpu()),
        "mean_positives_per_step": float(pos_per_step.float().mean().cpu()),
        "frac_zero_positive_steps": float((pos_per_step == 0).float().mean().cpu()),
        "pred_bucket_hist": _bucket_preds(pred_ids, tok, luts),
    }

    # ---- where is BCE mass?
    crit = FocalBCELoss.from_counts(tok.token_counts,
                                    token_weights=tok.token_weights,
                                    gamma=1.0, reduction="none").to(device)
    raw_bce = crit(pred_logits, (multi_hot * 0.95))          # label smoothing like your loop
    masked_bce = (raw_bce * allowed.float())
    stats.update({
        "mean_raw_bce_allV": float(raw_bce.mean().detach().cpu()),
        "mean_bce_allowed_only": float(masked_bce.sum().detach().cpu() /
                                       max(1, allowed.sum().item())),
        "share_loss_from_allowed":
            float(masked_bce.sum().detach().cpu() / (raw_bce.sum().detach().cpu() + 1e-9)),
    })
    # Class-level multi-hot prevalence in this batch
    stats["batch_target_sum_per_class_top"] = [
        (int(i), tok.id2token[int(i)], int(s))
        for (s, i) in sorted(
            [(int(multi_hot.sum(dim=(0,1))[i].item()), i) for i in range(V)],
            reverse=True
        )[:15]
    ]
    return stats

@torch.no_grad()
def inspect_epoch(model, loader, k_window: int, max_batches: int = 3):
    device = next(model.parameters()).device
    tok = model.embedder.tokenizer
    luts = build_luts(tok)
    luts = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in luts.items()}

    print("=== Token weights / counts snapshot ===")
    w = summarize_token_weights(tok)
    print("Top by count:", w["top_by_count"][:5])
    print("Rare by count:", w["rare_by_count"][:5])

    for bi, batch in enumerate(loader):
        if bi >= max_batches: break
        stats = inspect_minibatch(model, batch, luts, k_window)
        print(f"\n[Batch {bi}]")
        for k, v in stats.items():
            print(f"  {k}: {v}")