"""
loss.py
==============

Utility module that handles the transformer (phase 2) auxillary tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union


class FocalBCELoss(nn.Module):
    """
    Focal BCE - class-balanced BCE with focal modulation.

    Loss:  –αₜ · (1 - pₜ)^γ · log pₜ,   pₜ = σ(logits) of the ground-truth bit.

    Hyper-params
    ------------
    α / counts :  pass an α-vector directly **or** call `from_counts(counts, …)`
                  to derive it from token frequencies.
    beta       :  smoothing for α; larger β ⇒ rarer tokens get larger weights
                  (default 0.999).
    min_count  :  floor on counts before weighting (default 5).
    clip_max   :  hard cap on α to avoid extreme gradients (default 8.0).
    gamma      :  focal exponent. 0 → plain BCE; 0.5 mild; 1 standard;
                  >1 strongly focuses on hard / rare tokens.
    reduction  :  "none" | "mean" | "sum" (as in PyTorch losses).

    Usage
    -----
    >>> crit = FocalBCELoss.from_counts(token_counts, gamma=0.5).to(device)
    """
    # -------- factory ----------------------------------------------------
    @classmethod
    def from_counts(cls,
                    counts: Union[torch.Tensor, np.ndarray],
                    *,
                    token_weights: Optional[Union[torch.Tensor, np.ndarray]] = None,
                    beta: float = 0.999,
                    min_count: int = 5,
                    clip_max: float = 8.0,
                    gamma: float = 1.0,
                    reduction: str = "none"):
        alpha = cls._calc_alpha(counts, beta=beta,
                                min_count=min_count, clip_max=clip_max)
        if token_weights is not None:
            token_weights = torch.as_tensor(token_weights).float()
            alpha *= token_weights
        return cls(alpha_vector=alpha,
                   gamma=gamma, reduction=reduction)

    # -------- ctor -------------------------------------------------------
    def __init__(self,
                 alpha_vector: torch.Tensor,
                 *,
                 gamma: float = 1.5,
                 reduction: str = "mean",
                 clip_max: Optional[float] = 8.0):
        super().__init__()
        alpha = alpha_vector.float().clone()
        if clip_max is not None:
            alpha = alpha.clamp(max=clip_max)
        self.register_buffer("alpha", alpha)
        self.gamma = gamma
        self.reduction = reduction

    # -------- fwd --------------------------------------------------------
    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        targets = targets.float()

        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none")

        probs   = torch.sigmoid(logits)
        p_t     = probs*targets + (1.0 - probs)*(1.0 - targets)
        mod_fac = (1.0 - p_t).pow(self.gamma)

        alpha = self.alpha.view(*([1]*(logits.ndim - 1)), -1)
        alpha = alpha*targets + (1.0 - alpha)*(1.0 - targets)

        loss = alpha * mod_fac * bce
        if self.reduction == "mean": return loss.mean()
        if self.reduction == "sum":  return loss.sum()
        return loss  # "none"

    # -------- helper -----------------------------------------------------
    @staticmethod
    def _calc_alpha(counts, *, beta, min_count, clip_max):
        counts = torch.as_tensor(counts).float().clamp(min=min_count)
        eff    = 1.0 - torch.pow(beta, counts)
        alpha  = (1.0 - beta) / (eff + 1e-6)
        return alpha.clamp(max=clip_max)
    

class MaskedFocalBCE(nn.Module):
    """
    BCE-with-logits + focal weighting + class alpha, computed per-token,
    but reduced ONLY over (allowed & non-PAD) positions. Uses a pos/neg split
    with a batch-adaptive negative weight so negatives never swamp positives.

    forward(logits, targets, allowed, label_smoothing=0.0, tau=0.5, neg_bounds=(0.05,0.5), hard_neg_k=None)
      - logits: [B,T,V]
      - targets: float multi-hot [B,T,V] (illegal entries can be 0; we'll ignore via 'allowed')
      - allowed: bool mask [B,T,V] (True where a class is legal & timestep is non-PAD)
    Returns: loss (scalar), dict with diagnostics
    """
    def __init__(self, focal_impl: "FocalBCELoss", tau: float = 0.5,
                 neg_bounds=(0.05, 0.5), label_smoothing: float = 0.0,
                 hard_neg_k: Optional[int] = None):
        super().__init__()
        # ensure base focal loss returns per-element tensor
        focal_impl.reduction = "none"
        self.focal = focal_impl
        self.tau = float(tau)
        self.neg_bounds = tuple(neg_bounds)
        self.eps = float(label_smoothing)
        self.hard_neg_k = hard_neg_k

    @classmethod
    def from_counts(cls, counts, token_weights=None, beta=0.999, min_count=5, clip_max=8.0,
                    gamma=1.0, tau=0.5, neg_bounds=(0.05,0.5), label_smoothing=0.0, hard_neg_k=None):
        focal = FocalBCELoss.from_counts(
            counts=counts, token_weights=token_weights, beta=beta,
            min_count=min_count, clip_max=clip_max, gamma=gamma, reduction="none"
        )
        return cls(focal_impl=focal, tau=tau, neg_bounds=neg_bounds,
                   label_smoothing=label_smoothing, hard_neg_k=hard_neg_k)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, allowed: torch.Tensor):
        # targets smoothing (0/1 -> 0/(1-eps))
        if self.eps > 0:
            targets = targets * (1.0 - self.eps)

        # raw focal BCE per element
        raw_bce = self.focal(logits, targets)  # [B,T,V]

        # masks
        allowed = allowed.bool()
        pos_mask = (targets > 0) & allowed                  # [B,T,V]
        neg_mask = (~(targets > 0)) & allowed               # [B,T,V]

        # Optional hard-negative mining (keep top-K hardest negatives per step)
        if self.hard_neg_k is not None and self.hard_neg_k > 0:
            with torch.no_grad():
                neg_logits = logits.masked_fill(~neg_mask, -1e9)
                k = min(self.hard_neg_k, neg_logits.size(-1))
                topk_idx = neg_logits.topk(k=k, dim=-1).indices
                hard_mask = torch.zeros_like(neg_mask)
                hard_mask.scatter_(2, topk_idx, True)
                neg_mask &= hard_mask

        # denominators (avoid /0)
        Np = pos_mask.float().sum().clamp(min=1.0)
        Nn = neg_mask.float().sum().clamp(min=1.0)

        loss_pos = (raw_bce * pos_mask.float()).sum() / Np
        loss_neg = (raw_bce * neg_mask.float()).sum() / Nn

        # batch-adaptive negative weight; clamp for stability
        lambda_neg = (self.tau * (Np / Nn)).clamp(self.neg_bounds[0], self.neg_bounds[1])

        loss = loss_pos + lambda_neg * loss_neg

        info = {
            "loss_pos": float(loss_pos.detach().cpu()),
            "loss_neg": float(loss_neg.detach().cpu()),
            "lambda_neg": float(lambda_neg.detach().cpu()),
            "Np": float(Np.detach().cpu()),
            "Nn": float(Nn.detach().cpu()),
        }
        return loss, info


def pairwise_ranking_loss(
    outcome_logits: torch.Tensor,
    pos_mask: torch.Tensor,
    neg_mask: torch.Tensor,
    max_pos: int = 256,
    max_neg: int = 512,
) -> torch.Tensor:
    """
    Direct AUROC-proxy pairwise ranking loss over outcome-head logits.

    For each outcome k:
      L_k = mean over (pos, neg) pairs of  softplus(logit_neg - logit_pos)
    which is equivalent to BPR / pairwise logistic; minimising it raises the
    probability that a positive position is ranked above a negative one — the
    same quantity `evaluate_on_test_set` pools as AUROC.

    Args:
        outcome_logits ([B, T, K] float): outcome-head raw logits.
        pos_mask       ([B, T, K] bool):  True at positions where outcome k
            occurs within the (training) horizon — the "imminent" positions.
        neg_mask       ([B, T, K] bool):  True at non-pad positions where
            outcome k does not occur within the horizon — the "absent" cohort.
        max_pos (int): per-outcome cap on positives sampled per batch.
        max_neg (int): per-outcome cap on negatives sampled per batch.
            Bounds the dense [Npos, Nneg] pair matrix.

    Returns:
        scalar mean loss across outcomes that have at least one (pos, neg) pair
        available in the current batch. Outcomes with zero positives or zero
        negatives contribute nothing (no gradient, no denominator).
    """
    B, T, K = outcome_logits.shape
    device  = outcome_logits.device
    losses  = []
    for k in range(K):
        logits_k = outcome_logits[..., k].reshape(-1)
        p_idx = pos_mask[..., k].reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        n_idx = neg_mask[..., k].reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        if p_idx.numel() == 0 or n_idx.numel() == 0:
            continue

        if p_idx.numel() > max_pos:
            p_idx = p_idx[torch.randperm(p_idx.numel(), device=device)[:max_pos]]
        if n_idx.numel() > max_neg:
            n_idx = n_idx[torch.randperm(n_idx.numel(), device=device)[:max_neg]]

        pos_logits = logits_k[p_idx]                                   # [Npos]
        neg_logits = logits_k[n_idx]                                   # [Nneg]
        diff = neg_logits.unsqueeze(0) - pos_logits.unsqueeze(1)       # [Npos, Nneg]
        losses.append(F.softplus(diff).mean())

    if not losses:
        return outcome_logits.new_tensor(0.0)
    return torch.stack(losses).mean()


class MaskedSetCE(nn.Module):
    """
    Soft 'set cross-entropy' over the allowed classes where at least 1 target is allowed. 
    Encourages total probability mass on the target set S_t at each step t. 
    CE is an auxiliary “mass-shaper” that helps concentrate probability inside the true future set.
    Same interface as MaskedFocalBCE.
    """
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.eps = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, allowed: torch.Tensor):
        allowed = allowed.bool()                   # [B,T,V]
        tgt = (targets > 0).float()                # [B,T,V]

        # optional smoothing inside the allowed simplex
        if self.eps > 0:
            denom = allowed.float().sum(-1, keepdim=True).clamp_min(1.0)
            smooth = allowed.float() / denom
            tgt = (1.0 - self.eps) * tgt + self.eps * smooth

        # set membership within allowed classes
        in_set = (tgt > 0) & allowed               # [B,T,V]
        pos_steps = in_set.any(dim=-1)             # only steps that have ≥1 legal target

        if not pos_steps.any():
            # return a clean scalar with grad
            return logits.sum() * 0.0, {"denom": 0.0}

        # Use a large negative instead of -inf to keep logsumexp finite
        masked_allowed = logits.masked_fill(~allowed, -1e9)
        masked_in_set  = logits.masked_fill(~in_set,  -1e9)

        logZ   = torch.logsumexp(masked_allowed, dim=-1)          # [B,T], finite
        logSet = torch.logsumexp(masked_in_set,  dim=-1)          # [B,T], finite on pos_steps

        set_ce = (logZ - logSet)                                  # ≥ 0
        set_ce = set_ce.masked_fill(~pos_steps, 0.0)

        denom = pos_steps.float().sum().clamp_min(1.0)
        loss  = set_ce.sum() / denom
        return loss, {"denom": float(denom.item())}