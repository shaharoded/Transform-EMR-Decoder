"""
utils.py
==============

General util functions for the package
"""
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Optional, Union

# ───────── local code ─────────────────────────────────────────────────── #
from transform_emr.config.dataset_config import (
    ADMISSION_TOKEN, TERMINAL_OUTCOMES, OUTCOMES, MEAL_TOKENS
)


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
                    reduction: str = "mean"):
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
            alpha.clamp_(max=clip_max)
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


class F1Aggregator:
    """
    Epoch-level micro-F1 aggregator accross batches.

        0 = RELEASE
        1 = DEATH
        2.. = individual complication tokens

    • TP_c = min(pred_count_c , true_count_c)
    • FP_c = max(pred_count_c - true_count_c , 0)
    • FN_c = max(true_count_c - pred_count_c , 0)
    """
    def __init__(self, tokenizer, device="cpu"):
        self.tok = tokenizer
        self.dev = device

        # -- id look‑ups -----------------------------------------------------
        self.rel_id = tokenizer.token2id.get("RELEASE")
        self.dth_id = tokenizer.token2id.get("DEATH")

        self.cmp_tokens = [t for t in OUTCOMES
                           if t not in ("RELEASE", "DEATH") and t in tokenizer.token2id]

        self.cmp_ids = torch.tensor([tokenizer.token2id[t] for t in self.cmp_tokens],
                                    device=device)           # [C]

        n_slots = 2 + len(self.cmp_ids)                      # 2 terminals + C complications
        self.tp = torch.zeros(n_slots, device=device)
        self.fp = torch.zeros(n_slots, device=device)
        self.fn = torch.zeros(n_slots, device=device)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _counts(self, ids):
        """ids: tensor [...]; returns tensor [2+C] counts per class."""
        flat = ids.view(-1)

        n_rel = (flat == self.rel_id).sum() if self.rel_id is not None else 0
        n_dth = (flat == self.dth_id).sum() if self.dth_id is not None else 0

        # vectorised counts per complication token
        n_cmp = (flat.unsqueeze(1) == self.cmp_ids).sum(dim=0).float()  # [C]

        return torch.cat([n_rel.unsqueeze(0),
                          n_dth.unsqueeze(0),
                          n_cmp]).to(torch.float32)                     # [2+C]

    # ------------------------------------------------------------------
    def update(self, pred_ids, tgt_ids):
        p = self._counts(pred_ids)
        t = self._counts(tgt_ids)

        self.tp += torch.min(p, t)
        self.fp += torch.clamp(p - t, min=0)
        self.fn += torch.clamp(t - p, min=0)

    # ------------------------------------------------------------------
    def compute(self):
        # indices 0 and 1 are REL / DTH
        scores = {}
        for idx, lbl in enumerate(("REL", "DTH")):
            denom = 2 * self.tp[idx] + self.fp[idx] + self.fn[idx]
            scores[lbl] = None if denom == 0 else (2 * self.tp[idx] / denom).item()

        # aggregate micro counts over all complication tokens (idx ≥2)
        tp_cmp = self.tp[2:].sum()
        fp_cmp = self.fp[2:].sum()
        fn_cmp = self.fn[2:].sum()
        denom  = 2 * tp_cmp + fp_cmp + fn_cmp
        scores["CMP"] = None if denom == 0 else (2 * tp_cmp / denom).item()

        return scores

    # ------------------------------------------------------------------
    def reset(self):
        self.tp.zero_(); self.fp.zero_(); self.fn.zero_()
    

def get_multi_hot_targets(position_ids: torch.Tensor,
                          padding_idx: int,
                          vocab_size: int,
                          k: int) -> torch.Tensor:
    """
    For each timestep t, mark all tokens that appear in positions [t+1, t+k]
    with a multi-hot vector (0/1) over the vocabulary.

    Parameters
    ----------
    position_ids : LongTensor, shape [B, T]
        Token ids, right-padded with `padding_idx`.
    padding_idx  : int
        The pad token id. It will be excluded from the targets.
    vocab_size   : int
        Size of the vocabulary.
    k            : int
        Lookahead window size.

    Returns
    -------
    targets : FloatTensor, shape [B, T, V]
        Multi-hot matrix. targets[b, t, v] == 1 if token v appears in
        any of the next k positions after t (inclusive of t+1, exclusive of t).
    """
    B, T = position_ids.shape
    device = position_ids.device

    # One‐hot: [B, T, V]
    oh = F.one_hot(position_ids.clamp(min=0), num_classes=vocab_size).to(torch.float32)
    csum = oh.cumsum(dim=1)  # [B, T, V]

    # Build csum_k by shifting left k, then for tail positions repeat the last csum row
    # 1) core = csum[:, k:] gives shape [B, T–k, V]
    core   = csum[:, k:]                # sums through position k..T–1
    # 2) tail = last csum vector repeated k times
    last   = csum[:, -1:]               # shape [B, 1, V]
    tail   = last.repeat(1, k, 1)       # shape [B, k, V]
    # 3) concat back to length T
    csum_k = torch.cat([core, tail], dim=1)  # [B, T, V]

    # future counts in (t+1 .. t+k] = csum_k – csum
    future = (csum_k - csum).clamp_min_(0.0)

    # never supervise PAD
    if 0 <= padding_idx < vocab_size:
        future[..., padding_idx] = 0.0

    return (future > 0).to(torch.float32)


def build_mlm(ids, tokenizer, p=0.15):
    """
    Embedder MLM Mask Helper.
    Logic: Mask (later predict) tokens important to position that won't hurt the general timeline.
            Mask everything informative but admission, CTX or terminal tokens.
    ids:  (B,T) tensor - raw_concept_ids / concept_ids / value_ids / position_ids
    returns masked_ids (same shape)
    Masking strategy (BERT-style):
        80% → replace with [MASK]
        10% → keep original
        10% → replace with random token (not PAD)
    """
    device = ids.device
    never_mask_ids = {
        tokenizer.pad_token_id,
        tokenizer.ctx_token_id,
        tokenizer.null_token_id,
        tokenizer.token2id.get(ADMISSION_TOKEN),
        *[tokenizer.token2id[tok] for tok in TERMINAL_OUTCOMES],
    }
    keep = torch.zeros_like(ids, dtype=torch.bool)
    for tid in never_mask_ids:
        if tid is not None:
            keep |= (ids == tid)
    mask = (~keep) & (torch.rand_like(ids.float()) < p)
    masked = ids.clone()
    # 80 %
    rand = torch.rand_like(ids.float())
    mask80 = mask & (rand < 0.8)
    masked[mask80] = tokenizer.mask_token_id
    # 10 % random token
    mask10 = mask & (rand >= 0.8) & (rand < 0.9)
    vocab_size = len(tokenizer.token2id)
    random_tokens = torch.randint(1, vocab_size, size=ids.shape, device=device)
    masked[mask10] = random_tokens[mask10]
    # 10 % keep original – already satisfied
    return masked, mask        # mask is the bool tensor of *predict‑me* positions


def set_embedder_frozen(model, freeze: bool):
    for p in model.embedder.parameters():
        p.requires_grad = not freeze
    model.embedder.eval() if freeze else model.embedder.train()
    

def linear_schedule(epoch: int, warmup: int, max_val: float) -> float:
    """Simple linear ramp from 0→max_val over `warmup` epochs."""
    return max_val * min(epoch / warmup, 1.0)


def apply_cbm(batch, epoch, warmup_epochs, tokenizer, forbid_ids, max_p=0.25):
    """
    Transformer CBM (Curriculum by Masking) Helper.
    Logic: Mask tokens that won't hurt the general timeline or conflict with penalties.
        This list adds to the MLM forbidden list, as now we can't mask meals / intervals, 
        as we'll teach the model the contradicting them is OK.
    Randomly masks ratio% of *input* tokens (excluding forbid_ids list),
    replacing them with [MASK] token id and corresponding sub-ids.

    batch: dict of tensors
    tokenizer: EMRTokenizer
    epoch: int, epoch number
    warmup_epochs: int, total number of warmup epochs from training_config
    forbid_ids: LongTensor of ids that must never be masked (PAD, CTX, ADMISSION, TERMINALS...)
    max_ratio: float, max masking ratio of the input

    NOTE: This basically means that masked tokens are marked as acceptable noise. 
    """    
    p = linear_schedule(epoch, warmup_epochs, max_p)

    pos_ids = batch["position_ids"]
    device = pos_ids.device
    B, T = pos_ids.shape

    mask_tok = tokenizer.mask_token_id
    pad_id   = tokenizer.pad_token_id

    # Eligible positions: not in forbid list and not PAD
    forbid = torch.zeros(tokenizer.token_weights.numel(), dtype=torch.bool, device=device)
    forbid[pad_id] = True
    forbid[mask_tok] = True
    forbid[forbid_ids] = True

    eligible = ~forbid[pos_ids]
    # Sample exactly p * (#eligible) positions ---
    # total eligible count
    E = int(eligible.sum().item())
    # how many to mask
    N = int(round(p * E))
    if N > 0:
        # Flattened indices of all eligible slots
        # idx is shape [E, 2] of (batch_idx, time_idx)
        idx = eligible.nonzero(as_tuple=False)
        # shuffle and pick first N
        perm = torch.randperm(E, device=device)[:N]
        pick = idx[perm]              # shape [N,2]
        to_mask = torch.zeros_like(eligible)
        to_mask[pick[:,0], pick[:,1]] = True
    else:
        to_mask = torch.zeros_like(eligible)

    # random 80/10/10 (like BERT)?
    # Keep it simple: replace all with [MASK]
    pos_ids = pos_ids.clone()
    raw_ids = batch["raw_concept_ids"].clone()
    con_ids = batch["concept_ids"].clone()
    val_ids = batch["value_ids"].clone()

    pos_ids[to_mask] = mask_tok
    raw_ids[to_mask] = mask_tok
    con_ids[to_mask] = mask_tok
    val_ids[to_mask] = mask_tok

    batch["position_ids"]    = pos_ids
    batch["raw_concept_ids"] = raw_ids
    batch["concept_ids"]     = con_ids
    batch["value_ids"]       = val_ids
    return batch


def mix_with_predictions(
        gt_ids: torch.LongTensor,
        pred_ids: torch.LongTensor,
        epoch: int,
        warmup_epochs: int,
        protected_ids: torch.BoolTensor,
        max_rate: float=0.3
    ) -> tuple[torch.LongTensor, torch.BoolTensor]:
    """
    Utility to mix ground-truth and predicted tokens in-batch,
    while never replacing protected token IDs (e.g., START/END,
    MEAL, OUTCOME, NULL, PAD, ADMISSION, CTX).
    Will increase replacement ratio during warmup.

    Args:
      gt_ids        : [B, T] LongTensor of ground-truth token IDs
      pred_ids      : [B, T] LongTensor of argmax predictions
      epoch         : int [0, max_epochs], Current epochs in training process
      warmup_epochs : int, Number of warmup epochs in training process
      protected_ids : [V] BoolTensor, True for tokens to keep from GT
      max_rate      : float, Max ratio of real predictions in Teacher's Forcing batch.


    Returns:
      mixed_ids : [B, T] LongTensor
      mix_mask  : [B, T] BoolTensor, True where pred_ids replaced GT
    """    
    device = gt_ids.device
    B, T = gt_ids.shape
    ss_rate = linear_schedule(epoch, warmup_epochs, max_rate)

    # 1) Random swap mask
    rand_mask = torch.rand(B, T, device=device) < ss_rate
    # 2) Build safe-to-mix mask
    safe_mask = ~protected_ids[gt_ids]
    # 3) Final mix mask
    mix_mask = rand_mask & safe_mask
    mixed_ids = torch.where(mix_mask, pred_ids, gt_ids)
    return mixed_ids, mix_mask


def build_luts(tokenizer):
    """
    Pre-compute all LUTs (lookup tensors) needed for:
      • legality masks (intervals + meals + value-conflict)
      • CBM masking forbid list
      • Inference legality

    Returns
    -------
    luts : dict
        {
        # per-token
        "is_start"          : Bool[V]
        "is_end"            : Bool[V]
        "base_id"           : Long[V]   (-1 if not interval token)
        "meal_rank"         : Long[V]   (-1 non-meal, else 0..K-1)
        "meal_pred_rank"    : Long[V]   (-1 non-meal)

        # per-base (nb = #interval bases)
        "start_ids_per_base": Long[nb]  id of *_START  (-1 if missing)
        "end_ids_per_base"  : Long[nb]  id of *_END    (-1 if missing)
        "conflict_mat"      : Bool[nb, nb]  (same concept & different value)

        # misc
        "start_ids"         : Long[*]   all start ids (unordered)
        "end_ids"           : Long[*]   all end ids   (unordered)
        "K_meals"           : Long[]    scalar
        "forbid_mask_ids"   : Long[*]   tokens we never CBM-mask
        }
    """
    V = len(tokenizer.token2id)
    device = torch.device("cpu")  # keep CPU; move to GPU later where needed

    # --- per-token LUTs ------------------------------------------------------
    is_start   = torch.zeros(V, dtype=torch.bool, device=device)
    is_end     = torch.zeros(V, dtype=torch.bool, device=device)
    base_id    = torch.full((V,), -1, dtype=torch.long, device=device)
    tok2concept= torch.full((V,), -1, dtype=torch.long, device=device)
    tok2value  = torch.full((V,), -1, dtype=torch.long, device=device)

    # ---------- detect START/END tokens, map bases, and fill per‑token LUTs ----------
    base2idx = {}
    start_ids_list, end_ids_list = [], []

    for tok, tid in tokenizer.token2id.items():
        # Strip ONLY the suffix for interval tokens
        if tok.endswith("_START"):
            core = tok[:-6]
            is_start[tid] = True
        elif tok.endswith("_END"):
            core = tok[:-4]
            is_end[tid] = True
        else:
            core = tok

        # ----- concept & value ids -----
        parts = core.split("_") # A_STATE_High, A_TREND_Dec, events
        # concept  = everything except the final value segment
        #           e.g.  A_STATE_Low   ->  A_STATE
        #                 A_TREND_inc   ->  A_TREND
        concept_key = "_".join(parts[:-1])
        value_key   = core # Will also represent events, contexts

        # Use position-id on V to mark the unifying mark of that hierarchy
        tok2concept[tid] = tokenizer.concept2id.get(concept_key, -1)
        tok2value[tid]   = tokenizer.value2id.get(value_key,   -1)

        # ----- interval base bookkeeping -----
        if is_start[tid] or is_end[tid]:
            base_idx = base2idx.setdefault(core, len(base2idx))
            base_id[tid] = base_idx
            if is_start[tid]:
                start_ids_list.append(tid)
            else:  # END
                end_ids_list.append(tid)

    # tensors of all *_START / *_END ids (unordered)
    start_ids = torch.tensor(start_ids_list, dtype=torch.long, device=device)
    end_ids   = torch.tensor(end_ids_list,   dtype=torch.long, device=device)

    # ---------- per‑base LUTs ----------
    nb = len(base2idx)
    start_ids_per_base = torch.full((nb,), -1, dtype=torch.long, device=device)
    end_ids_per_base   = torch.full((nb,), -1, dtype=torch.long, device=device)
    base_concept       = torch.full((nb,), -1, dtype=torch.long, device=device)
    base_value         = torch.full((nb,), -1, dtype=torch.long, device=device)

    # Fill per‑base arrays (first seen wins; START/END of same base share concept/value)
    for tid in range(V):
        b = base_id[tid].item()
        if b < 0:
            continue
        if is_start[tid]:
            start_ids_per_base[b] = tid
        elif is_end[tid]:
            end_ids_per_base[b] = tid
        if base_concept[b] < 0:
            base_concept[b] = tok2concept[tid]
        if base_value[b] < 0:
            base_value[b] = tok2value[tid]

    # ---------- conflict matrix: same concept, different value ----------
    if nb > 0:
        conf_mat = (base_concept[:, None] == base_concept[None, :]) & \
                (base_value[:,  None]  != base_value[None,  :])
    else:
        conf_mat = torch.zeros(0, 0, dtype=torch.bool, device=device)

    # --- meals ---------------------------------------------------------------
    meal_rank = torch.full((V,), -1, dtype=torch.long, device=device)
    for r, name in enumerate(MEAL_TOKENS):
        tid = tokenizer.token2id.get(name)
        if tid is not None:
            meal_rank[tid] = r
    K = int(meal_rank.max().item()) + 1 if (meal_rank >= 0).any() else 0

    meal_pred_rank = torch.full((V,), -1, dtype=torch.long, device=device)
    if K > 0:
        meal_mask = meal_rank >= 0
        meal_pred_rank[meal_mask] = (meal_rank[meal_mask] - 1) % K

    # ---- forbid list for CBM ----
    forbid = {
        tokenizer.pad_token_id,
        getattr(tokenizer, "ctx_token_id", None),
        getattr(tokenizer, "null_token_id", None),
        tokenizer.token2id.get(ADMISSION_TOKEN),
        *[tokenizer.token2id.get(t) for t in TERMINAL_OUTCOMES],
        *[tokenizer.token2id.get(t) for t in OUTCOMES],
        *[tokenizer.token2id.get(t) for t in MEAL_TOKENS],
        *start_ids.tolist(),
        *end_ids.tolist(),
    }
    forbid_mask_ids = torch.tensor([tid for tid in forbid if tid is not None],
                                   dtype=torch.long)
    
    # ---- forbid list for Decoder ----
    forbid = {
        tokenizer.pad_token_id,
        tokenizer.mask_token_id,
        getattr(tokenizer, "ctx_token_id", None),
        tokenizer.token2id.get(ADMISSION_TOKEN),
    }
    predict_block = torch.tensor([tid for tid in forbid if tid is not None],
                                   dtype=torch.long)

    return {
        # per-token
        "is_start": is_start,
        "is_end":   is_end,
        "base_id":  base_id,
        "meal_rank":      meal_rank,
        "meal_pred_rank": meal_pred_rank,

        # per-base
        "start_ids_per_base": start_ids_per_base,
        "end_ids_per_base":   end_ids_per_base,
        "conflict_mat":       conf_mat,

        # misc
        "start_ids": start_ids,
        "end_ids":   end_ids,
        "K_meals":   torch.tensor(K, dtype=torch.long, device=device),
        "forbid_mask_ids": forbid_mask_ids,
        "predict_block": predict_block
    }


def compute_legality_masks_tf(position_ids: torch.LongTensor,
                              is_start: torch.BoolTensor,
                              is_end:   torch.BoolTensor,
                              base_id:  torch.LongTensor,
                              start_ids_per_base: torch.LongTensor,
                              end_ids_per_base:   torch.LongTensor,
                              meal_rank: torch.LongTensor,
                              meal_pred_rank: torch.LongTensor,
                              K_meals: torch.Tensor,
                              conflict_mat: torch.BoolTensor,
                              predict_block: torch.BoolTensor):
    """
    Vectorized legality/bonus masks from GOLD prefix (teacher forcing).

    illegal[B,T,V]  True → forbid v at step t
    bonus  [B,T,V]  True → boost v at step t

    Terms:
    If token== 'GLUCOSE_TREND_inc_START', base(tok) == 'GLUCOSE_TREND_inc'
    If token== 'GLUCOSE_TREND_inc_START', concept(tok) == 'GLUCOSE_TREND'

    Interval logic (per base):
      • END is illegal if base not open yet (You can't see 'GLUCOSE_TREND_inc_END' before opening 
      it 'GLUCOSE_TREND_inc_START'). Enforced using base(tok).
      • START illegal if base already open 
      (You can't have 'GLUCOSE_TREND_inc_START' after 'GLUCOSE_TREND_inc_START' without seeing
        'GLUCOSE_TREND_inc_END'). Enforced using base(tok).
      • END bonus  if base open (to push the model to close intervals). Enforced using base(tok).
      • START of concept(tok) is illegal if concept(tok) is still open 
        (meaning you can't have 'GLUCOSE_TREND_inc_START' after 'GLUCOSE_TREND_dec_START' without seeing
        'GLUCOSE_TREND_inc_END'). Enforced using concept(tok).

    Meal logic:
      cyclic order; meal m illegal if predecessor rank not seen yet, bonus if seen. 
      The first meal is never illegal, only the following ones.

    All done without loops over T (only broadcast/cumsums).

    position_ids : [B,T]
    """
    device = position_ids.device
    B, T = position_ids.shape
    V    = is_start.numel()
    nb   = start_ids_per_base.numel()

    # Map tokens → base / start / end
    tok_base = base_id[position_ids]    # [B,T]
    tok_s    = is_start[position_ids]   # [B,T]
    tok_e    = is_end[position_ids]     # [B,T]

    # Build one-hots over bases
    valid = tok_base >= 0
    scatter_idx = tok_base.clone()
    scatter_idx[~valid] = 0
    start_oh = torch.zeros(B, T, nb, device=device, dtype=torch.int16)
    end_oh   = start_oh.clone()
    b_idx = torch.arange(B, device=device)[:,None]
    t_idx = torch.arange(T, device=device)[None,:]
    start_oh[b_idx, t_idx, scatter_idx] = (tok_s & valid).to(start_oh.dtype)
    end_oh  [b_idx, t_idx, scatter_idx] = (tok_e & valid).to(end_oh.dtype)

    # Cumulative sums to know "open count" at each t
    starts_cum = start_oh.cumsum(dim=1)
    ends_cum   = end_oh.cumsum(dim=1)

    # Build "open state before t" by shifting right
    prev_s = torch.zeros_like(starts_cum); prev_s[:,1:,:] = starts_cum[:,:-1,:]
    prev_e = torch.zeros_like(ends_cum);   prev_e[:,1:,:] = ends_cum[:,:-1,:]
    open_before = (prev_s - prev_e) > 0  # [B,T,nb]

    # Prepare illegal / bonus matrices
    illegal = torch.zeros(B, T, V, device=device, dtype=torch.bool)
    bonus   = illegal.clone()

    # 1) END rules: illegal if not open_before, bonus if open_before
    end_ids = end_ids_per_base.view(1,1,nb).expand(B,T,nb)
    illegal.scatter_(2, end_ids, ~open_before)
    bonus.  scatter_(2, end_ids,  open_before)

    # 2) DUP-START: START when *that same* base was already open should be illegal
    #    clamp base_id to ≥0 so gather never errors
    base_idxs = tok_base.clamp(min=0).unsqueeze(-1)   # [B,T,1]
    was_open  = open_before.gather(2, base_idxs).squeeze(-1)  # [B,T]
    dup       = tok_s & (tok_base >= 0) & was_open
    if dup.any():
        b_d, t_d    = dup.nonzero(as_tuple=True)
        start_toks  = position_ids[b_d, t_d]    # the token‐IDs of those STARTs
        illegal[b_d, t_d, start_toks] = True

    # 3) CNF (conflicts): if any other value of same concept was open_before
    if nb > 0 and conflict_mat.any():
        # 3) CNF (conflicts): START of a value is illegal if any conflicting base is already open
        oc = open_before.to(torch.float32)                   # [B,T,nb]
        cm = conflict_mat.to(torch.float32).T                # [nb,nb]
        conflict_active = (oc @ cm) > 0                      # [B,T,nb]

        # only apply to actual START tokens of each base
        conflict_active &= start_oh.bool()                  # [B,T,nb]

        # OR‑add these into the illegal mask at the corresponding START token IDs
        ids = start_ids_per_base                           # shape [nb]
        illegal[:, :, ids] |= conflict_active              # in‑place OR

    # --- Meal logic: allow starting anywhere, then enforce absolute cycle order ---
    if K_meals > 0:
        # 1) build a one‑hot of *where* meals occur in the GOLD prefix
        mr   = meal_rank[position_ids]             # [B,T]
        mask = mr >= 0
        meal_oh = torch.zeros(B, T, K_meals, device=device, dtype=torch.bool)
        idx = mask.nonzero(as_tuple=False)
        if idx.numel():
            b_i, t_i = idx[:,0], idx[:,1]
            meal_oh[b_i, t_i, mr[mask]] = True

        # 2) compute “prefix_seen”: which ranks have appeared *before* time t
        #    (we shift the cumsum right by one)
        cum        = meal_oh.cumsum(dim=1) > 0          # [B,T,K_meals]
        prefix_seen = cum.clone()
        prefix_seen[:,1:,:] = cum[:,:-1,:]
        prefix_seen[:,0,:] = False

        # 3) decide legality:
        #    • if *no* meal has ever been seen in the prefix → any meal is OK
        #    • else → only the immediate successor (pred_rank) is OK
        mtok       = meal_rank >= 0                     # [V]
        if mtok.any():
            v_ids       = mtok.nonzero(as_tuple=False).squeeze(-1)  # meal token IDs
            pred_ranks  = meal_pred_rank[ mtok ]                  # [nv]
            # a) predecessor rule
            ok_pred     = prefix_seen[:,:,pred_ranks]             # [B,T,nv]
            # b) free‐pass if absolutely no meal seen yet
            any_seen    = prefix_seen.any(dim=2, keepdim=True)    # [B,T,1]
            ok_initial  = ~any_seen                                 # [B,T,1]
            ok          = ok_pred | ok_initial                     # [B,T,nv]

            # 4) apply to illegal/bonus
            illegal[:,:,v_ids] |= ~ok
            bonus  [:,:,v_ids] |=  ok
    
    # ---- Specials never to be predicted ----
    illegal |= predict_block.view(1, 1, -1)   # broadcast [V] -> [B,T,V]

    return illegal, bonus


def penalty_interval_structure(
    pred_ids: torch.LongTensor,
    gt_ids:   torch.LongTensor,
    is_start:             torch.BoolTensor,
    is_end:               torch.BoolTensor,
    base_id:              torch.LongTensor,
    start_ids_per_base:   torch.LongTensor,
    end_ids_per_base:     torch.LongTensor,
    meal_rank:            torch.LongTensor,
    meal_pred_rank:       torch.LongTensor,
    K_meals:              torch.Tensor,
    conflict_mat:         torch.BoolTensor,
    window:               int = 5,
) -> torch.Tensor:
    """
    Computes a structural-violation penalty for interval tokens, covering:
      FSM : END without an open START
      DUP : START when same base already open
      UNC : START never closed by END (sequence ended)
      CNF : START while a conflicting base is open (same concept, different value)

    Returns a scalar ∈ [0,1]:
      (new_timestep_violations + new_unc_violations) /
      (interval_token_count + batch_size)

    Forgiveness: if a violation of the same token_id and type
    occurs in the GT sequence within ±`window` time steps,
    it is forgiven (not counted as new). Unclosed (UNC) violations
    are forgiven if the same base remains unclosed in GT.

    Args:
      pred_ids, gt_ids: [B, T] LongTensor
      is_start, is_end, base_id: [V] Bool/Long tensors
      conflict_mat: [nb, nb] Bool
      window: integer radius for forgiving GT violations around each t
    """
    B, T = pred_ids.shape

    # 1) Compute illegal masks for pred & GT
    illegal_pred, _ = compute_legality_masks_tf(
        pred_ids, is_start, is_end,
        base_id, start_ids_per_base, end_ids_per_base,
        meal_rank, meal_pred_rank, K_meals, conflict_mat
    )
    illegal_gt, _ = compute_legality_masks_tf(
        gt_ids,   is_start, is_end,
        base_id, start_ids_per_base, end_ids_per_base,
        meal_rank, meal_pred_rank, K_meals, conflict_mat
    )

    # Gather illegal flags per token
    # 2a) Gather illegal flags per token for pred **and** GT
    pred_illegal = illegal_pred.gather(2, pred_ids.unsqueeze(-1)).squeeze(-1)  # [B,T]
    gt_illegal   = illegal_gt.  gather(2, gt_ids.  unsqueeze(-1)).squeeze(-1)  # [B,T]

    # 2) Forgiveness window over GT
    gt_win = F.max_pool1d(
        gt_illegal.float().unsqueeze(1),
        kernel_size=2*window+1, stride=1, padding=window
    ).squeeze(1).bool()  # [B,T]

    # 3) New-timestep violations
    new_ts_viol = pred_illegal & (~gt_win)
    count_ts_viol = new_ts_viol.float().sum()

    # 4) Unclosed (UNC) violations: count residual opens
    # Map tokens → base indices
    base_idx = base_id[pred_ids]  # [B,T]
    # Masks where start/end occurred and base_idx >=0
    start_mask = is_start[pred_ids] & (base_idx >= 0)
    end_mask   = is_end  [pred_ids] & (base_idx >= 0)

    nb = conflict_mat.shape[0]
    # Count starts/ends via scatter_add
    starts_pred = torch.zeros(B, nb, device=pred_ids.device, dtype=torch.float)
    ends_pred   = torch.zeros(B, nb, device=pred_ids.device, dtype=torch.float)
    starts_gt   = torch.zeros(B, nb, device=pred_ids.device, dtype=torch.float)
    ends_gt     = torch.zeros(B, nb, device=pred_ids.device, dtype=torch.float)

    # Predicted
    if start_mask.any():
        idxs = base_idx[start_mask]
        b_idxs, t_idxs = (start_mask).nonzero(as_tuple=True)
        # scatter per batch
        for b, tid in zip(b_idxs.tolist(), idxs.tolist()):
            starts_pred[b, tid] += 1.0
    if end_mask.any():
        idxs = base_idx[end_mask]
        b_idxs, t_idxs = (end_mask).nonzero(as_tuple=True)
        for b, tid in zip(b_idxs.tolist(), idxs.tolist()):
            ends_pred[b, tid] += 1.0
    # GT
    base_idx_gt = base_id[gt_ids]
    mask_s_gt = is_start[gt_ids] & (base_idx_gt >= 0)
    mask_e_gt = is_end  [gt_ids] & (base_idx_gt >= 0)
    if mask_s_gt.any():
        idxs = base_idx_gt[mask_s_gt]
        b_idxs, t_idxs = mask_s_gt.nonzero(as_tuple=True)
        for b, tid in zip(b_idxs.tolist(), idxs.tolist()):
            starts_gt[b, tid] += 1.0
    if mask_e_gt.any():
        idxs = base_idx_gt[mask_e_gt]
        b_idxs, t_idxs = mask_e_gt.nonzero(as_tuple=True)
        for b, tid in zip(b_idxs.tolist(), idxs.tolist()):
            ends_gt[b, tid] += 1.0

    open_pred = (starts_pred - ends_pred) > 0  # [B, nb]
    open_gt   = (starts_gt   - ends_gt)   > 0  # [B, nb]
    new_unc_viol = (open_pred & (~open_gt)).float().sum()

    # 5) Normalize by interval tokens + batch
    denom = ((base_idx >= 0).float().sum() + B).clamp_min(1.0)
    penalty = (count_ts_viol + new_unc_viol) / denom
    return penalty

def penalty_meal_order(pred_ids:  torch.LongTensor, meal_rank: torch.LongTensor) -> torch.Tensor:
    """
    Enforces cyclic order among meals: B→L→D→N→B...
    We ignore non-meal tokens (they can appear anywhere in between).

    Implementation:
      1. Extract only meal ids per sequence (mask ranks>=0).
      2. Compare each consecutive pair in that filtered list.

    pred_ids  : [B,T]
    meal_rank : [V]  (-1 non-meal, else 0..K-1)

    return scalar ∈ [0,1] = (#wrong meal transitions)/(#meal transitions)
    """

    """
    Computes a cyclic-meal-order penalty: B→L→D→N→B...
    Returns a scalar ∈ [0,1]: the fraction of adjacent meal transitions
    in the predicted sequence that violate the expected cycle.

    Steps:
      1. Map pred_ids → ranks (with -1 for non-meal tokens).
      2. Compress each batch sequence into its meal-only sequence M of shape [B, max_meals].
      3. Compute expected next-meal rank = (M[:, :-1] + 1) % K.
      4. Count wrong transitions (curr != expected) and normalize by total transitions.

    Fully batched, GPU-friendly, and differentiable.
    """
    device = pred_ids.device
    ranks = meal_rank[pred_ids]  # [B,T]
    B, T = ranks.shape
    mask = ranks >= 0            # [B,T]
    if not mask.any():
        return torch.tensor(0.0, device=device)

    # 1. Identify meal positions per sequence
    mask_int = mask.int()
    meal_idx = torch.cumsum(mask_int, dim=1)  # [B, T]
    meal_counts = meal_idx[:, -1]             # [B]
    max_meals = int(meal_counts.max().item())

    # 2. Scatter predicted ranks into a compact [B,max_meals] matrix
    b_t = mask.nonzero(as_tuple=False)        # [N,2] of (b, t)
    b_idx, t_idx = b_t[:, 0], b_t[:, 1]
    k_idx = meal_idx[b_idx, t_idx] - 1       # [N]

    M = torch.full((B, max_meals), -1, dtype=ranks.dtype, device=device)
    flat_M = M.view(-1)
    flat_idx = b_idx * max_meals + k_idx
    flat_M[flat_idx] = ranks[b_idx, t_idx]
    M = flat_M.view(B, max_meals)

    # 3. Compute transitions
    prev = M[:, :-1]
    curr = M[:, 1:]
    valid = (prev >= 0) & (curr >= 0)
    K = int(meal_rank.max().item()) + 1
    expected = (prev + 1) % K

    wrong = ((curr != expected) & valid).float().sum()
    total = valid.float().sum().clamp_min(1.0)
    return wrong / total



def apply_masks_to_logits(logits, illegal_mask, bonus_mask, bonus_boost=0.2):
    """
    logits: [B,T,V] *after* slicing (no [CTX])
    illegal_mask/bonus_mask: Bool [B,T,V]
    We add/subtract constants (broadcast).

    NOTE: Logits mask uses large negative number to avoid nan in BCE loss.
          Illegal targets must also be masked to avoid large backpropagations.
    """
    # illegal → -inf
    logits = logits.masked_fill(illegal_mask, -1e9)
    # bonus → add small boost
    if bonus_boost > 0:
        logits = logits + bonus_boost * bonus_mask.float()
    return logits


def plot_losses(train_losses, val_losses):
    """
    Plot train vs. validation loss to inspect training quality.
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross‑entropy loss")
    plt.title("Training vs. validation loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def build_rep_penalty(last_tokens, V, window=5, strength=0.6, device=None):
    """
    Soft repetition discourager on inference.
    last_tokens : list[int]   (already generated, newest at the end)
    V           : vocab size
    window      : how many recent tokens we look back
    strength    : scalar multiplier for the penalty (0..1 typical)
    Returns:
        rep_vec : [V] float tensor, 0 for unseen, higher for very recent repeats
    """
    if not last_tokens or strength <= 0:
        return torch.zeros(V, device=device)
    device = device or torch.device("cpu")
    k = min(window, len(last_tokens))
    # decay weights: newest gets 1.0, then 0.8, 0.6, ...
    decay = torch.linspace(1.0, 0.2, steps=window, device=device)[:k]
    idx = torch.tensor(last_tokens[-k:], device=device)

    rep_vec = torch.zeros(V, device=device)
    # reverse so newest aligns with decay[0]
    rep_vec.index_add_(0, idx.flip(0), decay)
    return rep_vec * strength


def audit_generated_stream(
        results_df: pd.DataFrame,
        tokenizer,
        token_w: int = 50,        # Max width of the “Token” column
    ) -> None:
    """
    Prints a step-by-step structural audit of a context+generation stream.
    Used for model manual tests after training, to validate poor results.

    After the row-by-row trace it prints the whole DataFrame (Token,
    IsInput, Note).  No return value.
    """
    # ------------- helpers ------------- #
    def _print(idx, tok, is_inp, problems, token_w):
        clipped = (tok[:token_w - 3] + "...") if len(tok) > token_w else tok
        note    = ";".join(problems)
        print(f"{idx:4d}  {clipped:<{token_w}}  {int(is_inp):>5d}  {note}")
    
    # ------------- look‑ups & runtime state ------------- #
    l = build_luts(tokenizer)
    is_start, is_end   = l['is_start'], l['is_end']
    base_id            = l['base_id']
    meal_rank          = l['meal_rank']
    conflict_mat       = l['conflict_mat']
    nb                 = int(l['start_ids_per_base'].numel())
    K                  = int(l['K_meals'].item())

    open_counts = torch.zeros(nb, dtype=torch.int16)
    seen_meals  = torch.zeros(K,  dtype=torch.bool) if K else None

    t2id = tokenizer.token2id
    notes = []                                            # per‑row notes

    # ------------- pretty header for live trace -------- #
    hdr = f"{'Idx':>4}  {'Token':<{token_w}}  IsInp  Note"
    print(hdr)
    print("-" * len(hdr))

    # ------------- iterate over stream ----------------- #
    for idx, (tok, is_inp) in enumerate(zip(results_df['Token'],
                                            results_df['IsInput'])):
        tid = t2id.get(tok, None)
        problems = []

        # UNK
        if tid is None:
            problems.append("UNK")
            notes.append(";".join(problems))
            _print(idx, tok, is_inp, problems, token_w)
            continue

        b = base_id[tid].item()
        s = bool(is_start[tid])
        e = bool(is_end[tid])
        r = meal_rank[tid].item()

        # interval FSM / DUP / CNF
        if e:
            if b < 0 or open_counts[b] == 0:
                problems.append("FSM")
            else:
                open_counts[b] -= 1
        elif s:
            if open_counts[b] > 0:
                problems.append("DUP")
            if conflict_mat.any() and (open_counts > 0).any():
                if (conflict_mat[b] & (open_counts > 0)).any():
                    problems.append("CNF")
            open_counts[b] += 1

        # meal order
        if r >= 0:
            if 'next_meal' not in locals():               # first meal we ever see
                next_meal = (r + 1) % K                   # set expectation
            else:
                if r != next_meal:                        # wrong meal → violation
                    problems.append("MEAL")
                next_meal = (r + 1) % K                   # advance expectation

        notes.append(";".join(problems))
        
    # ------------- final full table -------------------- #
    df_out = results_df.copy()
    df_out['Note'] = notes

    pd.set_option("display.max_rows", None)
    print("\n=== Full trajectory with annotations ===")
    print(df_out[['Token', 'IsInput', 'Note']]
          .to_string(index=True, col_space={'Token': token_w}))