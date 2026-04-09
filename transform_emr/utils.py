"""
utils.py
==============

General util functions for the package
"""
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import pandas as pd
from typing import Optional

# ───────── local code ─────────────────────────────────────────────────── #
from transform_emr.config.dataset_config import (
    ADMISSION_TOKEN, TERMINAL_OUTCOMES, OUTCOMES, MEAL_TOKENS
)
from transform_emr.schedulers import linear_schedule


@torch.no_grad()
def get_temporal_multi_hot_targets(
    target_ids: torch.Tensor,
    all_abs_ts: torch.Tensor,
    padding_idx: int,
    vocab_size: int,
    window_size: float,
    query_abs_ts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Build temporal multi-hot targets over a future time window using GPU-efficient
    searchsorted + prefix-sum approach.

    For each query step t, marks token ids that appear at any future step s such that:
        0 < (all_abs_ts[s] - query_abs_ts[t]) <= window_size

    IMPORTANT: This function assumes all_abs_ts is NON-DECREASING (sorted) per batch.
    The dataset MUST maintain this ordering. See dataset.py for sorting guarantees.

    Args:
        target_ids: [B, T_all] token ids whose occurrences will be marked as positives.
        all_abs_ts: [B, T_all] absolute timestamps, MUST be non-decreasing per batch.
        padding_idx: Token id used for PAD. PAD is excluded from targets.
        vocab_size: Vocabulary size V for output shape.
        window_size: Future window size (same normalized units as ``all_abs_ts``).
        query_abs_ts: [B, T_q] optional query timestamps. If omitted, uses ``all_abs_ts``.

    Returns:
        FloatTensor [B, T_q, V] with 0/1 multi-hot labels.
    """
    if query_abs_ts is None:
        query_abs_ts = all_abs_ts

    B, T_all = target_ids.shape
    T_q = query_abs_ts.size(1)

    # GPU-friendly searchsorted + prefix-sum approach (O(B * T * log T) instead of O(B * T^2)).
    # Assumes timestamps are sorted; violation will produce incorrect results silently.
    left_idx = torch.searchsorted(all_abs_ts, query_abs_ts, right=True)
    right_idx = torch.searchsorted(all_abs_ts, query_abs_ts + window_size, right=True)

    oh = F.one_hot(target_ids.clamp(min=0), num_classes=vocab_size).to(torch.float32)  # [B, T_all, V]
    csum = oh.cumsum(dim=1)

    # Prefix a zero row so window sum is prefix[right] - prefix[left].
    prefix = torch.cat(
        [torch.zeros(B, 1, vocab_size, device=target_ids.device, dtype=csum.dtype), csum],
        dim=1,
    )  # [B, T_all+1, V]

    left = left_idx.clamp(0, T_all).unsqueeze(-1).expand(B, T_q, vocab_size)
    right = right_idx.clamp(0, T_all).unsqueeze(-1).expand(B, T_q, vocab_size)

    future_counts = prefix.gather(1, right) - prefix.gather(1, left)
    multi_hot = (future_counts > 0).to(torch.float32)

    if 0 <= padding_idx < vocab_size:
        multi_hot[..., padding_idx] = 0.0

    return multi_hot


@torch.no_grad()
def get_future_outcome_targets(
    target_ids: torch.Tensor,      # [B, T] token ids
    outcome_ids: list,        # [K] list of outcome token IDs
    all_abs_ts: Optional[torch.Tensor] = None,  # [B, T] absolute timestamps
    query_abs_ts: Optional[torch.Tensor] = None, # [B, T_q] query timestamps
    tau: Optional[float] = None,      # decay time constant (hours / 336)
    horizon: Optional[float] = None,  # max lookahead horizon (hours / 336)
) -> torch.Tensor:
    """
    Builds time-aware outcome targets for auxiliary prediction head.

    Two modes:
    
    1. **Binary mode** (all_abs_ts=None): 
       target[b, t, k] = 1 if outcome_ids[k] appears in position_ids[b, t+1:] (anywhere after t).
       
    2. **Time-decayed mode** (all_abs_ts provided):
       target[b, t, k] = soft score ∈ [0, 1]:
           sum_s { exp(-dt(t,s) / tau) * 1[token_s == outcome_k] }.clamp(0, 1)
       where dt(t,s) is filtered by 0 < dt <= horizon.
       Maximum signal for outcomes very soon; decays exponentially to zero.

    Args:
        target_ids: [B, T] token sequence to search for outcomes.
        outcome_ids: [K] list of outcome token IDs.
        all_abs_ts: [B, T] absolute timestamps aligned with target_ids. If provided, enables time decay.
        query_abs_ts: [B, T_q] query timestamps. If omitted, uses all_abs_ts (same shape as target_ids).
        tau: Decay time constant (same units as timestamps). Only used if all_abs_ts is provided.
        horizon: Max lookahead window (same units as timestamps). Only used if all_abs_ts is provided.

    Returns:
        FloatTensor [B, T_q, K] with outcome probabilities (0/1 for binary, soft [0..1] for decayed).
    """
    # B, T = target_ids.shape
    K = len(outcome_ids)
    device = target_ids.device
    
    # Build outcome match matrix: [B, T, K]
    # matches[b, t, k] = 1.0 if target_ids[b, t] == outcome_ids[k]
    out_tensor = torch.tensor(outcome_ids, device=device, dtype=torch.long).view(1, 1, K)
    matches = (target_ids.unsqueeze(-1) == out_tensor).float()  # [B, T, K]

    # Binary mode: no time information
    if all_abs_ts is None:
        # Shift matches to get "future presence": target[t] = any match at s > t
        future_matches = torch.zeros_like(matches)
        future_matches[:, :-1, :] = matches[:, 1:, :]
        future_presence = future_matches.flip(dims=[1]).cummax(dim=1).values.flip(dims=[1])
        return future_presence

    # Time-decayed mode
    if query_abs_ts is None:
        query_abs_ts = all_abs_ts
    
    if tau is None or horizon is None:
        raise ValueError(
            "tau and horizon must be provided when all_abs_ts is given. "
            "Use model_config.TRAINING_SETTINGS for defaults."
        )

    # T_q = query_abs_ts.size(1)
    # Compute time differences: [B, T_q, T]
    dt = all_abs_ts.unsqueeze(1) - query_abs_ts.unsqueeze(2)
    
    # Horizon filtering: 0 < dt <= horizon
    in_horizon = (dt > 0) & (dt <= horizon)
    
    # Decay weights: exp(-dt / tau), masked to horizon
    decay_weights = torch.exp(-dt / tau).masked_fill(~in_horizon, 0.0)  # [B, T_q, T]
    
    # Aggregate outcomes via batch matrix multiply: [B, T_q, T] x [B, T, K] -> [B, T_q, K]
    outcome_targets = torch.bmm(decay_weights, matches).clamp(0.0, 1.0)
    
    return outcome_targets


def build_mlm(ids, tokenizer, p=0.15):
    """
    Embedder MLM Mask Helper.
    Logic: Mask (later predict) tokens important to position that won't hurt the general timeline.
            Mask everything informative but admission, CTX or terminal tokens.
    ids:  (B,T) tensor - concept_ids / value_ids / position_ids
    returns masked_ids (same shape)
    Masking strategy (BERT-style):
        80% → replace with [MASK]
        10% → keep original
        10% → replace with random token (not PAD)
    """
    device = ids.device
    never_mask_ids = {
        tokenizer.pad_token_id,
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


def apply_cbm(batch, tokenizer, forbid_ids, p=0.25):
    """
    Transformer CBM (Curriculum by Masking) Helper.
    Logic: Mask tokens that won't hurt the general timeline or conflict with penalties.
        This list adds to the MLM forbidden list, as now we can't mask meals / intervals, 
        as we'll teach the model the contradicting them is OK.
    Randomly masks ratio% of *input* tokens (excluding forbid_ids list),
    replacing them with [MASK] token id and corresponding sub-ids.

    batch: dict of tensors
    tokenizer: EMRTokenizer
    forbid_ids: LongTensor of ids that must never be masked (PAD, CTX, ADMISSION, TERMINALS...)
    p: float, masking ratio of the input

    NOTE: This basically means that masked tokens are marked as acceptable noise. 
    """    
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
    raw_ids = batch["parent_raw_ids"].clone()
    con_ids = batch["concept_ids"].clone()
    val_ids = batch["value_ids"].clone()

    pos_ids[to_mask] = mask_tok
    raw_ids[to_mask, :] = mask_tok
    con_ids[to_mask] = mask_tok
    val_ids[to_mask] = mask_tok

    batch["position_ids"]    = pos_ids
    batch["parent_raw_ids"]  = raw_ids
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
    NOTE: Currently not used in the codebase.
    """    
    device = gt_ids.device
    B, T = gt_ids.shape
    ss_rate = linear_schedule(epoch, 0, warmup_epochs, max_rate)

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
        "predict_block"     : Long[*]   tokens we never predict (PAD/MASK/CTX) - Forbids at all steps.
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
        tokenizer.null_token_id,
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
    block_ids = {
        tokenizer.pad_token_id,
        tokenizer.mask_token_id
    }
    block_ids = [tid for tid in block_ids if tid is not None]

    predict_block = torch.zeros(V, dtype=torch.bool, device=device)
    predict_block[torch.tensor(block_ids, dtype=torch.long, device=device)] = True

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
    Vectorized legality masks from GOLD prefix (teacher forcing).

    illegal[B,T,V]  True → forbid v at step t

    Terms:
    If token== 'GLUCOSE_TREND_inc_START', base(tok) == 'GLUCOSE_TREND_inc'
    If token== 'GLUCOSE_TREND_inc_START', concept(tok) == 'GLUCOSE_TREND'

    Interval logic (per base):
      • END is illegal if base not open yet (You can't see 'GLUCOSE_TREND_inc_END' before opening 
      it 'GLUCOSE_TREND_inc_START'). Enforced using base(tok).
      • START illegal if base already open 
      (You can't have 'GLUCOSE_TREND_inc_START' after 'GLUCOSE_TREND_inc_START' without seeing
        'GLUCOSE_TREND_inc_END'). Enforced using base(tok).
      • START of concept(tok) is illegal if concept(tok) is still open 
        (meaning you can't have 'GLUCOSE_TREND_inc_START' after 'GLUCOSE_TREND_dec_START' without seeing
        'GLUCOSE_TREND_inc_END'). Enforced using concept(tok).

    Meal logic:
            cyclic order; meal m illegal if predecessor rank not seen yet.
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

    # Prepare legality matrix
    illegal = torch.zeros(B, T, V, device=device, dtype=torch.bool)

    # 1) END rules: illegal if not open_before
    end_ids = end_ids_per_base.view(1,1,nb).expand(B,T,nb)
    illegal.scatter_(2, end_ids, ~open_before)

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

            # 4) apply to illegal mask
            illegal[:,:,v_ids] |= ~ok
    
    # ---- Specials never to be predicted ----
    illegal |= predict_block.view(1, 1, -1)   # broadcast [V] -> [B,T,V]

    return illegal


def masked_softmax(logits, allowed):
    """
    Softmax over allowed classes only; zeros on disallowed.
    Safe when a row is fully masked (no legal classes): returns zeros and stops grad.
    """
    allowed = allowed.to(torch.bool)

    # numeric stability: subtract row max
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    # push disallowed far down but finite
    shifted = shifted.masked_fill(~allowed, -1e9)

    # normal softmax on the shifted scores
    lse = torch.logsumexp(shifted, dim=-1, keepdim=True)     # [B,T,1]
    probs = torch.exp(shifted - lse) * allowed               # [B,T,V], zero on disallowed

    # handle rows where *all* entries are disallowed
    any_valid = allowed.any(dim=-1, keepdim=True)            # [B,T,1]
    probs = torch.where(any_valid, probs, torch.zeros_like(probs))
    # stop gradients on fully-masked rows (no learning signal there)
    probs = probs * any_valid.float()
    return probs


def apply_masks_to_logits(logits, illegal_mask):
    """
    logits: [B,T,V] *after* slicing (no [CTX])
    illegal_mask: Bool [B,T,V]

    Sets illegal-token logits to -1e9 so they are suppressed in both
    softmax and BCE without producing NaN gradients.

    Note: bonus boosting (+0.2 for legal-closure tokens) was removed.
    The bonus was a hardcoded nudge toward closing open intervals, but
    it was never calibrated and interfered with BCE learning the same
    signal organically. The illegal hard-mask is sufficient.
    """
    return logits.masked_fill(illegal_mask, -1e9)


def plot_losses(train_losses, val_losses):
    """
    Plot train vs. validation loss to inspect training quality.

    Ignores loss at first step (major instability).
    """
    train_losses = train_losses[1:]
    val_losses = val_losses[1:]

    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
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
    

# ---------------------------------------------------------------------------
# Legacy / diagnostics-only helpers (not used by current training path)
# These are kept for now for potential future use, but they are not called
# ---------------------------------------------------------------------------

@torch.no_grad()
def _gather_valid_ids(ids_tensor):
    """Return (ids_valid, mask_valid) where ids_valid has all ids >=0 and mask_valid selects those columns."""
    mask = ids_tensor >= 0
    return ids_tensor[mask], mask


def soft_interval_penalty(
    logits: torch.Tensor,           # [B,T,V] AFTER applying your legality/bonus mask to logits
    allowed: torch.Tensor,          # [B,T,V] bool (legal & non-PAD)
    start_ids_per_base: torch.Tensor, 
    end_ids_per_base: torch.Tensor,  # [nb] (may contain -1)
    conflict_mat: torch.Tensor,     # [nb,nb] bool
    alpha: float = 8.0,             # sharpness for sigmoids
    w_end_no_open: float = 1.0,
    w_dup_start: float = 1.0,
    w_conflict: float = 1.0,
    w_unclosed: float = 0.5,
    return_details: bool = False,
    eps: float = 1e-8,              # numerical stability for denominators
):
    """
    Differentiable interval structure penalty.
    Builds expected 'open mass' from probabilities of *_START/*_END and penalizes:
      • END with ~no open mass
      • START when already open (dup)
      • START when conflicting base open
      • residual open mass at the end (unclosed)
    Returns:
        penalty: scalar in [0,1]
        details: dict of raw & normalized components (if return_details=True)
    
    NOTE: DEPRECATED. Function is redundant given the function soft_illegal_mass_penalty.
    """
    B, T, V = logits.shape
    device = logits.device

    P = masked_softmax(logits, allowed)                       # [B,T,V]

    # Restrict to bases that have both START and END
    s_ids_all, s_mask = _gather_valid_ids(start_ids_per_base)
    e_ids_all, e_mask = _gather_valid_ids(end_ids_per_base)
    valid_mask = s_mask & e_mask
    if valid_mask.sum() == 0:
        return (logits.new_zeros(()), {}) if return_details else logits.new_zeros(())

    s_ids = start_ids_per_base[valid_mask]  # [nbv]
    e_ids = end_ids_per_base[valid_mask]    # [nbv]
    nbv   = s_ids.numel()

    # START/END probabilities per base
    p_s = P[:, :, s_ids]  # [B,T,nbv]
    p_e = P[:, :, e_ids]  # [B,T,nbv]

    # Exclusive prefix "open-before-t" mass
    cs_s = torch.cumsum(p_s, dim=1)
    cs_e = torch.cumsum(p_e, dim=1)
    open_before = F.relu(torch.cat(
        [torch.zeros(B,1,nbv, device=device, dtype=p_s.dtype),
         cs_s[:, :-1, :] - cs_e[:, :-1, :]],
        dim=1
    ))  # [B,T,nbv]

    # === Raw components (un-normalized) ===
    # END with no open
    pen_end_no_open = (p_e * torch.exp(-alpha * open_before)).sum()        # ≤ B*T*nbv

    # DUP START while already open
    pen_dup_start   = (p_s * (1.0 - torch.exp(-alpha * open_before))).sum()# ≤ B*T*nbv

    # CONFLICT START while conflicting base open
    if conflict_mat.numel() and nbv:
        cm_sub    = conflict_mat[valid_mask][:, valid_mask].float()        # [nbv,nbv]
        open_conf = torch.einsum('btn,nm->btm', open_before, cm_sub)        # [B,T,nbv]
        pen_conf  = (p_s * (1.0 - torch.exp(-alpha * open_conf))).sum()    # ≤ B*T*nbv
    else:
        pen_conf  = logits.new_zeros(())

    # UNCLOSED mass at sequence end
    open_final   = F.relu(cs_s[:, -1, :] - cs_e[:, -1, :])                  # [B,nbv]
    pen_unclosed = open_final.sum()                                         # ≤ B*nbv

    # === Strict [0,1] normalization per component ===
    denom_bt = (B * T * nbv) + eps
    denom_b  = (B *     nbv) + eps

    norm_end_no_open = pen_end_no_open / denom_bt
    norm_dup_start   = pen_dup_start   / denom_bt
    norm_conf        = pen_conf        / denom_bt
    norm_unclosed    = pen_unclosed    / denom_b

    # Weighted average in [0,1]
    wsum = (w_end_no_open + w_dup_start + w_conflict + w_unclosed) + eps
    penalty = (
        w_end_no_open * norm_end_no_open +
        w_dup_start   * norm_dup_start   +
        w_conflict    * norm_conf        +
        w_unclosed    * norm_unclosed
    ) / wsum

    if return_details:
        details = {
            "raw": {
                "end_no_open": pen_end_no_open.detach(),
                "dup_start":   pen_dup_start.detach(),
                "conflict":    pen_conf.detach(),
                "unclosed":    pen_unclosed.detach(),
            },
            "norm": {
                "end_no_open": norm_end_no_open.detach().item(),
                "dup_start":   norm_dup_start.detach().item(),
                "conflict":    norm_conf.detach().item(),
                "unclosed":    norm_unclosed.detach().item(),
            },
            "meta": {
                "B": B, "T": T, "nbv": int(nbv), "alpha": float(alpha),
                "ws": (float(w_end_no_open), float(w_dup_start), float(w_conflict), float(w_unclosed))
            },
            # light sanity stats for quick prints
            "stats": {
                "p_s_sum": float(p_s.sum().detach()),
                "p_e_sum": float(p_e.sum().detach()),
                "open_before_mean": float(open_before.mean().detach()),
                "open_final_mean": float(open_final.mean().detach())
            }
        }
        return penalty, details

    return penalty


def soft_meal_order_penalty(
    logits: torch.Tensor,             # [B,T,V] AFTER legality/bonus masking
    allowed: torch.Tensor,            # [B,T,V] bool
    meal_rank: torch.Tensor,          # [V]  (-1 non-meal else 0..K-1)
    decay: float = 0.8,               # recency decay (0<decay<1); closer to 1 = longer memory
    beta: float = 6.0,                # sharpness for “seen” squashing
    eps: float = 1e-8,
):
    """
    Differentiable cyclic meal-order penalty with *recency*:
      • Build soft distribution over the **most-recent** meal rank before t via
        an exponential-decay causal convolution over rank-level meal probabilities.
      • Allowed rank at t is then the **successor** of that soft last-seen distribution.
      • Penalize the probability mass on meal ranks that are NOT in that successor distribution.
      • First meal (no prior seen) has zero penalty.

    Returns: scalar penalty.

    NOTE: DEPRECATED. Function is redundant given the function soft_illegal_mass_penalty.
    """
    B, T, V = logits.shape
    device  = logits.device
    # No meals -> no penalty
    if not (meal_rank >= 0).any():
        return logits.new_zeros(())

    # 1) Probabilities over allowed classes
    l = logits.masked_fill(~allowed, -1e9)          # [B,T,V]
    P = torch.exp(l - torch.logsumexp(l, dim=-1, keepdim=True))  # [B,T,V]
    P = P * allowed.to(P.dtype)

    # 2) Collapse token-level meal probs to rank-level probs: P_rank[b,t,r]
    meal_ids = (meal_rank >= 0).nonzero(as_tuple=False).squeeze(-1)   # [N_meal_tokens]
    ranks    = meal_rank[meal_ids].to(torch.long)                     # [N] each in 0..K-1
    K        = int(meal_rank[meal_rank >= 0].max().item()) + 1

    P_meal_tok = P[:, :, meal_ids]                                    # [B,T,N]
    P_rank = torch.zeros(B, T, K, device=device)
    # scatter_add tokens into their ranks
    P_rank.scatter_add_(dim=2, index=ranks.view(1,1,-1).expand(B,T,-1), src=P_meal_tok)  # [B,T,K]

    # 3) Build soft “most-recent rank before t” via exponential-decay causal conv
    #    y[:, :, t] = sum_{j <= t} P_rank[:, j] * decay^(t-j)
    kernel = (decay ** torch.arange(T, device=device, dtype=P_rank.dtype)).view(1,1,T)  # [1,1,T]
    x = P_rank.transpose(1,2)                                       # [B,K,T]  (channels=K)

    # depthwise kernel: [K, 1, T]
    base = (decay ** torch.arange(T, device=device, dtype=x.dtype)).view(1, 1, T)
    w = base.repeat(K, 1, 1)

    # causal conv: padding=T-1 then trim right
    y_full = F.conv1d(x, w, padding=T-1, groups=K)  # [B, K, T+T-1]
    y = y_full[:, :, :T]                            # [B, K, T]

    # exclusive prefix: last-seen BEFORE t
    last_seen = torch.zeros_like(y)
    last_seen[:, :, 1:] = y[:, :, :-1]                              # [B,K,T]
    last_seen = last_seen.transpose(1,2)                             # [B,T,K]

    # squash to [0,1], emphasize larger mass
    seen_soft = 1.0 - torch.exp(-beta * last_seen)                   # [B,T,K]
    any_seen  = (seen_soft.sum(dim=2, keepdim=True) > 0).float()     # [B,T,1]

    # 4) Successor distribution over ranks (shift by +1 mod K), normalized
    succ = torch.roll(seen_soft, shifts=+1, dims=2)                  # successor of rank r is (r+1)%K
    succ = succ / (succ.sum(dim=2, keepdim=True) + eps)              # [B,T,K]; undefined -> handled by any_seen

    # 5) Current meal rank mass vs allowed successor ranks
    #    penalty = sum_r P_rank[b,t,r] * (1 - succ[b,t,r])  when any_seen>0;  else 0
    wrong = 1.0 - succ                                               # [B,T,K]
    pen = (P_rank * wrong * any_seen).sum()                          # scalar

    denom = (B * T + 1.0)
    return pen / denom


def soft_illegal_mass_penalty(
    logits_pre_mask: torch.Tensor,    # [B,T,V] BEFORE apply_masks_to_logits
    illegal_mask: torch.Tensor,       # [B,T,V] booleans from compute_legality_masks_tf
    nonpad_mask: torch.Tensor,        # [B,T]   booleans (targets != PAD)
    margin: float = 0.03,             # small safety margin; set 0 to disable
    power: float = 1.0                # >1 accentuates heavy offenders
) -> torch.Tensor:
    """
    Penalize probability mass placed on illegal classes *before* masking.
    Scale ~[0,1]. Fully differentiable w.r.t. logits_pre_mask.

    NOTE: DEPRECATED — kept for diagnostic use only; removed from the training loss.

    Why: The apply_masks_to_logits hard-mask already enforces legality at every step
    (illegal logits → -1e9), so the model never selects illegal tokens regardless.
    The penalty's only purpose would be to make the *pre-mask* distribution internally
    avoid illegal regions — but BCE/CE gradients drive allowed logits up, which in
    softmax space *relatively* raises illegal logits. The penalty gradient fights this
    without enough λ budget to win. Net effect: the penalty fluctuates near its initial
    value and contributes noise rather than useful signal.
    Use for inspection (how much illegal mass does the raw network assign?) but not as
    a training objective.
    """
    P = torch.softmax(logits_pre_mask, dim=-1)                     # [B,T,V]
    mass = (P * illegal_mask).sum(dim=-1)                          # [B,T]
    if margin > 0:
        mass = F.relu(mass - margin)                               # hinge around margin
    num = (mass * nonpad_mask.float()).sum()
    den = nonpad_mask.float().sum().clamp_min(1.0)
    pen = num / den
    return pen.pow(power)


def soft_unclosed_interval_penalty(
    logits: torch.Tensor,           # [B,T,V] AFTER masking
    allowed: torch.Tensor,          # [B,T,V] bool
    start_ids_per_base: torch.Tensor,
    end_ids_per_base: torch.Tensor, # [nb]
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Penalizes intervals that remain open at the end of the sequence (step T).
    This captures global consistency that step-wise masks cannot enforce.

    NOTE: DEPRECATED — kept for diagnostic use only; removed from the training loss.

    Why: The penalty aggregates total probability mass assigned to start tokens vs
    end tokens across ALL time steps, then penalises relu(total_starts - total_ends).
    This is a global approximation of a local constraint ("every start must be followed
    by an end"). The resulting gradient tells the model "assign less mass to starts /
    more mass to ends everywhere" — a diffuse signal that does not correspond to any
    specific positional violation. A model satisfying global mass balance (total_starts
    ≈ total_ends) can still generate structurally invalid sequences at individual steps.
    The step-wise illegal mask already prevents the worst violations; the penalty adds
    noise without improving structural correctness in practice.
    Use for inspection (how much unclosed-interval mass accumulates over a batch?) but
    not as a training objective.

    Returns: scalar penalty in [0, 1]
    """
    B, T, V = logits.shape
    
    # 1. Probabilities from MASKED logits (we must respect the mask's decisions)
    P = masked_softmax(logits, allowed)

    # 2. Gather valid start/end IDs
    s_ids_all, s_mask = _gather_valid_ids(start_ids_per_base)
    e_ids_all, e_mask = _gather_valid_ids(end_ids_per_base)
    valid_mask = s_mask & e_mask
    
    if valid_mask.sum() == 0:
        return logits.new_zeros(())

    s_ids = start_ids_per_base[valid_mask]
    e_ids = end_ids_per_base[valid_mask]
    nbv   = s_ids.numel()

    # 3. Calculate total mass given to STARTs and ENDs
    p_s = P[:, :, s_ids]  # [B,T,nbv]
    p_e = P[:, :, e_ids]  # [B,T,nbv]

    total_starts = torch.sum(p_s, dim=1) # [B, nbv]
    total_ends   = torch.sum(p_e, dim=1) # [B, nbv]

    # 4. Penalty = Excess starts that were never closed
    # Relu ensures we don't penalize "extra ends" here (that's handled by illegal mask)
    open_final = F.relu(total_starts - total_ends) # [B, nbv]
    
    # 5. Normalize
    # We normalize by Batch * Num_Bases. 
    # (Max penalty is 1.0 if every base is fully open for every patient)
    pen_unclosed = open_final.sum() / (B * nbv + eps)
    
    return pen_unclosed


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
    predict_block:       torch.BoolTensor,
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
    
    NOTE: Currently not used in the codebase. Found to be less effective than expected, and may need rethinking.
    """
    B, T = pred_ids.shape

    # 1) Compute illegal masks for pred & GT
    illegal_pred = compute_legality_masks_tf(
        pred_ids, is_start, is_end,
        base_id, start_ids_per_base, end_ids_per_base,
        meal_rank, meal_pred_rank, K_meals, conflict_mat, predict_block
    )
    illegal_gt = compute_legality_masks_tf(
        gt_ids,   is_start, is_end,
        base_id, start_ids_per_base, end_ids_per_base,
        meal_rank, meal_pred_rank, K_meals, conflict_mat, predict_block
    )

    # Gather illegal flags per token
    # 2a) Gather illegal flags per token for pred **and** GT
    pred_illegal = illegal_pred.gather(2, pred_ids.unsqueeze(-1)).squeeze(-1)  # [B,T]
    gt_illegal   = illegal_gt.  gather(2, gt_ids.  unsqueeze(-1)).squeeze(-1)  # [B,T]

    # Do not let specials (PAD/CTX/MASK) create forgiveness windows
    gt_special = predict_block[gt_ids]      # [B,T] True where GT token is a special
    gt_illegal = gt_illegal & (~gt_special) # drop specials from GT-illegal used for forgiveness

    # Do not count specials on the PRED side as “new” violations either
    pred_special = predict_block[pred_ids]     # specials in PRED
    pred_illegal = pred_illegal & (~pred_special)

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
    Computes a cyclic-meal-order penalty: B→L→D→N→B...
    Returns a scalar ∈ [0,1]: the fraction of adjacent meal transitions
    in the predicted sequence that violate the expected cycle.

    Steps:
      1. Map pred_ids → ranks (with -1 for non-meal tokens).
      2. Compress each batch sequence into its meal-only sequence M of shape [B, max_meals].
      3. Compute expected next-meal rank = (M[:, :-1] + 1) % K.
      4. Count wrong transitions (curr != expected) and normalize by total transitions.

    Fully batched, GPU-friendly, non differentiable (due to argmax -> pred_ids).

    NOTE: Currently not used in the codebase. Found to be less effective than expected, and may need rethinking.
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