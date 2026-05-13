"""
utils.py
==============

General util functions for the package
"""
import sys
import os
import datetime
import functools
import inspect
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


class _TeeStream:
    """Wraps sys.stdout so every write goes to both the terminal and a log file."""

    def __init__(self, log_path, original_stream):
        self._original = original_stream
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._file = open(log_path, "a", encoding="utf-8", buffering=1)  # line-buffered

    def write(self, text):
        self._original.write(text)
        self._file.write(text)

    def flush(self):
        self._original.flush()
        self._file.flush()

    def close(self):
        self._file.close()

    # Proxy everything else (isatty, fileno, etc.) to the real stream so tqdm
    # and other tools that inspect stdout continue to work correctly.
    def __getattr__(self, name):
        return getattr(self._original, name)


_active_tee: Optional[_TeeStream] = None  # shared across all decorated training functions


def _ensure_tee_active():
    global _active_tee
    if _active_tee is not None:
        return
    # Tee log goes to /tmp (local fs) — the workspace mfs has intermittent OSError [Errno 5]
    # that crashes training mid-run. stdout redirect to /tmp/run.log already captures everything.
    log_path = "/tmp/training.log"
    _active_tee = _TeeStream(log_path, sys.stdout)
    sys.stdout = _active_tee
    print(f"[Logger] Logging to: {log_path}")


def logger(func):
    """
    Decorator for training functions.  On first call it activates the stdout
    tee (append mode).  Every call prints a timestamped header containing the
    function name, model config, and training settings so the log is
    self-contained and searchable across runs.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _ensure_tee_active()

        # Pull training_settings out of the call if present
        try:
            bound = inspect.signature(func).bind(*args, **kwargs)
            bound.apply_defaults()
            ts = bound.arguments.get("training_settings")
        except Exception:
            ts = None

        try:
            from transform_emr.config.model_config import MODEL_CONFIG
        except Exception:
            MODEL_CONFIG = None

        sep = "=" * 70
        print(f"\n{sep}")
        print(f"  {func.__name__}  |  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if MODEL_CONFIG:
            print(f"  model config      : {MODEL_CONFIG}")
        if ts:
            print(f"  training settings : {ts}")
        print(sep)

        return func(*args, **kwargs)
    return wrapper


@torch.no_grad()
def get_temporal_multi_hot_targets(
    target_ids: torch.Tensor,
    all_abs_ts: torch.Tensor,
    padding_idx: int,
    vocab_size: int,
    window_size: float,
    query_abs_ts: Optional[torch.Tensor] = None,
    outcome_ids: Optional[torch.Tensor] = None,
    next_token_ids: Optional[torch.Tensor] = None,
    wide_token_ids: Optional[torch.Tensor] = None,
    wide_window_size: Optional[float] = None,
    wide_tiers: Optional[list] = None,
) -> torch.Tensor:
    """
    Build temporal multi-hot targets over a future time window using GPU-efficient
    searchsorted + prefix-sum approach.

    For each query step t, marks token ids that appear at any future step s such that:
        0 < (all_abs_ts[s] - query_abs_ts[t]) <= window_size

    Outcome override: when ``outcome_ids`` and ``next_token_ids`` are both provided,
    any query position whose immediate next token is an outcome/terminal token has its
    entire multi-hot row replaced with a 1-hot on just that token. This eliminates
    gradient dilution at exactly the positions where precise supervision matters most.

    IMPORTANT: This function assumes all_abs_ts is NON-DECREASING (sorted) per batch.
    The dataset MUST maintain this ordering. See dataset.py for sorting guarantees.

    Args:
        target_ids: [B, T_all] token ids whose occurrences will be marked as positives.
        all_abs_ts: [B, T_all] absolute timestamps, MUST be non-decreasing per batch.
        padding_idx: Token id used for PAD. PAD is excluded from targets.
        vocab_size: Vocabulary size V for output shape.
        window_size: Future window size (same normalized units as ``all_abs_ts``).
        query_abs_ts: [B, T_q] optional query timestamps. If omitted, uses ``all_abs_ts``.
        outcome_ids: 1-D LongTensor of outcome + terminal token ids. When provided
            together with ``next_token_ids``, enables the 1-hot override.
        next_token_ids: [B, T_q] immediate next token at each query position.
            Use ``padding_idx`` where no next token exists (e.g. last position in phase-1).

    Returns:
        FloatTensor [B, T_q, V] with 0/1 multi-hot labels.
    """
    if query_abs_ts is None:
        query_abs_ts = all_abs_ts

    B, T_all = target_ids.shape
    T_q = query_abs_ts.size(1)

    # GPU-friendly searchsorted + prefix-sum approach (O(B * T * log T) instead of O(B * T^2)).
    # Assumes timestamps are sorted; violation will produce incorrect results silently.
    all_abs_ts = all_abs_ts.contiguous()
    query_abs_ts = query_abs_ts.contiguous()
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

    # Optionally apply wider future windows to specific subsets of tokens. Two entry
    # points are supported:
    #   - single-tier: pass wide_token_ids + wide_window_size
    #   - multi-tier:  pass wide_tiers = [(ids, window), ...]
    # Each tier reuses the same prefix sums — one extra searchsorted + gather per
    # tier. The wider window provides denser positive signal for rare/critical
    # tokens (e.g. terminals), so the LM head learns to assign higher logits at
    # pre-event positions instead of only the immediate-next position.
    _wide_specs = []
    if wide_token_ids is not None and wide_window_size is not None and wide_token_ids.numel() > 0 and wide_window_size > window_size:
        _wide_specs.append((wide_token_ids, wide_window_size))
    if wide_tiers is not None:
        for _tier_ids, _tier_w in wide_tiers:
            if _tier_ids is not None and _tier_w is not None and _tier_ids.numel() > 0 and _tier_w > window_size:
                _wide_specs.append((_tier_ids, _tier_w))
    for _wide_ids, _w_size in _wide_specs:
        right_wide_idx = torch.searchsorted(all_abs_ts, query_abs_ts + _w_size, right=True)
        right_wide = right_wide_idx.clamp(0, T_all).unsqueeze(-1).expand(B, T_q, vocab_size)
        future_counts_wide = prefix.gather(1, right_wide) - prefix.gather(1, left)
        wide_multi_hot = (future_counts_wide > 0).to(torch.float32)
        if 0 <= padding_idx < vocab_size:
            wide_multi_hot[..., padding_idx] = 0.0
        wide_ids = _wide_ids.to(target_ids.device)
        multi_hot[:, :, wide_ids] = wide_multi_hot[:, :, wide_ids]

    # Outcome 1-hot override: replace the broad window multi-hot with a strict 1-hot at
    # positions where the immediate next token is an outcome or terminal. Outcome tokens
    # are never illegal so the legality mask does not conflict with this override.
    if outcome_ids is not None and next_token_ids is not None and outcome_ids.numel() > 0:
        assert next_token_ids.shape == (B, T_q), (
            f"next_token_ids shape {next_token_ids.shape} must match (B={B}, T_q={T_q})"
        )
        oids = outcome_ids.to(target_ids.device)
        # is_outcome_next[b, q] = True iff next_token_ids[b, q] is in outcome_ids
        is_outcome_next = (next_token_ids.unsqueeze(-1) == oids.view(1, 1, -1)).any(-1)  # [B, T_q]
        if is_outcome_next.any():
            bi, qi = is_outcome_next.nonzero(as_tuple=True)
            tok = next_token_ids[bi, qi]   # [N] the specific outcome token ids
            multi_hot[bi, qi] = 0.0        # wipe window
            multi_hot[bi, qi, tok] = 1.0   # replace with 1-hot

    return multi_hot


def get_temporal_soft_targets(
    target_ids: torch.Tensor,
    all_abs_ts: torch.Tensor,
    query_abs_ts: torch.Tensor,
    padding_idx: int,
    vocab_size: int,
    tau: torch.Tensor,
    horizon: float,
) -> torch.Tensor:
    """
    Soft-kernel LM-head BCE targets — exp59's two-tier window replaced by a
    learnable per-token-class decay constant.

    For each query step t and each token id v in the vocabulary:
        target[b, t, v] = clamp_{0..1}( sum_{s : 0 < dt(t,s) <= horizon}
                                        exp(-dt(t,s) / tau[v])
                                        * 1[target_ids[b, s] == v] )

    Matches the formula already used by `get_future_outcome_targets` for the
    outcome head, extended to the full LM-head vocabulary. Implementation
    uses scatter_add along the V dimension to avoid materialising a
    [B, T, V] one-hot intermediate — memory cost is the same as the binary
    version's [B, T_q, V] target tensor, plus a [B, T_q, T] decay matrix.

    NOTE: NOT decorated with @torch.no_grad(): gradient flows through tau so
    the learnable `log_tau_lm` parameter trains end-to-end with the LM-head
    BCE. The binary `get_temporal_multi_hot_targets` path remains
    non-differentiable (boolean → float cast) so removing its no_grad would
    be a no-op.

    Args:
        target_ids:    [B, T] token ids whose occurrences mark positives.
        all_abs_ts:    [B, T] absolute timestamps (normalised, same units as horizon).
        query_abs_ts:  [B, T_q] query timestamps.
        padding_idx:   PAD id. PAD positions in target_ids contribute 0, and
                       target[..., padding_idx] is zeroed.
        vocab_size:    V.
        tau:           [V] per-token-class decay constants (positive, same
                       normalised units as horizon). Differentiable.
        horizon:       Hard horizon — positives at dt > horizon contribute 0.

    Returns:
        FloatTensor [B, T_q, V] with soft targets in [0, 1].
    """
    B, T_all = target_ids.shape
    T_q = query_abs_ts.size(1)
    V = vocab_size
    device = target_ids.device

    # Per-source-position tau: tau_per_s[b, s] = tau[target_ids[b, s]].
    # Clamp index to V-1 so the lookup is safe even if rare ids land outside.
    safe_ids = target_ids.clamp(0, V - 1)
    tau_per_s = tau[safe_ids]  # [B, T_all], differentiable

    # Δt: [B, T_q, T_all]. Positive = future.
    dt = all_abs_ts.unsqueeze(1) - query_abs_ts.unsqueeze(2)
    in_horizon = (dt > 0) & (dt <= horizon)

    # decay[b, t, s] = exp(-Δt / tau_per_s[b, s]) inside the horizon, else 0.
    # Clamp tau ≥ 1e-6 to keep the division finite if any tau gets pushed to 0.
    decay = torch.exp(-dt / tau_per_s.unsqueeze(1).clamp(min=1e-6))
    decay = decay.masked_fill(~in_horizon, 0.0)

    # Source-side PAD mask: PAD never contributes a positive.
    if 0 <= padding_idx < V:
        nonpad_src = (target_ids != padding_idx).to(decay.dtype)
        decay = decay * nonpad_src.unsqueeze(1)

    # Scatter-add into the V dim using target_ids as the column index.
    target = torch.zeros(B, T_q, V, device=device, dtype=decay.dtype)
    idx = target_ids.unsqueeze(1).expand(B, T_q, T_all)
    target.scatter_add_(2, idx, decay)
    target = target.clamp(0.0, 1.0)

    if 0 <= padding_idx < V:
        pad_mask = torch.ones(V, device=device, dtype=target.dtype)
        pad_mask[padding_idx] = 0.0
        target = target * pad_mask

    return target


@torch.no_grad()
def get_future_outcome_targets(
    target_ids: torch.Tensor,      # [B, T] token ids
    outcome_ids: list,        # [K] list of outcome token IDs
    all_abs_ts: Optional[torch.Tensor] = None,  # [B, T] absolute timestamps
    query_abs_ts: Optional[torch.Tensor] = None, # [B, T_q] query timestamps
    tau = None,        # scalar OR [K] tensor — decay constant in normalised units
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

    # tau may be a scalar or a per-outcome tensor of shape [K]. Per-outcome tau lets
    # each outcome adapt its own decay timescale — RELEASE-type events have different
    # dynamics from clinical complications, and a single scalar undersupervises one.
    is_per_k_tau = torch.is_tensor(tau) and tau.dim() == 1 and tau.numel() == K

    if not is_per_k_tau:
        decay_weights = torch.exp(-dt / tau).masked_fill(~in_horizon, 0.0)  # [B, T_q, T]
        outcome_targets = torch.bmm(decay_weights, matches).clamp(0.0, 1.0)
    else:
        # Per-outcome decay: bmm in a loop over K to keep memory bounded.
        # Out-of-horizon entries are suppressed with masked_fill, NOT multiplication —
        # exp(-dt/tau) can overflow to inf on padded/out-of-horizon dt values, and
        # 0 * inf would produce NaN. masked_fill replaces those entries cleanly.
        outcome_cols = []
        for k_idx in range(K):
            decay_k = torch.exp(-dt / tau[k_idx]).masked_fill(~in_horizon, 0.0)  # [B,T_q,T]
            col = torch.bmm(decay_k, matches[..., k_idx:k_idx + 1]).clamp(0.0, 1.0)  # [B,T_q,1]
            outcome_cols.append(col)
        outcome_targets = torch.cat(outcome_cols, dim=-1)

    return outcome_targets


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
        "tok2concept": tok2concept,   # Long[V], -1 if no concept mapping
        "tok2value":   tok2value,     # Long[V], -1 if no value mapping

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


def init_legality_state_batched(luts: dict, position_ids: torch.LongTensor):
    """
    Compute initial batched legality state from a seed sequence (may be padded).

    Mirrors the per-token scalar state in the old inference helpers but fully vectorized.
    Padding tokens (base_id == -1, meal_rank == -1) are ignored automatically.

    Args:
        luts: dict returned by build_luts(), already on the target device.
        position_ids: [B, T_seed] — may include right-padding with pad_token_id.

    Returns:
        open_counts  : LongTensor [B, nb]  — net open count per interval base.
        next_meal_rank: LongTensor [B]     — required next meal rank (-1 = no meal seen yet,
                                             any meal is allowed as the first one).
    """
    device = position_ids.device
    B, T  = position_ids.shape
    nb    = luts["start_ids_per_base"].numel()
    K     = int(luts["K_meals"].item())

    # ── interval open counts ──────────────────────────────────────────────────
    tok_base = luts["base_id"][position_ids]   # [B, T], -1 for non-interval / pad
    tok_s    = luts["is_start"][position_ids]  # [B, T]
    tok_e    = luts["is_end"  ][position_ids]  # [B, T]

    valid        = tok_base >= 0
    scatter_idx  = tok_base.clamp(min=0)       # avoid -1 index; gated by `valid`

    start_oh = torch.zeros(B, T, nb, dtype=torch.int32, device=device)
    end_oh   = torch.zeros(B, T, nb, dtype=torch.int32, device=device)
    b_idx    = torch.arange(B, device=device)[:, None]
    t_idx    = torch.arange(T, device=device)[None, :]
    start_oh[b_idx, t_idx, scatter_idx] = (tok_s & valid).to(torch.int32)
    end_oh  [b_idx, t_idx, scatter_idx] = (tok_e & valid).to(torch.int32)

    open_counts = (start_oh.sum(dim=1) - end_oh.sum(dim=1)).clamp(min=0).long()  # [B, nb]

    # ── meal cycle state ──────────────────────────────────────────────────────
    next_meal_rank = torch.full((B,), -1, dtype=torch.long, device=device)
    if K > 0:
        mr      = luts["meal_rank"][position_ids]          # [B, T], -1 for non-meal
        is_meal = mr >= 0                                   # [B, T]
        if is_meal.any():
            # Find the LAST meal position per batch item (vectorized argmax trick).
            t_range  = torch.arange(T, dtype=torch.float32, device=device)
            weighted = torch.where(is_meal, t_range.unsqueeze(0), torch.full_like(t_range, -1.0).unsqueeze(0))
            last_t   = weighted.argmax(dim=1)              # [B]
            has_meal = is_meal.any(dim=1)                  # [B]
            last_rank = mr[torch.arange(B, device=device), last_t]   # [B]
            next_meal_rank = torch.where(
                has_meal,
                (last_rank + 1) % K,
                torch.full((B,), -1, dtype=torch.long, device=device)
            )

    return open_counts, next_meal_rank


def build_illegal_mask_batched(luts: dict, open_counts: torch.LongTensor,
                                next_meal_rank: torch.LongTensor,
                                pad_id: int, mask_id: int) -> torch.BoolTensor:
    """
    Build a [B, V] illegal-token mask for the *next* generation step, given the
    current batched legality state.

    This is the batched, stateful counterpart to the scalar ``_build_illegal_mask``
    that was previously inlined in inference.py.

    Args:
        luts           : dict from build_luts(), on the correct device.
        open_counts    : LongTensor [B, nb] — net open-count per interval base.
        next_meal_rank : LongTensor [B]     — required meal rank (-1 = free choice).
        pad_id, mask_id: always-illegal special tokens.

    Returns:
        BoolTensor [B, V], True → token is illegal for that batch item.
    """
    device  = open_counts.device
    B       = open_counts.shape[0]
    V       = int(luts["is_start"].numel())
    nb      = int(luts["start_ids_per_base"].numel())
    K       = int(luts["K_meals"].item())

    illegal = torch.zeros(B, V, dtype=torch.bool, device=device)

    closed = open_counts <= 0   # [B, nb]
    opened = open_counts  > 0   # [B, nb]

    # 1) END illegal when its base is not open
    end_ids_pb  = luts["end_ids_per_base"]    # [nb]
    valid_end   = end_ids_pb >= 0             # [nb]
    if valid_end.any() and closed.any():
        mask           = closed & valid_end.unsqueeze(0)     # [B, nb]
        b_idx, nb_idx  = mask.nonzero(as_tuple=True)
        if b_idx.numel():
            illegal[b_idx, end_ids_pb[nb_idx]] = True

    # 2) START illegal when its base is already open (duplicate start)
    start_ids_pb = luts["start_ids_per_base"]   # [nb]
    valid_start  = start_ids_pb >= 0
    if valid_start.any() and opened.any():
        mask           = opened & valid_start.unsqueeze(0)   # [B, nb]
        b_idx, nb_idx  = mask.nonzero(as_tuple=True)
        if b_idx.numel():
            illegal[b_idx, start_ids_pb[nb_idx]] = True

    # 3) Conflict: START of another value of an already-open concept
    conf_mat = luts["conflict_mat"]   # [nb, nb]
    if valid_start.any() and opened.any() and conf_mat.any():
        oc              = opened.to(torch.float32)             # [B, nb]
        conflict_active = (oc @ conf_mat.to(torch.float32)) > 0   # [B, nb]
        mask            = conflict_active & valid_start.unsqueeze(0)
        b_idx, nb_idx   = mask.nonzero(as_tuple=True)
        if b_idx.numel():
            illegal[b_idx, start_ids_pb[nb_idx]] = True

    # 4) Meal cycle
    if K > 0:
        meal_rank    = luts["meal_rank"]        # [V]
        is_meal_tok  = meal_rank >= 0           # [V]
        if is_meal_tok.any():
            meal_v_ids   = is_meal_tok.nonzero(as_tuple=False).squeeze(-1)  # [n_meals]
            meal_v_ranks = meal_rank[meal_v_ids]                             # [n_meals]
            # Patients that have seen at least one meal must follow the cycle
            has_seen     = next_meal_rank >= 0                               # [B]
            if has_seen.any():
                required     = next_meal_rank[has_seen].unsqueeze(1)         # [B_act, 1]
                ok           = meal_v_ranks.unsqueeze(0) == required         # [B_act, n_meals]
                b_active     = has_seen.nonzero(as_tuple=False).squeeze(-1)  # [B_act]
                bad_b, bad_v = (~ok).nonzero(as_tuple=True)
                if bad_b.numel():
                    illegal[b_active[bad_b], meal_v_ids[bad_v]] = True

    # 5) Always block pad / mask tokens
    illegal[:, pad_id]  = True
    illegal[:, mask_id] = True

    return illegal


def update_legality_state_batched(luts: dict, next_token_ids: torch.LongTensor,
                                   open_counts: torch.LongTensor,
                                   next_meal_rank: torch.LongTensor,
                                   finished: torch.BoolTensor):
    """
    Update open_counts and next_meal_rank in-place after a batch generation step.

    Finished patients are skipped so their state stays frozen.

    Args:
        luts            : dict from build_luts(), on the correct device.
        next_token_ids  : LongTensor [B] — the token chosen for each batch item.
        open_counts     : LongTensor [B, nb] — mutated in-place.
        next_meal_rank  : LongTensor [B]     — mutated in-place.
        finished        : BoolTensor [B]     — True for already-finished patients.

    Returns:
        (open_counts, next_meal_rank) — same tensors, updated in-place.
    """
    K      = int(luts["K_meals"].item())
    device = next_token_ids.device
    B      = next_token_ids.shape[0]

    active     = ~finished
    active_idx = active.nonzero(as_tuple=False).view(-1)   # [B_act]
    if active_idx.numel() == 0:
        return open_counts, next_meal_rank

    active_toks = next_token_ids[active_idx]  # [B_act]

    # ── interval state ────────────────────────────────────────────────────────
    is_s   = luts["is_start"][active_toks]   # [B_act]
    is_e   = luts["is_end"  ][active_toks]   # [B_act]
    b_ids  = luts["base_id" ][active_toks]   # [B_act], -1 if not interval
    valid  = b_ids >= 0

    start_mask = is_s & valid
    if start_mask.any():
        bi = active_idx[start_mask]
        ba = b_ids[start_mask]
        open_counts.index_put_((bi, ba),
                               torch.ones(start_mask.sum(), dtype=open_counts.dtype, device=device),
                               accumulate=True)

    end_mask = is_e & valid
    if end_mask.any():
        bi = active_idx[end_mask]
        ba = b_ids[end_mask]
        open_counts.index_put_((bi, ba),
                               torch.full((end_mask.sum(),), -1, dtype=open_counts.dtype, device=device),
                               accumulate=True)
        open_counts.clamp_(min=0)

    # ── meal cycle ────────────────────────────────────────────────────────────
    if K > 0:
        mr       = luts["meal_rank"][active_toks]   # [B_act]
        is_meal  = mr >= 0
        if is_meal.any():
            meal_active = active_idx[is_meal]
            next_meal_rank[meal_active] = (mr[is_meal] + 1) % K

    return open_counts, next_meal_rank


def build_rep_penalty_batched(last_tokens_batch: list, V: int,
                               window: int = 5, strength: float = 0.6,
                               device=None) -> torch.Tensor:
    """
    Batched repetition-penalty vector for inference.

    Vectorised version of build_rep_penalty for a whole batch of patients.

    Args:
        last_tokens_batch : List[List[int]], one list per patient (newest last).
        V                 : Vocabulary size.
        window, strength  : Same semantics as build_rep_penalty.
        device            : Target device.

    Returns:
        FloatTensor [B, V].
    """
    B   = len(last_tokens_batch)
    rep = torch.zeros(B, V, device=device)
    if strength <= 0 or all(not t for t in last_tokens_batch):
        return rep

    decay = torch.linspace(1.0, 0.2, steps=window, device=device)  # newest=1.0

    for b, last_toks in enumerate(last_tokens_batch):
        if not last_toks:
            continue
        k   = min(window, len(last_toks))
        idx = torch.tensor(last_toks[-k:], dtype=torch.long, device=device)
        rep[b].index_add_(0, idx.flip(0), decay[:k])

    return rep * strength


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


def compute_soft_outcome_labels(gen_abs_ts_hours, gt_df, outcome_names,
                                 tau_hours, horizon_hours, device):
    """
    Compute soft outcome labels for a single patient's generated trajectory,
    using the same time-decayed formula as Phase-2 training:

        target_k(t) = clamp( Σ_s exp(-dt(t,s)/τ) * 1[token_s == outcome_k], 0, 1 )

    where s iterates over ground-truth FUTURE events within `horizon_hours` of t.
    Used during phase-3 training (finetuning on generated trajectories) to provide a learning signal for outcomes,
    even when the exact outcome token is not generated at the exact time in the future.
    
    Parameters
    ----------
    gen_abs_ts_hours : 1-D tensor of absolute times (hours from admission) for
                       each generated step.
    gt_df            : full (untruncated) token DataFrame for this patient.
                       Expected to have a 'PositionToken' (or 'Token') column and
                       a 'TimePoint' column (normalised to [0,1] by /336).
    outcome_names    : list of outcome token strings (model.outcome_names).
    tau_hours        : decay constant in hours (TRAINING_SETTINGS['outcome_decay_tau_hours']).
    horizon_hours    : max lookahead in hours (TRAINING_SETTINGS['outcome_horizon_hours']).
    device           : torch device for the returned tensor.

    Returns
    -------
    torch.Tensor of shape [T_gen, len(outcome_names)], dtype float32.
    """
    if gt_df is None or gen_abs_ts_hours.numel() == 0:
        return torch.zeros(0, len(outcome_names), device=device)

    T_gen   = gen_abs_ts_hours.shape[0]
    labels  = torch.zeros(T_gen, len(outcome_names))
    tok_col = 'PositionToken' if 'PositionToken' in gt_df.columns else 'Token'

    for k, name in enumerate(outcome_names):
        occ = gt_df[gt_df[tok_col] == name]
        if occ.empty:
            continue
        occ_times = torch.tensor(occ['TimePoint'].values * 336.0, dtype=torch.float32)
        for t_idx in range(T_gen):
            t      = gen_abs_ts_hours[t_idx].item()
            dt     = occ_times - t
            future = (dt > 0) & (dt <= horizon_hours)
            if not future.any():
                continue
            labels[t_idx, k] = torch.clamp(
                torch.exp(-dt[future] / tau_hours).sum(), 0.0, 1.0
            )

    return labels.to(device)


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
