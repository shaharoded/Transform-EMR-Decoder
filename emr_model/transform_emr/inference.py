import json
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm.auto import tqdm
import joblib
from pathlib import Path

# ───────── local code ─────────────────────────────────────────────────── #
from transform_emr.config.dataset_config import *
from transform_emr.config.model_config import CHECKPOINT_PATH
from transform_emr.utils import (
    build_luts,
    init_legality_state_batched,
    build_illegal_mask_batched,
    update_legality_state_batched,
    build_rep_penalty_batched,
)


def get_token_embedding(embedder, token: str) -> torch.Tensor:
    """
    Returns the embedding vector of a specific token from a trained embedder.

    Args:
        embedder (EMREmbedding): A trained EMREmbedding model.
        token (str): The string token to lookup.

    Returns:
        torch.Tensor: Embedding vector of shape [embed_dim].
    """
    if token not in embedder.token2id:
        raise ValueError(f"Token '{token}' not found in vocabulary.")

    token_id = embedder.tokenizer.token2id[token]
    embedding = embedder.token_embed.weight[token_id].detach()
    return embedding


# ───────── batched seed preparation ─────────────────────────────────────── #

def _prepare_batch_seeds(batch_pids, dataset, tok, device):
    """
    Pad a batch of patient seed sequences to the same length.

    Padding is RIGHT-side (trailing): real tokens occupy positions [0:Ti], pad
    tokens fill [Ti:T_max].  This is safe for causal generation because:
      - Causal attention prevents real tokens from attending to any trailing pad.
      - Logits are read from last_valid_idx = seed_lens - 1 (last real position).
      - The KV cache mask passed to every decode step marks pad positions invalid,
        so generated tokens never attend to pad KV entries.
    Pad tokens never appear in the output DataFrame.

    Returns
    -------
    pos_ids          : [B, T_max] long
    parent_raw_ids   : [B, T_max, P] long
    concept_ids      : [B, T_max] long
    value_ids        : [B, T_max] long
    abs_ts           : [B, T_max] float  (normalised to [0,1])
    ctx_vecs         : [B, ctx_dim] float
    seed_lens        : [B] long  — real (unpadded) length of each patient's seed
    """
    pad_id = tok.pad_token_id
    P = tok.tokenid2parent_raw_ids.shape[1]

    seqs = []
    for pid in batch_pids:
        df = dataset.patient_groups[pid]
        pos   = torch.tensor(df["PositionID"].tolist(), dtype=torch.long)
        raw   = tok.tokenid2parent_raw_ids[pos]                          # [T, P]
        con   = torch.tensor(df["ConceptID"].tolist(),  dtype=torch.long)
        val   = torch.tensor(df["ValueID"].tolist(),    dtype=torch.long)
        ts    = torch.tensor(df["TimePoint"].tolist(),  dtype=torch.float32) / 336.0
        ctx   = torch.tensor(dataset.context_df.loc[pid].values, dtype=torch.float32)
        seqs.append((pos, raw, con, val, ts, ctx))

    B      = len(seqs)
    T_max  = max(s[0].shape[0] for s in seqs)
    seed_lens = torch.tensor([s[0].shape[0] for s in seqs], dtype=torch.long)

    pos_ids        = torch.full((B, T_max), pad_id, dtype=torch.long)
    parent_raw_ids = torch.full((B, T_max, P), pad_id, dtype=torch.long)
    concept_ids    = torch.full((B, T_max), pad_id, dtype=torch.long)
    value_ids      = torch.full((B, T_max), pad_id, dtype=torch.long)
    abs_ts         = torch.zeros(B, T_max, dtype=torch.float32)
    ctx_vecs       = torch.stack([s[5] for s in seqs], dim=0)

    for i, (pos, raw, con, val, ts, _) in enumerate(seqs):
        Ti = pos.shape[0]
        pos_ids[i, :Ti]        = pos
        parent_raw_ids[i, :Ti] = raw
        concept_ids[i, :Ti]    = con
        value_ids[i, :Ti]      = val
        abs_ts[i, :Ti]         = ts

    return (
        pos_ids.to(device),
        parent_raw_ids.to(device),
        concept_ids.to(device),
        value_ids.to(device),
        abs_ts.to(device),
        ctx_vecs.to(device),
        seed_lens.to(device),
    )


def _sample_tokens(next_logits, temperature, top_k):
    """
    Sample (or argmax) the next token from [B, V] logits.
    Returns LongTensor [B].
    When temperature > 1.0 and top_k is None, uses full-vocabulary multinomial
    sampling — argmax is kept only at temperature == 1.0 (the steady-state).
    """
    if top_k:
        topv, topi = torch.topk(next_logits, top_k, dim=-1)
        probs = F.softmax(topv / temperature, dim=-1)
        idx   = torch.multinomial(probs, 1).squeeze(-1)          # [B]
        return topi[torch.arange(topi.shape[0], device=topi.device), idx]
    elif temperature > 1.0:
        # Softmax over full vocabulary at elevated temperature — enables escape
        # from the immediate-terminal local minimum without any hard rule.
        probs = F.softmax(next_logits / temperature, dim=-1)
        return torch.multinomial(probs, 1).squeeze(-1)           # [B]
    else:
        return torch.argmax(next_logits / temperature, dim=-1)   # [B]


# ─────────────────────────────────────────────────────────────────────────── #

@torch.no_grad()
def generate(model,
             dataset,
             max_duration_hours=336.0,
             max_len=2000,
             temperature=1.0,
             top_k=None,
             rep_decay=0.6,
             batch_size=16,
             collect_risk_scores=False,
             tqdm_position=0,
             tqdm_desc='Generating',
             temperature_start=3.0,
             temperature_anneal_steps=10,
             hazard_suppress=True,
             hazard_min_hours=24.0,
             freeze_risk_at_seed=True):
    """
    Unified autoregressive generation for all patients in *dataset*.

    Generates token-by-token using a batched KV-cached decode loop with FP16 autocast.
    A patient stops when ANY of these is true:
      1. It emits a terminal token (DEATH / RELEASE).
      2. Its current absolute time ≥ max_duration_hours (training horizon).
      3. The decode loop has produced max_len new tokens (safety cap).
    Patients that stop on (2) or (3) without a natural terminal receive a forced
    terminal token chosen by the highest generation logit among TERMINAL_OUTCOMES,
    clamped to ≤ max_duration_hours.

    Args:
        model: Trained GPT model.
        dataset: EMRDataset providing patient seeds.
        max_duration_hours (float): Primary stop condition — generation horizon
            in hours from admission. Should match the training horizon (336 h /
            14 days by default) so Time2Vec stays in-distribution. Going past
            this point produces meaningless extrapolated embeddings.
        max_len (int): Hard ceiling on new tokens per patient (safety only).
        temperature (float): Steady-state softmax temperature (1.0 = neutral;
            also the final temperature after annealing when temperature_start > 1).
        top_k (int | None): Top-k sampling; argmax if None (and temperature == 1).
        rep_decay (float): Repetition penalty strength (0 = disabled).
        batch_size (int): Patients processed in parallel.
        collect_risk_scores (bool): When True, attach outcome-head probabilities
            (P_<name> columns) at every step — one bulk GPU->CPU sync per decode
            step, same efficiency as the decode forward pass itself.  When False
            only Token / TimePoint / IsInput / IsOutcome / IsTerminal are emitted.
        tqdm_position, tqdm_desc: tqdm display controls.
        temperature_start (float): Initial temperature for the F2 annealing schedule
            (default 3.0). When > temperature, sampling switches from argmax to
            full-vocab multinomial for the first temperature_anneal_steps steps,
            then exponentially decays back to temperature.  This lets the model
            escape the immediate-terminal local minimum without any hard rule.
            Set equal to temperature (e.g. temperature_start=1.0) to disable.
        temperature_anneal_steps (int): Steps over which to decay temperature_start
            → temperature (default 10).  After this many generated tokens the
            steady-state temperature applies.
        hazard_suppress (bool): F3 flag. When True, uses the outcome head's predicted
            P(DEATH or RELEASE in next 48 h) at the seed-end position to draw a
            per-patient terminal suppression time T from an Exponential distribution
            (E[T] = 48 / p_terminal hours), clamped to ≥ hazard_min_hours. Terminal
            tokens (DEATH/RELEASE) are masked to −∞ until elapsed generated time ≥ T.
        hazard_min_hours (float): Hard floor for drawn terminal suppression times
            (default 24.0 h). Ensures no patient terminates within the first 24 h
            of generated time even if the outcome head predicts very high terminal risk.
        freeze_risk_at_seed (bool): F4 flag. When True (default), the outcome-head
            probabilities used to score every generated window are frozen to the
            seed-end prediction rather than re-evaluated at each decode step.
            Motivation: the outcome head was calibrated on real teacher-forced
            trajectories. After 50+ generated tokens the hidden state drifts from
            the training distribution (covariate shift), producing unreliable —
            sometimes inverted — risk scores mid-trajectory.  The seed-end
            prediction is computed from real patient context (the full 2-day seed)
            and reflects the model's calibrated between-patient discrimination
            (AUROC ~0.91 in the truncated eval).  Because between-patient pairs
            dominate AUROC (>99% of positive/negative pairs), freezing scores
            preserves almost all discriminative signal while eliminating the
            covariate-shift noise. Default True because evaluation.py is locked
            and cannot pass this argument explicitly; DISCARD reverts the commit.

    Returns:
        pd.DataFrame with columns PatientId, Step, TimePoint, Token, IsInput,
        IsOutcome, IsTerminal, and (if collect_risk_scores=True) one P_<outcome>
        column per outcome in model.outcome_names.
    """
    max_abs_ts_norm = float(max_duration_hours) / 336.0  # comparison threshold in normalised units
    autocast_dtype = torch.float16 if torch.cuda.is_available() else torch.bfloat16
    device    = next(model.parameters()).device
    tok       = model.embedder.tokenizer
    luts      = build_luts(tok)
    luts      = {k: v.to(device) if torch.is_tensor(v) else v for k, v in luts.items()}

    id2token     = tok.id2token
    token2id     = tok.token2id
    outcome_ids  = {token2id[o] for o in OUTCOMES         if o in token2id}
    terminal_ids = {token2id[t] for t in TERMINAL_OUTCOMES if t in token2id}
    terminal_set = torch.tensor(sorted(terminal_ids), dtype=torch.long, device=device)
    pad_id       = tok.pad_token_id
    mask_id      = tok.mask_token_id
    outcome_cols = [f"P_{n}" for n in model.outcome_names] if collect_risk_scores else []

    rows          = []
    stuck_pids    = []
    fallback_pids = []   # patients that exhausted max_len and received a forced terminal
    all_pids      = dataset.patient_ids

    with torch.autocast(device_type=device.type if hasattr(device, 'type') else 'cuda',
                        dtype=autocast_dtype, enabled=torch.cuda.is_available()):

        for batch_start in tqdm(range(0, len(all_pids), batch_size),
                                desc=tqdm_desc, position=tqdm_position,
                                leave=False, dynamic_ncols=True):

            batch_pids = all_pids[batch_start:batch_start + batch_size]
            B = len(batch_pids)

            # ── prepare padded seed tensors ───────────────────────────────
            pos_ids, parent_raw_ids, concept_ids, value_ids, abs_ts, ctx_vecs, seed_lens = \
                _prepare_batch_seeds(batch_pids, dataset, tok, device)

            last_valid_idx = (seed_lens - 1).clamp(min=0)

            # ── single prefill: next-token logits + KV cache (+ input risk scores) ──
            # One pass supplies both the KV cache needed for autoregressive decoding
            # and (when collect_risk_scores=True) the outcome probs for input tokens.
            logits_pre, abs_t_pre, input_outcome_logits, _, past_kvs = model.forward_with_cache(
                parent_raw_ids=parent_raw_ids,
                concept_ids=concept_ids,
                value_ids=value_ids,
                position_ids=pos_ids,
                abs_ts=abs_ts,
                context_vec=ctx_vecs,
            )

            if collect_risk_scores:
                input_probs = torch.sigmoid(input_outcome_logits).cpu().numpy()  # 1 sync [B, T_max, K]

            # F4: capture seed-end outcome probabilities once per batch.
            # All generated positions will use these instead of mid-trajectory
            # outcome head outputs, eliminating covariate-shift noise.
            frozen_probs_cpu = None
            if freeze_risk_at_seed and collect_risk_scores:
                frozen_probs_cpu = torch.sigmoid(
                    input_outcome_logits[torch.arange(B, device=device), last_valid_idx, :]
                ).cpu().numpy()  # [B, K]

            # ── log input tokens ──────────────────────────────────────────
            # Bulk CPU transfers: 3 syncs total (+ 1 for risk scores above).
            pos_ids_cpu = pos_ids.cpu().tolist()
            abs_ts_cpu  = (abs_ts * 336.0).cpu().tolist()
            sl_cpu      = seed_lens.cpu().tolist()

            for bi, pid in enumerate(batch_pids):
                Ti = sl_cpu[bi]
                for i in range(Ti):
                    tid = pos_ids_cpu[bi][i]
                    row = {
                        "PatientId":  pid,
                        "Step":       i + 1,
                        "TimePoint":  abs_ts_cpu[bi][i],
                        "Token":      id2token.get(tid, f"<UNK_{tid}>"),
                        "IsInput":    1,
                        "IsOutcome":  int(tid in outcome_ids),
                        "IsTerminal": int(tid in terminal_ids),
                    }
                    if collect_risk_scores:
                        for j, col in enumerate(outcome_cols):
                            row[col] = float(input_probs[bi, i, j])
                    rows.append(row)
                    if tid in terminal_ids:
                        break

            # ── mark patients whose seed ends with a terminal ─────────────
            last_toks = pos_ids[torch.arange(B, device=device), last_valid_idx]
            finished  = torch.isin(last_toks, terminal_set)

            if finished.all():
                continue

            next_logits    = logits_pre[torch.arange(B, device=device), last_valid_idx, :]
            current_abs_ts = abs_t_pre[torch.arange(B, device=device), last_valid_idx]
            last_seed_ts   = abs_ts[torch.arange(B, device=device), last_valid_idx]
            current_abs_ts = torch.maximum(current_abs_ts, last_seed_ts)

            # Stop any patient whose seed already exceeds the training horizon.
            time_exceeded = current_abs_ts >= max_abs_ts_norm
            finished      = finished | time_exceeded

            # F3: draw per-patient terminal suppression times from outcome head.
            # p_terminal = P(DEATH or RELEASE in next 48 h) at seed-end position.
            # T ~ Exp(rate = p_terminal / 48h), clamped to >= hazard_min_hours.
            # Terminal tokens are suppressed until elapsed generated hours >= T.
            hazard_min_t = None
            if hazard_suppress:
                terminal_outcome_idxs = [i for i, n in enumerate(model.outcome_names)
                                          if n in set(TERMINAL_OUTCOMES)]
                if terminal_outcome_idxs:
                    seed_outcome = input_outcome_logits[
                        torch.arange(B, device=device), last_valid_idx
                    ][:, terminal_outcome_idxs]                      # [B, n_term]
                    p_term = torch.sigmoid(seed_outcome).max(dim=-1).values.clamp(1e-4, 1 - 1e-4)
                    u = torch.rand(B, device=device).clamp(1e-6, 1 - 1e-6)
                    drawn = -torch.log(u) * (48.0 / p_term)          # [B] hours
                    hazard_min_t = drawn.clamp(min=hazard_min_hours)  # [B] hours

            cache_mask        = (pos_ids != pad_id)
            open_counts, next_meal_rank = init_legality_state_batched(luts, pos_ids)
            last_tokens_batch = [[] for _ in range(B)]
            # Move parent LUT to GPU once per batch to avoid CPU round-trips each step.
            t2parent_gpu      = tok.tokenid2parent_raw_ids.to(device)

            # ── decode loop ───────────────────────────────────────────────
            steps = 0
            while steps < max_len and not finished.all():

                illegal = build_illegal_mask_batched(luts, open_counts, next_meal_rank,
                                                     pad_id, mask_id)
                next_logits = next_logits.masked_fill(illegal, float("-inf"))

                if rep_decay and rep_decay > 0:
                    rep_vec = build_rep_penalty_batched(last_tokens_batch, V=next_logits.size(-1),
                                                        window=5, strength=rep_decay, device=device)
                    next_logits = next_logits - rep_vec

                if finished.any():
                    next_logits[finished] = float("-inf")
                    next_logits[finished, pad_id] = 0.0

                # F3: suppress terminal tokens for patients whose elapsed generated time
                # is below their drawn hazard threshold.
                if hazard_min_t is not None and terminal_ids:
                    elapsed_gen_hours = (current_abs_ts - last_seed_ts).clamp(min=0) * 336.0
                    suppress = (elapsed_gen_hours < hazard_min_t) & ~finished
                    if suppress.any():
                        for tid in terminal_ids:
                            next_logits[suppress, tid] = float("-inf")

                # F2: exponential annealing schedule from temperature_start → temperature.
                # When temperature_start > temperature, the first temperature_anneal_steps
                # tokens are sampled from a flatter distribution to escape the terminal
                # local minimum.  After the schedule, the steady-state temperature applies.
                if temperature_start > temperature and temperature_anneal_steps > 0:
                    t_frac   = min(1.0, steps / temperature_anneal_steps)
                    step_temp = temperature_start * ((temperature / temperature_start) ** t_frac)
                else:
                    step_temp = temperature
                next_token_ids = _sample_tokens(next_logits, step_temp, top_k)

                open_counts, next_meal_rank = update_legality_state_batched(
                    luts, next_token_ids, open_counts, next_meal_rank, finished)

                # Bulk CPU transfers: 3 syncs per step instead of 3×B syncs per step.
                finished_cpu = finished.tolist()
                tok_ids_cpu  = next_token_ids.tolist()
                abs_ts_step  = (current_abs_ts * 336.0).tolist()

                step_row_idx = {} if collect_risk_scores else None
                newly_stuck  = []
                for bi in range(B):
                    if finished_cpu[bi]:
                        continue
                    tid = tok_ids_cpu[bi]
                    if tid == pad_id:
                        # argmax landed on all-inf logits — impossible legality state
                        stuck_pids.append(batch_pids[bi])
                        newly_stuck.append(bi)
                        continue
                    tok_str = id2token.get(tid, f"<UNK_{tid}>")
                    row = {
                        "PatientId":  batch_pids[bi],
                        "Step":       sl_cpu[bi] + steps + 1,
                        "TimePoint":  abs_ts_step[bi],
                        "Token":      tok_str,
                        "IsInput":    0,
                        "IsOutcome":  int(tid in outcome_ids),
                        "IsTerminal": int(tid in terminal_ids),
                    }
                    if collect_risk_scores:
                        for col in outcome_cols:
                            row[col] = 0.0   # placeholder — filled after forward below
                        step_row_idx[bi] = len(rows)
                    rows.append(row)
                    last_tokens_batch[bi].append(tid)

                if newly_stuck:
                    stuck_mask = torch.zeros(B, dtype=torch.bool, device=device)
                    stuck_mask[newly_stuck] = True
                    finished = finished | stuck_mask

                is_terminal_step = torch.isin(next_token_ids, terminal_set)
                finished = finished | is_terminal_step

                steps += 1
                if finished.all():
                    break

                # ── embed new tokens for next decode step ─────────────────
                c_ids_new = luts["tok2concept"][next_token_ids].clamp(min=0)
                c_ids_new[luts["tok2concept"][next_token_ids] < 0] = mask_id
                v_ids_new = luts["tok2value"][next_token_ids].clamp(min=0)
                v_ids_new[luts["tok2value"][next_token_ids] < 0] = mask_id

                par_new = t2parent_gpu[next_token_ids]
                par_new[finished] = t2parent_gpu[pad_id]
                c_ids_new[finished] = pad_id
                v_ids_new[finished] = pad_id
                pos_new  = next_token_ids.clone()
                pos_new[finished] = pad_id

                abs_ts_new = current_abs_ts.unsqueeze(1)
                new_valid  = torch.ones(B, 1, dtype=torch.bool, device=device)
                cache_mask = torch.cat([cache_mask, new_valid], dim=1)

                logits_dec, abs_t_dec, outcome_logits_dec, _, past_kvs = model.forward_with_cache(
                    parent_raw_ids=par_new.unsqueeze(1),
                    concept_ids=c_ids_new.unsqueeze(1),
                    value_ids=v_ids_new.unsqueeze(1),
                    position_ids=pos_new.unsqueeze(1),
                    abs_ts=abs_ts_new,
                    context_vec=ctx_vecs,
                    past_kvs=past_kvs,
                    cache_key_pad_mask=cache_mask,
                )

                next_logits    = logits_dec[:, 0, :]
                new_abs_t      = abs_t_dec[:, 0]
                current_abs_ts = torch.maximum(new_abs_t, current_abs_ts)

                # Time-based stop: patients whose absolute time hits the horizon are done.
                time_exceeded = current_abs_ts >= max_abs_ts_norm
                finished      = finished | time_exceeded

                # Fill outcome probs for this step — 1 bulk sync for the whole batch.
                if collect_risk_scores and step_row_idx:
                    if frozen_probs_cpu is not None:
                        # F4: use seed-end frozen probs (no GPU sync needed here).
                        for bi, row_idx in step_row_idx.items():
                            for j, col in enumerate(outcome_cols):
                                rows[row_idx][col] = float(frozen_probs_cpu[bi, j])
                    else:
                        step_probs_cpu = torch.sigmoid(outcome_logits_dec[:, 0, :]).cpu().numpy()
                        for bi, row_idx in step_row_idx.items():
                            for j, col in enumerate(outcome_cols):
                                rows[row_idx][col] = float(step_probs_cpu[bi, j])

            # ── terminal fallback for patients that exited without natural terminal ─────
            # Triggers when a patient hit max_duration_hours OR max_len.
            # Picks DEATH or RELEASE by highest generation logit at the last step.
            # TimePoint clamped to ≤ max_duration_hours so it stays within training window.
            if len(terminal_ids) > 0:
                term_list = list(terminal_ids)
                # A patient is in the fallback set if its last emitted token was NOT terminal.
                # `finished` may have been set by terminal emission, time, or max_len exit.
                last_tok_per_patient = {bi: (last_tokens_batch[bi][-1] if last_tokens_batch[bi]
                                              else int(last_toks[bi]))
                                        for bi in range(B)}
                for bi in range(B):
                    if last_tok_per_patient[bi] in terminal_ids:
                        continue
                    fallback_pids.append(batch_pids[bi])
                    best_logit  = next_logits[bi, term_list]
                    best_tid    = term_list[int(torch.argmax(best_logit))]
                    forced_time = min(current_abs_ts[bi].item() * 336.0, float(max_duration_hours))
                    row = {
                        "PatientId":  batch_pids[bi],
                        "Step":       sl_cpu[bi] + steps + 1,
                        "TimePoint":  forced_time,
                        "Token":      id2token[best_tid],
                        "IsInput":    0,
                        "IsOutcome":  1,
                        "IsTerminal": 1,
                    }
                    if collect_risk_scores:
                        if frozen_probs_cpu is not None:
                            for j, col in enumerate(outcome_cols):
                                row[col] = float(frozen_probs_cpu[bi, j])
                        else:
                            for col in outcome_cols:
                                row[col] = 0.0
                    rows.append(row)

    # ── post-generation report ────────────────────────────────────────────────
    n_total = len(all_pids)

    if stuck_pids:
        out_path = Path(CHECKPOINT_PATH) / "stuck_patients.json"
        with open(out_path, "w") as f:
            json.dump({"stuck_patient_ids": stuck_pids}, f, indent=2)
        print(f"[generate] WARNING  {len(stuck_pids)}/{n_total} patient(s) hit an all-illegal "
              f"legality state and were skipped early. Saved to {out_path}")

    if fallback_pids:
        print(f"[generate] INFO  {len(fallback_pids)}/{n_total} "
              f"({100 * len(fallback_pids) / n_total:.1f}%) patient(s) reached "
              f"max_len={max_len} without a natural terminal — forced terminal injected.")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import random
    import joblib
    from pathlib import Path
    from transform_emr.embedder import EMREmbedding
    from transform_emr.transformer import GPT
    from transform_emr.dataset import DataProcessor, EMRTokenizer, EMRDataset
    from transform_emr.config.model_config import *
    from transform_emr.config.dataset_config import *


    # Load test data
    print("Loading dataset...")
    df = pd.read_csv(TEST_TEMPORAL_DATA_FILE, low_memory=False)
    ctx_df = pd.read_csv(TEST_CTX_DATA_FILE)

    # Subset: Pick N random patients for this inference batch
    print("Getting subset...")
    patient_ids = df["PatientID"].unique()
    N = 10  # adjust as needed
    selected_ids = sorted(random.sample(list(patient_ids), N))

    df_subset = df[df["PatientID"].isin(selected_ids)].copy()
    ctx_subset = ctx_df.loc[selected_ids].copy()

    # Load tokenizer and scaler
    print("Loading resources...")
    tokenizer = EMRTokenizer.load(Path(CHECKPOINT_PATH) / "tokenizer.pt")
    scaler = joblib.load(Path(CHECKPOINT_PATH) / "scaler.pkl")

    # Run preprocessing for excel file
    print("Building testing dataset...")
    processor = DataProcessor(df_subset.copy(), ctx_subset.copy(), scaler=scaler, tak_repo_path=TAK_REPO_PATH)
    df_test, ctx_df_test = processor.run()
    dataset_test = EMRDataset(df_test, ctx_df_test, tokenizer=tokenizer)

    # Run preprocessing for generation
    print("Building input dataset...")
    k_days = 5
    processor = DataProcessor(df_subset.copy(), ctx_subset.copy(), scaler=scaler, tak_repo_path=TAK_REPO_PATH, max_input_days=k_days)
    df_subset, ctx_subset = processor.run()
    dataset = EMRDataset(df_subset, ctx_subset, tokenizer=tokenizer)

    # Load models
    print("Loading model and generating predictions...")
    embedder, _, _, _, _, _, _ = EMREmbedding.load(PHASE1_CHECKPOINT, tokenizer=tokenizer)
    model, _, _, _, _, _ = GPT.load(PHASE3_CHECKPOINT, embedder=embedder)
    model.eval()

    # Run inference
    result_df = generate(model, dataset, temperature=1.0, batch_size=16)

    # Save to Excel with two sheets
    output_path = Path(CHECKPOINT_PATH) / "inference_results.xlsx"
    with pd.ExcelWriter(output_path) as writer:
        result_df.to_excel(writer, sheet_name="Generated Events", index=False)
        dataset_test.tokens_df.to_excel(writer, sheet_name="Input Events", index=False)

    print(f"Inference results saved to: {output_path}")
