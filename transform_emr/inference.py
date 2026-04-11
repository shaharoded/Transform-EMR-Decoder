import torch
import torch.nn.functional as F
import pandas as pd
from tqdm.auto import tqdm
import joblib
from pathlib import Path

# ───────── local code ─────────────────────────────────────────────────── #
from transform_emr.config.dataset_config import *
from transform_emr.utils import build_luts, build_rep_penalty


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



# ───────── shared generation helpers ────────────────────────────────────── #

def _decode_token_components(tok, token_str: str):
    """Decode a position-token string into (concept_id, value_id)."""
    parts = token_str.split("_")
    concept = (
        "_".join(parts[:-2])
        if len(parts) >= 2 and parts[-2] in ("STATE", "TREND", "CONTEXT", "EVENT", "PATTERN")
        else "_".join(parts)
    )
    value = (
        "_".join(parts[:-1])
        if len(parts) >= 2 and parts[-1] in ("START", "END")
        else "_".join(parts)
    )
    return (
        tok.concept2id.get(concept, tok.mask_token_id),
        tok.value2id.get(value, tok.mask_token_id)
    )


def _build_illegal_mask(luts, open_counts, next_meal_rank, pad_id, mask_id, device):
    """Return a Boolean [V] mask of token ids that are illegal to generate next."""
    V       = luts["is_start"].numel()
    illegal = torch.zeros(V, dtype=torch.bool, device=device)
    closed  = (open_counts <= 0)
    opened  = (open_counts >  0)
    if closed.any():
        illegal[luts["end_ids_per_base"][closed]] = True
    if opened.any():
        illegal[luts["start_ids_per_base"][opened]] = True
    if opened.any():
        conflicting = luts["conflict_mat"][opened].any(dim=0) & ~opened
        if conflicting.any():
            illegal[luts["start_ids_per_base"][conflicting]] = True
    ranks   = luts["meal_rank"]
    is_meal = ranks >= 0
    if is_meal.any() and next_meal_rank is not None:
        illegal |= is_meal & (ranks != next_meal_rank)
    illegal[pad_id]  = True
    illegal[mask_id] = True
    return illegal


def _update_legality_state(luts, tid: int, open_counts, next_meal_rank, K: int):
    """Update interval/meal legality state after token `tid` was generated.
    Mutates open_counts in place. Returns updated next_meal_rank."""
    if luts["is_start"][tid]:
        open_counts[luts["base_id"][tid]] += 1
    elif luts["is_end"][tid]:
        bid = luts["base_id"][tid]
        if open_counts[bid] > 0:
            open_counts[bid] -= 1
    if K > 0:
        mr = luts["meal_rank"][tid]
        if mr >= 0:
            next_meal_rank = (mr + 1) % K
    return next_meal_rank


# ─────────────────────────────────────────────────────────────────────────── #

@torch.no_grad()
def infer_event_stream(model,
                       dataset,
                       max_len=500,
                       temperature=1.0,
                       top_k=None,
                       rep_decay=0.6,          # repetition filter, set None to disable
                       tqdm_position=0,
                       tqdm_desc='Generating'):
    """
    Generates a stream of events for each patient in the dataset, using the predicted abs time as input to the next token.
    If max_len is reached without generating a terminal token, the most probable terminal token is injected.
    Strictly legal autoregressive generation.

    • Interval legality: no END without START, no duplicate START.
    • Concept legality: no START on conceptX_value1 interval if another conceptX_value2 is open.
    • Meal legality: a meal m is illegal if its predecessor in the cycle hasn't been seen yet.
    • Repetition: Reduce probability of the last N generated tokens.

    Args:
        model: Trained GPT model.
        dataset: EMRDataset object (must contain all token components and context).
        max_len: Number of new tokens to generate.
        temperature, top_k: sampling controls (uses argmax if None)
        rep_decay: Max decay scaler to reduce from recent (5) token's logits -> avoid repetition.
        tqdm_position: Controls the TQDM hierarchy for the function that can be activated directly or externally.
        tqdm_desc: Controls the TQDM description for the function that can be activated directly or externally.
    
    Returns:
        DataFrame with PatientID, Step, Token, TimePoint, IsInput, IsOutcome, IsTerminal
    """
    device    = next(model.parameters()).device
    tok = model.embedder.tokenizer
    luts = build_luts(tok)        # is_start, is_end, base_id, meal_rank, ...
    luts = {k: v.to(device) if torch.is_tensor(v) else v for k,v in luts.items()}

    id2token  = tok.id2token
    token2id  = tok.token2id
    outcome_ids  = {token2id[o] for o in OUTCOMES if o in token2id}
    terminal_ids = {token2id[t] for t in TERMINAL_OUTCOMES if t in token2id}
    pad_id = tok.pad_token_id
    mask_id = tok.mask_token_id

    rows = []

    # -------------------------------------------------------------------------------


    for pid in tqdm(dataset.patient_ids, desc=tqdm_desc, position=tqdm_position, leave=False, dynamic_ncols=True):
        df = dataset.patient_groups[pid]
        ctx_vec = torch.tensor(dataset.context_df.loc[pid].values, dtype=torch.float32).unsqueeze(0).to(device)

        # Prepare input (same as before)
        pos_ids        = torch.tensor([df["PositionID"].tolist()],    dtype=torch.long, device=device)
        # Use the pre-padded LUT [V, Pmax] so every token maps to a fixed-length parent vector
        parent_raw_ids = tok.tokenid2parent_raw_ids[pos_ids[0]].unsqueeze(0).to(device)  # [1, T, P]
        concept_ids    = torch.tensor([df["ConceptID"].tolist()],     dtype=torch.long, device=device)
        value_ids      = torch.tensor([df["ValueID"].tolist()],       dtype=torch.long, device=device)
        # Re-normalize hours → [0,1] using the same 336h window (which were de-normalized for the df output)
        abs_ts         = torch.tensor([df["TimePoint"].tolist()],     dtype=torch.float32, device=device) / 336.0

        # Log inputs
        for i in range(pos_ids.size(1)):
            tid = pos_ids[0, i].item()
            rows.append({
                "PatientId": pid,
                "Step": i + 1,
                "TimePoint": abs_ts[0, i].item()*336.0,
                "Token": id2token.get(tid, f"<UNK_{tid}>"),
                "IsInput": 1,
                "IsOutcome": int(tid in outcome_ids),
                "IsTerminal": int(tid in terminal_ids)
            })
            if tid in terminal_ids:
                break

        # If the seed already ended, skip generation
        if pos_ids[0, -1].item() in terminal_ids:
            continue

        # === init legality state from the seed sequence =====================
        n_b = luts["start_ids"].numel()
        open_counts = torch.zeros(n_b, dtype=torch.int32, device=device)

        K = int((luts["meal_rank"] >= 0).any()) and int(luts["meal_rank"].max().item()) + 1 or 0
        next_meal_rank = None            # strict meal cycle pointer

        # walk through seed tokens to set state
        for t in range(pos_ids.size(1)):
            next_meal_rank = _update_legality_state(luts, pos_ids[0, t].item(), open_counts, next_meal_rank, K)

        # list of generated tokens for repetition filter
        last_tokens = []

        # === generation loop =================================================
        steps = 0
        while steps < max_len:
            # Unpack 4 values: logits, time, outcome_logits (ignored here), gate_logit (ignored here)
            logits, abs_t_pred, _, _ = model(
                parent_raw_ids=parent_raw_ids,
                concept_ids=concept_ids,
                value_ids=value_ids,
                position_ids=pos_ids,
                abs_ts=abs_ts,
                context_vec=ctx_vec
            ) # Need next token and time, not the binary expected outcomes

            next_logits = logits[:, -1, :].clone()  # [1,V]

            # hard legality mask
            illegal = _build_illegal_mask(luts, open_counts, next_meal_rank, pad_id, mask_id, device)
            next_logits.masked_fill_(illegal.unsqueeze(0), -float("inf"))

            # Avoid repetition (soft):
            rep_vec = build_rep_penalty(last_tokens, V=next_logits.size(-1),
                                        window=5, strength=rep_decay,
                                        device=device)
            if rep_vec.any():
                next_logits[0] -= rep_vec

            # pick token
            if top_k:
                topv, topi = torch.topk(next_logits, top_k, dim=-1)
                probs = F.softmax(topv / temperature, dim=-1)
                idx = torch.multinomial(probs, 1).item()
                next_token_id = topi[0, idx].item()
            else:
                next_token_id = torch.argmax(next_logits / temperature, dim=-1).item()

            tok_str = id2token.get(next_token_id, f"<UNK_{next_token_id}>")
            is_terminal = next_token_id in terminal_ids
            is_outcome  = next_token_id in outcome_ids
            next_parent_vec = tok.tokenid2parent_raw_ids[next_token_id].view(1, 1, -1)  # [1,1,P]

            # time prediction (normalized)
            pred_abs_norm = abs_t_pred[0, -1].item()
            pred_abs_norm = max(pred_abs_norm, abs_ts[0, -1].item())  # monotonic restriction
            pred_abs = pred_abs_norm * 336.0

            rows.append({
                "PatientId": pid,
                "Step": pos_ids.shape[1] + 1,
                "TimePoint": pred_abs,
                "Token": tok_str,
                "IsInput": 0,
                "IsOutcome": int(is_outcome),
                "IsTerminal": int(is_terminal)
            })

            # update tensors
            c_id, v_id = _decode_token_components(tok, tok_str)
            parent_raw_ids = torch.cat([parent_raw_ids, next_parent_vec.to(device)], dim=1)  # [1,T+1,P]
            concept_ids    = torch.cat([concept_ids, torch.tensor([[c_id]], device=device)], dim=1)
            value_ids      = torch.cat([value_ids,   torch.tensor([[v_id]], device=device)], dim=1)
            pos_ids        = torch.cat([pos_ids,     torch.tensor([[next_token_id]], device=device)], dim=1)
            abs_ts         = torch.cat([abs_ts,      torch.tensor([[pred_abs_norm]], device=device)], dim=1)

            # update state
            next_meal_rank = _update_legality_state(luts, next_token_id, open_counts, next_meal_rank, K)

            last_tokens.append(next_token_id)

            steps += 1
            if is_terminal:
                break

        # fallback: force a terminal if never reached
        if steps == max_len and len(terminal_ids) > 0:
            term_list = list(terminal_ids)
            term_logits = logits[:, -1, term_list]
            best = term_list[int(torch.argmax(term_logits))]
            rows.append({
                "PatientId": pid,
                "Step": pos_ids.shape[1] + 1,
                "Token": id2token[best],
                "TimePoint": pred_abs,
                "IsInput": 0,
                "IsOutcome": 1,
                "IsTerminal": 1
            })

    return pd.DataFrame(rows)


@torch.no_grad()
def generate_risk_curves(model,
                         dataset,
                         max_len=500,
                         temperature=1.0,
                         top_k=None,
                         rep_decay=0.6,
                         tqdm_position=0,
                         tqdm_desc='Risk Curves'):
    """
    Single-trajectory generation (K=1) with the outcome head read at every step.

    At each input position a single teacher-forced forward pass over the full input
    sequence supplies the outcome head probabilities.  At each generated position
    the outcome head is read from the incremental autoregressive pass.

    Returns
    -------
    pd.DataFrame with columns:
        PatientId, Step, TimePoint, Token, IsInput, IsTerminal,
        P_<outcome_name>  (one float column per outcome in model.outcome_names)

    Use the IsInput==0 rows for complication-risk evaluation; the IsInput==1 rows
    show how the model's concern evolved over the observed history.
    """
    device   = next(model.parameters()).device
    tok      = model.embedder.tokenizer
    luts     = build_luts(tok)
    luts     = {k: v.to(device) if torch.is_tensor(v) else v for k, v in luts.items()}

    id2token     = tok.id2token
    token2id     = tok.token2id
    terminal_ids = {token2id[t] for t in TERMINAL_OUTCOMES if t in token2id}
    pad_id       = tok.pad_token_id
    mask_id      = tok.mask_token_id
    outcome_cols = [f"P_{n}" for n in model.outcome_names]

    rows = []

    # ── patient loop ───────────────────────────────────────────────────── #

    for pid in tqdm(dataset.patient_ids, desc=tqdm_desc, position=tqdm_position, leave=False, dynamic_ncols=True):
        df      = dataset.patient_groups[pid]
        ctx_vec = torch.tensor(dataset.context_df.loc[pid].values, dtype=torch.float32).unsqueeze(0).to(device)

        pos_ids        = torch.tensor([df["PositionID"].tolist()],   dtype=torch.long,    device=device)
        parent_raw_ids = tok.tokenid2parent_raw_ids[pos_ids[0]].unsqueeze(0).to(device)
        concept_ids    = torch.tensor([df["ConceptID"].tolist()],    dtype=torch.long,    device=device)
        value_ids      = torch.tensor([df["ValueID"].tolist()],      dtype=torch.long,    device=device)
        abs_ts         = torch.tensor([df["TimePoint"].tolist()],    dtype=torch.float32, device=device) / 336.0

        # Single teacher-forced pass over full input → risk at each input position
        _, _, input_outcome_logits, _ = model(
            parent_raw_ids=parent_raw_ids, concept_ids=concept_ids,
            value_ids=value_ids, position_ids=pos_ids,
            abs_ts=abs_ts, context_vec=ctx_vec
        )
        input_probs = torch.sigmoid(input_outcome_logits[0])  # [T_input, num_outcomes]

        for i in range(pos_ids.size(1)):
            tid = pos_ids[0, i].item()
            row = {
                "PatientId":  pid,
                "Step":       i + 1,
                "TimePoint":  abs_ts[0, i].item() * 336.0,
                "Token":      id2token.get(tid, f"<UNK_{tid}>"),
                "IsInput":    1,
                "IsTerminal": int(tid in terminal_ids),
            }
            for j, col in enumerate(outcome_cols):
                row[col] = input_probs[i, j].item()
            rows.append(row)
            if tid in terminal_ids:
                break

        if pos_ids[0, -1].item() in terminal_ids:
            continue

        # === init legality state from the seed sequence =====================
        n_b = luts["start_ids"].numel()
        open_counts = torch.zeros(n_b, dtype=torch.int32, device=device)

        K = int((luts["meal_rank"] >= 0).any()) and int(luts["meal_rank"].max().item()) + 1 or 0
        next_meal_rank = None            # strict meal cycle pointer

        # walk through seed tokens to set state
        for t in range(pos_ids.size(1)):
            next_meal_rank = _update_legality_state(luts, pos_ids[0, t].item(), open_counts, next_meal_rank, K)

        last_tokens = []
        steps = 0

        while steps < max_len:
            logits, abs_t_pred, outcome_logits, _ = model(
                parent_raw_ids=parent_raw_ids, concept_ids=concept_ids,
                value_ids=value_ids, position_ids=pos_ids,
                abs_ts=abs_ts, context_vec=ctx_vec
            )
            next_logits = logits[:, -1, :].clone()
            next_probs  = torch.sigmoid(outcome_logits[0, -1, :])  # [num_outcomes]

            illegal = _build_illegal_mask(luts, open_counts, next_meal_rank, pad_id, mask_id, device)
            next_logits.masked_fill_(illegal.unsqueeze(0), -float("inf"))

            rep_vec = build_rep_penalty(last_tokens, V=next_logits.size(-1),
                                        window=5, strength=rep_decay, device=device)
            if rep_vec.any():
                next_logits[0] -= rep_vec

            if top_k:
                topv, topi = torch.topk(next_logits, top_k, dim=-1)
                probs = F.softmax(topv / temperature, dim=-1)
                idx   = torch.multinomial(probs, 1).item()
                next_token_id = topi[0, idx].item()
            else:
                next_token_id = torch.argmax(next_logits / temperature, dim=-1).item()

            tok_str     = id2token.get(next_token_id, f"<UNK_{next_token_id}>")
            is_terminal = next_token_id in terminal_ids

            pred_abs_norm = abs_t_pred[0, -1].item()
            pred_abs_norm = max(pred_abs_norm, abs_ts[0, -1].item())
            pred_abs      = pred_abs_norm * 336.0

            row = {
                "PatientId":  pid,
                "Step":       pos_ids.shape[1] + 1,
                "TimePoint":  pred_abs,
                "Token":      tok_str,
                "IsInput":    0,
                "IsTerminal": int(is_terminal),
            }
            for j, col in enumerate(outcome_cols):
                row[col] = next_probs[j].item()
            rows.append(row)

            c_id, v_id = _decode_token_components(tok, tok_str)
            next_parent_vec = tok.tokenid2parent_raw_ids[next_token_id].view(1, 1, -1)
            parent_raw_ids = torch.cat([parent_raw_ids, next_parent_vec.to(device)], dim=1)
            concept_ids    = torch.cat([concept_ids, torch.tensor([[c_id]], device=device)], dim=1)
            value_ids      = torch.cat([value_ids,   torch.tensor([[v_id]], device=device)], dim=1)
            pos_ids        = torch.cat([pos_ids,     torch.tensor([[next_token_id]], device=device)], dim=1)
            abs_ts         = torch.cat([abs_ts,      torch.tensor([[pred_abs_norm]], device=device)], dim=1)

            next_meal_rank = _update_legality_state(luts, next_token_id, open_counts, next_meal_rank, K)

            last_tokens.append(next_token_id)
            steps += 1
            if is_terminal:
                break

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
    k_days=5
    processor = DataProcessor(df_subset.copy(), ctx_subset.copy(), scaler=scaler, tak_repo_path=TAK_REPO_PATH, max_input_days=k_days)
    df_subset, ctx_subset = processor.run()
    dataset = EMRDataset(df_subset, ctx_subset, tokenizer=tokenizer)

    # Load models
    print("Loading model and generating predictions...")
    embedder, _, _, _, _, _, _ = EMREmbedding.load(PHASE1_CHECKPOINT, tokenizer=tokenizer)
    model, _, _, _, _, _ = GPT.load(PHASE3_CHECKPOINT, embedder=embedder)
    model.eval()

    # Run inference
    result_df = infer_event_stream(model, dataset, temperature=1.0)  # optional: adjust temperature

    # Save to Excel with two sheets
    output_path = Path(CHECKPOINT_PATH) / "inference_results.xlsx"
    with pd.ExcelWriter(output_path) as writer:
        result_df.to_excel(writer, sheet_name="Generated Events", index=False)
        dataset_test.tokens_df.to_excel(writer, sheet_name="Input Events", index=False)

    print(f"Inference results saved to: {output_path}")

