import torch
from torch.nn import functional as F
import pandas as pd
from tqdm import tqdm
import joblib
from pathlib import Path

# ───────── local code ─────────────────────────────────────────────────── #
from transform_emr.config.dataset_config import *


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


def infer_event_stream(model, dataset, max_len=500, temperature=1.0):
    """
    Generates a stream of events for each patient in the dataset.
    NOTE: Inference generates a token every 1h for current version. An upgrade might try to infer the next token's 
          relative time and use that time for generation.
    Args:
        model: Trained GPT model.
        dataset: EMRDataset object (must contain all token components and context).
        max_len: Number of new tokens to generate.
        temprature: Controls the model's creativity. >1 makes model more creative but can cause hallucinations
    
    Returns:
        DataFrame with PatientID, Step, Token, IsInput, IsOutcome, IsTerminal
    """
    tokenizer = model.embedder.tokenizer
    token2id = tokenizer.token2id
    id2token = tokenizer.id2token
    outcome_ids = {token2id[o] for o in OUTCOMES if o in token2id}
    terminal_ids = {token2id[t] for t in TERMINAL_OUTCOMES if t in token2id}
    
    device = next(model.parameters()).device
    rows = []

    def decode_token_components(token):
        parts = token.split("_")
        raw = parts[0]
        concept = "_".join(parts[:2]) if len(parts) > 1 else parts[0]
        value = "_".join(parts[:-1]) if parts[-1] in ("START", "END") else "_".join(parts)
        return (
            tokenizer.rawconcept2id.get(raw, tokenizer.mask_token_id),
            tokenizer.concept2id.get(concept, tokenizer.mask_token_id),
            tokenizer.value2id.get(value, tokenizer.mask_token_id)
        )

    for pid in tqdm(dataset.patient_ids, desc="Generating"):
        df = dataset.patient_groups[pid]
        ctx_vec = torch.tensor(dataset.context_df.loc[pid].values, dtype=torch.float32).unsqueeze(0).to(device)

        # Prepare input tensors
        raw_ids     = torch.tensor([df["RawConceptID"].tolist()], dtype=torch.long, device=device)
        concept_ids = torch.tensor([df["ConceptID"].tolist()], dtype=torch.long, device=device)
        value_ids   = torch.tensor([df["ValueID"].tolist()], dtype=torch.long, device=device)
        pos_ids     = torch.tensor([df["PositionID"].tolist()], dtype=torch.long, device=device)
        delta_ts    = torch.tensor([df["TimeDelta"].tolist()], dtype=torch.float32, device=device)
        abs_ts      = torch.tensor([df["TimePoint"].tolist()], dtype=torch.float32, device=device)

        seq_len = pos_ids.size(1)

        # Log all inputs
        rows.append({
            "PatientID": pid, "Step": 0, "Token": "[CTX]",
            "IsInput": 1, "IsOutcome": 0, "IsTerminal": 0
        })
        for i in range(seq_len):
            tok_id = pos_ids[0, i].item()
            rows.append({
                "PatientID": pid,
                "Step": i + 1,
                "TimeDelta": delta_ts[0, i].item(),
                "TimePoint": abs_ts[0, i].item(),
                "Token": id2token.get(tok_id, f"<UNK_{tok_id}>"),
                "IsInput": 1,
                "IsOutcome": int(tok_id in outcome_ids),
                "IsTerminal": int(tok_id in terminal_ids)
            })
            if tok_id in terminal_ids:
                break

        # Begin autoregressive generation
        steps = 0
        while steps < max_len:
            with torch.no_grad():
                logits, delta_t_preds = model(
                    raw_concept_ids=raw_ids,
                    concept_ids=concept_ids,
                    value_ids=value_ids,
                    position_ids=pos_ids,
                    delta_ts=delta_ts,
                    abs_ts=abs_ts,
                    context_vec=ctx_vec
                )
                next_logits = logits[:, -1, :]  # shape: [1, V]
                
                # Ensure no [MASK] as output
                mask_id = model.embedder.tokenizer.token2id.get("[MASK]")
                next_logits[0, mask_id] = -float("inf")

                # Softmax
                next_probs = F.softmax(next_logits / temperature, dim=-1)

                # Avoid selecting [PAD] token
                pad_id = model.embedder.tokenizer.token2id["[PAD]"]
                next_probs[0, pad_id] = 0.0
                next_probs = next_probs / next_probs.sum()  # re-normalize to 1.0

                next_token_id = torch.multinomial(next_probs, num_samples=1).item()

            tok_str = id2token.get(next_token_id, f"<UNK_{next_token_id}>")
            is_outcome = next_token_id in outcome_ids
            is_terminal = next_token_id in terminal_ids

            # Predicted delta for next token
            pred_delta_t = delta_t_preds[0, -1].item()
            pred_delta_t = min(max(pred_delta_t, 0.0), 720.0)  # Avoid negative or NaN, limit next token to 30 days

            rows.append({
                "PatientID": pid,
                "Step": raw_ids.shape[1] + 1,
                "TimeDelta": pred_delta_t,
                "TimePoint": abs_ts[0, -1].item(),
                "Token": tok_str,
                "IsInput": 0,
                "IsOutcome": int(is_outcome),
                "IsTerminal": int(is_terminal)
            })

            if is_terminal:
                break

            # Get generated tokens components for future decode
            raw_id, concept_id, value_id = decode_token_components(tok_str)

            raw_ids     = torch.cat([raw_ids, torch.tensor([[raw_id]], device=device)], dim=1)
            concept_ids = torch.cat([concept_ids, torch.tensor([[concept_id]], device=device)], dim=1)
            value_ids   = torch.cat([value_ids, torch.tensor([[value_id]], device=device)], dim=1)
            pos_ids     = torch.cat([pos_ids, torch.tensor([[next_token_id]], device=device)], dim=1)
            delta_ts = torch.cat([delta_ts, torch.tensor([[pred_delta_t]], device=device)], dim=1)
            abs_ts = torch.cat([abs_ts, abs_ts[:, -1:] + pred_delta_t], dim=1)

            steps += 1

    return pd.DataFrame(rows)



if __name__ == "__main__":
    import random
    import joblib
    from pathlib import Path
    from transform_emr.embedder import EMREmbedding
    from transform_emr.transformer import GPT
    from transform_emr.dataset import DataProcessor, EMRTokenizer, EMRDataset
    from transform_emr.config.model_config import *

    # Load test data
    print("Loading dataset...")
    df = pd.read_csv(TEST_TEMPORAL_DATA_FILE, low_memory=False)
    ctx_df = pd.read_csv(TEST_CTX_DATA_FILE)

    # ⚠️ Subset: Pick N random patients for this inference batch
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
    processor = DataProcessor(df_subset.copy(), ctx_subset.copy(), scaler=scaler)
    df_test, ctx_df_test = processor.run()
    dataset_test = EMRDataset(df_test, ctx_df_test, tokenizer=tokenizer)
    
    # Run preprocessing for generation
    print("Building input dataset...")
    k_days=5
    processor = DataProcessor(df_subset.copy(), ctx_subset.copy(), scaler=scaler, max_input_days=k_days)
    df_subset, ctx_subset = processor.run()
    dataset = EMRDataset(df_subset, ctx_subset, tokenizer=tokenizer)

    # Load models
    print("Loading model and generating predictions...")
    embedder, _, _, _, _ = EMREmbedding.load(EMBEDDER_CHECKPOINT, tokenizer=tokenizer)
    model, _, _, _, _ = GPT.load(TRANSFORMER_CHECKPOINT, embedder=embedder)
    model.eval()

    # Run inference
    result_df = infer_event_stream(model, dataset, temperature=1.0)  # optional: adjust temperature

    # Save to Excel with two sheets
    output_path = Path(CHECKPOINT_PATH) / "inference_results.xlsx"
    with pd.ExcelWriter(output_path) as writer:
        result_df.to_excel(writer, sheet_name="Generated Events", index=False)
        dataset_test.tokens_df.to_excel(writer, sheet_name="Input Events", index=False)

    print(f"Inference results saved to: {output_path}")

