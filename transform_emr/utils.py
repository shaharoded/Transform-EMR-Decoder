"""
utils.py
==============

general util functions for the package
"""
import matplotlib.pyplot as plt
import torch
from collections import Counter
from math import exp


# ───────── local code ─────────────────────────────────────────────────── #
from transform_emr.config.dataset_config import meal2rank
from transform_emr.config.model_config import TRAINING_SETTINGS



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


def get_multi_hot_targets(position_ids, padding_idx, vocab_size, k):
    """
    For each timestep t, mark all tokens in [t+1, t+k] in a multi-hot vector.
    """
    B, T = position_ids.shape
    targets = torch.zeros((B, T, vocab_size), dtype=torch.float32, device=position_ids.device)

    for step in range(1, k + 1):
        if T - step <= 0:
            continue
        for b in range(B):
            for t in range(T - step):
                token = position_ids[b, t + step].item()
                if token != padding_idx:
                    targets[b, t, token] = 1.0
    return targets


def get_penalty_weight(epoch, max_weight=TRAINING_SETTINGS.get("penalty_weight")):
    """
    Penalty weight schedule:
    - [0, warmup): 0.0 (no penalty)
    - [warmup, 2*warmup): sharp sigmoid ramp-up
    - [2*warmup, ∞): max_weight
    """
    warmup = TRAINING_SETTINGS.get("warmup_epochs")

    if epoch < warmup:
        return 0.0
    elif epoch >= 2 * warmup:
        return max_weight

    # progress ∈ [0, 1] over the ramp phase
    progress = (epoch - warmup) / warmup
    sharpness = 16  # Increase for steeper burst
    weight = 1 / (1 + exp(-sharpness * (progress - 0.75)))

    return max_weight * weight


def penalty_meal_order(predicted_tokens, id2token, device=None):
    """
    Penalizes incorrect meal ordering in a cyclic daily schedule.
    Expects meals to follow MEAL_BREAKFAST → MEAL_LUNCH → MEAL_DINNER → MEAL_NIGHT → MEAL_BREAKFAST.

    Args:
        predicted_tokens (Tensor): [B, T] tensor of predicted token IDs.
        id2token (dict): Mapping from token ID to token string.

    Returns:
        Torch.Tensor: Total penalty score.
    """
    total_penalty = 0.0
    total_transitions = 0

    for b in range(predicted_tokens.size(0)):
        sequence = [
            meal2rank[id2token.get(predicted_tokens[b, t].item())]
            for t in range(predicted_tokens.size(1))
            if id2token.get(predicted_tokens[b, t].item()) in meal2rank
        ]

        for i in range(1, len(sequence)):
            expected = (sequence[i - 1] + 1) % len(meal2rank)
            total_transitions += 1
            if sequence[i] != expected:
                total_penalty += 1.0

    if total_transitions == 0:
        return torch.tensor(0.0, device=device)

    normalized = total_penalty / total_transitions
    return torch.tensor(normalized, dtype=torch.float32, device=device)


def penalty_hallucinated_intervals(predicted_tokens, target_tokens, id2token, start_suffix="_START", end_suffix="_END", device=None):
    """
    Computes a normalized penalty for hallucinated or structurally invalid intervals.

    This function penalizes the model for:
      - Predicting unmatched START/END pairs (e.g., START without END).
      - Predicting END before a corresponding START.
      - Generating interval markers that do not appear in the target sequence.

    The penalty is normalized as:
        unmatched predicted intervals / total predicted intervals

    Args:
        predicted_tokens (Tensor): [B, T] tensor of predicted token IDs.
        target_tokens    (Tensor): [B, T] tensor of ground-truth token IDs.
        id2token (dict): Mapping from token ID to token string.
        start_suffix (str): Suffix indicating interval start (default="_START").
        end_suffix (str): Suffix indicating interval end (default="_END").
        device (torch.device): Device for returned tensor.

    Returns:
        torch.Tensor: A scalar tensor with normalized penalty in [0, 1].
    """
    total_unmatched = 0      # total unmatched (hallucinated or malformed) intervals across batch
    total_predicted = 0      # total interval tokens predicted across batch
    B, T = predicted_tokens.shape

    def extract_intervals(tokens):
        """
        Extracts a sequence of (concept, marker) pairs like ('HYPO', 'START').

        Args:
            tokens (Tensor): [T] vector of token IDs.

        Returns:
            List[Tuple[str, str]]: e.g., [('HYPO', 'START'), ('HYPO', 'END')]
        """
        sequence = []
        for t in range(T):
            tok_str = id2token[tokens[t].item()]
            if tok_str.endswith(start_suffix):
                base = tok_str[:-len(start_suffix)]
                sequence.append((base, "START"))
            elif tok_str.endswith(end_suffix):
                base = tok_str[:-len(end_suffix)]
                sequence.append((base, "END"))
        return sequence

    def decompose(sequence):
        """
        Identifies unmatched interval markers in a sequence using a simple stack-based parser.

        Args:
            sequence (List[Tuple[str, str]]): extracted interval sequence

        Returns:
            List[Tuple[str, str]]: unmatched (base, marker) items (e.g., ('HYPO', 'END'))
        """
        incomplete = []
        stack = []
        for base, kind in sequence:
            if kind == "START":
                stack.append(base)
            else:  # kind == "END"
                if base in stack:
                    stack.remove(base)  # matched START/END pair
                else:
                    incomplete.append((base, "END"))
        # Remaining STARTs in stack are unmatched
        incomplete += [(base, "START") for base in stack]
        return incomplete

    for b in range(B):
        pred_seq = extract_intervals(predicted_tokens[b])
        tgt_seq = extract_intervals(target_tokens[b])

        # Decompose both sequences to unmatched START/END tokens
        pred_incomplete = decompose(pred_seq)
        tgt_incomplete = decompose(tgt_seq)

        total_predicted += len(pred_seq)

        # Convert unmatched sequences to counters for filtering
        pred_filtered = Counter(pred_incomplete)
        tgt_filtered = Counter(tgt_incomplete)

        # Remove hallucinations that also appear in the target as incomplete (no penalty)
        for k in tgt_filtered:
            if k in pred_filtered:
                pred_filtered[k] -= min(pred_filtered[k], tgt_filtered[k])
                if pred_filtered[k] <= 0:
                    del pred_filtered[k]

        # Remaining are hallucinated/unjustified by target
        total_unmatched += sum(pred_filtered.values())

    if total_predicted == 0:
        return torch.tensor(0.0, device=device)

    # Normalize penalty: unmatched / predicted
    normalized = total_unmatched / total_predicted
    return torch.tensor(normalized, dtype=torch.float32, device=device)


def penalty_false_positives(predictions, targets, token_weights, important_token_ids, threshold=0.5):
    """
    Penalizes overgeneration of important tokens, scaled by their importance weights.

    Only considers tokens in `important_token_ids`. Penalizes if a token is predicted more times
    than it appears in the ground truth (multi-hot). This prevents false positives on critical concepts.

    Args:
        predictions (Tensor): [B, T, V] — raw logits or probabilities
        targets     (Tensor): [B, T, V] — ground-truth multi-hot labels
        token_weights (Tensor): [V] — per-token importance weights
        important_token_ids (Iterable[int]): list or set of token IDs to consider for penalty
        threshold (float): threshold above which a token is considered predicted

    Returns:
        Torch.Tensor: penalty
    """
    pred_bin = (predictions > threshold).float()         # [B, T, V]
    false_pos = torch.clamp(pred_bin - targets, min=0.0) # [B, T, V]
    fp_counts = false_pos.sum(dim=(0, 1))                # [V]

    # Mask and weight only the important token IDs
    important_ids = torch.tensor(list(important_token_ids), device=predictions.device)
    penalties = fp_counts[important_ids] * token_weights[important_ids]

    penalty = penalties.sum()
    max_possible_fp = predictions.size(0) * predictions.size(1)  # B * T norm
    normalized_penalty = penalty / max_possible_fp
    return normalized_penalty.clamp(0.0, 1.0)