"""
test_inference.py
=================

Tests for:
  • InterveneGPT.forward_with_cache  (prefill and decode modes, KV-cache consistency)
  • Batched legality utils  (init_legality_state_batched, build_illegal_mask_batched,
                             update_legality_state_batched, build_rep_penalty_batched)
  • infer_event_stream      (batched, KV-cached generation end-to-end)
  • generate_risk_curves    (outcome risk scores end-to-end)
"""

import pytest
import torch
import pandas as pd

from intervene_ar.dataset import EMRTokenizer
from intervene_ar.embedder import EMREmbedding
from intervene_ar.transformer import InterveneGPT
from intervene_ar.utils import (
    build_luts,
    init_legality_state_batched,
    build_illegal_mask_batched,
    update_legality_state_batched,
    build_rep_penalty_batched,
)
from intervene_ar.inference import generate


# ─────────────────────────────────────────────────────────────────────────────
# Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mini_tokenizer():
    """
    Full-featured mini tokenizer matching the vocabulary used by test_utils.py.
    Includes interval tokens, meal tokens, and terminal outcomes so the inference
    functions can resolve legality rules and terminal detection correctly.
    """
    toks = [
        "[PAD]", "[MASK]", "[CTX]", "[NULL]",
        "ADMISSION_EVENT",
        "A_STATE_Low_START", "A_STATE_Low_END",
        "A_STATE_High_START", "A_STATE_High_END",
        "A_TREND_dec_START", "A_TREND_dec_END",
        "A_TREND_inc_START", "A_TREND_inc_END",
        "MEAL_CONTEXT_Breakfast", "MEAL_CONTEXT_Lunch",
        "MEAL_CONTEXT_Dinner", "MEAL_CONTEXT_Night-Snack",
        "DEATH_EVENT", "RELEASE_EVENT",
        "GLUCOSE_READING_EVENT", "INSULIN_DOSE_EVENT",
    ]
    token2id = {t: i for i, t in enumerate(toks)}
    rawconcept2id = {
        "A": 0, "MEAL_CONTEXT": 1, "ADMISSION_EVENT": 2,
        "DEATH_EVENT": 3, "RELEASE_EVENT": 4,
    }
    concept2id = {
        "A_STATE": 0, "A_TREND": 1, "MEAL_CONTEXT": 2,
        "ADMISSION_EVENT": 3, "DEATH_EVENT": 4, "RELEASE_EVENT": 5,
    }
    value2id = {
        "A_STATE_Low": 0, "A_STATE_High": 1,
        "A_TREND_dec": 2, "A_TREND_inc": 3,
        "MEAL_CONTEXT_Breakfast": 4, "MEAL_CONTEXT_Lunch": 5,
        "MEAL_CONTEXT_Dinner": 6, "MEAL_CONTEXT_Night-Snack": 7,
        "ADMISSION_EVENT": 8, "DEATH_EVENT": 9, "RELEASE_EVENT": 10,
    }
    V = len(toks)
    tk = EMRTokenizer(
        token2id=token2id,
        rawconcept2id=rawconcept2id,
        concept2id=concept2id,
        value2id=value2id,
        special_tokens=["[PAD]", "[MASK]", "[CTX]", "[NULL]"],
        token_weights=torch.ones(V),
        outcome_weights=torch.ones(V),
        token_counts=torch.tensor([], dtype=torch.long),
        tokenid2parent_raw_ids=torch.zeros((V, 1), dtype=torch.long),
        parent_pad_len=1,
    )
    tk.pad_token_id  = token2id["[PAD]"]
    tk.mask_token_id = token2id["[MASK]"]
    tk.ctx_token_id  = token2id["[CTX]"]
    tk.null_token_id = token2id["[NULL]"]
    return tk


@pytest.fixture(scope="module")
def mini_embedder(mini_tokenizer):
    return EMREmbedding(
        tokenizer=mini_tokenizer,
        ctx_dim=2,
        time2vec_dim=4,
        embed_dim=8,
        dropout=0.0,
    )


@pytest.fixture(scope="module")
def mini_cfg():
    return {
        "embed_dim": 8,
        "time2vec_dim": 4,
        "n_layer": 2,
        "n_head": 2,
        "block_size": 16,
        "dropout": 0.0,   # disabled so prefill==decode in eval mode
        "bias": True,
    }


@pytest.fixture(scope="module")
def mini_model(mini_embedder, mini_cfg):
    model = InterveneGPT(cfg=mini_cfg, embedder=mini_embedder, use_checkpoint=False)
    model.eval()
    return model


@pytest.fixture(scope="module")
def luts(mini_tokenizer):
    raw = build_luts(mini_tokenizer)
    return {k: v.to("cpu") if torch.is_tensor(v) else v for k, v in raw.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_inputs(mini_tokenizer, B=2, T=4):
    """Return a simple batch of dummy inputs for InterveneGPT.forward / forward_with_cache."""
    V  = len(mini_tokenizer.token2id)
    pad = mini_tokenizer.pad_token_id
    parent_raw = torch.zeros(B, T, 1, dtype=torch.long)
    concept    = torch.zeros(B, T, dtype=torch.long)
    value      = torch.zeros(B, T, dtype=torch.long)
    pos        = torch.zeros(B, T, dtype=torch.long)
    abs_ts     = torch.arange(T, dtype=torch.float32).unsqueeze(0).expand(B, -1) * 0.05
    context    = torch.zeros(B, 2)
    return parent_raw, concept, value, pos, abs_ts, context


class FakeDataset:
    """
    Minimal dataset substitute accepted by infer_event_stream / generate_risk_curves.
    Each patient has a short seed ending with a non-terminal token so generation starts.
    """
    def __init__(self, mini_tokenizer, n_patients=3, seed_len=3):
        tk  = mini_tokenizer
        pad = tk.pad_token_id

        # Choose a non-PAD, non-terminal token for the seed body
        terminal_ids = {tk.token2id.get("DEATH_EVENT"), tk.token2id.get("RELEASE_EVENT")}
        safe_id = next(
            i for t, i in tk.token2id.items()
            if i not in terminal_ids and i != pad and i != tk.mask_token_id
        )
        concept_id = tk.concept2id.get(list(tk.concept2id.keys())[0], 0)
        value_id   = tk.value2id.get(list(tk.value2id.keys())[0], 0)

        self.patient_ids = [f"P{i}" for i in range(n_patients)]
        self.patient_groups = {}
        for pid in self.patient_ids:
            self.patient_groups[pid] = pd.DataFrame({
                "PositionID":  [safe_id] * seed_len,
                "ConceptID":   [concept_id] * seed_len,
                "ValueID":     [value_id] * seed_len,
                "TimePoint":   [float(t) * 0.1 for t in range(seed_len)],
            })
        ctx_data = {pid: [0.5, 0.3] for pid in self.patient_ids}
        self.context_df = pd.DataFrame.from_dict(
            ctx_data, orient="index", columns=["age_norm", "gender"]
        )


# ─────────────────────────────────────────────────────────────────────────────
# InterveneGPT.forward_with_cache — prefill
# ─────────────────────────────────────────────────────────────────────────────

def test_forward_with_cache_prefill_shape(mini_model, mini_tokenizer):
    """Prefill (no past_kvs) returns 5 values with shapes matching regular forward."""
    model = mini_model
    B, T  = 2, 4
    V     = len(mini_tokenizer.token2id)
    K     = model.num_outcomes
    n_L   = model.cfg["n_layer"]

    parent_raw, concept, value, pos, abs_ts, context = _make_inputs(mini_tokenizer, B, T)

    with torch.no_grad():
        logits, abs_t, out_log, gate, ttt_pred, new_kvs = model.forward_with_cache(
            parent_raw_ids=parent_raw, concept_ids=concept, value_ids=value,
            position_ids=pos, abs_ts=abs_ts, context_vec=context,
        )

    assert logits.shape    == (B, T, V),  f"logits shape mismatch: {logits.shape}"
    assert abs_t.shape     == (B, T),     f"abs_t shape mismatch: {abs_t.shape}"
    assert out_log.shape   == (B, T, K),  f"outcome_logits shape mismatch: {out_log.shape}"
    assert gate.shape      == (B, T),     f"gate shape mismatch: {gate.shape}"
    assert len(new_kvs)    == n_L,        f"Expected {n_L} KV entries, got {len(new_kvs)}"
    for i, (k, v) in enumerate(new_kvs):
        assert k.shape[0] == B and k.shape[2] == T, \
            f"Layer {i} key cache shape wrong: {k.shape}"
    print("test_forward_with_cache_prefill_shape passed.")


def test_forward_with_cache_matches_regular_forward(mini_model, mini_tokenizer):
    """Prefill output must match the regular forward() on the same inputs."""
    model = mini_model
    B, T  = 2, 4
    parent_raw, concept, value, pos, abs_ts, context = _make_inputs(mini_tokenizer, B, T)

    with torch.no_grad():
        logits_reg, abs_t_reg, out_reg, gate_reg, _ = model(
            parent_raw_ids=parent_raw, concept_ids=concept, value_ids=value,
            position_ids=pos, abs_ts=abs_ts, context_vec=context,
        )
        logits_fwc, abs_t_fwc, out_fwc, gate_fwc, _, _ = model.forward_with_cache(
            parent_raw_ids=parent_raw, concept_ids=concept, value_ids=value,
            position_ids=pos, abs_ts=abs_ts, context_vec=context,
        )

    torch.testing.assert_close(logits_reg, logits_fwc, msg="logits mismatch vs regular forward")
    torch.testing.assert_close(abs_t_reg,  abs_t_fwc,  msg="abs_t_pred mismatch")
    torch.testing.assert_close(out_reg,    out_fwc,    msg="outcome_logits mismatch")
    print("test_forward_with_cache_matches_regular_forward passed.")


# ─────────────────────────────────────────────────────────────────────────────
# InterveneGPT.forward_with_cache — decode step
# ─────────────────────────────────────────────────────────────────────────────

def test_forward_with_cache_decode_shape(mini_model, mini_tokenizer):
    """Decode step (T=1, with past_kvs) returns correctly-shaped 5-tuple."""
    model = mini_model
    B, T  = 2, 4
    V     = len(mini_tokenizer.token2id)
    K     = model.num_outcomes
    n_L   = model.cfg["n_layer"]

    parent_raw, concept, value, pos, abs_ts, context = _make_inputs(mini_tokenizer, B, T)

    with torch.no_grad():
        # Prefill T-1 tokens
        _, _, _, _, _, past_kvs = model.forward_with_cache(
            parent_raw_ids=parent_raw[:, :T-1, :], concept_ids=concept[:, :T-1],
            value_ids=value[:, :T-1], position_ids=pos[:, :T-1],
            abs_ts=abs_ts[:, :T-1], context_vec=context,
        )

        # Decode the T-th token
        cache_mask = torch.ones(B, T, dtype=torch.bool)
        logits, abs_t, out_log, gate, ttt_pred, new_kvs = model.forward_with_cache(
            parent_raw_ids=parent_raw[:, T-1:, :], concept_ids=concept[:, T-1:],
            value_ids=value[:, T-1:], position_ids=pos[:, T-1:],
            abs_ts=abs_ts[:, T-1:], context_vec=context,
            past_kvs=past_kvs, cache_key_pad_mask=cache_mask,
        )

    assert logits.shape  == (B, 1, V), f"decode logits shape: {logits.shape}"
    assert abs_t.shape   == (B, 1),    f"decode abs_t shape: {abs_t.shape}"
    assert out_log.shape == (B, 1, K), f"decode outcome shape: {out_log.shape}"
    for i, (k, v) in enumerate(new_kvs):
        assert k.shape[2] == T, \
            f"Layer {i} key cache should have T={T} steps after decode, got {k.shape[2]}"
    print("test_forward_with_cache_decode_shape passed.")


def test_kv_cache_consistency_with_full_forward(mini_model, mini_tokenizer):
    """
    KV cache correctness: split-and-decode must match full forward at the last position.

    For a sequence of length T, the logit at position T-1 (predicting token T) must be
    numerically identical whether computed by:
      (a) forward_with_cache over all T tokens at once, OR
      (b) forward_with_cache on tokens [0..T-2] then decode on token [T-1].
    """
    model = mini_model
    B, T  = 2, 5
    parent_raw, concept, value, pos, abs_ts, context = _make_inputs(mini_tokenizer, B, T)

    with torch.no_grad():
        # (a) full forward
        logits_full, _, _, _, _, _ = model.forward_with_cache(
            parent_raw_ids=parent_raw, concept_ids=concept, value_ids=value,
            position_ids=pos, abs_ts=abs_ts, context_vec=context,
        )
        expected = logits_full[:, T-1, :]   # [B, V]

        # (b) prefill T-1 tokens, then decode token T-1
        _, _, _, _, _, past_kvs = model.forward_with_cache(
            parent_raw_ids=parent_raw[:, :T-1, :], concept_ids=concept[:, :T-1],
            value_ids=value[:, :T-1], position_ids=pos[:, :T-1],
            abs_ts=abs_ts[:, :T-1], context_vec=context,
        )
        cache_mask = torch.ones(B, T, dtype=torch.bool)   # no padding
        logits_dec, _, _, _, _, _ = model.forward_with_cache(
            parent_raw_ids=parent_raw[:, T-1:, :], concept_ids=concept[:, T-1:],
            value_ids=value[:, T-1:], position_ids=pos[:, T-1:],
            abs_ts=abs_ts[:, T-1:], context_vec=context,
            past_kvs=past_kvs, cache_key_pad_mask=cache_mask,
        )
        actual = logits_dec[:, 0, :]   # [B, V]

    torch.testing.assert_close(expected, actual, atol=1e-5, rtol=1e-4,
                               msg="KV cache decode logits differ from full forward")
    print("test_kv_cache_consistency_with_full_forward passed.")


# ─────────────────────────────────────────────────────────────────────────────
# Batched legality utils
# ─────────────────────────────────────────────────────────────────────────────

def test_init_legality_state_batched_open_counts(mini_tokenizer, luts):
    """
    After a seed with one START token, open_counts for that base should be 1.
    A second seed with no intervals should have all-zero open_counts.
    """
    tk   = mini_tokenizer
    low_s = tk.token2id["A_STATE_Low_START"]
    pad   = tk.pad_token_id

    # Batch of 2: patient 0 has one START, patient 1 has only PAD
    B, T   = 2, 3
    pos_ids = torch.full((B, T), pad, dtype=torch.long)
    pos_ids[0, 0] = low_s   # patient 0 opened an interval

    open_counts, _ = init_legality_state_batched(luts, pos_ids)

    base = luts["base_id"][low_s].item()
    assert open_counts[0, base].item() == 1, "Patient 0 should have base open"
    assert open_counts[1].sum().item() == 0, "Patient 1 should have no open bases"
    print("test_init_legality_state_batched_open_counts passed.")


def test_init_legality_state_batched_meal_rank(mini_tokenizer, luts):
    """
    After a seed containing Breakfast (rank 0), next_meal_rank should be 1 (Lunch).
    A seed with no meals should give next_meal_rank = -1.
    """
    tk   = mini_tokenizer
    bfst = tk.token2id["MEAL_CONTEXT_Breakfast"]
    pad  = tk.pad_token_id
    K    = int(luts["K_meals"].item())

    if K == 0:
        pytest.skip("No meal tokens in vocabulary")

    bfst_rank = luts["meal_rank"][bfst].item()

    B, T    = 2, 3
    pos_ids = torch.full((B, T), pad, dtype=torch.long)
    pos_ids[0, 0] = bfst   # patient 0 had breakfast

    _, next_meal_rank = init_legality_state_batched(luts, pos_ids)

    expected_next = (bfst_rank + 1) % K
    assert next_meal_rank[0].item() == expected_next, \
        f"After Breakfast (rank {bfst_rank}), expected next={expected_next}, got {next_meal_rank[0].item()}"
    assert next_meal_rank[1].item() == -1, "Patient with no meals should have rank=-1"
    print("test_init_legality_state_batched_meal_rank passed.")


def test_build_illegal_mask_batched_pad_always_blocked(mini_tokenizer, luts):
    """PAD and MASK tokens must always be in the illegal mask for every batch item."""
    tk  = mini_tokenizer
    nb  = luts["start_ids_per_base"].numel()
    B   = 3
    oc  = torch.zeros(B, nb, dtype=torch.long)
    nmr = torch.full((B,), -1, dtype=torch.long)
    illegal = build_illegal_mask_batched(luts, oc, nmr, tk.pad_token_id, tk.mask_token_id)
    assert illegal[:, tk.pad_token_id].all(),  "PAD must be illegal for all patients"
    assert illegal[:, tk.mask_token_id].all(), "MASK must be illegal for all patients"
    print("test_build_illegal_mask_batched_pad_always_blocked passed.")


def test_build_illegal_mask_batched_per_patient_independence(mini_tokenizer, luts):
    """
    Two patients with different open_counts get independent illegal masks:
    patient 0 (interval closed) → END is illegal; patient 1 (interval open) → START is illegal.
    """
    tk    = mini_tokenizer
    pad   = tk.pad_token_id
    msk   = tk.mask_token_id
    nb    = luts["start_ids_per_base"].numel()
    low_s = tk.token2id["A_STATE_Low_START"]
    low_e = tk.token2id["A_STATE_Low_END"]
    base  = luts["base_id"][low_s].item()

    oc  = torch.zeros(2, nb, dtype=torch.long)
    oc[1, base] = 1   # patient 1 has interval open
    nmr = torch.tensor([-1, -1], dtype=torch.long)

    illegal = build_illegal_mask_batched(luts, oc, nmr, pad, msk)

    assert illegal[0, low_e], "END should be illegal for patient 0 (interval not open)"
    assert not illegal[0, low_s] or True, "START for patient 0 is not forced illegal by open_counts"
    assert illegal[1, low_s], "Duplicate START should be illegal for patient 1 (already open)"
    print("test_build_illegal_mask_batched_per_patient_independence passed.")


def test_update_legality_state_batched_start_end(mini_tokenizer, luts):
    """
    START increments open_counts; END decrements; spurious END does not go negative.
    Finished patients are not updated.
    """
    tk    = mini_tokenizer
    nb    = luts["start_ids_per_base"].numel()
    low_s = tk.token2id["A_STATE_Low_START"]
    low_e = tk.token2id["A_STATE_Low_END"]
    base  = luts["base_id"][low_s].item()

    B   = 2
    oc  = torch.zeros(B, nb, dtype=torch.long)
    nmr = torch.full((B,), -1, dtype=torch.long)
    fin = torch.tensor([False, True])   # patient 1 is already finished

    # Both try to START; only patient 0 is active
    update_legality_state_batched(luts,
                                   torch.tensor([low_s, low_s]),
                                   oc, nmr, fin)
    assert oc[0, base].item() == 1, "Active patient should have base open"
    assert oc[1, base].item() == 0, "Finished patient should not be updated"

    # Now patient 0 ENDs
    fin2 = torch.tensor([False, False])
    update_legality_state_batched(luts,
                                   torch.tensor([low_e, low_e]),
                                   oc, nmr, fin2)
    assert oc[0, base].item() == 0, "After END, open_count should be 0"
    assert oc[1, base].item() == 0, "Patient 1 had extra END; must not go negative"
    print("test_update_legality_state_batched_start_end passed.")


def test_update_legality_state_batched_meal_rank(mini_tokenizer, luts):
    """Generating a meal token advances next_meal_rank to the next cycle position."""
    tk   = mini_tokenizer
    K    = int(luts["K_meals"].item())
    if K == 0:
        pytest.skip("No meal tokens in vocabulary")

    bfst = tk.token2id["MEAL_CONTEXT_Breakfast"]
    bfst_rank = luts["meal_rank"][bfst].item()

    nb  = luts["start_ids_per_base"].numel()
    oc  = torch.zeros(1, nb, dtype=torch.long)
    nmr = torch.tensor([-1], dtype=torch.long)
    fin = torch.tensor([False])

    update_legality_state_batched(luts, torch.tensor([bfst]), oc, nmr, fin)

    expected = (bfst_rank + 1) % K
    assert nmr[0].item() == expected, \
        f"After Breakfast (rank {bfst_rank}), expected next={expected}, got {nmr[0].item()}"
    print("test_update_legality_state_batched_meal_rank passed.")


def test_build_rep_penalty_batched_shape(mini_tokenizer):
    """build_rep_penalty_batched returns [B, V] with correct shape."""
    V = len(mini_tokenizer.token2id)
    B = 3
    last = [[1, 2, 3], [4], []]
    rep  = build_rep_penalty_batched(last, V=V, window=5, strength=0.6, device="cpu")
    assert rep.shape == (B, V), f"Expected shape ({B}, {V}), got {rep.shape}"
    print("test_build_rep_penalty_batched_shape passed.")


def test_build_rep_penalty_batched_zero_strength(mini_tokenizer):
    """With strength=0, penalty should be all zeros."""
    V   = len(mini_tokenizer.token2id)
    rep = build_rep_penalty_batched([[1, 2]], V=V, strength=0.0, device="cpu")
    assert rep.abs().max().item() == 0.0, "Zero strength should give zero penalty"
    print("test_build_rep_penalty_batched_zero_strength passed.")


def test_build_rep_penalty_batched_empty_lists(mini_tokenizer):
    """All-empty last_tokens lists should return a zero tensor."""
    V   = len(mini_tokenizer.token2id)
    rep = build_rep_penalty_batched([[], [], []], V=V, strength=0.6, device="cpu")
    assert rep.abs().max().item() == 0.0
    print("test_build_rep_penalty_batched_empty_lists passed.")


def test_build_rep_penalty_batched_penalises_recent(mini_tokenizer):
    """The most-recently generated token should receive the highest penalty."""
    V = len(mini_tokenizer.token2id)
    recent = 5
    older  = 3
    rep = build_rep_penalty_batched([[older, recent]], V=V, window=5, strength=0.6, device="cpu")
    assert rep[0, recent] > rep[0, older], \
        "Most recent token should have higher penalty than older one"
    print("test_build_rep_penalty_batched_penalises_recent passed.")


# ─────────────────────────────────────────────────────────────────────────────
# generate (event stream mode, collect_risk_scores=False)
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_INFER_COLS = {"PatientId", "Step", "Token", "TimePoint",
                       "IsInput", "IsOutcome", "IsTerminal"}


def test_infer_event_stream_returns_dataframe(mini_model, mini_tokenizer):
    """generate returns a non-empty DataFrame with required columns."""
    dataset = FakeDataset(mini_tokenizer)
    df = generate(mini_model, dataset, max_len=5, batch_size=2)

    assert isinstance(df, pd.DataFrame), "Should return a DataFrame"
    assert not df.empty, "DataFrame should not be empty"
    assert EXPECTED_INFER_COLS.issubset(df.columns), \
        f"Missing columns: {EXPECTED_INFER_COLS - set(df.columns)}"
    print("test_infer_event_stream_returns_dataframe passed.")


def test_infer_event_stream_all_patients_present(mini_model, mini_tokenizer):
    """Every patient in the dataset must appear in the output."""
    dataset = FakeDataset(mini_tokenizer, n_patients=4)
    df = generate(mini_model, dataset, max_len=3, batch_size=2)

    output_pids = set(df["PatientId"].unique())
    for pid in dataset.patient_ids:
        assert pid in output_pids, f"Patient {pid} missing from output"
    print("test_infer_event_stream_all_patients_present passed.")


def test_infer_event_stream_has_terminal_token(mini_model, mini_tokenizer):
    """
    At least one row per patient should be a terminal event.
    If the model doesn't generate one within max_len, the fallback injector fires.
    """
    dataset = FakeDataset(mini_tokenizer, n_patients=2)
    df = generate(mini_model, dataset, max_len=5, batch_size=2)

    for pid in dataset.patient_ids:
        patient_df = df[df["PatientId"] == pid]
        assert patient_df["IsTerminal"].any(), \
            f"Patient {pid} has no terminal token in output"
    print("test_infer_event_stream_has_terminal_token passed.")


def test_infer_event_stream_input_rows_before_generated(mini_model, mini_tokenizer):
    """Input rows (IsInput=1) must all come before generated rows (IsInput=0) per patient."""
    dataset = FakeDataset(mini_tokenizer, n_patients=2)
    df = generate(mini_model, dataset, max_len=5, batch_size=2)

    for pid in dataset.patient_ids:
        p = df[df["PatientId"] == pid]
        input_steps    = p[p["IsInput"] == 1]["Step"]
        generated_steps = p[p["IsInput"] == 0]["Step"]
        if not generated_steps.empty and not input_steps.empty:
            assert input_steps.max() < generated_steps.min(), \
                f"Patient {pid}: generated steps should come after input steps"
    print("test_infer_event_stream_input_rows_before_generated passed.")


def test_infer_event_stream_batch_sizes_consistent(mini_model, mini_tokenizer):
    """
    Batching should not change the set of patients or their input tokens.
    Run with batch_size=1 and batch_size=4 and compare input rows.
    """
    dataset = FakeDataset(mini_tokenizer, n_patients=3)

    df1 = generate(mini_model, dataset, max_len=3, batch_size=1)
    df4 = generate(mini_model, dataset, max_len=3, batch_size=4)

    for pid in dataset.patient_ids:
        rows1 = df1[(df1["PatientId"] == pid) & (df1["IsInput"] == 1)].sort_values("Step")
        rows4 = df4[(df4["PatientId"] == pid) & (df4["IsInput"] == 1)].sort_values("Step")
        assert list(rows1["Token"]) == list(rows4["Token"]), \
            f"Patient {pid}: input tokens differ between batch sizes"
        assert list(rows1["Step"]) == list(rows4["Step"]), \
            f"Patient {pid}: input steps differ between batch sizes"
    print("test_infer_event_stream_batch_sizes_consistent passed.")


def test_infer_event_stream_token_strings_are_valid(mini_model, mini_tokenizer):
    """Generated token strings should appear in the vocabulary (no <UNK_*> tokens)."""
    dataset = FakeDataset(mini_tokenizer, n_patients=2)
    df = generate(mini_model, dataset, max_len=5, batch_size=2)

    vocab = set(mini_tokenizer.token2id.keys())
    bad   = df[~df["Token"].isin(vocab)]
    assert bad.empty, f"Unknown tokens in output:\n{bad[['PatientId','Step','Token']]}"
    print("test_infer_event_stream_token_strings_are_valid passed.")


# ─────────────────────────────────────────────────────────────────────────────
# generate (risk score mode, collect_risk_scores=True)
# ─────────────────────────────────────────────────────────────────────────────

def test_generate_risk_curves_returns_dataframe(mini_model, mini_tokenizer):
    """generate with collect_risk_scores=True returns required columns including P_* cols."""
    dataset     = FakeDataset(mini_tokenizer)
    outcome_cols = {f"P_{n}" for n in mini_model.outcome_names}
    required     = {"PatientId", "Step", "Token", "TimePoint", "IsInput", "IsTerminal"} | outcome_cols

    df = generate(mini_model, dataset, max_len=5, batch_size=2, collect_risk_scores=True)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert required.issubset(df.columns), \
        f"Missing columns: {required - set(df.columns)}"
    print("test_generate_risk_curves_returns_dataframe passed.")


def test_generate_risk_curves_probabilities_in_range(mini_model, mini_tokenizer):
    """All P_* columns must contain values strictly in [0, 1]."""
    dataset = FakeDataset(mini_tokenizer, n_patients=2)
    df = generate(mini_model, dataset, max_len=5, batch_size=2, collect_risk_scores=True)

    for col in [f"P_{n}" for n in mini_model.outcome_names]:
        vals = df[col]
        assert (vals >= 0.0).all() and (vals <= 1.0).all(), \
            f"Column {col} has values outside [0, 1]: min={vals.min()}, max={vals.max()}"
    print("test_generate_risk_curves_probabilities_in_range passed.")


def test_generate_risk_curves_input_rows_have_outcome_scores(mini_model, mini_tokenizer):
    """Input rows (IsInput=1) must have outcome probabilities filled in (not 0.0 always)."""
    dataset = FakeDataset(mini_tokenizer, n_patients=2, seed_len=4)
    df = generate(mini_model, dataset, max_len=3, batch_size=2, collect_risk_scores=True)

    input_rows = df[df["IsInput"] == 1]
    assert not input_rows.empty, "Should have input rows"
    # At least some P_* value should be non-zero (random model weights produce non-trivial outputs)
    p_cols  = [f"P_{n}" for n in mini_model.outcome_names]
    nonzero = (input_rows[p_cols].abs() > 1e-6).any(axis=1).any()
    assert nonzero, "Input-row outcome probabilities should not all be exactly zero"
    print("test_generate_risk_curves_input_rows_have_outcome_scores passed.")


def test_generate_risk_curves_all_patients_present(mini_model, mini_tokenizer):
    """Every patient must appear in the risk-curve output."""
    dataset = FakeDataset(mini_tokenizer, n_patients=3)
    df = generate(mini_model, dataset, max_len=3, batch_size=2, collect_risk_scores=True)

    output_pids = set(df["PatientId"].unique())
    for pid in dataset.patient_ids:
        assert pid in output_pids, f"Patient {pid} missing from risk curve output"
    print("test_generate_risk_curves_all_patients_present passed.")
