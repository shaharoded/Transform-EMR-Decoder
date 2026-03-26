import torch
import torch.nn.functional as F
import pytest
import transform_emr.utils as utils_module

from transform_emr.utils import (
    get_multi_hot_targets,
    get_future_outcome_targets,
    build_mlm,
    linear_schedule,
    apply_cbm,
    mix_with_predictions,
    penalty_interval_structure,
    penalty_meal_order,
    soft_interval_penalty, 
    soft_meal_order_penalty,
    build_luts,
    compute_legality_masks_tf,
    apply_masks_to_logits,
    build_rep_penalty,
    soft_unclosed_interval_penalty
)
from transform_emr.schedulers import LambdaScheduleController, linear_schedule as sched_linear_schedule
from transform_emr.dataset import EMRTokenizer


def test_linear_schedule_import_is_from_schedulers():
    """Ensure utils.linear_schedule remains the scheduler-exported function (import compatibility)."""
    assert linear_schedule is sched_linear_schedule
    assert utils_module.linear_schedule is sched_linear_schedule


def test_unified_lambda_schedule_controller_phase1_alias_and_immediate_activation():
    """
    Phase-1 scheduling contract:
      - Aux keys are mlm/dt with ramp=1 (immediate full λ after calibration).
      - update() accepts training-style aliases (vl_mlm_raw, vl_dt_raw).
    """
    cfg = {
        "aux_max_fraction_default": 0.20,
        "phase1_aux_fraction_caps": {"mlm": 0.20, "dt": 0.20},
        "phase1_dynamic_schedule": {"mlm_ramp_epochs": 1, "dt_ramp_epochs": 1},
    }
    controller = LambdaScheduleController(training_settings=cfg, start_epoch=0)

    # Before calibration everything should be zero.
    l0 = controller.get_lambdas(epoch=0)
    assert l0["mlm"] == 0.0 and l0["dt"] == 0.0

    # Calibrate using alias keys exactly as training loop provides.
    events = controller.update(epoch=0, vl_main=2.0, vl_mlm_raw=1.0, vl_dt_raw=0.5)
    assert len(events) >= 2

    # With ramp=1 and start_epoch=0, λ should be at full calibrated value immediately.
    l1 = controller.get_lambdas(epoch=0)
    assert l1["mlm"] == pytest.approx(0.4)   # 0.2 * 2.0 / 1.0
    assert l1["dt"] == pytest.approx(0.8)    # 0.2 * 2.0 / 0.5

    # Calibration is frozen; second update with different losses should not change λ_max.
    controller.update(epoch=1, vl_main=10.0, vl_mlm_raw=10.0, vl_dt_raw=10.0)
    l2 = controller.get_lambdas(epoch=1)
    assert l2["mlm"] == pytest.approx(l1["mlm"])
    assert l2["dt"] == pytest.approx(l1["dt"])


def test_unified_lambda_schedule_controller_phase2_ramp_progression():
    """
    Phase-2 scheduling contract:
      - CE/DT start together at epoch 0.
      - Ramp follows configured epochs.
      - Penalty/Outcome keep zero before unlock (dynamic mode enabled).
    """
    cfg = {
        "aux_max_fraction_default": 0.20,
        "phase2_aux_fraction_caps": {"ce": 0.20, "dt": 0.20, "penalty": 0.20, "outcome": 0.20},
        "phase2_dynamic_schedule": {
            "enabled": True,
            "ce_ramp_epochs": 5,
            "dt_ramp_epochs": 5,
            "penalty_ramp_epochs": 5,
            "outcome_ramp_epochs": 5,
            "plateau_min_delta": 1e-4,
            "base_plateau_patience": 3,
            "penalty_plateau_patience": 3,
            "min_base_epochs": 5,
            "min_penalty_epochs": 5,
        },
    }
    controller = LambdaScheduleController(training_settings=cfg, start_epoch=0)

    # Calibrate CE/DT using training-style alias keys.
    controller.update(epoch=0, vl_main=2.0, vl_ce_raw=1.0, vl_dt_raw=0.5)

    # Target lambdas: ce=0.4, dt=0.8; at epoch=0 ramp starts at 0.
    lam0 = controller.get_lambdas(epoch=0)
    assert lam0["ce"] == pytest.approx(0.0)
    assert lam0["dt"] == pytest.approx(0.0)
    assert lam0["penalty"] == pytest.approx(0.0)
    assert lam0["outcome"] == pytest.approx(0.0)

    # Mid-ramp check at epoch=2 with ramp=5.
    lam2 = controller.get_lambdas(epoch=2)
    assert lam2["ce"] == pytest.approx(0.4 * (2 / 5), rel=1e-6)
    assert lam2["dt"] == pytest.approx(0.8 * (2 / 5), rel=1e-6)

    # End-ramp at epoch=5 reaches full λ_max.
    lam5 = controller.get_lambdas(epoch=5)
    assert lam5["ce"] == pytest.approx(0.4, rel=1e-6)
    assert lam5["dt"] == pytest.approx(0.8, rel=1e-6)

@pytest.fixture(scope="module")
def mini_tokenizer():
    """
    Simulated tokenizer with a three-level hierarchy:
      - Raw concepts: A (0), MEAL (1), ADMISSION (2), OUTCOME (3)
      - Concepts: A_X (0), A_Y (1), MEAL_B/L/D (2), ADMISSION (3), DEATH/RELEASE (4)
      - Values: VAL1 (0), VAL2 (1) per concept where applicable
      - Position tokens: <concept>_<value>_START/END, plus single tokens for meals/outcomes.
    """
    toks = [
        "[PAD]", "[MASK]", "[CTX]", "[NULL]",
        # Admission context
        "ADMISSION_EVENT",
        # Intervals for A_X with two values
        "A_STATE_Low_START", "A_STATE_Low_END",
        "A_STATE_High_START", "A_STATE_High_END",
        # Intervals for A_Y with two values
        "A_TREND_dec_START", "A_TREND_dec_END",
        "A_TREND_inc_START", "A_TREND_inc_END",
        # Meals
        "MEAL_CONTEXT_Breakfast", "MEAL_CONTEXT_Lunch", "MEAL_CONTEXT_Dinner", "MEAL_CONTEXT_Night-Snack",
        # Outcomes
        "DEATH_EVENT", "RELEASE_EVENT",
        # Regular events (eligible for CBM masking)
        "GLUCOSE_READING_EVENT", "INSULIN_DOSE_EVENT"
    ]
    token2id = {tok: i for i, tok in enumerate(toks)}
    # Raw concept mapping: group by top-level concept
    rawconcept2id = {
        "A": 0,
        "MEAL_CONTEXT": 1,
        "ADMISSION_EVENT": 2,
        "DEATH_EVENT": 3,
        "RELEASE_EVENT": 4
    }
    # Concept-level mapping
    concept2id = {
        "A_STATE": 0,
        "A_TREND": 1,
        "MEAL_CONTEXT": 2,
        "ADMISSION_EVENT": 3,
        "DEATH_EVENT": 4,
        "RELEASE_EVENT": 5
    }
    # Value-level mapping (e.g., high/low categories)
    value2id = {
        "A_STATE_Low": 0,
        "A_STATE_High": 1,
        "A_TREND_dec": 2,
        "A_TREND_inc": 3,
        "MEAL_CONTEXT_Breakfast": 4,
        "MEAL_CONTEXT_Lunch": 5,
        "MEAL_CONTEXT_Dinner": 6,
        "MEAL_CONTEXT_Night-Snack": 7,
        "ADMISSION_EVENT": 8,
        "DEATH_EVENT": 9,
        "RELEASE_EVENT": 10
    }
    special_tokens = ["[PAD]", "[MASK]", "[CTX]", "[NULL]"]
    token_weights = torch.ones(len(toks))
    outcome_weights = torch.ones(len(toks))
    important_token_ids = torch.tensor([], dtype=torch.long)
    token_counts = torch.tensor([], dtype=torch.long)

    # Dummy parent raw mapping for testing
    vocab_size = len(token2id)
    tokenid2parent_raw_ids = torch.zeros((vocab_size, 1), dtype=torch.long)
    parent_pad_len = 1

    tk = EMRTokenizer(
        token2id=token2id,
        rawconcept2id=rawconcept2id,
        concept2id=concept2id,
        value2id=value2id,
        special_tokens=special_tokens,
        token_weights=token_weights,
        outcome_weights=outcome_weights,
        important_token_ids=important_token_ids,
        token_counts = token_counts,
        tokenid2parent_raw_ids=tokenid2parent_raw_ids,
        parent_pad_len=parent_pad_len
    )
    # assign special attributes
    tk.pad_token_id  = token2id['[PAD]']
    tk.mask_token_id = token2id['[MASK]']
    tk.ctx_token_id  = token2id['[CTX]']
    tk.null_token_id = token2id['[NULL]']
    return tk


def test_multi_hot_targets_visual_and_assert():
    """
    For each t in a longer sequence, print the
    true future IDs vs. the multi-hot IDs, then
    assert they exactly match.

    Expectation: At every position t, the targets are the positions t+1 up to t+k, until the first padding (0) token.
    """
    # --- Setup a toy sequence: 1..10 then two PADs (0) ---
    seq = torch.tensor([[1,2,3,4,5,6,7,8,9,10,0,0]])
    B, T = seq.shape
    V    = seq.max().item() + 1  # 11 = tokens 0..10
    k    = 5

    # --- Compute multi-hot targets ---
    mh = get_multi_hot_targets(seq, padding_idx=0, vocab_size=V, k=k)
    assert mh.shape == (B, T, V)

    # --- For each timestep, print & assert correctness ---
    for t in range(T):
        # curr
        curr = seq[0, t]
        # ground-truth future slice
        future = seq[0, t+1 : t+1+k].tolist()
        # drop pads & dedupe
        expected = sorted({x for x in future if x != 0})

        # what the function actually marked
        hot_ids = mh[0, t].nonzero(as_tuple=False).squeeze(-1).tolist()
        hot_ids.sort()

        # print for human verification
        print(f"t={t}, curr={curr} | future={future} | hot_ids={hot_ids}")

        # pytest assertion
        assert hot_ids == expected, (
            f"At t={t}, expected {expected} but got {hot_ids}"
        )

def test_get_future_outcome_targets():
    """
    Verifies that get_future_outcome_targets correctly flags future events.
    Scenario: Sequence [A, Sepsis, B, Death, Pad]
    """
    # 1. Setup
    # Let 1=Sepsis, 2=Death, 9=Pad, others=random
    seq = torch.tensor([[10, 1, 11, 2, 9]]) # [B=1, T=5]
    outcome_ids = [1, 2] # Sepsis, Death
    
    # 2. Run
    targets = get_future_outcome_targets(seq, outcome_ids) # [1, 5, 2]
    
    # 3. Assertions
    # T=0 (Token 10): Future has Sepsis(1) and Death(2) -> Both True
    assert targets[0, 0, 0].item() == 1.0, "T=0 should predict future Sepsis"
    assert targets[0, 0, 1].item() == 1.0, "T=0 should predict future Death"
    
    # T=1 (Token 1/Sepsis): Future has Death(2). Sepsis is *current*, not future.
    # So Sepsis target should be 0 (unless another Sepsis occurs later).
    assert targets[0, 1, 0].item() == 0.0, "T=1 (Sepsis) should NOT predict future Sepsis (unless another occurs)"
    assert targets[0, 1, 1].item() == 1.0, "T=1 (Sepsis) should predict future Death"
    
    # T=2 (Token 11): Future has Death(2).
    assert targets[0, 2, 1].item() == 1.0, "T=2 should predict future Death"
    
    # T=3 (Token 2/Death): No future outcomes.
    assert targets[0, 3, 1].item() == 0.0, "T=3 (Death) should have no future Death"
    
    # T=4 (Pad): No future.
    assert targets[0, 4, 0].item() == 0.0
    
    print("test_get_future_outcome_targets: passed.")

def test_linear_schedule_visual_and_assert():
    """
    For a linear ramp over warmup epochs:
    - When epoch <= warmup: value = (epoch / warmup) * max
    - When epoch >  warmup: value = max
    """
    warmup = 5
    maxv   = 1.0
    # test a range of epochs before, at, and after warmup
    for epoch in [0, 1, 2, 5, 6, 10]:
        val = linear_schedule(epoch=epoch, start_epoch=0, end_epoch=warmup, max_val=maxv)
        expected = min(epoch / warmup, 1.0) * maxv
        # print for visual inspection
        print(f"epoch={epoch} | expected={expected:.3f} | actual={val:.3f}")
        # assert correctness
        assert val == pytest.approx(expected)


def test_build_mlm_masking_visual_and_assert(mini_tokenizer):
    """
    Print each position’s token, mask flag, and new ID,
    then assert forbidden tokens stay unchanged and eligible positions are flagged.
    """
    tk = mini_tokenizer

    # One-row batch covering forbidden & eligible IDs
    seq_ids = [
        tk.pad_token_id,                      # forbidden
        tk.null_token_id,                     # forbidden
        tk.token2id["ADMISSION_EVENT"],       # forbidden
        tk.token2id["DEATH_EVENT"],           # forbidden
        tk.token2id["RELEASE_EVENT"],         # forbidden
        tk.token2id["A_STATE_High_START"],    # eligible
        tk.token2id["A_TREND_inc_START"],     # eligible
        tk.token2id["MEAL_CONTEXT_Breakfast"],# eligible
        tk.token2id["A_TREND_inc_END"],       # eligible
        tk.token2id["A_STATE_High_END"],      # eligible
    ]
    ids = torch.tensor([seq_ids], dtype=torch.long)

    masked, mask = build_mlm(ids, tokenizer=tk, p=1.0)

    forbidden = {
        tk.pad_token_id,
        tk.null_token_id,
        tk.token2id["ADMISSION_EVENT"],
        tk.token2id["DEATH_EVENT"],
        tk.token2id["RELEASE_EVENT"],
    }

    for pos, orig in enumerate(seq_ids):
        token    = tk.id2token[orig]
        was_mask = bool(mask[0, pos].item())
        new_id   = masked[0, pos].item()

        print(f"pos={pos:<2} token={token:<24}"
              f"orig={orig:<2} masked={was_mask:<5} new={new_id}")

        if orig in forbidden:
            # these must never be masked or changed
            assert not was_mask, f"❌ Forbidden {token} was masked"
            assert new_id == orig, f"❌ Forbidden {token} changed to {new_id}"
        else:
            # eligible positions must have mask flag True
            assert was_mask, f"❌ Expected {token} to be masked"

    # (We no longer assert new_id != orig, since 10% of the time BERT-style
    # keeps the original even when flagged masked.)


def test_apply_cbm_visual_and_assert(mini_tokenizer):
    """
    With p=1.0 (epoch == warmup_epochs), verify that:
      - Forbidden tokens (pad, mask, forbid_mask_ids) remain unchanged
      - Eligible tokens are always replaced by mask_token_id
      - Covers both no-eligible and with-eligible scenarios
    """
    tk   = mini_tokenizer
    luts = build_luts(tk)
    pad_id   = tk.pad_token_id
    mask_tok = tk.mask_token_id
    forbid_ids = set(luts["forbid_mask_ids"].tolist()) | {pad_id, mask_tok}

    V = len(tk.token_weights)
    # Find any eligible token
    eligible = next((i for i in range(V) if i not in forbid_ids), None)

    # Case 1: no eligible tokens => sequence unchanged
    if eligible is None:
        in_seq = torch.tensor([list(forbid_ids)[:4]], dtype=torch.long)
        batch = {
            "position_ids": in_seq.clone(),
            "parent_raw_ids": torch.zeros(in_seq.shape[0], in_seq.shape[1], 1, dtype=torch.long),
            "concept_ids": in_seq.clone(),
            "value_ids": in_seq.clone()
        }
        out = apply_cbm(
            batch.copy(),
            tokenizer=tk,
            forbid_ids=torch.tensor(sorted(luts["forbid_mask_ids"].tolist()), dtype=torch.long),
            p=1.0
        )
        out_seq = out["position_ids"][0]
        print(f"test_apply_cbm (no eligible): input={in_seq[0].tolist()}, output={out_seq.tolist()}")
        assert torch.equal(out_seq, in_seq[0]), "All tokens forbidden: sequence must remain unchanged"
    else:
        # Case 2: eligible tokens exist
        candidates = [i for i in range(V) if i not in forbid_ids and i != eligible]
        if not candidates:
            pytest.skip("Only one eligible token in vocab, skipping ‘with‑eligible’ CBM test")
        other = candidates[0]
        in_seq = torch.tensor([[pad_id, eligible, other, mask_tok]], dtype=torch.long)
        batch = {
            "position_ids": in_seq.clone(),
            "parent_raw_ids": torch.zeros(in_seq.shape[0], in_seq.shape[1], 1, dtype=torch.long),
            "concept_ids": in_seq.clone(),
            "value_ids": in_seq.clone()
        }
        out = apply_cbm(
            batch.copy(),
            tokenizer=tk,
            forbid_ids=torch.tensor(sorted(luts["forbid_mask_ids"].tolist()), dtype=torch.long),
            p=1.0
        )
        out_seq = out["position_ids"][0]
        print(f"test_apply_cbm (with eligible): input ={in_seq[0].tolist()}")
        print(f"test_apply_cbm (with eligible): output={out_seq.tolist()}")
        for idx, orig in enumerate(in_seq[0].tolist()):
            new = int(out_seq[idx])
            is_forb = orig in forbid_ids
            print(f" pos={idx} | orig={orig:<3} | new={new:<3} | forbidden={is_forb}")
            if is_forb:
                assert new == orig, f"Forbidden token {orig} was changed to {new}"
            else:
                assert new == mask_tok, f"Eligible token {orig} not masked, got {new}"
            # 3) Minimal-forbid scenario: only PAD and MASK tokens are forbidden
    minimal_forbid = torch.tensor([pad_id, mask_tok], dtype=torch.long)
    # Now any other token is eligible; pick two distinct ones
    eligible_min = next((i for i in range(V) if i not in {pad_id, mask_tok}), None)
    other_min    = next((i for i in range(V) if i not in {pad_id, mask_tok, eligible_min}), None)
    if eligible_min is not None and other_min is not None:
        in_seq3 = torch.tensor([[pad_id, eligible_min, other_min, pad_id]], dtype=torch.long)
        batch3 = {
            "position_ids": in_seq3.clone(),
            "parent_raw_ids": torch.zeros(in_seq3.shape[0], in_seq3.shape[1], 1, dtype=torch.long),
            "concept_ids": in_seq3.clone(),
            "value_ids": in_seq3.clone()
        }
        out3 = apply_cbm(
            batch3.copy(),
            tokenizer=tk, forbid_ids=minimal_forbid, p=1.0
        )
        out3_seq = out3["position_ids"][0]
        print(f"test_apply_cbm (minimal forbid): input ={in_seq3[0].tolist()}, output={out3_seq.tolist()}")
        # pad positions unchanged, eligible and other masked
        assert out3_seq[0].item() == pad_id
        assert out3_seq[3].item() == pad_id
        assert out3_seq[1].item() == mask_tok, f"Eligible token {eligible_min} not masked"
        assert out3_seq[2].item() == mask_tok, f"Eligible token {other_min} not masked"
        print("test_apply_cbm: minimal-forbid case passed.")
    else:
        print("test_apply_cbm: minimal-forbid scenario skipped (not enough eligible tokens)")


def test_mix_with_predictions_visual_and_assert(mini_tokenizer):
    """
    Protected GT tokens stay, unprotected are replaced by pred.
    """
    tk = mini_tokenizer

    # Single sequence of length 3
    gt   = torch.tensor([[1, 2, 3]])
    pred = torch.tensor([[9, 9, 9]])
    prot = torch.zeros(len(tk.token2id), dtype=torch.bool)
    prot[1] = True  # protect ID=1

    mixed, mask = mix_with_predictions(
        gt, pred,
        epoch=5,
        warmup_epochs=5,
        protected_ids=prot,
        max_rate=1.0
    )
    print("MIXED BATCH:\n", mixed)

    for j, (g, p, m_flag) in enumerate(zip(gt[0], pred[0], mask[0])):
        g_id     = int(g.item())
        p_id     = int(p.item())
        mixed_id = int(mixed[0,j].item())

        print(f"pos={j} | gt={g_id} | pred={p_id} | mask={bool(m_flag.item())} | mixed={mixed_id}")

        if prot[g_id]:
            assert mixed_id == g_id, f"❌ Protected {g_id} was replaced"
            assert not m_flag,       f"❌ Protected {g_id} should not be masked"
        else:
            assert mixed_id == p_id, f"❌ Unprotected {g_id} not replaced by {p_id}"
            assert m_flag,           f"❌ Unprotected {g_id} should be masked"


def test_build_luts_and_legality_visual_and_assert(mini_tokenizer):
    """
    Comprehensive verification of LUTs and legality masks:
    1. **LUT correctness**: prints and asserts the per-token and per-base lookups:
       - `is_start`/`is_end` flags for interval tokens
       - `base_id` grouping for each interval value
       - `conflict_mat` marking conflicting bases (same concept, different values)
    2. **Interval legality**:
       - **Correct** start→end sequence should yield no illegal flags (FSM/DUP/CNF pass)
       - **Reversed** end→start should flag FSM violation at t=0
       - **Conflict** start of one value while another is open should flag CNF violation at t=1
    3. **Meal ordering**:
       - Legal cycle (L→D→N→B→L) yields no illegal flags
       - Illegal short cycle (L→B→D) yields a violation at t=1 (Breakfast follows Lunch)
    4. **Interleaving robustness**:
       - Interval logic unaffected by unrelated meal tokens
       - Meal logic unaffected by unrelated interval tokens
    5. **Penalties**:
       - `penalty_interval_structure`: >0 for reversed sequence (UNC/FSM)
       - `penalty_meal_order`: zero for legal, >0 for illegal meal cycles
    """
    tk = mini_tokenizer
    l  = build_luts(tk)

    # --- Dump LUT contents for visual verification ---
    print("Vocab size:", len(tk.token2id))
    print("is_start:", l['is_start'].tolist())
    print("is_end:",   l['is_end'].tolist())
    print("base_id:",  l['base_id'].tolist())
    print("start_ids_per_base:", l['start_ids_per_base'].tolist())
    print("end_ids_per_base:",   l['end_ids_per_base'].tolist())
    print("conflict_mat:",        l['conflict_mat'].tolist())

    # --- Interval token IDs ---
    low_s  = tk.token2id['A_STATE_Low_START']
    low_e  = tk.token2id['A_STATE_Low_END']
    high_s = tk.token2id['A_STATE_High_START']
    high_e = tk.token2id['A_STATE_High_END']
    inc_s  = tk.token2id['A_TREND_inc_START']
    inc_e  = tk.token2id['A_TREND_inc_END']
    dec_s  = tk.token2id['A_TREND_dec_START']
    dec_e  = tk.token2id['A_TREND_dec_END']
    pad    = tk.pad_token_id

    # --- Verify start/end flags and base conflicts ---
    assert l['is_start'][low_s]
    assert l['is_start'][high_s]
    assert l['is_end'][low_e]
    assert l['is_end'][high_e]

    # convert *all* four interval groups to base‑ids
    low_b   = l['base_id'][low_s].item()
    high_b  = l['base_id'][high_s].item()
    inc_b   = l['base_id'][inc_s].item()
    dec_b   = l['base_id'][dec_s].item()

    assert low_b != high_b, "Low and High should be separate bases"
    assert l['conflict_mat'][low_b, high_b], "CNF should be True for Low vs High"
    assert l['conflict_mat'][inc_b, dec_b], "CNF should be True for inc vs dec"
    assert not l['conflict_mat'][inc_b, inc_b], "CNF should be False for same value"
    assert not l['conflict_mat'][inc_b, high_b], "CNF should be False for STATE vs TREND"

    # 1) Correct interval order
    seq_ok = torch.tensor([[low_s, low_e, pad]])
    illegal_ok, bonus_ok = compute_legality_masks_tf(
        seq_ok, l['is_start'], l['is_end'], l['base_id'],
        l['start_ids_per_base'], l['end_ids_per_base'], l['meal_rank'],
        l['meal_pred_rank'], l['K_meals'], l['conflict_mat'], l['predict_block']
    )
    # Only check actual event tokens; PAD is intentionally blocked from prediction.
    for t_idx, tok in enumerate([low_s, low_e]):
        assert not illegal_ok[0, t_idx, tok], (
            f"Token {tk.id2token[tok]} wrongly flagged illegal at position {t_idx}")
    # PAD should be illegal to predict
    assert illegal_ok[0, 2, pad], "PAD should be blocked from prediction"
    assert bonus_ok[0, 1, low_e], "Bonus mask for END token missing"

    # 2) FSM violation: END before START
    seq_rev = torch.tensor([[low_e, low_s, pad]])
    illegal_rev, _ = compute_legality_masks_tf(
        seq_rev, l['is_start'], l['is_end'], l['base_id'],
        l['start_ids_per_base'], l['end_ids_per_base'], l['meal_rank'],
        l['meal_pred_rank'], l['K_meals'], l['conflict_mat'], l['predict_block']
    )
    assert illegal_rev[0,0,low_e], "FSM violation should flag END when no START"

    # 3) CNF violation: conflicting High START while Low open
    seq_conf = torch.tensor([[low_s, high_s, pad]])
    illegal_conf, _ = compute_legality_masks_tf(
        seq_conf, l['is_start'], l['is_end'], l['base_id'],
        l['start_ids_per_base'], l['end_ids_per_base'], l['meal_rank'],
        l['meal_pred_rank'], l['K_meals'], l['conflict_mat'], l['predict_block']
    )
    assert illegal_conf[0,1,high_s], "CNF violation should flag conflicting High START"

    # 4) DUP violation: START twice in a row on same base
    seq_dup = torch.tensor([[low_s, low_s, pad]])
    illegal_dup, _ = compute_legality_masks_tf(
        seq_dup, l['is_start'], l['is_end'], l['base_id'],
        l['start_ids_per_base'], l['end_ids_per_base'], l['meal_rank'],
        l['meal_pred_rank'], l['K_meals'], l['conflict_mat'], l['predict_block']
    )
    assert illegal_dup[0,1,low_s], "DUP violation should flag second START illegal"

    # --- Meal ordering ---
    print("K_meals =", l['K_meals'].item(), "meal_rank:", l['meal_rank'])
    b    = tk.token2id['MEAL_CONTEXT_Breakfast']
    l_id = tk.token2id['MEAL_CONTEXT_Lunch']
    d    = tk.token2id['MEAL_CONTEXT_Dinner']
    n    = tk.token2id['MEAL_CONTEXT_Night-Snack']

    # 5) Legal full cycle: L → D → N → B → L
    seq_cycle = torch.tensor([[l_id, d, n, b, l_id]])
    illegal_cycle, _ = compute_legality_masks_tf(
        seq_cycle, l['is_start'], l['is_end'], l['base_id'],
        l['start_ids_per_base'], l['end_ids_per_base'], l['meal_rank'],
        l['meal_pred_rank'], l['K_meals'], l['conflict_mat'], l['predict_block']
    )
    for t_idx, tok in enumerate([l_id, d, n, b, l_id]):
        assert not illegal_cycle[0, t_idx, tok], (
            f"Token {tk.id2token[tok]} wrongly flagged illegal at position {t_idx}")

    # 6) Illegal short cycle: L → B → D — Breakfast at t=1 should be flagged illegal
    seq_bad = torch.tensor([[l_id, b, d, pad]])
    illegal_bad, _ = compute_legality_masks_tf(
        seq_bad, l['is_start'], l['is_end'], l['base_id'],
        l['start_ids_per_base'], l['end_ids_per_base'], l['meal_rank'],
        l['meal_pred_rank'], l['K_meals'], l['conflict_mat'], l['predict_block']
    )
    assert illegal_bad[0, 1, b], "Meal order violation should flag Breakfast at t=1 for L→B"

    # 7) Interval+Meal interleaving
    seq_mix1 = torch.tensor([[low_s, b, low_e, pad]])
    illegal_mix1, _ = compute_legality_masks_tf(
        seq_mix1, l['is_start'], l['is_end'], l['base_id'],
        l['start_ids_per_base'], l['end_ids_per_base'], l['meal_rank'],
        l['meal_pred_rank'], l['K_meals'], l['conflict_mat'], l['predict_block']
    )
    # low_e at pos2 should be legal despite an unrelated meal token at pos1
    assert not illegal_mix1[0, 2, low_e], (
        "Interval END should be legal even with unrelated meal token in between")

    # 8) Meal+Interval interleaving
    seq_mix2 = torch.tensor([[l_id, low_s, d, n, b, l_id]])
    illegal_mix2, _ = compute_legality_masks_tf(
        seq_mix2, l['is_start'], l['is_end'], l['base_id'],
        l['start_ids_per_base'], l['end_ids_per_base'], l['meal_rank'],
        l['meal_pred_rank'], l['K_meals'], l['conflict_mat'], l['predict_block']
    )
    # dinner at pos2 should be legal (Lunch at pos0, low_s in between shouldn't matter)
    assert not illegal_mix2[0, 2, d], (
        "Meal order should not be broken by unrelated interval token in between")
    

def test_penalty_interval_structure_and_meal_order(mini_tokenizer):
    tk = mini_tokenizer
    l  = build_luts(tk)
    
    # --- Define tokens ---
    b    = tk.token2id['MEAL_CONTEXT_Breakfast']
    lu = tk.token2id['MEAL_CONTEXT_Lunch']
    d    = tk.token2id['MEAL_CONTEXT_Dinner']
    n    = tk.token2id['MEAL_CONTEXT_Night-Snack']
    low_s  = tk.token2id['A_STATE_Low_START']
    low_e  = tk.token2id['A_STATE_Low_END']
    pad    = tk.pad_token_id

    # --- Penalty: interval structure with forgiveness ---
    window=1
    # GT and pred share same FSM violation at t=0 => forgiven (penalty=0)
    gt = torch.tensor([[low_e, low_s, pad]])
    pred = gt.clone()
    p_forgiven = penalty_interval_structure(
        pred, gt,
        l['is_start'], l['is_end'], l['base_id'],
        l['start_ids_per_base'], l['end_ids_per_base'],
        l['meal_rank'], l['meal_pred_rank'],
        l['K_meals'], l['conflict_mat'], l['predict_block'], window=1
    )
    assert p_forgiven == 0, f"Violation should be forgiven, got {p_forgiven}"

    # Pred has an extra FSM violation at t=1 not in GT => penalty > 0
    # GT has a low_e violation at t=0
    gt2   = torch.tensor([[ b, low_e, pad, pad, pad, pad]])
    # Put your new bad END at t=3, which is >1 step away
    pred2 = torch.tensor([[ b, lu, d, n, b, low_e]])

    p_new = penalty_interval_structure(
        pred2, gt2,
        l['is_start'], l['is_end'], l['base_id'],
        l['start_ids_per_base'], l['end_ids_per_base'],
        l['meal_rank'], l['meal_pred_rank'],
        l['K_meals'], l['conflict_mat'], l['predict_block'],
        window=1
    )
    illegal_pred, _ = compute_legality_masks_tf(pred2, l['is_start'], l['is_end'], l['base_id'],
        l['start_ids_per_base'], l['end_ids_per_base'], l['meal_rank'],
        l['meal_pred_rank'], l['K_meals'], l['conflict_mat'], l['predict_block'])
    pred_illegal = illegal_pred.gather(2, pred2.unsqueeze(-1)).squeeze(-1)
    print("pred_illegal:", pred_illegal)          # e.g. tensor([[False, False,  True]])
    gt_illegal = compute_legality_masks_tf(gt2, l['is_start'], l['is_end'], l['base_id'],
        l['start_ids_per_base'], l['end_ids_per_base'], l['meal_rank'],
        l['meal_pred_rank'], l['K_meals'], l['conflict_mat'], l['predict_block'])[0] \
                    .gather(2, gt2.unsqueeze(-1)).squeeze(-1)
    print("gt_illegal:  ", gt_illegal)            # e.g. tensor([[ True, False, False]])
    gt_win = F.max_pool1d(
        gt_illegal.float().unsqueeze(1),
        kernel_size=2*window+1,  # e.g. 3
        stride=1,
        padding=window           # e.g. 1
    ).squeeze(1).bool()          # → [B, T]
    print("gt_win:      ", gt_win)                # e.g. tensor([[ True,  True, False]])
    new_ts_viol = pred_illegal & (~gt_win)
    print("new_ts_viol: ", new_ts_viol)           # → must be [False,False,True]
    print("p_new:       ", p_new)
    assert p_new > 0, "Violation more than 1 step away should not be forgiven"

    # --- Penalty: meal order ---
    # Legal transitions: L→D→N→B => 0 penalty
    seq_ok = torch.tensor([[lu, low_s, d, n, low_e, b]])
    p_meal_ok = penalty_meal_order(seq_ok, l['meal_rank'])
    print('legal transition: L→D→N→B => penalty=', p_meal_ok)
    assert p_meal_ok == 0, f"No penalty for legal cycle, got {p_meal_ok}"

    # Illegal transition: B→L→N→D→L => penalty > 0
    seq_bad = torch.tensor([[b, lu, low_s, low_e, n, d, lu]])
    p_meal_bad = penalty_meal_order(seq_bad, l['meal_rank'])
    print('Illegal transition: B→L→N→D→L => penalty=', p_meal_bad)
    assert p_meal_bad > 0, f"Penalty should be positive for incorrect meal cycle, got {p_meal_bad}"

    # Illegal transition: B→B→L→D => penalty > 0
    seq_bad = torch.tensor([[b, low_e, low_s, b, low_e, lu, d]])
    p_meal_bad = penalty_meal_order(seq_bad, l['meal_rank'])
    print('Illegal transition: B→B→L→D => penalty=', p_meal_bad)
    assert p_meal_bad > 0, f"Penalty should be positive for repeating meal, got {p_meal_bad}"


def test_soft_interval_penalty_legal_vs_violation_and_grad(mini_tokenizer):
    tk = mini_tokenizer
    l  = build_luts(tk)

    V = len(tk.token2id)
    pad = tk.pad_token_id
    low_s = tk.token2id['A_STATE_Low_START']
    low_e = tk.token2id['A_STATE_Low_END']
    high_s = tk.token2id['A_STATE_High_START']  # conflicting base to 'low'

    # Helper: make allowed mask (block PAD/MASK/CTX/NULL like in training)
    allowed = torch.ones(1, 2, V, dtype=torch.bool)
    for bid in [tk.pad_token_id, tk.mask_token_id, tk.ctx_token_id, tk.null_token_id]:
        allowed[:, :, bid] = False

    # Case A (legal): START then END → small/near-zero penalty
    logits_A = torch.full((1,2,V), -8.0)
    with torch.no_grad():
        logits_A[0,0,low_s] = 8.0
        logits_A[0,1,low_e] = 8.0
    logits_A.requires_grad_()                      # enable grad AFTER edits
    pen_A = soft_interval_penalty(
        logits_A, allowed,
        l['start_ids_per_base'], l['end_ids_per_base'],
        l['conflict_mat'],
        alpha=10.0
    )
    pen_A.backward()
    # penalty should be (very) small; exact zero is not required
    assert pen_A.item() < 1e-3, f"Legal START→END should not be penalized, got {pen_A.item():.6f}"
    # Grad should exist on both steps (differentiable path)
    assert logits_A.grad.abs().sum().item() > 0, "Expected non-zero gradients for soft penalty (legal case)"

    # Case B (FSM): END without prior START at t=0 → positive penalty
    logits_B = torch.full((1,2,V), -8.0)
    with torch.no_grad():
        logits_B[0,0,low_e] = 8.0     # illegal END first
        logits_B[0,1,low_s] = 8.0     # then legal START
    logits_B.requires_grad_()                      # enable grad AFTER edits
    pen_B = soft_interval_penalty(
        logits_B, allowed,
        l['start_ids_per_base'], l['end_ids_per_base'],
        l['conflict_mat'],
        alpha=10.0
    )
    pen_B.backward()
    assert pen_B.item() > 1e-3, "END without open must be penalized"

    # Case C (CNF): conflict START while first base open
    logits_C = torch.full((1,2,V), -8.0)
    with torch.no_grad():
        logits_C[0,0,low_s] = 8.0    # open Low
        logits_C[0,1,high_s] = 8.0   # conflicting START while Low still open  <-- CHANGED
    logits_C.requires_grad_()
    pen_C = soft_interval_penalty(
        logits_C, allowed,
        l['start_ids_per_base'], l['end_ids_per_base'],
        l['conflict_mat'],
        alpha=10.0
    )
    pen_C.backward()
    assert pen_C.item() > 1e-3, "Conflicting START while open must be penalized"

    # Grad must flow back to the *previous* step via the prefix (open_before)
    # Check that the gradient on the first step logits is non-zero in the violating cases
    assert logits_B.grad[0,0].abs().sum().item() > 0, "Grad should reach prior step (FSM case)"
    assert logits_C.grad[0,0].abs().sum().item() > 0, "Grad should reach prior step (CNF case)"


def test_soft_meal_penalty_recency_and_successor(mini_tokenizer):
    tk = mini_tokenizer
    l  = build_luts(tk)

    V = len(tk.token2id)
    b = tk.token2id['MEAL_CONTEXT_Breakfast']
    lu = tk.token2id['MEAL_CONTEXT_Lunch']
    d = tk.token2id['MEAL_CONTEXT_Dinner']
    n = tk.token2id['MEAL_CONTEXT_Night-Snack']

    allowed = torch.ones(1, 3, V, dtype=torch.bool)
    for bid in [tk.pad_token_id, tk.mask_token_id, tk.ctx_token_id, tk.null_token_id]:
        allowed[:, :, bid] = False

    # A) First meal at t=0: free (no prior seen)
    logits_A = torch.full((1,1,V), -8.0)
    with torch.no_grad():
        logits_A[0,0,b] = 8.0
    logits_A.requires_grad_()  # enable grad AFTER edits
    penA = soft_meal_order_penalty(logits_A, allowed[:, :1, :], l['meal_rank'], decay=0.9, beta=6.0)
    penA.backward()
    assert penA.item() < 1e-4, f"First meal should be free; got {penA.item():.6f}"

    # B) Legal successor: Lunch→Dinner (successor of Lunch is Dinner)
    logits_B = torch.full((1,2,V), -8.0)
    with torch.no_grad():
        logits_B[0,0,lu] = 8.0
        logits_B[0,1,d]  = 8.0
    logits_B.requires_grad_()  # enable grad AFTER edits
    penB = soft_meal_order_penalty(logits_B, allowed[:, :2, :], l['meal_rank'], decay=0.9, beta=6.0)
    penB.backward()
    assert penB.item() < 1e-3, f"Legal successor L→D should be near zero penalty; got {penB.item():.6f}"

    # C) Illegal order: Lunch→Breakfast (Breakfast is not successor of Lunch)
    logits_C = torch.full((1,2,V), -8.0)
    with torch.no_grad():
        logits_C[0,0,lu] = 8.0
        logits_C[0,1,b]  = 8.0
    logits_C.requires_grad_()  # enable grad AFTER edits
    penC = soft_meal_order_penalty(logits_C, allowed[:, :2, :], l['meal_rank'], decay=0.9, beta=6.0)
    penC.backward()
    assert penC.item() > 1e-3, "Illegal successor L→B should be penalized"

    # D) Recency: Lunch then Night; right after Lunch, successor mass favors Dinner.
    # If we put mass on Night immediately, penalty should be higher than Dinner.
    logits_Dd = torch.full((1,2,V), -8.0)  # L→D
    with torch.no_grad():
        logits_Dd[0,0,lu] = 8.0 
        logits_Dd[0,1,d] = 8.0
    logits_Dd.requires_grad_()  # enable grad AFTER edits
    pen_Dd = soft_meal_order_penalty(logits_Dd, allowed[:, :2, :], l['meal_rank'], decay=0.9, beta=6.0)

    logits_Dn = torch.full((1,2,V), -8.0)  # L→N (skip Dinner)
    with torch.no_grad():
        logits_Dn[0,0,lu] = 8.0
        logits_Dn[0,1,n] = 8.0
    logits_Dn.requires_grad_()  # enable grad AFTER edits
    pen_Dn = soft_meal_order_penalty(logits_Dn, allowed[:, :2, :], l['meal_rank'], decay=0.9, beta=6.0)

    assert pen_Dn.item() > pen_Dd.item(), "Recency should prefer Dinner over Night immediately after Lunch"

def test_soft_unclosed_interval_penalty(mini_tokenizer):
    """
    Test that soft_unclosed_interval_penalty:
      - Returns 0 for perfectly paired START/END.
      - Returns >0 for START without END.
      - Ignores masked/illegal logits (relies on P).
    """
    tk = mini_tokenizer
    l = build_luts(tk)
    
    # Pick a valid base (e.g. A_STATE_Low)
    base_idx = 0
    s_id = l["start_ids_per_base"][base_idx].item()
    e_id = l["end_ids_per_base"][base_idx].item()
    
    # 1 Batch, 5 Timesteps, V vocab
    B, T, V = 1, 5, len(tk.token2id)
    allowed = torch.ones(B, T, V, dtype=torch.bool)
    
    # --- Scenario 1: Closed Interval (Start t=0, End t=4) ---
    logits_closed = torch.full((B, T, V), -100.0)
    logits_closed[0, 0, s_id] = 100.0
    logits_closed[0, 4, e_id] = 100.0
    
    pen_closed = soft_unclosed_interval_penalty(
        logits_closed, allowed, 
        l["start_ids_per_base"], l["end_ids_per_base"]
    )
    # Mass(Start) ≈ 1, Mass(End) ≈ 1 => Diff ≈ 0
    assert pen_closed.item() < 1e-4, f"Closed sequence should have ~0 penalty, got {pen_closed.item()}"

    # --- Scenario 2: Unclosed Interval (Start t=0, No End) ---
    logits_open = torch.full((B, T, V), -100.0)
    logits_open[0, 0, s_id] = 100.0
    
    pen_open = soft_unclosed_interval_penalty(
        logits_open, allowed, 
        l["start_ids_per_base"], l["end_ids_per_base"]
    )
    
    # Mass(Start) ≈ 1, Mass(End) ≈ 0 => Diff ≈ 1
    # Normalized by (Batch * Num_Bases)
    nbv = (l["start_ids_per_base"] >= 0).sum().item()
    expected = 1.0 / nbv
    
    assert pen_open.item() > 0
    assert abs(pen_open.item() - expected) < 1e-2, f"Expected ~{expected}, got {pen_open.item()}"
    
    print("test_soft_unclosed_interval_penalty: passed.")
    

def test_soft_penalties_agree_with_hard_in_peaked_limit(mini_tokenizer):
    tk = mini_tokenizer
    l  = build_luts(tk)

    V = len(tk.token2id)
    pad = tk.pad_token_id
    low_s = tk.token2id['A_STATE_Low_START']
    low_e = tk.token2id['A_STATE_Low_END']
    lu = tk.token2id['MEAL_CONTEXT_Lunch']
    b = tk.token2id['MEAL_CONTEXT_Breakfast']

    allowed = torch.ones(1, 2, V, dtype=torch.bool)
    for bid in [tk.pad_token_id, tk.mask_token_id, tk.ctx_token_id, tk.null_token_id]:
        allowed[:, :, bid] = False

    scale = 30.0  # make distributions ~one-hot

    # Interval: END→START (hard violation) vs START→END (legal)
    seq_bad = torch.tensor([[low_e, low_s]])      # argmax sequence
    logits_bad = torch.full((1,2,V), -scale)
    logits_bad[0,0,low_e] = scale; logits_bad[0,1,low_s] = scale

    seq_ok = torch.tensor([[low_s, low_e]])
    logits_ok = torch.full((1,2,V), -scale)
    logits_ok[0,0,low_s] = scale; logits_ok[0,1,low_e] = scale

    # Hard penalties on argmax sequences (for direction)
    hard_bad = penalty_interval_structure(seq_bad, seq_ok, l['is_start'], l['is_end'], l['base_id'],
                                          l['start_ids_per_base'], l['end_ids_per_base'],
                                          l['meal_rank'], l['meal_pred_rank'], l['K_meals'],
                                          l['conflict_mat'], l['predict_block'], window=1)
    hard_ok = penalty_interval_structure(seq_ok, seq_ok, l['is_start'], l['is_end'], l['base_id'],
                                         l['start_ids_per_base'], l['end_ids_per_base'],
                                         l['meal_rank'], l['meal_pred_rank'], l['K_meals'],
                                         l['conflict_mat'], l['predict_block'], window=1)
    soft_bad = soft_interval_penalty(logits_bad, allowed, l['start_ids_per_base'], l['end_ids_per_base'], l['conflict_mat'], alpha=12.0)
    soft_ok  = soft_interval_penalty(logits_ok,  allowed, l['start_ids_per_base'], l['end_ids_per_base'], l['conflict_mat'], alpha=12.0)

    assert hard_ok == 0 and soft_ok.item() < 1e-4, "Legal case should be (near) zero in both"
    assert hard_bad > 0 and soft_bad.item() > soft_ok.item(), "Violation should increase soft penalty vs legal"

    # Meals: L→B (bad) vs L→D (ok)
    logits_mb = torch.full((1,2,V), -scale); logits_mb[0,0,lu]=scale; logits_mb[0,1,b]=scale
    logits_md = torch.full((1,2,V), -scale); logits_md[0,0,lu]=scale; logits_md[0,1,tk.token2id['MEAL_CONTEXT_Dinner']]=scale
    soft_mb = soft_meal_order_penalty(logits_mb, allowed, l['meal_rank'], decay=0.9, beta=8.0)
    soft_md = soft_meal_order_penalty(logits_md, allowed, l['meal_rank'], decay=0.9, beta=8.0)
    assert soft_mb.item() > soft_md.item(), "Illegal L→B should be penalized more than legal L→D"


def test_apply_masks_to_logits():
    """
    Test that apply_masks_to_logits:
      - Masks illegal positions to -inf
      - Applies bonus boosting correctly
      - Handles multi-batch, multi-vocab cases
    """
    # 2 batches, 2 timesteps, vocab size 5
    logits = torch.arange(20.0).view(2,2,5)
    # Mark some illegal positions
    illegal = torch.zeros_like(logits, dtype=torch.bool)
    illegal[0,0,1] = True   # batch0, timestep0, token1
    illegal[1,1,4] = True   # batch1, timestep1, token4
    # Mark some bonus positions
    bonus = torch.zeros_like(logits, dtype=torch.bool)
    bonus[0,1,0] = True     # batch0, timestep1, token0
    bonus[1,0,2] = True     # batch1, timestep0, token2

    # Apply with boost=0.3
    boost = 0.3
    out = apply_masks_to_logits(logits.clone(), illegal, bonus, bonus_boost=boost)
    # Expected: illegal -> -inf
    assert out[0,0,1].item() == -1e9, f"Expected -1e9 at [0,0,1], got {out[0,0,1]}"
    assert out[1,1,4].item() == -1e9, f"Expected -1e9 at [1,1,4], got {out[1,1,4]}"
    # Expected: bonus -> logits + boost
    exp00 = logits[0,1,0] + boost
    exp12 = logits[1,0,2] + boost
    print(f"test_apply_masks_to_logits: expected out[0,1,0]={exp00}, got {out[0,1,0]}")
    print(f"test_apply_masks_to_logits: expected out[1,0,2]={exp12}, got {out[1,0,2]}")
    assert abs(out[0,1,0].item() - exp00) < 1e-6
    assert abs(out[1,0,2].item() - exp12) < 1e-6

    # Apply with boost=0.0
    out2 = apply_masks_to_logits(logits.clone(), illegal, bonus, bonus_boost=0.0)
    # Expect same as logits except illegal
    for b in range(2):
        for t in range(2):
            for v in range(5):
                if illegal[b,t,v]:
                    assert out2[b,t,v].item() == -1e9
                else:
                    assert abs(out2[b,t,v].item() - logits[b,t,v].item()) < 1e-6
    print("test_apply_masks_to_logits: all cases passed.")


def test_build_rep_penalty():
    """
    Test that build_rep_penalty:
      - Returns zeros for empty history or strength<=0
      - Correctly applies decay over a sliding window
      - Handles repeated tokens accumulating penalty
    """
    V = 4
    # 1) Empty history
    rep0 = build_rep_penalty([], V, window=3, strength=0.5)
    print(f"test_build_rep_penalty: empty -> {rep0}")
    assert torch.all(rep0 == 0), f"Expected zeros for empty, got {rep0}"

        # 2) Longer history with repeats
    # Use last_tokens = [0,1,2,1,0,1], window=4, strength=0.5
    last = [0,1,2,1,0,1]
    window = 4
    strength = 0.5
    rep = build_rep_penalty(last, V, window=window, strength=strength)
    # Compute expected via the same decay logic:
    k = min(window, len(last))
    decay = torch.linspace(1.0, 0.2, steps=window)[:k] * strength
    idx = torch.tensor(last[-k:])
    flip_idx = idx.flip(0)
    expected = torch.zeros(V)
    for i, tok in enumerate(flip_idx.tolist()):
        expected[tok] += decay[i]
    print(f"test_build_rep_penalty: rep={rep}, expected={expected}")
    for i in range(V):
        assert abs(rep[i].item() - expected[i].item()) < 1e-6, (
            f"Index {i} expected {expected[i]}, got {rep[i]}"
        )
    print("test_build_rep_penalty: all cases passed.")