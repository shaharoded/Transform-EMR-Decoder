import torch
import torch.nn.functional as F
import pytest
import transform_emr.utils as utils_module

from transform_emr.utils import (
    get_temporal_multi_hot_targets,
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
from transform_emr.utils import compute_soft_outcome_labels
from transform_emr.inference import _build_illegal_mask, _update_legality_state


def test_linear_schedule_import_is_from_schedulers():
    """Ensure utils.linear_schedule remains the scheduler-exported function (import compatibility)."""
    assert linear_schedule is sched_linear_schedule
    assert utils_module.linear_schedule is sched_linear_schedule


def test_unified_lambda_schedule_controller_phase1_alias_and_immediate_activation():
    """
    Phase-1 scheduling contract:
      - bce_only_epochs=1 means aux activates at epoch 1, not 0.
      - Calibration uses tr_main (training BCE) not validation.
      - update() uses vl_total for plateau, tr_main+aux names for calibration.
      - Single-stage: has_dynamic=False, current_warmup_end_epoch() returns concrete epoch.
    """
    cfg = {
        "bce_only_epochs": 1,
        "aux_fraction_caps": {"mlm": 0.20, "dt": 0.20},
        "order": [["mlm", "dt"]],
        "ramp_epochs": {"mlm": 1, "dt": 1},
    }
    controller = LambdaScheduleController(schedule_config=cfg, start_epoch=0)
    assert controller.has_dynamic is False
    # For single-stage, warmup_end_epoch returns the epoch after ramp completes
    warmup_epoch = controller.current_warmup_end_epoch()
    assert warmup_epoch == 1, f"Expected warmup to end at epoch 1, got {warmup_epoch}"

    # Before bce_only_epochs passes, lambdas = 0.
    assert controller.get_lambdas(epoch=0)["mlm"] == 0.0

    # First active epoch (1): calibrate using training losses.
    events = controller.update(epoch=1, vl_total=1.8, tr_main=2.0, mlm=1.0, dt=0.5)
    assert len(events) >= 2  # one calibration message per aux

    # lambda_max = fraction * tr_main / tr_aux
    # mlm: 0.20 * 2.0 / 1.0 = 0.40;  dt: 0.20 * 2.0 / 0.5 = 0.80
    l1 = controller.get_lambdas(epoch=1)
    assert l1["mlm"] == pytest.approx(0.40)
    assert l1["dt"] == pytest.approx(0.80)

    # Calibration is frozen — second update must not change lambda_max.
    controller.update(epoch=2, vl_total=1.0, tr_main=10.0, mlm=10.0, dt=10.0)
    l2 = controller.get_lambdas(epoch=2)
    assert l2["mlm"] == pytest.approx(l1["mlm"])
    assert l2["dt"] == pytest.approx(l1["dt"])


def test_unified_lambda_schedule_controller_phase2_ramp_progression():
    """
    Phase-2 scheduling contract:
      - Stage-0 (ce, dt) activates at epoch 1 (bce_only_epochs=1), ramp_epochs=5.
      - Ramp runs from start_epoch to start_epoch+ramp_epochs (epoch 1 to 6).
      - Penalty/Outcome stay zero before unlock.
      - has_dynamic=True; current_warmup_end_epoch() is inf until outcome unlocked.
    """
    cfg = {
        "bce_only_epochs": 1,
        "aux_fraction_caps": {"ce": 0.20, "dt": 0.20, "penalty": 0.20, "outcome": 0.20},
        "order": [["ce", "dt"], ["penalty"], ["outcome"]],
        "ramp_epochs": {"ce": 5, "dt": 5, "penalty": 5, "outcome": 5},
        "plateau_min_delta": 1e-4,
        "plateau_patience": [3, 3],
    }
    controller = LambdaScheduleController(schedule_config=cfg, start_epoch=0)
    assert controller.has_dynamic is True
    assert controller.current_warmup_end_epoch() == float("inf")

    # Calibrate at first active epoch (1) using training losses.
    controller.update(epoch=1, vl_total=2.0, tr_main=2.0, ce=1.0, dt=0.5)
    # lambda_max: ce=0.40, dt=0.80. ramp start=1, end=6.

    # At start of ramp (epoch 1): linear_schedule(1, 1, 6, max) = 0.
    lam1 = controller.get_lambdas(epoch=1)
    assert lam1["ce"] == pytest.approx(0.0)
    assert lam1["dt"] == pytest.approx(0.0)
    assert lam1["penalty"] == pytest.approx(0.0)
    assert lam1["outcome"] == pytest.approx(0.0)

    # Mid-ramp at epoch 3: (3-1)/(6-1) = 2/5.
    lam3 = controller.get_lambdas(epoch=3)
    assert lam3["ce"] == pytest.approx(0.40 * (2 / 5), rel=1e-6)
    assert lam3["dt"] == pytest.approx(0.80 * (2 / 5), rel=1e-6)

    # End-ramp at epoch 6: (6-1)/(6-1) = 1 → full lambda_max.
    lam6 = controller.get_lambdas(epoch=6)
    assert lam6["ce"] == pytest.approx(0.40, rel=1e-6)
    assert lam6["dt"] == pytest.approx(0.80, rel=1e-6)


def test_scheduler_stage_transitions_and_warmup():
    """
    Phase-2 dynamic curriculum with ramp-delay gating:
      - bce_only_epochs=1: stage-0 (ce, dt) starts at epoch 1, ramp=1 → ramp_end=1.
      - Plateau check for stage 0→1 starts at epoch >= ramp_end (epoch 1).
      - Penalty unlocks after patience=2 non-improving vl_total epochs.
      - Plateau check for stage 1→2 is SKIPPED during penalty ramp (epochs penalty_start to penalty_start+3).
      - Outcome unlocks after patience=2 non-improving epochs post-ramp.
      - warmup_complete_epoch = outcome_start + outcome_ramp_epochs.
    """
    cfg = {
        "bce_only_epochs": 1,
        "aux_fraction_caps": {"ce": 0.20, "dt": 0.20, "penalty": 0.20, "outcome": 0.20},
        "order": [["ce", "dt"], ["penalty"], ["outcome"]],
        "ramp_epochs": {"ce": 1, "dt": 1, "penalty": 3, "outcome": 3},
        "plateau_min_delta": 1e-4,
        "plateau_patience": [2, 2],
    }
    sc = LambdaScheduleController(schedule_config=cfg, start_epoch=0)

    # Epoch 1: first active epoch. Calibrates + plateau check starts.
    # vl_total=1.8 is improvement over inf → bad=0.
    sc.update(epoch=1, vl_total=1.8, tr_main=1.0, ce=0.5, dt=0.3)
    assert sc._auxiliaries["penalty"]["start_epoch"] is None

    # Epoch 2: vl_total flat → bad=1 < patience=2 → locked.
    sc.update(epoch=2, vl_total=1.8, tr_main=1.0, ce=0.5, dt=0.3)
    assert sc._auxiliaries["penalty"]["start_epoch"] is None, "bad=1 < patience=2"

    # Epoch 3: bad=2 = patience → penalty unlocks at epoch 4.
    sc.update(epoch=3, vl_total=1.8, tr_main=1.0, ce=0.5, dt=0.3)
    penalty_start = sc._auxiliaries["penalty"]["start_epoch"]
    assert penalty_start == 4, f"Expected penalty_start=4, got {penalty_start}"

    # During penalty ramp (penalty_start=4, ramp_epochs=3, ramp_end=7):
    # plateau check for outcome is SKIPPED at epochs 4, 5, 6.
    sc.update(epoch=4, vl_total=1.5, tr_main=0.7, ce=0.4, dt=0.2, penalty=0.4)
    sc.update(epoch=5, vl_total=1.5, tr_main=0.7, ce=0.4, dt=0.2, penalty=0.4)
    sc.update(epoch=6, vl_total=1.5, tr_main=0.7, ce=0.4, dt=0.2, penalty=0.4)
    assert sc._auxiliaries["outcome"]["start_epoch"] is None, "Must not unlock during penalty ramp"

    # Epoch 7: ramp ends (ramp_end=7). First plateau check for stage 1→2.
    # vl_total=1.0 is improvement over inf → bad=0.
    sc.update(epoch=7, vl_total=1.0, tr_main=0.6, ce=0.3, dt=0.2, penalty=0.4)
    assert sc._auxiliaries["outcome"]["start_epoch"] is None

    # Epoch 8: flat → bad=1.
    sc.update(epoch=8, vl_total=1.0, tr_main=0.6, ce=0.3, dt=0.2, penalty=0.4)
    assert sc._auxiliaries["outcome"]["start_epoch"] is None, "bad=1 < patience=2"

    # Epoch 9: bad=2 = patience → outcome unlocks at epoch 10.
    sc.update(epoch=9, vl_total=1.0, tr_main=0.6, ce=0.3, dt=0.2, penalty=0.4)
    outcome_start = sc._auxiliaries["outcome"]["start_epoch"]
    assert outcome_start == 10, f"Expected outcome_start=10, got {outcome_start}"

    # Warmup = outcome_start + ramp_epochs(3).
    assert sc.current_warmup_end_epoch() == 13

    # Penalty lambda ramps: 0 at start, lambda_max at start+3.
    pen_max = sc._auxiliaries["penalty"]["lambda_max"]
    assert sc.get_lambdas(penalty_start)["penalty"] == pytest.approx(0.0)
    assert sc.get_lambdas(penalty_start + 3)["penalty"] == pytest.approx(pen_max)


def test_scheduler_state_dict_roundtrip():
    """state_dict() / load_state_dict() restores calibration and stage state exactly."""
    cfg = {
        "bce_only_epochs": 1,
        "aux_fraction_caps": {"ce": 0.20, "dt": 0.20, "penalty": 0.20, "outcome": 0.20},
        "order": [["ce", "dt"], ["penalty"], ["outcome"]],
        "ramp_epochs": {"ce": 1, "dt": 1, "penalty": 3, "outcome": 3},
        "plateau_min_delta": 1e-4,
        "plateau_patience": [2, 2],
    }
    sc = LambdaScheduleController(schedule_config=cfg, start_epoch=0)
    sc.update(epoch=1, vl_total=1.8, tr_main=1.0, ce=0.5, dt=0.3)
    sc.update(epoch=2, vl_total=1.8, tr_main=1.0, ce=0.5, dt=0.3)
    sc.update(epoch=3, vl_total=1.8, tr_main=1.0, ce=0.5, dt=0.3)

    state = sc.state_dict()

    # New controller with same config, restore state.
    sc2 = LambdaScheduleController(schedule_config=cfg, start_epoch=0)
    sc2.load_state_dict(state)

    assert sc2._auxiliaries["ce"]["lambda_max"] == pytest.approx(sc._auxiliaries["ce"]["lambda_max"])
    assert sc2._auxiliaries["penalty"]["start_epoch"] == sc._auxiliaries["penalty"]["start_epoch"]
    assert sc2._current_stage == sc._current_stage
    assert sc2._stage_bad_epochs == sc._stage_bad_epochs
    assert sc2.get_lambdas(4) == sc.get_lambdas(4)


def test_scheduler_missing_cap_raises():
    """Missing aux_fraction_caps entry raises KeyError at construction, not silently."""
    cfg = {
        "aux_fraction_caps": {"ce": 0.20},  # dt missing
        "order": [["ce", "dt"]],
        "ramp_epochs": {"ce": 1, "dt": 1},
    }
    with pytest.raises(KeyError, match="dt"):
        LambdaScheduleController(schedule_config=cfg, start_epoch=0)

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
        token_counts=token_counts,
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
    assert they match within a temporal window.

    Expectation: At every position t, the targets are the IDs that appear within a time window 
    after t, until padding (0) token is encountered.
    """
    # --- Setup a toy sequence: 1..10 then two PADs (0) ---
    seq = torch.tensor([[1,2,3,4,5,6,7,8,9,10,0,0]], dtype=torch.long)
    B, T = seq.shape
    V    = seq.max().item() + 1  # 11 = tokens 0..10
    
    # --- Create timestamps: each token is 1 unit apart ---
    # This simulates a sequence where each event occurs 1 time unit after the previous
    abs_ts = torch.arange(T, dtype=torch.float32).unsqueeze(0)  # [B, T]
    
    # --- Window size of 5 units matches the previous k=5 behavior ---
    window_size = 5.0

    # --- Compute temporal multi-hot targets ---
    mh = get_temporal_multi_hot_targets(
        target_ids=seq, 
        all_abs_ts=abs_ts, 
        padding_idx=0, 
        vocab_size=V, 
        window_size=window_size
    )
    assert mh.shape == (B, T, V)

    # --- For each timestep, print & assert correctness ---
    for t in range(T):
        # curr
        curr = seq[0, t]
        # ground-truth: future tokens within window_size time units, excluding pads
        # tokens at position s where 0 < (abs_ts[s] - abs_ts[t]) <= window_size
        future_mask = (abs_ts[0] > abs_ts[0, t]) & (abs_ts[0] <= abs_ts[0, t] + window_size)
        future_ids = seq[0, future_mask].tolist()
        # drop pads & dedupe
        expected = sorted({x for x in future_ids if x != 0})

        # what the function actually marked
        hot_ids = mh[0, t].nonzero(as_tuple=False).squeeze(-1).tolist()
        hot_ids.sort()

        # print for human verification
        print(f"t={t}, curr={curr} | future_ids={future_ids} | hot_ids={hot_ids}")

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
    illegal_ok = compute_legality_masks_tf(
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

    # 2) FSM violation: END before START
    seq_rev = torch.tensor([[low_e, low_s, pad]])
    illegal_rev = compute_legality_masks_tf(
        seq_rev, l['is_start'], l['is_end'], l['base_id'],
        l['start_ids_per_base'], l['end_ids_per_base'], l['meal_rank'],
        l['meal_pred_rank'], l['K_meals'], l['conflict_mat'], l['predict_block']
    )
    assert illegal_rev[0,0,low_e], "FSM violation should flag END when no START"

    # 3) CNF violation: conflicting High START while Low open
    seq_conf = torch.tensor([[low_s, high_s, pad]])
    illegal_conf = compute_legality_masks_tf(
        seq_conf, l['is_start'], l['is_end'], l['base_id'],
        l['start_ids_per_base'], l['end_ids_per_base'], l['meal_rank'],
        l['meal_pred_rank'], l['K_meals'], l['conflict_mat'], l['predict_block']
    )
    assert illegal_conf[0,1,high_s], "CNF violation should flag conflicting High START"

    # 4) DUP violation: START twice in a row on same base
    seq_dup = torch.tensor([[low_s, low_s, pad]])
    illegal_dup = compute_legality_masks_tf(
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
    illegal_cycle = compute_legality_masks_tf(
        seq_cycle, l['is_start'], l['is_end'], l['base_id'],
        l['start_ids_per_base'], l['end_ids_per_base'], l['meal_rank'],
        l['meal_pred_rank'], l['K_meals'], l['conflict_mat'], l['predict_block']
    )
    for t_idx, tok in enumerate([l_id, d, n, b, l_id]):
        assert not illegal_cycle[0, t_idx, tok], (
            f"Token {tk.id2token[tok]} wrongly flagged illegal at position {t_idx}")

    # 6) Illegal short cycle: L → B → D — Breakfast at t=1 should be flagged illegal
    seq_bad = torch.tensor([[l_id, b, d, pad]])
    illegal_bad = compute_legality_masks_tf(
        seq_bad, l['is_start'], l['is_end'], l['base_id'],
        l['start_ids_per_base'], l['end_ids_per_base'], l['meal_rank'],
        l['meal_pred_rank'], l['K_meals'], l['conflict_mat'], l['predict_block']
    )
    assert illegal_bad[0, 1, b], "Meal order violation should flag Breakfast at t=1 for L→B"

    # 7) Interval+Meal interleaving
    seq_mix1 = torch.tensor([[low_s, b, low_e, pad]])
    illegal_mix1 = compute_legality_masks_tf(
        seq_mix1, l['is_start'], l['is_end'], l['base_id'],
        l['start_ids_per_base'], l['end_ids_per_base'], l['meal_rank'],
        l['meal_pred_rank'], l['K_meals'], l['conflict_mat'], l['predict_block']
    )
    # low_e at pos2 should be legal despite an unrelated meal token at pos1
    assert not illegal_mix1[0, 2, low_e], (
        "Interval END should be legal even with unrelated meal token in between")

    # 8) Meal+Interval interleaving
    seq_mix2 = torch.tensor([[l_id, low_s, d, n, b, l_id]])
    illegal_mix2 = compute_legality_masks_tf(
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
    illegal_pred = compute_legality_masks_tf(pred2, l['is_start'], l['is_end'], l['base_id'],
        l['start_ids_per_base'], l['end_ids_per_base'], l['meal_rank'],
        l['meal_pred_rank'], l['K_meals'], l['conflict_mat'], l['predict_block'])
    pred_illegal = illegal_pred.gather(2, pred2.unsqueeze(-1)).squeeze(-1)
    print("pred_illegal:", pred_illegal)          # e.g. tensor([[False, False,  True]])
    gt_illegal = compute_legality_masks_tf(gt2, l['is_start'], l['is_end'], l['base_id'],
        l['start_ids_per_base'], l['end_ids_per_base'], l['meal_rank'],
        l['meal_pred_rank'], l['K_meals'], l['conflict_mat'], l['predict_block']) \
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
      - Masks illegal positions to -1e9 (effectively -inf for suppression)
      - Preserves legal positions unchanged
      - Handles multi-batch, multi-vocab cases
      
    Note: Bonus boosting was removed as it interfered with BCE learning.
    """
    # 2 batches, 2 timesteps, vocab size 5
    logits = torch.arange(20.0).view(2,2,5)
    # Mark some illegal positions
    illegal = torch.zeros_like(logits, dtype=torch.bool)
    illegal[0,0,1] = True   # batch0, timestep0, token1
    illegal[1,1,4] = True   # batch1, timestep1, token4

    # Apply masking
    out = apply_masks_to_logits(logits.clone(), illegal)
    
    # Expected: illegal -> -1e9
    assert out[0,0,1].item() == -1e9, f"Expected -1e9 at [0,0,1], got {out[0,0,1]}"
    assert out[1,1,4].item() == -1e9, f"Expected -1e9 at [1,1,4], got {out[1,1,4]}"
    
    # Expected: legal positions preserve original logits
    for b in range(2):
        for t in range(2):
            for v in range(5):
                if illegal[b,t,v]:
                    assert out[b,t,v].item() == -1e9, f"Illegal [{ b},{t},{v}] not masked"
                else:
                    assert abs(out[b,t,v].item() - logits[b,t,v].item()) < 1e-6, \
                        f"Legal position [{b},{t},{v}] should preserve logits"
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


# ─────────────────────────────────────────────────────────────────────────────
# compute_soft_outcome_labels
# ─────────────────────────────────────────────────────────────────────────────

def _make_gt_df(tokens_and_times):
    """Helper: build a minimal ground-truth DataFrame with PositionToken and TimePoint columns.
    TimePoint is normalised (hours / 336) to match the DataProcessor output convention."""
    import pandas as pd
    rows = [{"PositionToken": tok, "TimePoint": t / 336.0} for tok, t in tokens_and_times]
    return pd.DataFrame(rows)


def test_compute_soft_outcome_labels_single_future_event():
    """
    One outcome event at t=48 h.  Generated steps at t=24, 36, 48, 60 h.
    tau=12, horizon=48.

    Expected behaviour:
    - t=24 : dt=24 → exp(-24/12)=exp(-2) ≈ 0.135  (within horizon)
    - t=36 : dt=12 → exp(-1)           ≈ 0.368
    - t=48 : dt=0  → NOT future (dt==0, condition is dt>0), so 0
    - t=60 : dt=-12 → past event, 0
    """
    outcome_name = "OUTCOME_EVENT"
    gt_df = _make_gt_df([(outcome_name, 48.0)])  # outcome at 48 h

    gen_abs_ts = torch.tensor([24.0, 36.0, 48.0, 60.0])
    tau_hours     = 12.0
    horizon_hours = 48.0

    labels = compute_soft_outcome_labels(
        gen_abs_ts, gt_df, [outcome_name], tau_hours, horizon_hours, device=torch.device("cpu")
    )

    assert labels.shape == (4, 1), f"Expected [4,1], got {labels.shape}"

    expected_24 = torch.exp(torch.tensor(-24.0 / tau_hours)).clamp(0, 1).item()
    expected_36 = torch.exp(torch.tensor(-12.0 / tau_hours)).clamp(0, 1).item()

    assert abs(labels[0, 0].item() - expected_24) < 1e-5, \
        f"t=24 expected {expected_24:.4f}, got {labels[0,0].item():.4f}"
    assert abs(labels[1, 0].item() - expected_36) < 1e-5, \
        f"t=36 expected {expected_36:.4f}, got {labels[1,0].item():.4f}"
    assert labels[2, 0].item() == 0.0, "dt==0 should not count as future"
    assert labels[3, 0].item() == 0.0, "Past event should give 0"
    print(f"labels: {labels.squeeze().tolist()} — test_compute_soft_outcome_labels_single_future_event passed.")


def test_compute_soft_outcome_labels_beyond_horizon():
    """Event at t=100 h, generated step at t=0 h.  horizon=48 → contribution should be 0."""
    outcome_name = "OUTCOME_EVENT"
    gt_df = _make_gt_df([(outcome_name, 100.0)])
    gen_abs_ts = torch.tensor([0.0])
    labels = compute_soft_outcome_labels(
        gen_abs_ts, gt_df, [outcome_name], tau_hours=12.0, horizon_hours=48.0,
        device=torch.device("cpu")
    )
    assert labels[0, 0].item() == 0.0, "Event beyond horizon should give 0"
    print("test_compute_soft_outcome_labels_beyond_horizon passed.")


def test_compute_soft_outcome_labels_multiple_events_clamped():
    """Two outcome events close together — sum may exceed 1 and must be clamped."""
    outcome_name = "OUTCOME_EVENT"
    gt_df = _make_gt_df([(outcome_name, 10.0), (outcome_name, 12.0)])
    gen_abs_ts = torch.tensor([0.0])  # 10 h and 12 h ahead
    labels = compute_soft_outcome_labels(
        gen_abs_ts, gt_df, [outcome_name], tau_hours=12.0, horizon_hours=48.0,
        device=torch.device("cpu")
    )
    assert labels[0, 0].item() <= 1.0, "Labels must be clamped to [0, 1]"
    assert labels[0, 0].item() > 0.0, "Close future events should produce non-zero label"
    print(f"clamped label={labels[0,0].item():.4f} — test_compute_soft_outcome_labels_multiple_events_clamped passed.")


def test_compute_soft_outcome_labels_absent_outcome():
    """Outcome name not present in gt_df → all zeros."""
    import pandas as pd
    gt_df = _make_gt_df([("OTHER_EVENT", 10.0)])
    gen_abs_ts = torch.tensor([0.0, 5.0])
    labels = compute_soft_outcome_labels(
        gen_abs_ts, gt_df, ["OUTCOME_EVENT"], tau_hours=12.0, horizon_hours=48.0,
        device=torch.device("cpu")
    )
    assert (labels == 0).all(), "Absent outcome should produce all-zero labels"
    print("test_compute_soft_outcome_labels_absent_outcome passed.")


# ─────────────────────────────────────────────────────────────────────────────
# _build_illegal_mask  and  _update_legality_state
# ─────────────────────────────────────────────────────────────────────────────

def test_build_illegal_mask_always_blocks_pad_and_mask(mini_tokenizer):
    """PAD and MASK must always appear in the illegal mask regardless of state."""
    tk  = mini_tokenizer
    luts = build_luts(tk)
    luts = {k: v.to("cpu") if torch.is_tensor(v) else v for k, v in luts.items()}

    n_b         = luts["start_ids"].numel()
    open_counts = torch.zeros(n_b, dtype=torch.int32)
    K           = int((luts["meal_rank"] >= 0).any()) and int(luts["meal_rank"].max().item()) + 1 or 0

    illegal = _build_illegal_mask(luts, open_counts, None, tk.pad_token_id, tk.mask_token_id, "cpu")

    assert illegal[tk.pad_token_id],  "PAD must always be illegal"
    assert illegal[tk.mask_token_id], "MASK must always be illegal"
    print("test_build_illegal_mask_always_blocks_pad_and_mask passed.")


def test_build_illegal_mask_end_blocked_when_no_start(mini_tokenizer):
    """END token should be illegal when the interval has never been opened."""
    tk  = mini_tokenizer
    luts = build_luts(tk)
    luts = {k: v.to("cpu") if torch.is_tensor(v) else v for k, v in luts.items()}

    n_b         = luts["start_ids"].numel()
    open_counts = torch.zeros(n_b, dtype=torch.int32)  # nothing open

    low_e = tk.token2id["A_STATE_Low_END"]
    illegal = _build_illegal_mask(luts, open_counts, None, tk.pad_token_id, tk.mask_token_id, "cpu")

    assert illegal[low_e], "END should be illegal when interval is not open"
    print("test_build_illegal_mask_end_blocked_when_no_start passed.")


def test_build_illegal_mask_start_blocked_when_open(mini_tokenizer):
    """Duplicate START should be illegal once the interval is already open."""
    tk  = mini_tokenizer
    luts = build_luts(tk)
    luts = {k: v.to("cpu") if torch.is_tensor(v) else v for k, v in luts.items()}

    n_b         = luts["start_ids"].numel()
    open_counts = torch.zeros(n_b, dtype=torch.int32)

    low_s = tk.token2id["A_STATE_Low_START"]
    # Open the Low_STATE interval
    open_counts = open_counts.clone()
    open_counts[luts["base_id"][low_s]] = 1

    illegal = _build_illegal_mask(luts, open_counts, None, tk.pad_token_id, tk.mask_token_id, "cpu")

    assert illegal[low_s], "Duplicate START should be illegal when interval is already open"
    print("test_build_illegal_mask_start_blocked_when_open passed.")


def test_build_illegal_mask_conflict_blocked(mini_tokenizer):
    """Starting a conflicting value while another is open must be illegal."""
    tk  = mini_tokenizer
    luts = build_luts(tk)
    luts = {k: v.to("cpu") if torch.is_tensor(v) else v for k, v in luts.items()}

    n_b         = luts["start_ids"].numel()
    open_counts = torch.zeros(n_b, dtype=torch.int32)

    low_s  = tk.token2id["A_STATE_Low_START"]
    high_s = tk.token2id["A_STATE_High_START"]
    # Open Low; High is a conflict
    open_counts[luts["base_id"][low_s]] = 1

    illegal = _build_illegal_mask(luts, open_counts, None, tk.pad_token_id, tk.mask_token_id, "cpu")

    assert illegal[high_s], "Conflicting START (High while Low open) should be illegal"
    print("test_build_illegal_mask_conflict_blocked passed.")


def test_build_illegal_mask_meal_order(mini_tokenizer):
    """When next_meal_rank is set, only the expected meal token should be allowed."""
    tk  = mini_tokenizer
    luts = build_luts(tk)
    luts = {k: v.to("cpu") if torch.is_tensor(v) else v for k, v in luts.items()}

    n_b         = luts["start_ids"].numel()
    open_counts = torch.zeros(n_b, dtype=torch.int32)
    K           = int((luts["meal_rank"] >= 0).any()) and int(luts["meal_rank"].max().item()) + 1 or 0

    b    = tk.token2id["MEAL_CONTEXT_Breakfast"]
    l_id = tk.token2id["MEAL_CONTEXT_Lunch"]
    d    = tk.token2id["MEAL_CONTEXT_Dinner"]
    n    = tk.token2id["MEAL_CONTEXT_Night-Snack"]
    meal_ranks = {b: luts["meal_rank"][b].item(),
                  l_id: luts["meal_rank"][l_id].item(),
                  d: luts["meal_rank"][d].item(),
                  n: luts["meal_rank"][n].item()}
    # Find which rank Lunch has, then set next_meal_rank to that
    lunch_rank = meal_ranks[l_id]
    illegal = _build_illegal_mask(luts, open_counts, lunch_rank, tk.pad_token_id, tk.mask_token_id, "cpu")

    # Lunch should be legal; all other meals should be illegal
    assert not illegal[l_id], "Expected meal (Lunch) should be legal"
    for tok, rank in meal_ranks.items():
        if rank != lunch_rank:
            assert illegal[tok], f"Non-expected meal (rank {rank}) should be illegal"
    print("test_build_illegal_mask_meal_order passed.")


def test_update_legality_state_start_increments_open(mini_tokenizer):
    """After a START token, open_counts for that base should increase by 1."""
    tk  = mini_tokenizer
    luts = build_luts(tk)
    luts = {k: v.to("cpu") if torch.is_tensor(v) else v for k, v in luts.items()}

    n_b         = luts["start_ids"].numel()
    open_counts = torch.zeros(n_b, dtype=torch.int32)
    K           = int((luts["meal_rank"] >= 0).any()) and int(luts["meal_rank"].max().item()) + 1 or 0

    low_s = tk.token2id["A_STATE_Low_START"]
    base  = luts["base_id"][low_s].item()

    _update_legality_state(luts, low_s, open_counts, None, K)

    assert open_counts[base].item() == 1, f"Expected open_counts[{base}]=1, got {open_counts[base].item()}"
    print("test_update_legality_state_start_increments_open passed.")


def test_update_legality_state_end_decrements_open(mini_tokenizer):
    """After a matching END, open_counts should decrease back to 0."""
    tk  = mini_tokenizer
    luts = build_luts(tk)
    luts = {k: v.to("cpu") if torch.is_tensor(v) else v for k, v in luts.items()}

    n_b         = luts["start_ids"].numel()
    open_counts = torch.zeros(n_b, dtype=torch.int32)
    K           = int((luts["meal_rank"] >= 0).any()) and int(luts["meal_rank"].max().item()) + 1 or 0

    low_s = tk.token2id["A_STATE_Low_START"]
    low_e = tk.token2id["A_STATE_Low_END"]
    base  = luts["base_id"][low_s].item()

    _update_legality_state(luts, low_s, open_counts, None, K)
    assert open_counts[base].item() == 1
    _update_legality_state(luts, low_e, open_counts, None, K)
    assert open_counts[base].item() == 0, "END should bring open_counts back to 0"
    print("test_update_legality_state_end_decrements_open passed.")


def test_update_legality_state_end_does_not_go_negative(mini_tokenizer):
    """END without prior START should not decrement below 0."""
    tk  = mini_tokenizer
    luts = build_luts(tk)
    luts = {k: v.to("cpu") if torch.is_tensor(v) else v for k, v in luts.items()}

    n_b         = luts["start_ids"].numel()
    open_counts = torch.zeros(n_b, dtype=torch.int32)
    K           = int((luts["meal_rank"] >= 0).any()) and int(luts["meal_rank"].max().item()) + 1 or 0

    low_e = tk.token2id["A_STATE_Low_END"]
    _update_legality_state(luts, low_e, open_counts, None, K)
    base = luts["base_id"][low_e].item()
    assert open_counts[base].item() >= 0, "open_counts must never go negative"
    print("test_update_legality_state_end_does_not_go_negative passed.")


def test_update_legality_state_meal_advances_rank(mini_tokenizer):
    """After a meal token, next_meal_rank should advance to the next rank in the cycle."""
    tk  = mini_tokenizer
    luts = build_luts(tk)
    luts = {k: v.to("cpu") if torch.is_tensor(v) else v for k, v in luts.items()}

    n_b         = luts["start_ids"].numel()
    open_counts = torch.zeros(n_b, dtype=torch.int32)
    K           = int((luts["meal_rank"] >= 0).any()) and int(luts["meal_rank"].max().item()) + 1 or 0

    l_id = tk.token2id["MEAL_CONTEXT_Lunch"]
    lunch_rank = luts["meal_rank"][l_id].item()

    next_rank = _update_legality_state(luts, l_id, open_counts, None, K)

    assert next_rank == (lunch_rank + 1) % K, \
        f"Expected rank {(lunch_rank+1)%K}, got {next_rank}"
    print(f"Lunch rank={lunch_rank}, next={next_rank} — test_update_legality_state_meal_advances_rank passed.")