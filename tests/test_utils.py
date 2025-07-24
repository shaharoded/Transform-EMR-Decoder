import torch
import math
import pytest

from transform_emr.utils import (
    get_multi_hot_targets,
    build_mlm,
    linear_schedule,
    apply_cbm,
    mix_with_predictions,
    penalty_interval_structure,
    penalty_meal_order,
    build_luts
)
from transform_emr.dataset import EMRTokenizer

@pytest.fixture(scope="module")
def mini_tokenizer():
    # very small vocab with PAD, MASK, CTX, ADMISSION, two intervals, 3 meals, TERMINAL
    toks = ["[PAD]","[MASK]","[CTX]","ADMISSION",
            "A_X_START","A_X_END","A_Y_START","A_Y_END",
            "MEAL_B","MEAL_L","MEAL_D","TERMINAL"]
    token2id = {t:i for i,t in enumerate(toks)}
    tk = EMRTokenizer(
        token2id=token2id,
        rawconcept2id={"A_X":0,"A_Y":1,"MEAL":2,"ADMISSION":3,"TERMINAL":4},
        concept2id=   {"A_X":0,"A_Y":1,"MEAL":2,"ADMISSION":3,"TERMINAL":4},
        value2id=     {"A_X":0,"A_Y":1,"MEAL":2,"ADMISSION":3,"TERMINAL":4},
        special_tokens=["[PAD]","[MASK]","[CTX]"],
        token_weights=torch.ones(len(toks)),
        important_token_ids=torch.tensor([],dtype=torch.long)
    )
    tk.pad_token_id  = token2id["[PAD]"]
    tk.mask_token_id = token2id["[MASK]"]
    tk.ctx_token_id  = token2id["[CTX]"]
    return tk

def test_multi_hot_and_shift(mini_tokenizer):
    tk = mini_tokenizer
    # B=1, T=4, window=2
    seq = torch.tensor([[1,2,3,0]])
    mh = get_multi_hot_targets(seq, padding_idx=0, vocab_size=len(tk), k=2)
    # at t=0: look at [1,2] → positions {2,3}
    assert mh[0,0,2]==1 and mh[0,0,3]==1
    # pad never target
    assert mh[...,0].sum()==0

def test_build_mlm_never_mask(mini_tokenizer):
    tk = mini_tokenizer
    # feed IDs including PAD, CTX, ADMISSION, TERMINAL → these must never be masked
    ids = torch.tensor([[tk.pad_token_id, tk.ctx_token_id, 
                         tk.token2id["ADMISSION"], tk.token2id["TERMINAL"], 5]])
    masked, mask = build_mlm(ids, tokenizer=tk, p=1.0)
    # forbidden positions should remain unchanged and mask=False
    for idx in [0,1,2,3]:
        assert masked[0,idx]==ids[0,idx] and mask[0,idx]==False
    # last position must have been masked (mask=True)
    assert mask[0,4] and masked[0,4] in {tk.mask_token_id}

@pytest.mark.parametrize("epoch,warmup,maxv,expected",[
    (0,10,0.5,0.0),(5,10,0.5,0.25),(10,10,0.5,0.5)
])
def test_linear(epoch,warmup,maxv,expected):
    assert math.isclose(linear_schedule(epoch,warmup,maxv), expected)
    i = linear_schedule(epoch,warmup,maxv)
    assert math.isclose(i, expected)

def test_apply_cbm_and_protection(mini_tokenizer):
    tk = mini_tokenizer
    luts = build_luts(tk)
    # batch with position_ids all 0..4
    batch = {
      "position_ids":    torch.arange(5).unsqueeze(0),
      "raw_concept_ids": torch.arange(5).unsqueeze(0),
      "concept_ids":     torch.arange(5).unsqueeze(0),
      "value_ids":       torch.arange(5).unsqueeze(0),
    }
    # forbid masking token 2 → should remain
    out = apply_cbm(batch.copy(), epoch=5, warmup_epochs=10, tokenizer=tk,
                    forbid_ids=torch.tensor([2]), max_p=1.0)
    assert out["position_ids"][0,2] == 2

def test_mix_with_predictions(mini_tokenizer):
    tk = mini_tokenizer
    gt   = torch.tensor([[0,1,2,3]])
    pred = torch.tensor([[9,9,9,9]])
    prot = torch.zeros(len(tk),dtype=torch.bool)
    prot[1]=True
    mixed,mask = mix_with_predictions(gt,pred,epoch=5,warmup_epochs=10,protected_ids=prot)
    # protected slot not replaced
    assert mixed[0,1]==1 and mask[0,1]==False
    # some unprotected likely replaced
    assert mixed.shape==gt.shape and mask.dtype==torch.bool

def test_vectorized_penalties_zero(mini_tokenizer):
    tk = mini_tokenizer
    l = build_luts(tk)
    seq = torch.zeros(1,5,dtype=torch.long)
    # perfectly legal
    p1 = penalty_interval_structure(seq, seq,
        is_start=l["is_start"], is_end=l["is_end"],
        base_id=l["base_id"], start_ids_per_base=l["start_ids_per_base"],
        end_ids_per_base=l["end_ids_per_base"],
        meal_rank=l["meal_rank"], meal_pred_rank=l["meal_pred_rank"],
        K_meals=l["K_meals"], conflict_mat=l["conflict_mat"],
        window=1
    )
    assert pytest.approx(0.0) == p1.item()

    # meal order OK: no meals → zero
    p2 = penalty_meal_order(seq, l["meal_rank"])
    assert p2.item() == 0.0
