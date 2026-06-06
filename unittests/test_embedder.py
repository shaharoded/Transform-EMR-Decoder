import torch
import pytest

from intervene_ar.embedder import EMREmbedding
from intervene_ar.dataset import EMRTokenizer

@pytest.fixture(scope="module")
def mini_tokenizer():
    # Minimal tokenizer for testing embedder
    toks = ["[PAD]", "[MASK]", "[CTX]", "[NULL]", "A_START", "A_END"]
    token2id = {t:i for i,t in enumerate(toks)}
    rawconcept2id = {"A":0, "[NULL]":1}
    concept2id    = {"A":0, "[NULL]":1}
    value2id      = {"A":0, "[NULL]":1}
    special_tokens = ["[PAD]","[MASK]","[CTX]","[NULL]"]
    token_weights = torch.ones(len(toks))
    outcome_weights = torch.ones(len(toks))
    token_counts = torch.tensor([], dtype=torch.long)

    # Dummy parent raw mapping
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
    # set special token attributes
    tk.pad_token_id  = token2id['[PAD]']
    tk.mask_token_id = token2id['[MASK]']
    tk.ctx_token_id  = token2id['[CTX]']
    tk.null_token_id = token2id['[NULL]']
    return tk

@pytest.mark.order(2)
def test_embedder_initialization(mini_tokenizer):
    cfg = {"ctx_dim":2, "time2vec_dim":2, "embed_dim":8}
    model = EMREmbedding(
        tokenizer=mini_tokenizer,
        ctx_dim=cfg['ctx_dim'],
        time2vec_dim=cfg['time2vec_dim'],
        embed_dim=cfg['embed_dim']
    )
    assert isinstance(model, torch.nn.Module)
    assert model.output_dim == cfg['embed_dim']

@pytest.mark.order(3)
def test_embedder_forward_and_mask_predict(mini_tokenizer):
    """
    Verify shapes for AdaLN-Zero embedder:
    - Returns (seq, cond) tuple by default.
    - seq shape is [B, T, D] (NO prepended CTX).
    - cond shape is [B, D].
    - Returns (seq, cond, mask) if requested.
    """
    tokenizer = mini_tokenizer
    ctx_dim = 2
    embed_dim = 7
    model = EMREmbedding(
        tokenizer=tokenizer,
        ctx_dim=ctx_dim,
        time2vec_dim=2,
        embed_dim=embed_dim
    )
    B, T = 2, 5
    dummy = {
        'parent_raw_ids':  torch.zeros(B, T, 1, dtype=torch.long),
        'concept_ids':     torch.zeros(B, T, dtype=torch.long),
        'value_ids':       torch.zeros(B, T, dtype=torch.long),
        'position_ids':    torch.zeros(B, T, dtype=torch.long),
        'abs_ts':          torch.zeros(B, T),
        'patient_contexts': torch.zeros(B, ctx_dim)
    }
    
    # 1. Forward without mask -> Expect (seq, cond)
    seq, cond = model(**dummy)
    
    # Assert sequence length is preserved (T), not T+1
    assert seq.shape == (B, T, embed_dim), f"Expected seq shape {(B,T,embed_dim)}, got {seq.shape}"
    # Assert condition embedding is correct
    assert cond.shape == (B, embed_dim), f"Expected cond shape {(B,embed_dim)}, got {cond.shape}"

    # 2. Forward with mask -> Expect (seq, cond, mask)
    seq2, cond2, mask = model.forward(**dummy, return_mask=True)
    
    assert seq2.shape == (B, T, embed_dim)
    assert cond2.shape == (B, embed_dim)
    assert mask.shape == (B, T), f"Expected mask shape {(B,T)}, got {mask.shape}"

    # 3. Test predict_time (unchanged)
    pred_t = model.predict_time(dummy['abs_ts'])
    assert pred_t.shape == (B, T, 1)
    assert (pred_t >= 0).all() and (pred_t <= 1).all()

@pytest.mark.order(4)
def test_forward_with_decoder_logits(mini_tokenizer):
    tokenizer = mini_tokenizer
    ctx_dim = 2
    model = EMREmbedding(
        tokenizer=tokenizer,
        ctx_dim=ctx_dim,
        time2vec_dim=2,
        embed_dim=8
    )
    B, T = 2, 4
    dummy = {
        'parent_raw_ids':  torch.zeros(B, T, 1, dtype=torch.long),
        'concept_ids':     torch.zeros(B, T, dtype=torch.long),
        'value_ids':       torch.zeros(B, T, dtype=torch.long),
        'position_ids':    torch.zeros(B, T, dtype=torch.long),
        'abs_ts':          torch.zeros(B, T),
        'patient_contexts': torch.zeros(B, ctx_dim)
    }
    batch = {
     "parent_raw_ids":  dummy['parent_raw_ids'],
     "concept_ids":     dummy['concept_ids'],
     "value_ids":       dummy['value_ids'],
     "position_ids":    dummy['position_ids'],
     "abs_ts":          dummy['abs_ts'],
     # forward_with_decoder pulls from "context_vec" key
     "context_vec":     dummy['patient_contexts'],
    }
    
    # forward_with_decoder returns (logits, seq_embeddings)
    logits, seq_emb = model.forward_with_decoder(batch)
    
    # forward_with_decoder predicts next-token logits: [B, T, vocab_size]
    # No [CTX] prepended, so T remains T.
    vocab_size = len(tokenizer.token2id)
    assert logits.shape == (B, T, vocab_size), f"Expected logits shape {(B,T,vocab_size)}, got {logits.shape}"


@pytest.mark.order(5)
def test_embedder_checkpoint_persists_configs(mini_tokenizer, tmp_path):
    model = EMREmbedding(
        tokenizer=mini_tokenizer,
        ctx_dim=2,
        time2vec_dim=2,
        embed_dim=8,
        dropout=0.2,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")

    ckpt_path = tmp_path / "embedder_ckpt.pt"
    training_settings = {
        "phase1_learning_rate": 1e-3,
        "phase1_n_epochs": 5,
    }

    model.save(
        epoch=3,
        best_val=0.42,
        optimizer=optimizer,
        scheduler=scheduler,
        path=ckpt_path,
        lambda_schedule_state={"stage": 1},
        bad_epochs=2,
        training_settings=training_settings,
    )

    loaded_model, epoch, best_val, optim_state, scheduler_state, lambda_state, bad_epochs = EMREmbedding.load(
        ckpt_path,
        tokenizer=mini_tokenizer,
    )

    assert isinstance(loaded_model, EMREmbedding)
    assert epoch == 3
    assert best_val == pytest.approx(0.42)
    assert optim_state is not None
    assert scheduler_state is not None
    assert lambda_state == {"stage": 1}
    assert bad_epochs == 2
    assert loaded_model.checkpoint_model_config["embed_dim"] == 8
    assert loaded_model.checkpoint_training_settings == training_settings
