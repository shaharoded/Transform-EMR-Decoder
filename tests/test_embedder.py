import torch
from transform_emr.embedder import EMREmbedding
from transform_emr.dataset import EMRTokenizer
import pytest


@pytest.mark.order(2)
def test_embedder_initialization():
    tokenizer = EMRTokenizer.load()
    cfg = {"ctx_dim":4, "time2vec_dim":2, "embed_dim":8}
    model = EMREmbedding(
        tokenizer=tokenizer,
        ctx_dim=cfg["ctx_dim"],
        time2vec_dim=cfg["time2vec_dim"],
        embed_dim=cfg["embed_dim"]
    )
    assert isinstance(model, torch.nn.Module)
    assert model.output_dim == cfg["embed_dim"]

@pytest.mark.order(3)
def test_embedder_forward_shapes():
    # use minimal batch of B=2, T=5
    tokenizer = EMRTokenizer.load()
    model = EMREmbedding(tokenizer=tokenizer, ctx_dim=3, time2vec_dim=2, embed_dim=7)
    B, T = 2, 5
    dummy = {
        "raw_concept_ids": torch.zeros(B, T, dtype=torch.long),
        "concept_ids":     torch.zeros(B, T, dtype=torch.long),
        "value_ids":       torch.zeros(B, T, dtype=torch.long),
        "position_ids":    torch.zeros(B, T, dtype=torch.long),
        "abs_ts":          torch.zeros(B, T),
        "context_vec":     torch.zeros(B, model.tokenizer.ctx_dim)
    }
    logits, abs_pred = model(**dummy)
    # EMREmbedding prepends CTX → length T+1
    assert logits.shape == (B, T+1, model.output_dim)
    assert abs_pred.shape == (B, T+1,)