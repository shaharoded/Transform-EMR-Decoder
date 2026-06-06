import pytest
import torch

from intervene_ar.transformer import InterveneGPT
from intervene_ar.embedder import EMREmbedding
from intervene_ar.dataset import EMRTokenizer

@pytest.fixture(scope="module")
def mini_tokenizer():
    # Minimal EMRTokenizer fixture for Transformer tests
    toks = ["[PAD]", "[MASK]", "[NULL]", "A_START", "A_END", "DEATH_EVENT", "RELEASE_EVENT"]
    token2id = {t: i for i, t in enumerate(toks)}
    rawconcept2id = {"A": 0, "[NULL]": 1, "DEATH_EVENT": 2, "RELEASE_EVENT": 3}
    concept2id = {"A": 0, "[NULL]": 1, "DEATH_EVENT": 2, "RELEASE_EVENT": 3}
    value2id = {"A": 0, "[NULL]": 1, "DEATH_EVENT": 2, "RELEASE_EVENT": 3}
    special_tokens = ["[PAD]", "[MASK]", "[NULL]"]
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
    # assign special token attributes
    tk.pad_token_id = token2id['[PAD]']
    tk.mask_token_id = token2id['[MASK]']
    tk.null_token_id = token2id['[NULL]']
    return tk

@pytest.fixture(scope="module")
def mini_embedder(mini_tokenizer):
    # EMREmbedding with small dims for testing
    return EMREmbedding(
        tokenizer=mini_tokenizer,
        ctx_dim=2,
        time2vec_dim=4,
        embed_dim=8,
        dropout=0.0
    )

@pytest.fixture(scope="module")
def transformer_cfg():
    # Minimal InterveneGPT configuration including time2vec_dim
    return {
        'embed_dim': 8,
        'time2vec_dim': 4,
        'n_layer': 2,
        'n_head': 2,
        'block_size': 4,
        'dropout': 0.1,
        'bias': True,
        'compile': False
    }

@pytest.fixture(scope="module")
def mini_transformer(mini_embedder, transformer_cfg):
    # Instantiate InterveneGPT decoder with fixed small configuration
    return InterveneGPT(cfg=transformer_cfg, embedder=mini_embedder, use_checkpoint=False)


def test_transformer_initialization(mini_transformer, transformer_cfg, mini_embedder):
    """
    Verify InterveneGPT initialization:
      - Model is instance of InterveneGPT
      - Config stored correctly
      - Embedder attached and dimensions match
    """
    model = mini_transformer
    assert isinstance(model, InterveneGPT)
    # Config should be preserved
    for k, v in transformer_cfg.items():
        assert model.cfg[k] == v, f"Config mismatch for {k}: expected {v}, got {model.cfg[k]}"
    # Embedder output dim matches embed_dim
    assert model.embedder.output_dim == transformer_cfg['embed_dim']


def test_transformer_forward_cpu(mini_transformer, mini_tokenizer):
    """
    Forward pass on CPU yields correctly shaped outputs:
      - logits: [B, T+1, V]
      - abs_t_pred: [B, T+1]
    """
    model = mini_transformer
    model.eval()
    B, T = 2, 5
    V = len(mini_tokenizer.token2id)
    # Dummy inputs
    parent_raw = torch.zeros(B, T, 1, dtype=torch.long) # 3D tensor
    concept = torch.zeros(B, T, dtype=torch.long)
    value = torch.zeros(B, T, dtype=torch.long)
    pos = torch.zeros(B, T, dtype=torch.long)
    abs_ts = torch.zeros(B, T)
    context = torch.zeros(B, 2)

    with torch.no_grad():
        logits, abs_t, outcomes, dt_gate, ttt_pred = model(
            parent_raw_ids=parent_raw,
            concept_ids=concept,
            value_ids=value,
            position_ids=pos,
            abs_ts=abs_ts,
            context_vec=context
        )
    # Check shapes
    assert logits.shape == (B, T, V), f"Expected logits shape {(B, T, V)}, got {logits.shape}"
    assert abs_t.shape == (B, T), f"Expected abs_t shape {(B, T)}, got {abs_t.shape}"
    assert outcomes.shape == (B, T, model.num_outcomes), f"Expected outcomes shape {(B, T, model.num_outcomes)}, got {outcomes.shape}"
    assert dt_gate.shape == (B, T), f"Expected dt_gate shape {(B, T)}, got {dt_gate.shape}"
    assert ttt_pred.shape == (B, T), f"Expected ttt_pred shape {(B, T)}, got {ttt_pred.shape}"


def test_transformer_checkpoint_persists_configs(mini_transformer, mini_embedder, tmp_path):
    model = mini_transformer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    ckpt_path = tmp_path / "transformer_ckpt.pt"
    training_settings = {
        "phase2_learning_rate": 1e-3,
        "phase2_n_epochs": 3,
        "phase2_scheduler": {"bce_only_epochs": 1},
    }

    model.save(
        path=ckpt_path,
        epoch=2,
        best_val=0.11,
        optimizer=optimizer,
        scheduler=scheduler,
        lambda_schedule_state={"enabled": True},
        training_settings=training_settings,
        bad_epochs=1,
    )

    loaded_model, epoch, best_val, optim_state, scheduler_state, lambda_state = InterveneGPT.load(
        ckpt_path,
        embedder=mini_embedder,
    )

    assert isinstance(loaded_model, InterveneGPT)
    assert epoch == 2
    assert best_val == pytest.approx(0.11)
    assert optim_state is not None
    assert scheduler_state is not None
    assert lambda_state == {"enabled": True}
    assert loaded_model.checkpoint_model_config["embed_dim"] == model.cfg["embed_dim"]
    assert loaded_model.checkpoint_training_settings == training_settings

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_transformer_forward_gpu(mini_transformer, mini_tokenizer):
    """
    Forward pass on GPU yields correctly shaped outputs and uses GPU tensor types.
    """
    model = mini_transformer.to('cuda')
    model.eval()
    B, T = 2, 5
    V = len(mini_tokenizer.token2id)
    parent_raw = torch.zeros(B, T, 1, dtype=torch.long, device='cuda') # 3D tensor
    concept = torch.zeros(B, T, dtype=torch.long, device='cuda')
    value = torch.zeros(B, T, dtype=torch.long, device='cuda')
    pos = torch.zeros(B, T, dtype=torch.long, device='cuda')
    abs_ts = torch.zeros(B, T, device='cuda')
    context = torch.zeros(B, 2, device='cuda')

    with torch.no_grad():
        logits, abs_t = model(
            parent_raw_ids=parent_raw,
            concept_ids=concept,
            value_ids=value,
            position_ids=pos,
            abs_ts=abs_ts,
            context_vec=context
        )
    # Check device and shapes
    assert logits.device.type == 'cuda'
    assert abs_t.device.type == 'cuda'
    assert logits.shape == (B, T, V)
    assert abs_t.shape == (B, T)