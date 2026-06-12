"""
train.py
=====================
Three-phase transformer training pipeline:

    Phase-1 : train_embedder()        →  checkpoints/phase1/ckpt_best.pt
    Phase-2 : pretrain_transformer()  →  checkpoints/phase2/ckpt_best.pt
    Phase-3 : finetune_transformer()  →  checkpoints/phase3/ckpt_best.pt

Contract:
  • Three-way patient split: train / val / test. Test is held out for
    evaluation.ipynb and NEVER seen during training or early-stop selection.
  • DataProcessor fits the scaler on train and reuses it on val.
  • Tokenizer is built once from train and cached at checkpoints/tokenizer.pt.
  • Phase-2 uses oversample=True; Phase-1 and Phase-3 use natural distribution.
  • Phase-1 embedder is reused across runs when (embed_dim, time2vec_dim,
    ctx_dim) match the cached checkpoint.
"""
import os
import gc
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import load as joblib_load
from sklearn.model_selection import train_test_split

from intervene_ar.dataset import (
    DataProcessor, EMRTokenizer, EMRDataset, collate_emr, get_dataloader,
)
from intervene_ar.embedder import EMREmbedding, train_embedder
from intervene_ar.transformer import InterveneGPT, pretrain_transformer, finetune_transformer
from intervene_ar.config.model_config import (
    MODEL_CONFIG, TRAINING_SETTINGS,
    CHECKPOINT_PATH, PHASE1_CHECKPOINT, PHASE2_CHECKPOINT, PHASE3_CHECKPOINT,
)
from intervene_ar.config.dataset_config import (
    TRAIN_TEMPORAL_DATA_FILE, TRAIN_CTX_DATA_FILE, TAK_REPO_PATH,
)

TOKENIZER_PATH = os.path.join(CHECKPOINT_PATH, "tokenizer.pt")
TEST_SPLIT     = 0.15
VAL_SPLIT      = 0.15
RANDOM_SEED    = 42


def summarize_split(train_ds, val_ds, n_train, n_val, n_test, tokenizer):
    """
    Purpose: Print train/val/test sizes and tokenizer vocab stats.
    Method: Aggregate counts from the EMRDataset.tokens_df + tokenizer dicts.

    Args:
        train_ds (EMRDataset): training split.
        val_ds (EMRDataset): validation split.
        n_train (int), n_val (int), n_test (int): patient counts.
        tokenizer (EMRTokenizer): fitted tokenizer.
    """
    print("Data Split Summary")
    print(f"  - Train patients: {n_train}   ({len(train_ds.tokens_df):,} records)")
    print(f"  - Val   patients: {n_val}    ({len(val_ds.tokens_df):,} records)")
    print(f"  - Test  patients: {n_test}   (held out for evaluation.py)")

    train_counts = train_ds.tokens_df.groupby("PatientId").size()
    print(f"\nTrain per-patient record count: "
          f"min={train_counts.min()}, max={train_counts.max()}, "
          f"mean={train_counts.mean():.1f}, median={train_counts.median()}")

    print(f"\nVocabulary sizes:")
    print(f"  - Raw concepts:   {len(tokenizer.rawconcept2id):,}")
    print(f"  - Concepts:       {len(tokenizer.concept2id):,}")
    print(f"  - Concept+Value:  {len(tokenizer.value2id):,}")
    print(f"  - Full Tokens:    {len(tokenizer.token2id):,}")


def prepare_data(sample=None, batch_size=None):
    """
    Purpose: Load source CSVs, split patients three ways, run DataProcessor on
             train/val (test stays raw for evaluation.py), build tokenizer.
    Method: Mirrors api.load_data — train_test_split twice with the same seed
            for reproducibility, fit scaler on train and apply to val.

    Args:
        sample (int or None): optional patient subset for smoke tests.
        batch_size (int): passed through to the dataloaders.

    Returns:
        train_dl, oversampled_train_dl, val_dl : DataLoaders.
        tokenizer (EMRTokenizer).
        test_raw (tuple): (test_temporal_df, test_ctx_df) raw, unprocessed.
    """
    batch_size = batch_size or TRAINING_SETTINGS["batch_size"]

    print("[Data]: Reading source CSVs...")
    temporal_raw = pd.read_csv(TRAIN_TEMPORAL_DATA_FILE, low_memory=False)
    ctx_raw      = pd.read_csv(TRAIN_CTX_DATA_FILE)

    if sample is not None:
        pids = temporal_raw["PatientId"].unique()
        rng  = np.random.RandomState(RANDOM_SEED)
        chosen = rng.choice(pids, size=min(sample, len(pids)), replace=False)
        temporal_raw = temporal_raw[temporal_raw["PatientId"].isin(chosen)]
        ctx_raw      = ctx_raw[ctx_raw["PatientId"].isin(chosen)]

    # Three-way patient split (mirrors api.py / evaluation contract).
    all_pids = np.sort(temporal_raw["PatientId"].unique())  # sorted for reproducibility across CSV row order
    trainval_ids, test_ids = train_test_split(
        all_pids, test_size=TEST_SPLIT, random_state=RANDOM_SEED
    )
    val_relative = VAL_SPLIT / (1.0 - TEST_SPLIT)
    train_ids, val_ids = train_test_split(
        trainval_ids, test_size=val_relative, random_state=RANDOM_SEED
    )

    train_temporal_raw = temporal_raw[temporal_raw["PatientId"].isin(train_ids)].copy()
    train_ctx_raw      = ctx_raw[ctx_raw["PatientId"].isin(train_ids)].copy()
    val_temporal_raw   = temporal_raw[temporal_raw["PatientId"].isin(val_ids)].copy()
    val_ctx_raw        = ctx_raw[ctx_raw["PatientId"].isin(val_ids)].copy()
    test_temporal_raw  = temporal_raw[temporal_raw["PatientId"].isin(test_ids)].copy()
    test_ctx_raw       = ctx_raw[ctx_raw["PatientId"].isin(test_ids)].copy()

    # Fit scaler on train, apply to val (test is processed at eval time).
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    print("[Data]: Processing train split (fitting scaler)...")
    train_processor = DataProcessor(
        train_temporal_raw, train_ctx_raw,
        scaler=None, tak_repo_path=TAK_REPO_PATH,
        checkpoint_path=CHECKPOINT_PATH,
    )
    train_temporal_df, train_ctx_df = train_processor.run()

    scaler = joblib_load(os.path.join(CHECKPOINT_PATH, "scaler.pkl"))
    print("[Data]: Processing val split (applying fitted scaler)...")
    val_processor = DataProcessor(
        val_temporal_raw, val_ctx_raw,
        scaler=scaler, tak_repo_path=TAK_REPO_PATH,
        checkpoint_path=CHECKPOINT_PATH,
    )
    val_temporal_df, val_ctx_df = val_processor.run()

    # Tokenizer cached by checkpoints/tokenizer.pt; rebuilt only on cold start.
    tokenizer_path = Path(TOKENIZER_PATH)
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    if tokenizer_path.exists():
        print("[Data]: Loading tokenizer from cache...")
        tokenizer = EMRTokenizer.load(str(tokenizer_path))
    else:
        print("[Data]: Building tokenizer...")
        tokenizer = EMRTokenizer.from_processed_df(train_temporal_df)
        tokenizer.save(str(tokenizer_path))

    train_ds = EMRDataset(train_temporal_df, train_ctx_df, tokenizer=tokenizer)
    val_ds   = EMRDataset(val_temporal_df,   val_ctx_df,   tokenizer=tokenizer)
    summarize_split(train_ds, val_ds, len(train_ids), len(val_ids), len(test_ids), tokenizer)

    train_dl              = get_dataloader(train_ds, batch_size=batch_size,
                                           collate_fn=collate_emr, oversample=False, bucket_batching=True)
    oversampled_train_dl  = get_dataloader(train_ds, batch_size=batch_size,
                                           collate_fn=collate_emr, oversample=True,  bucket_batching=True)
    val_dl                = get_dataloader(val_ds,   batch_size=batch_size,
                                           collate_fn=collate_emr, oversample=False, bucket_batching=True)

    return train_dl, oversampled_train_dl, val_dl, tokenizer, (test_temporal_raw, test_ctx_raw)


def _load_or_build_embedder(tokenizer):
    """
    Purpose: Reuse cached Phase-1 embedder when arch dims match; else build fresh.
    Method:  Compares (embed_dim, time2vec_dim, ctx_dim) against the checkpoint's
             saved config — same gate api.py uses.
    """
    key = (MODEL_CONFIG["embed_dim"], MODEL_CONFIG["time2vec_dim"], MODEL_CONFIG["ctx_dim"])
    ckpt = Path(PHASE1_CHECKPOINT)
    if ckpt.exists():
        try:
            cfg = torch.load(str(ckpt), map_location="cpu", weights_only=True)["config"]
            if (cfg["embed_dim"], cfg["time2vec_dim"], cfg["ctx_dim"]) == key:
                print("[Phase 1]: Config unchanged — loading cached embedder.")
                embedder, *_ = EMREmbedding.load(str(ckpt), tokenizer=tokenizer)
                return embedder, True
        except Exception as e:
            print(f"[Phase 1]: Cached embedder unusable ({e}); rebuilding.")
    embedder = EMREmbedding(
        tokenizer=tokenizer,
        ctx_dim=MODEL_CONFIG["ctx_dim"],
        time2vec_dim=MODEL_CONFIG["time2vec_dim"],
        embed_dim=MODEL_CONFIG["embed_dim"],
        dropout=MODEL_CONFIG["dropout"],
    )
    return embedder, False


def run_training(reset_phase23=True):
    """
    Purpose: Run the full three-phase training pipeline.
    Method:  Loads data, optionally clears Phase-2/3 checkpoints for a fresh run,
             trains each phase via the same entry points api.py uses.

    Args:
        reset_phase23 (bool): when True (default), clears checkpoints/phase2 and
            phase3 directories so each call retrains them from scratch. Phase-1
            is always preserved when arch dims match (cheap to keep, expensive
            to retrain).

    Returns:
        (embedder, model_p2, model_p3, test_raw): the trained components plus
        the raw test split, ready to feed into evaluation.evaluate_on_test_set.
    """
    if reset_phase23:
        for _phase in ("phase2", "phase3"):
            p = Path(CHECKPOINT_PATH) / _phase
            if p.exists():
                shutil.rmtree(p)
            p.mkdir(parents=True, exist_ok=True)
    (Path(CHECKPOINT_PATH) / "phase1").mkdir(parents=True, exist_ok=True)

    train_dl, oversampled_train_dl, val_dl, tokenizer, test_raw = prepare_data(
        sample=TRAINING_SETTINGS.get("sample"),
        batch_size=TRAINING_SETTINGS["batch_size"],
    )

    # Auto-detect ctx_dim from the first batch (handles QA features being on/off).
    for batch in train_dl:
        MODEL_CONFIG["ctx_dim"] = int(batch["context_vec"].shape[-1])
        break
    print(f"Model config: {MODEL_CONFIG}")

    # Phase 1 — embedder
    embedder, reused = _load_or_build_embedder(tokenizer)
    if not reused:
        embedder, _, _ = train_embedder(
            embedder          = embedder,
            train_loader      = train_dl,
            val_loader        = val_dl,
            resume            = False,
            checkpoint_path   = PHASE1_CHECKPOINT,
            training_settings = TRAINING_SETTINGS,
        )

    # Phase 2 — backbone (oversampled batches for rare-outcome balance)
    model_p2 = InterveneGPT(cfg=MODEL_CONFIG, embedder=embedder)
    model_p2, _, _ = pretrain_transformer(
        model             = model_p2,
        train_dl          = oversampled_train_dl,
        val_dl            = val_dl,
        resume            = False,
        checkpoint_path   = PHASE2_CHECKPOINT,
        training_settings = TRAINING_SETTINGS,
    )

    # Phase 3 — outcome head fine-tune (natural distribution; pos_weight handles imbalance).
    # Load best Phase-2 checkpoint when present, else continue with in-memory model.
    _p2_best = Path(PHASE2_CHECKPOINT)
    _p2_last = _p2_best.parent / "ckpt_last.pt"
    _p2_ckpt = _p2_best if _p2_best.exists() else (_p2_last if _p2_last.exists() else None)
    if _p2_ckpt is not None:
        model_p3, *_ = InterveneGPT.load(str(_p2_ckpt), embedder=embedder)
    else:
        model_p3 = model_p2

    model_p3, _, _ = finetune_transformer(
        model             = model_p3,
        train_dl          = train_dl,
        val_dl            = val_dl,
        resume            = False,
        checkpoint_path   = PHASE3_CHECKPOINT,
        training_settings = TRAINING_SETTINGS,
    )

    return embedder, model_p2, model_p3, test_raw


if __name__ == "__main__":
    run_training()
    # Release training references; evaluation lives in evaluation.ipynb /
    # evaluation.py and is intentionally run in a separate process so peak
    # train-time RAM doesn't bleed into autoregressive generation.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[Train] Done. Open evaluation.ipynb to score the held-out test split.")
