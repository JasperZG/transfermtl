"""Tests for training loops, checkpoints, and prediction IO (A3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader

from tests.synthetic_fixture.featurize import build_synthetic_loader
from transfermtl.models import MultiHeadModel
from transfermtl.training import (
    TrainConfig,
    load_checkpoint,
    save_checkpoint,
    train_all_task_mtl,
    train_pairwise_mtl,
    train_stl,
)
from transfermtl.utils.registry import build
from transfermtl.utils.schemas import PredictionSchema


def _train_auc(model: MultiHeadModel, dataset: list, task: str) -> float:
    model.eval()
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    y_true: list[float] = []
    y_pred: list[float] = []
    ti = model.task_names.index(task)
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            probs = torch.sigmoid(out[task]).cpu().tolist()
            ys = batch.y[:, ti].cpu().tolist()
            y_pred.extend(probs)
            y_true.extend(ys)
    return float(roc_auc_score(y_true, y_pred))


# ---------------------------------------------------------------------------
# Convergence on the synthetic fixture
# ---------------------------------------------------------------------------


def test_stl_converges_on_fixture(synthetic_dataset: pd.DataFrame) -> None:
    train = build_synthetic_loader(synthetic_dataset, "train", ["task_1"])
    val = build_synthetic_loader(synthetic_dataset, "val", ["task_1"])
    cfg = TrainConfig(max_epochs=20, patience=10)
    art = train_stl(train, val, task="task_1", seed=0, cfg=cfg)

    val_auc = _train_auc(art.model, val, "task_1")
    assert val_auc > 0.7, f"val AUC {val_auc:.3f}"


def test_mtl_converges_on_fixture(synthetic_dataset: pd.DataFrame) -> None:
    train = build_synthetic_loader(synthetic_dataset, "train", ["task_1", "task_2"])
    val = build_synthetic_loader(synthetic_dataset, "val", ["task_1", "task_2"])
    cfg = TrainConfig(max_epochs=20, patience=10)
    art = train_pairwise_mtl(train, val, task_i="task_1", task_j="task_2", seed=0, cfg=cfg)
    auc1 = _train_auc(art.model, val, "task_1")
    auc2 = _train_auc(art.model, val, "task_2")
    assert auc1 > 0.7, f"task_1 val AUC {auc1:.3f}"
    assert auc2 > 0.7, f"task_2 val AUC {auc2:.3f}"


def test_all_task_mtl_runs(synthetic_dataset: pd.DataFrame) -> None:
    train = build_synthetic_loader(synthetic_dataset, "train", ["task_1", "task_2"])
    val = build_synthetic_loader(synthetic_dataset, "val", ["task_1", "task_2"])
    cfg = TrainConfig(max_epochs=1, patience=1)
    art = train_all_task_mtl(train, val, tasks=["task_1", "task_2"], seed=0, cfg=cfg)
    assert len(art.outcome.history) == 1


# ---------------------------------------------------------------------------
# Checkpoint roundtrip + schema
# ---------------------------------------------------------------------------


def test_checkpoint_roundtrip(tmp_path: Path, synthetic_dataset: pd.DataFrame) -> None:
    train = build_synthetic_loader(synthetic_dataset, "train", ["task_1"])
    val = build_synthetic_loader(synthetic_dataset, "val", ["task_1"])
    cfg = TrainConfig(max_epochs=2, patience=1)
    ckpt_path = tmp_path / "ckpt.pt"
    art = train_stl(train, val, task="task_1", seed=7, cfg=cfg, checkpoint_path=ckpt_path)
    assert ckpt_path.exists()
    bundle = load_checkpoint(ckpt_path)
    assert bundle.seed == 7
    assert bundle.cfg_dict["max_epochs"] == 2
    assert isinstance(bundle.git_sha, str)
    # Same parameters round-trip exactly.
    for k, v in art.model.state_dict().items():
        assert torch.allclose(v, bundle.model_state[k])


def test_predictions_match_schema(tmp_path: Path, synthetic_dataset: pd.DataFrame) -> None:
    train = build_synthetic_loader(synthetic_dataset, "train", ["task_1"])
    val = build_synthetic_loader(synthetic_dataset, "val", ["task_1"])
    test = build_synthetic_loader(synthetic_dataset, "test", ["task_1"])
    cfg = TrainConfig(max_epochs=2, patience=1)
    pred_path = tmp_path / "pred.parquet"
    train_stl(
        train,
        val,
        task="task_1",
        seed=0,
        cfg=cfg,
        test_data=test,
        predictions_path=pred_path,
    )
    df = pd.read_parquet(pred_path)
    PredictionSchema.validate(df)
    assert len(df) == len(test)
    assert set(df["task"].unique()) == {"task_1"}


# ---------------------------------------------------------------------------
# Determinism (same seed -> same val_loss)
# ---------------------------------------------------------------------------


def test_seeded_determinism(synthetic_dataset: pd.DataFrame) -> None:
    train = build_synthetic_loader(synthetic_dataset, "train", ["task_1"])
    val = build_synthetic_loader(synthetic_dataset, "val", ["task_1"])
    cfg = TrainConfig(max_epochs=3, patience=10)

    a = train_stl(train, val, task="task_1", seed=42, cfg=cfg)
    b = train_stl(train, val, task="task_1", seed=42, cfg=cfg)
    assert abs(a.outcome.best_val_loss - b.outcome.best_val_loss) < 1e-5


# ---------------------------------------------------------------------------
# NaN-label masking
# ---------------------------------------------------------------------------


def test_mtl_handles_missing_labels(synthetic_dataset: pd.DataFrame) -> None:
    df = synthetic_dataset.copy()
    rng = np.random.default_rng(0)
    mask = rng.random(len(df)) < 0.3
    df.loc[mask, "task_2"] = np.nan

    train = build_synthetic_loader(df, "train", ["task_1", "task_2"])
    val = build_synthetic_loader(df, "val", ["task_1", "task_2"])
    cfg = TrainConfig(max_epochs=2, patience=1)
    art = train_pairwise_mtl(train, val, task_i="task_1", task_j="task_2", seed=0, cfg=cfg)
    assert np.isfinite(art.outcome.best_val_loss)


# ---------------------------------------------------------------------------
# Direct save_checkpoint (no training) — covers the standalone API
# ---------------------------------------------------------------------------


def test_save_checkpoint_standalone(tmp_path: Path) -> None:
    encoder = build("encoder", "gcn")
    model = MultiHeadModel(encoder, ["t1"])
    cfg = TrainConfig(max_epochs=1)
    p = tmp_path / "x.pt"
    save_checkpoint(p, model=model, optim=None, epoch=0, val_loss=0.5, cfg=cfg, seed=1)
    bundle = load_checkpoint(p)
    assert bundle.epoch == 0
    assert bundle.val_loss == pytest.approx(0.5)
    assert bundle.seed == 1
