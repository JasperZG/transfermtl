"""Single-task training (binary classification).

`task` names a column in the SplitSchema parquet. Encoder + a single TaskHead
are trained with BCE-with-logits. The label leak/mask handles NaN entries: if
no compounds in a batch have a valid label, the batch contributes no gradient.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from transfermtl.models.multihead import MultiHeadModel
from transfermtl.training.checkpoint import save_checkpoint
from transfermtl.training.loops import TrainConfig, TrainOutcome, train_loop
from transfermtl.training.predict import save_predictions
from transfermtl.utils.registry import build
from transfermtl.utils.seeding import set_seed


@dataclass(frozen=True)
class STLArtifacts:
    model: MultiHeadModel
    outcome: TrainOutcome
    checkpoint_path: Path | None
    predictions_path: Path | None


def _make_loss_fn(task: str) -> Callable[[nn.Module, Any], torch.Tensor]:
    def compute_loss(model: nn.Module, batch: Any) -> torch.Tensor:
        out = model(batch)
        logits: torch.Tensor = out[task]
        targets: torch.Tensor = batch.y[:, 0]
        valid = ~torch.isnan(targets)
        if int(valid.sum().item()) == 0:
            return torch.tensor(float("nan"), device=logits.device)
        return F.binary_cross_entropy_with_logits(logits[valid], targets[valid])

    return compute_loss


def train_stl(
    train_data: list[Data],
    val_data: list[Data],
    task: str,
    seed: int,
    cfg: TrainConfig | None = None,
    encoder_name: str = "gcn",
    encoder_kwargs: dict[str, object] | None = None,
    device: torch.device | str = "cpu",
    checkpoint_path: str | Path | None = None,
    test_data: Iterable[Data] | None = None,
    predictions_path: str | Path | None = None,
) -> STLArtifacts:
    """Train STL on `task`. Optionally save checkpoint + test predictions."""
    cfg = cfg or TrainConfig()
    set_seed(seed)

    encoder = build("encoder", encoder_name, **(encoder_kwargs or {}))
    model = MultiHeadModel(encoder, [task])

    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=cfg.batch_size, shuffle=False)

    outcome = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        compute_loss=_make_loss_fn(task),
        cfg=cfg,
        device=device,
    )

    ckpt_path: Path | None = None
    if checkpoint_path is not None:
        ckpt_path = save_checkpoint(
            checkpoint_path,
            model=model,
            optim=None,
            epoch=outcome.best_epoch,
            val_loss=outcome.best_val_loss,
            cfg=cfg,
            seed=seed,
        )

    pred_path: Path | None = None
    if predictions_path is not None and test_data is not None:
        pred_path = save_predictions(
            predictions_path,
            model=model,
            dataset=test_data,
            seed=seed,
            batch_size=cfg.batch_size,
            device=device,
        )

    return STLArtifacts(
        model=model,
        outcome=outcome,
        checkpoint_path=ckpt_path,
        predictions_path=pred_path,
    )
