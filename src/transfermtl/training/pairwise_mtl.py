"""Pairwise MTL (plan §2.5).

Each batch contains compounds with a valid label for at least one of `(task_i,
task_j)`. Compounds with both labels contribute to both task losses. The total
loss is the sum of per-task means over their valid subsets.
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
class MTLArtifacts:
    model: MultiHeadModel
    outcome: TrainOutcome
    checkpoint_path: Path | None
    predictions_path: Path | None


def _make_loss_fn(
    tasks: list[str],
) -> Callable[[nn.Module, Any], torch.Tensor]:
    def compute_loss(model: nn.Module, batch: Any) -> torch.Tensor:
        out = model(batch)
        y: torch.Tensor = batch.y
        device = y.device

        per_task: list[torch.Tensor] = []
        for ti, task in enumerate(tasks):
            logits = out[task]
            targets = y[:, ti]
            valid = ~torch.isnan(targets)
            if int(valid.sum().item()) == 0:
                continue
            per_task.append(F.binary_cross_entropy_with_logits(logits[valid], targets[valid]))

        if not per_task:
            return torch.tensor(float("nan"), device=device)
        return torch.stack(per_task).sum()

    return compute_loss


def train_pairwise_mtl(
    train_data: list[Data],
    val_data: list[Data],
    task_i: str,
    task_j: str,
    seed: int,
    cfg: TrainConfig | None = None,
    encoder_name: str = "gcn",
    encoder_kwargs: dict[str, object] | None = None,
    device: torch.device | str = "cpu",
    checkpoint_path: str | Path | None = None,
    test_data: Iterable[Data] | None = None,
    predictions_path: str | Path | None = None,
) -> MTLArtifacts:
    cfg = cfg or TrainConfig()
    set_seed(seed)

    tasks = [task_i, task_j]
    encoder = build("encoder", encoder_name, **(encoder_kwargs or {}))
    model = MultiHeadModel(encoder, tasks)

    train_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=cfg.batch_size, shuffle=False)

    outcome = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        compute_loss=_make_loss_fn(tasks),
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

    return MTLArtifacts(
        model=model,
        outcome=outcome,
        checkpoint_path=ckpt_path,
        predictions_path=pred_path,
    )
