"""Generic train loop with AdamW, cosine LR schedule, early stopping, grad clip.

Plan §2.4 fixes the optimizer/schedule/early-stopping settings; values live in
`configs/_shared/train_default.yaml` and are mirrored by `TrainConfig` defaults.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader

LossFn = Callable[[nn.Module, Any], torch.Tensor]


@dataclass(frozen=True)
class TrainConfig:
    """Mirrors `configs/_shared/train_default.yaml` (frozen by A1)."""

    optimizer: str = "adamw"
    lr: float = 1.0e-3
    weight_decay: float = 1.0e-2
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 25
    lr_schedule: str = "cosine"
    lr_min: float = 1.0e-5
    grad_clip: float = 1.0


@dataclass
class TrainOutcome:
    best_val_loss: float
    best_epoch: int
    final_model_state: dict[str, torch.Tensor]
    history: list[dict[str, float]] = field(default_factory=list)


def make_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    if cfg.optimizer != "adamw":
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )


def make_scheduler(
    optim: torch.optim.Optimizer, cfg: TrainConfig
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if cfg.lr_schedule == "cosine":
        return CosineAnnealingLR(optim, T_max=cfg.max_epochs, eta_min=cfg.lr_min)
    if cfg.lr_schedule in ("none", "constant", None):
        return None
    raise ValueError(f"Unsupported lr_schedule: {cfg.lr_schedule}")


def _clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    compute_loss: LossFn,
    cfg: TrainConfig,
    device: torch.device | str = "cpu",
    log_fn: Callable[[int, dict[str, float]], None] | None = None,
) -> TrainOutcome:
    """Train `model` against `compute_loss` with early stopping on val loss.

    `compute_loss(model, batch)` must return a scalar tensor (or NaN tensor when
    a batch contributes no valid loss — that batch is silently skipped).
    """
    device = torch.device(device)
    model.to(device)
    optim = make_optimizer(model, cfg)
    sched = make_scheduler(optim, cfg)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = _clone_state(model)
    history: list[dict[str, float]] = []
    epochs_no_improve = 0

    for epoch in range(cfg.max_epochs):
        # ---- train ----
        model.train()
        train_total, train_n = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            optim.zero_grad()
            loss = compute_loss(model, batch)
            if not torch.isfinite(loss):
                continue
            loss.backward()  # type: ignore[no-untyped-call]
            if cfg.grad_clip and cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optim.step()
            train_total += float(loss.item())
            train_n += 1
        train_loss = train_total / max(train_n, 1)

        # ---- validate ----
        model.eval()
        val_total, val_n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                loss = compute_loss(model, batch)
                if not torch.isfinite(loss):
                    continue
                val_total += float(loss.item())
                val_n += 1
        val_loss = val_total / max(val_n, 1) if val_n > 0 else float("inf")

        if sched is not None:
            sched.step()

        record = {"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss}
        history.append(record)
        if log_fn is not None:
            log_fn(epoch, record)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = _clone_state(model)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= cfg.patience:
            break

    model.load_state_dict(best_state)
    return TrainOutcome(
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
        final_model_state=best_state,
        history=history,
    )
