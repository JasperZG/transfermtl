"""Checkpoint serialization with run metadata (cfg, seed, git SHA, timestamp)."""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from transfermtl.utils.git import current_sha
from transfermtl.utils.io import ensure_parent


@dataclass(frozen=True)
class CheckpointBundle:
    model_state: dict[str, torch.Tensor]
    optim_state: dict[str, Any] | None
    epoch: int
    val_loss: float
    cfg_dict: dict[str, Any]
    seed: int
    git_sha: str
    timestamp: str


def _cfg_to_dict(cfg: Any) -> dict[str, Any]:
    if isinstance(cfg, dict):
        return dict(cfg)
    if is_dataclass(cfg) and not isinstance(cfg, type):
        return asdict(cfg)
    raise TypeError(f"cfg must be dict or dataclass, got {type(cfg).__name__}")


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optim: torch.optim.Optimizer | None,
    epoch: int,
    val_loss: float,
    cfg: Any,
    seed: int,
) -> Path:
    p = ensure_parent(path)
    bundle = {
        "model_state": model.state_dict(),
        "optim_state": optim.state_dict() if optim is not None else None,
        "epoch": int(epoch),
        "val_loss": float(val_loss),
        "cfg_dict": _cfg_to_dict(cfg),
        "seed": int(seed),
        "git_sha": current_sha(),
        "timestamp": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    torch.save(bundle, p)
    return p


def load_checkpoint(path: str | Path) -> CheckpointBundle:
    obj: Any = torch.load(path, weights_only=False, map_location="cpu")
    return CheckpointBundle(
        model_state=obj["model_state"],
        optim_state=obj.get("optim_state"),
        epoch=int(obj["epoch"]),
        val_loss=float(obj["val_loss"]),
        cfg_dict=dict(obj["cfg_dict"]),
        seed=int(obj["seed"]),
        git_sha=str(obj["git_sha"]),
        timestamp=str(obj["timestamp"]),
    )
