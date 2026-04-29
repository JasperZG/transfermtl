"""Thin W&B wrapper with no-op fallback when offline or wandb is missing.

Project tag is `tsh-mtl` (plan §2.19). Captures git SHA via utils.git.
"""

from __future__ import annotations

import os
from typing import Any

from transfermtl.utils.git import current_sha, dirty_tree_warning

PROJECT_TAG = "tsh-mtl"

_ACTIVE_RUN: Any | None = None


def init(
    config: dict[str, Any],
    run_name: str | None = None,
    tags: list[str] | None = None,
    mode: str | None = None,
) -> Any | None:
    """Initialize a W&B run; returns the run handle or None if unavailable.

    Snapshots the supplied config, attaches git SHA, and warns on dirty tree.
    Falls back to a no-op (returns None) if wandb is not installed or
    `WANDB_MODE=disabled`.
    """
    global _ACTIVE_RUN

    enriched = dict(config)
    enriched["_git_sha"] = current_sha()
    enriched["_git_dirty"] = dirty_tree_warning()

    effective_mode = mode or os.environ.get("WANDB_MODE")
    if effective_mode == "disabled":
        _ACTIVE_RUN = None
        return None

    try:
        import wandb
    except ImportError:
        _ACTIVE_RUN = None
        return None

    _ACTIVE_RUN = wandb.init(
        project=PROJECT_TAG,
        name=run_name,
        tags=tags or [],
        config=enriched,
        mode=effective_mode,
    )
    return _ACTIVE_RUN


def log_metrics(step: int | None = None, **kw: Any) -> None:
    """Log scalar metrics to the active W&B run; no-op if not initialized."""
    if _ACTIVE_RUN is None:
        return
    try:
        import wandb

        if step is not None:
            wandb.log(kw, step=step)
        else:
            wandb.log(kw)
    except ImportError:
        return


def finish() -> None:
    global _ACTIVE_RUN
    if _ACTIVE_RUN is None:
        return
    try:
        import wandb

        wandb.finish()
    except ImportError:
        pass
    _ACTIVE_RUN = None
