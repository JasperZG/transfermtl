"""Compose an encoder with one task head per task name.

Used by STL (single head), pairwise MTL (two heads), and all-task MTL (one head
per dataset task).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch.nn as nn
from torch import Tensor

from transfermtl.models.heads import TaskHead


class MultiHeadModel(nn.Module):
    """Encoder shared across `task_names`; one `TaskHead` per task name.

    Forward returns a dict {task_name: logits[B]}.
    """

    def __init__(
        self,
        encoder: nn.Module,
        task_names: Iterable[str],
        head_hidden_dim: int = 128,
        head_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.task_names: list[str] = list(task_names)
        in_dim: int = int(getattr(encoder, "hidden_dim", 256))
        self.heads = nn.ModuleDict(
            {
                t: TaskHead(in_dim=in_dim, hidden_dim=head_hidden_dim, dropout=head_dropout)
                for t in self.task_names
            }
        )

    def forward(self, batch: Any) -> dict[str, Tensor]:
        h: Tensor = self.encoder(batch)
        return {t: self.heads[t](h).squeeze(-1) for t in self.task_names}
