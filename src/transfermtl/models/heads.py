"""Task-specific MLP heads on top of the shared encoder representation."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TaskHead(nn.Module):
    """2-layer MLP that maps a pooled graph embedding to a single logit."""

    def __init__(self, in_dim: int = 256, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, h: Tensor) -> Tensor:
        x = F.relu(self.fc1(h))
        x = self.dropout(x)
        out: Tensor = self.fc2(x)
        return out
