"""3-layer GCN encoder per plan §2.3.

Architecture: input projection (atom_feat -> hidden) + N GCNConv layers + output
projection. The default config (atom_feature_dim=74, hidden_dim=256, n_layers=3)
yields ~282K parameters, which fits the [200K, 800K] band the brief asks for.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv, global_mean_pool


class GCNEncoder(nn.Module):
    """Encoder that maps a PyG batch to a per-graph representation [B, hidden_dim]."""

    def __init__(
        self,
        hidden_dim: int = 256,
        n_layers: int = 3,
        dropout: float = 0.1,
        atom_feature_dim: int = 74,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.atom_feature_dim = atom_feature_dim

        self.input_proj = nn.Linear(atom_feature_dim, hidden_dim)
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def encode(self, data: Any) -> Tensor:
        x: Tensor = data.x.float()
        edge_index: Tensor = data.edge_index
        batch_idx = getattr(data, "batch", None)

        x = F.relu(self.input_proj(x))
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.n_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        if batch_idx is None:
            batch_idx = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h: Tensor = global_mean_pool(x, batch_idx)
        h = self.output_proj(h)
        return h

    def forward(self, data: Any) -> Tensor:
        return self.encode(data)
