"""Tests for the encoder + heads (A3, plan §2.3)."""

from __future__ import annotations

import torch
from torch_geometric.data import Batch, Data

from transfermtl.models import GCNEncoder, MultiHeadModel, TaskHead
from transfermtl.utils.registry import build
from transfermtl.utils.seeding import set_seed


def _toy_batch(n_graphs: int = 8, n_nodes: int = 5, atom_dim: int = 74) -> Batch:
    set_seed(0)
    data_list: list[Data] = []
    for _ in range(n_graphs):
        x = torch.randn(n_nodes, atom_dim)
        edges = []
        for i in range(n_nodes):
            edges.append([i, (i + 1) % n_nodes])
            edges.append([(i + 1) % n_nodes, i])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        data_list.append(Data(x=x, edge_index=edge_index))
    return Batch.from_data_list(data_list)


def test_gcn_output_shape() -> None:
    encoder = build("encoder", "gcn")
    batch = _toy_batch(n_graphs=8)
    h = encoder(batch)
    assert h.shape == (8, 256)


def test_gcn_param_count_in_range() -> None:
    encoder = build("encoder", "gcn")
    n = sum(p.numel() for p in encoder.parameters())
    assert 200_000 <= n <= 800_000, f"got {n}"


def test_head_output_shape() -> None:
    head = TaskHead(in_dim=256, hidden_dim=128, dropout=0.1)
    h = torch.randn(7, 256)
    out = head(h)
    assert out.shape == (7, 1)


def test_encoder_deterministic_with_seed() -> None:
    set_seed(123)
    enc_a = GCNEncoder()
    set_seed(123)
    enc_b = GCNEncoder()
    for (n_a, p_a), (n_b, p_b) in zip(
        enc_a.named_parameters(), enc_b.named_parameters(), strict=True
    ):
        assert n_a == n_b
        assert torch.allclose(p_a, p_b), f"diverge at {n_a}"


def test_multihead_returns_dict_per_task() -> None:
    encoder = build("encoder", "gcn")
    model = MultiHeadModel(encoder, ["t1", "t2", "t3"])
    batch = _toy_batch(n_graphs=4)
    out = model(batch)
    assert set(out.keys()) == {"t1", "t2", "t3"}
    for t in ("t1", "t2", "t3"):
        assert out[t].shape == (4,)


def test_registry_lists_gcn() -> None:
    from transfermtl.utils.registry import REGISTRY

    assert "gcn" in REGISTRY["encoder"]
