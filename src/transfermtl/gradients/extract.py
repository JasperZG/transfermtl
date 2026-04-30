"""Regional gradient extraction (plan §2.7).

The mean encoder gradient over the compounds in a region:

    g_i(r) = (1/|D_i(r)|) Σ_{x ∈ D_i(r)} ∇_θ ℓ_i(x; θ, φ_i)

where θ is the encoder parameters and φ_i is the task-i head. The full
gradient vector is the concatenation of `parameter.grad.flatten()` over
`encoder.parameters()` in iteration order — this order is *locked* (asserted
by `test_encoder_param_order_locked`) so two reruns produce identical
flattenings, which is required for cosine affinity to be reproducible.

Subsamples to `max_subsample` (default 500) when the region is larger.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from transfermtl.models.multihead import MultiHeadModel
from transfermtl.training.checkpoint import load_checkpoint
from transfermtl.utils.registry import build


def encoder_param_order_hash(model: MultiHeadModel) -> str:
    """SHA-256 of the iteration order of encoder parameter names + shapes.

    Tests assert this hash is stable across runs so the flattening order in
    `compute_gradient_vector` is deterministic.
    """
    h = hashlib.sha256()
    for name, p in model.encoder.named_parameters():
        h.update(name.encode())
        h.update(b":")
        h.update(str(tuple(p.shape)).encode())
        h.update(b";")
    return h.hexdigest()


def _flatten_encoder_grads(model: MultiHeadModel) -> np.ndarray:
    parts: list[np.ndarray] = []
    for p in model.encoder.parameters():
        if p.grad is None:
            parts.append(np.zeros(p.numel(), dtype=np.float64))
        else:
            parts.append(p.grad.detach().cpu().numpy().astype(np.float64).flatten())
    if not parts:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(parts)


def _zero_encoder_grads(model: MultiHeadModel) -> None:
    for p in model.encoder.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()


def _bce_loss(model: MultiHeadModel, batch: Data, task: str, task_index: int) -> torch.Tensor:
    out = model(batch)
    logits: torch.Tensor = out[task]
    targets: torch.Tensor = batch.y[:, task_index]
    valid = ~torch.isnan(targets)
    if int(valid.sum().item()) == 0:
        return torch.tensor(float("nan"), device=logits.device)
    return F.binary_cross_entropy_with_logits(logits[valid], targets[valid])


def compute_gradient_vector(
    model: MultiHeadModel,
    data: list[Data],
    task: str,
    task_index: int = 0,
    max_subsample: int = 500,
    rng_seed: int = 0,
    batch_size: int = 32,
    loss_fn: Callable[[MultiHeadModel, Data, str, int], torch.Tensor] | None = None,
) -> tuple[np.ndarray, int, float]:
    """Mean encoder gradient over `data` for `task`.

    Returns (g_vec, n_used, grad_norm). `n_used` reflects subsampling and
    NaN-label dropout. `grad_norm` is L2 of `g_vec`.
    """
    if not data:
        n_params = sum(p.numel() for p in model.encoder.parameters())
        return np.zeros(n_params, dtype=np.float64), 0, 0.0

    rng = np.random.default_rng(rng_seed)
    if len(data) > max_subsample:
        idx = rng.choice(len(data), size=max_subsample, replace=False)
        data = [data[i] for i in idx.tolist()]

    loss_fn = loss_fn or _bce_loss

    model.train()  # gradients flow; dropout active
    accum: np.ndarray | None = None
    n_used = 0

    loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    for batch in loader:
        _zero_encoder_grads(model)
        for p in model.encoder.parameters():
            p.requires_grad_(True)

        loss = loss_fn(model, batch, task, task_index)
        if not torch.isfinite(loss):
            continue
        loss.backward()  # type: ignore[no-untyped-call]
        flat = _flatten_encoder_grads(model)
        # Number of compounds with valid label in this batch.
        targets = batch.y[:, task_index]
        n_valid = int((~torch.isnan(targets)).sum().item())
        if n_valid == 0:
            continue
        if accum is None:
            accum = flat * n_valid
        else:
            accum += flat * n_valid
        n_used += n_valid

    if accum is None or n_used == 0:
        n_params = sum(p.numel() for p in model.encoder.parameters())
        return np.zeros(n_params, dtype=np.float64), 0, 0.0

    g_vec = accum / float(n_used)
    grad_norm = float(np.linalg.norm(g_vec))
    return g_vec, n_used, grad_norm


def _build_model_from_state(
    model_state: dict[str, torch.Tensor],
    encoder_kwargs: dict[str, object] | None = None,
) -> MultiHeadModel:
    """Rebuild a MultiHeadModel from a saved state dict.

    Task names are inferred from `heads.<task>.fc1.weight` keys; the encoder
    is the registered `gcn` factory unless `encoder_kwargs` overrides.
    """
    task_names = sorted({k.split(".")[1] for k in model_state if k.startswith("heads.")})
    if not task_names:
        raise ValueError("model_state contains no `heads.<task>.*` entries")
    encoder = build("encoder", "gcn", **(encoder_kwargs or {}))
    model = MultiHeadModel(encoder, task_names)
    model.load_state_dict(model_state)
    return model


def compute_regional_gradient(
    checkpoint_path: str | Path,
    task: str,
    region_data: list[Data],
    task_index: int = 0,
    max_subsample: int = 500,
    rng_seed: int = 0,
    encoder_kwargs: dict[str, object] | None = None,
) -> tuple[np.ndarray, int, float]:
    """Load a checkpoint and compute the mean regional encoder gradient for `task`."""
    bundle = load_checkpoint(checkpoint_path)
    model = _build_model_from_state(bundle.model_state, encoder_kwargs=encoder_kwargs)
    return compute_gradient_vector(
        model,
        region_data,
        task=task,
        task_index=task_index,
        max_subsample=max_subsample,
        rng_seed=rng_seed,
    )
