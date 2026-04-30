"""Sign-heterogeneity indicators (plan §2.9).

A pair (i, j) is *sign-heterogeneous* iff there exist two valid regions
r_a, r_b such that:
    Δ(r_a) > +ε AND CI(r_a) excludes 0
    Δ(r_b) < −ε AND CI(r_b) excludes 0

The pair-averaged variant uses Δ_ij(r); the task-specific variants use
Δ_{i ← j}(r) and Δ_{j ← i}(r) separately and report S_i and S_j.

`epsilon` defaults to 1.5 AUC points (plan §2.21). Δ inputs are expected in
the same scale as ε — callers reporting raw AUC differences should multiply
by 100 first, or pass `epsilon=0.015` to compare in raw units.
"""

from __future__ import annotations

from collections.abc import Mapping

from transfermtl.utils.types import BootstrapResult


def _ci_excludes_zero(b: BootstrapResult) -> bool:
    return (b.ci_lower > 0) or (b.ci_upper < 0)


def compute_S_pair(
    deltas: Mapping[int, BootstrapResult],
    epsilon: float = 1.5,
) -> bool:
    """True iff at least one positive AND one negative region with |Δ| > ε and CI excluding 0."""
    has_pos = any(b.estimate > epsilon and _ci_excludes_zero(b) for b in deltas.values())
    has_neg = any(b.estimate < -epsilon and _ci_excludes_zero(b) for b in deltas.values())
    return bool(has_pos and has_neg)


def compute_S_task_specific(
    deltas_i: Mapping[int, BootstrapResult],
    deltas_j: Mapping[int, BootstrapResult],
    epsilon: float = 1.5,
) -> tuple[bool, bool]:
    """Per-task sign heterogeneity flags `(S_i, S_j)` using Δ_{i←j} and Δ_{j←i}."""
    return compute_S_pair(deltas_i, epsilon), compute_S_pair(deltas_j, epsilon)
