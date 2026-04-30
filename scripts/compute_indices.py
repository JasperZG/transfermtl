"""Compute pair-level S, H, C indices from per-region benefits and gradients.

Reads:
  outputs/benefits/{dataset}/{task_i}_{task_j}/region_benefits.parquet  (A6)
  outputs/gradients/{dataset}/{task_i}_{task_j}/seed*/region_affinity.parquet (A6)

Writes:
  outputs/analysis/{dataset}/pair_indices.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from transfermtl.indices.cancellation import compute_C
from transfermtl.indices.heterogeneity_index import compute_H
from transfermtl.indices.io import write_pair_indices
from transfermtl.indices.sign_heterogeneity import compute_S_pair, compute_S_task_specific
from transfermtl.utils.schemas import RegionBenefitSchema
from transfermtl.utils.types import BootstrapResult

log = logging.getLogger("compute_indices")


def _benefits_to_bootstrap(df: pd.DataFrame, key: str = "delta_pair") -> dict[int, BootstrapResult]:
    out: dict[int, BootstrapResult] = {}
    for _, row in df.iterrows():
        out[int(row["region_id"])] = BootstrapResult(
            estimate=float(row[key]),
            ci_lower=float(row["ci_lo"]),
            ci_upper=float(row["ci_hi"]),
            samples=None,
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--epsilon", type=float, default=1.5)
    parser.add_argument("--eta", type=float, default=0.5)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    bench_root = Path("outputs/benefits") / args.dataset
    rows: list[dict[str, object]] = []

    for pair_dir in sorted(bench_root.iterdir() if bench_root.exists() else []):
        if not pair_dir.is_dir():
            continue
        df = RegionBenefitSchema.validate(pd.read_parquet(pair_dir / "region_benefits.parquet"))
        deltas = _benefits_to_bootstrap(df, "delta_pair")
        deltas_i = _benefits_to_bootstrap(df, "delta_i_from_j")
        deltas_j = _benefits_to_bootstrap(df, "delta_j_from_i")

        S = compute_S_pair(deltas, epsilon=args.epsilon)
        S_i, S_j = compute_S_task_specific(deltas_i, deltas_j, epsilon=args.epsilon)

        delta_means = {rid: b.estimate for rid, b in deltas.items() if not np.isnan(b.estimate)}
        n_test = dict(zip(df["region_id"].astype(int), df["n_test"].astype(int), strict=True))
        H = compute_H(delta_means, n_test)
        C = compute_C(delta_means, n_test, eta=args.eta)
        n_valid = int((~df[["delta_pair", "ci_lo", "ci_hi"]].isna().any(axis=1)).sum())

        rows.append(
            {
                "pair_id": pair_dir.name,
                "S_ij": bool(S),
                "S_i": bool(S_i),
                "S_j": bool(S_j),
                "H_ij": float(H),
                "C_ij": float(C),
                "n_valid_regions": n_valid,
            }
        )

    if not rows:
        log.warning("no benefit files found under %s", bench_root)
        return 1
    out = write_pair_indices(args.dataset, rows)
    log.info("wrote %s (%d pairs)", out, len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
