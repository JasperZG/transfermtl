"""Select 20-40 task pairs per dataset for the Phase 0 pilot.

The selection is deterministic given a seed, so two pilot reruns produce the
exact same pair set. The output frame uses *friendly* task names (e.g.
``NR-AR``); downstream training/measurement scripts call
``transfermtl.data.manifest.resolve_task_name`` to resolve those to the
``task_*`` columns A2 writes into the split parquet.

Usage:
    python scripts/select_pilot_pairs.py [--seed 42] [--per-dataset 25] \
        [--out outputs/analysis/pilot_pairs.parquet]
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from transfermtl.data.datasets import TOX21_TASKS
from transfermtl.utils.io import write_parquet

log = logging.getLogger("select_pilot_pairs")

# SIDER's 27 tasks are themselves SOC labels (System Organ Classes). Without
# an authoritative ATC grouping shipped with the CSV, we treat every pair as
# cross-SOC. A future refinement (or A9) can swap in a finer mechanism map.
SIDER_TASKS: tuple[str, ...] = (
    "Hepatobiliary disorders",
    "Metabolism and nutrition disorders",
    "Product issues",
    "Eye disorders",
    "Investigations",
    "Musculoskeletal and connective tissue disorders",
    "Gastrointestinal disorders",
    "Social circumstances",
    "Immune system disorders",
    "Reproductive system and breast disorders",
    "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
    "General disorders and administration site conditions",
    "Endocrine disorders",
    "Surgical and medical procedures",
    "Vascular disorders",
    "Blood and lymphatic system disorders",
    "Skin and subcutaneous tissue disorders",
    "Congenital, familial and genetic disorders",
    "Infections and infestations",
    "Respiratory, thoracic and mediastinal disorders",
    "Psychiatric disorders",
    "Renal and urinary disorders",
    "Pregnancy, puerperium and perinatal conditions",
    "Ear and labyrinth disorders",
    "Cardiac disorders",
    "Nervous system disorders",
    "Injury, poisoning and procedural complications",
)


def _tox21_mechanism(task: str) -> str:
    """Tox21 mechanism family from the task prefix: 'NR' (nuclear receptor) or
    'SR' (stress response).
    """
    if task.startswith("NR-"):
        return "NR"
    if task.startswith("SR-"):
        return "SR"
    raise ValueError(f"Unrecognized Tox21 task family: {task!r}")


def _select_with_caps(
    rng: np.random.Generator,
    pool: list[tuple[str, str]],
    cap: int,
) -> list[tuple[str, str]]:
    """Deterministically take min(cap, len(pool)) pairs from `pool`."""
    if cap <= 0 or not pool:
        return []
    n = min(cap, len(pool))
    if n == len(pool):
        return list(pool)
    idx = rng.choice(len(pool), size=n, replace=False)
    return [pool[i] for i in sorted(int(k) for k in idx)]


def select_tox21_pairs(
    n_within_nr: int = 8,
    n_within_sr: int = 5,
    n_cross: int = 10,
    n_random: int = 2,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Compose Tox21 pair categories per the brief.

    - within_nr: NR-* x NR-* pairs (mechanism-aligned)
    - within_sr: SR-* x SR-* pairs (mechanism-aligned)
    - cross_mechanism: NR-* x SR-*  (different families)
    - random: anything else, sampled uniformly without replacement to top up
    """
    rng = np.random.default_rng(seed)
    nr = [t for t in TOX21_TASKS if _tox21_mechanism(t) == "NR"]
    sr = [t for t in TOX21_TASKS if _tox21_mechanism(t) == "SR"]

    within_nr_pool: list[tuple[str, str]] = list(itertools.combinations(nr, 2))
    within_sr_pool: list[tuple[str, str]] = list(itertools.combinations(sr, 2))
    cross_pool: list[tuple[str, str]] = [(a, b) for a in nr for b in sr]

    selected: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(pairs: list[tuple[str, str]], category: str) -> None:
        for ti, tj in pairs:
            key = (ti, tj) if ti <= tj else (tj, ti)
            if key in seen:
                continue
            seen.add(key)
            selected.append(
                {"dataset": "tox21", "task_i": key[0], "task_j": key[1], "category": category}
            )

    add(_select_with_caps(rng, within_nr_pool, n_within_nr), "within_nr")
    add(_select_with_caps(rng, within_sr_pool, n_within_sr), "within_sr")
    add(_select_with_caps(rng, cross_pool, n_cross), "cross_mechanism")

    if n_random > 0:
        full_pool = list(itertools.combinations(TOX21_TASKS, 2))
        remaining = [p for p in full_pool if (p[0], p[1]) not in seen]
        add(_select_with_caps(rng, remaining, n_random), "random")
    return selected


def select_sider_pairs(n_pairs: int = 25, seed: int = 42) -> list[dict[str, str]]:
    """Pick `n_pairs` task pairs from SIDER's 27 SOC columns deterministically.

    Each pair is labelled ``cross_soc`` because every column is itself a SOC.
    """
    rng = np.random.default_rng(seed)
    pool = list(itertools.combinations(SIDER_TASKS, 2))
    chosen = _select_with_caps(rng, pool, n_pairs)
    return [
        {"dataset": "sider", "task_i": a, "task_j": b, "category": "cross_soc"} for (a, b) in chosen
    ]


def select_pilot_pairs(
    seed: int = 42,
    tox21_within_nr: int = 8,
    tox21_within_sr: int = 5,
    tox21_cross: int = 10,
    tox21_random: int = 2,
    sider_n: int = 25,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    rows.extend(
        select_tox21_pairs(
            n_within_nr=tox21_within_nr,
            n_within_sr=tox21_within_sr,
            n_cross=tox21_cross,
            n_random=tox21_random,
            seed=seed,
        )
    )
    rows.extend(select_sider_pairs(n_pairs=sider_n, seed=seed))
    return pd.DataFrame(rows, columns=["dataset", "task_i", "task_j", "category"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/analysis/pilot_pairs.parquet"),
    )
    parser.add_argument("--tox21-within-nr", type=int, default=8)
    parser.add_argument("--tox21-within-sr", type=int, default=5)
    parser.add_argument("--tox21-cross", type=int, default=10)
    parser.add_argument("--tox21-random", type=int, default=2)
    parser.add_argument("--sider-n", type=int, default=25)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    df = select_pilot_pairs(
        seed=args.seed,
        tox21_within_nr=args.tox21_within_nr,
        tox21_within_sr=args.tox21_within_sr,
        tox21_cross=args.tox21_cross,
        tox21_random=args.tox21_random,
        sider_n=args.sider_n,
    )
    out = write_parquet(args.out, df)
    log.info("wrote %d pairs to %s", len(df), out)
    log.info("per-dataset: %s", df.groupby("dataset").size().to_dict())
    log.info("per-category: %s", df.groupby(["dataset", "category"]).size().to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
