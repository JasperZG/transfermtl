"""CLI entry point for pairwise MTL training (plan §2.5).

Usage:
    python scripts/train_pairwise_mtl.py --dataset tox21 \
        --task-i NR-AR --task-j NR-AR-LBD --seed 0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

from transfermtl.training import (
    MTLArtifacts,
    TrainConfig,
    load_pyg_dataset,
    train_pairwise_mtl,
)
from transfermtl.utils import logging as wandb_logging
from transfermtl.utils.git import current_sha, dirty_tree_warning
from transfermtl.utils.io import read_parquet
from transfermtl.utils.schemas import SplitSchema

log = logging.getLogger("train_pairwise_mtl")

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = REPO_ROOT / "outputs" / "splits"
CACHE_DIR = REPO_ROOT / "outputs" / "cache" / "featurized"
CKPT_DIR = REPO_ROOT / "outputs" / "checkpoints"
PRED_DIR = REPO_ROOT / "outputs" / "predictions"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task-i", required=True, dest="task_i")
    parser.add_argument("--task-j", required=True, dest="task_j")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log.info("git=%s dirty=%s", current_sha(), dirty_tree_warning())

    cfg = TrainConfig(max_epochs=args.max_epochs, batch_size=args.batch_size)
    tasks = [args.task_i, args.task_j]

    split_path = SPLITS_DIR / args.dataset / "split.parquet"
    split_df = read_parquet(split_path, schema=SplitSchema)

    train = load_pyg_dataset(split_df, "train", tasks, cache_dir=CACHE_DIR)
    val = load_pyg_dataset(split_df, "val", tasks, cache_dir=CACHE_DIR)
    test = load_pyg_dataset(split_df, "test", tasks, cache_dir=CACHE_DIR)
    log.info("train=%d val=%d test=%d", len(train), len(val), len(test))

    wandb_logging.init(
        config={
            "dataset": args.dataset,
            "task_i": args.task_i,
            "task_j": args.task_j,
            "seed": args.seed,
            **cfg.__dict__,
        },
        run_name=f"mtl/{args.dataset}/{args.task_i}_{args.task_j}/seed{args.seed}",
        tags=["mtl", args.dataset, args.task_i, args.task_j],
    )

    pair = f"{args.task_i}_{args.task_j}"
    ckpt_path = CKPT_DIR / args.dataset / "mtl" / pair / f"seed{args.seed}.pt"
    pred_path = PRED_DIR / args.dataset / "mtl" / pair / f"seed{args.seed}.parquet"

    art: MTLArtifacts = train_pairwise_mtl(
        train_data=train,
        val_data=val,
        task_i=args.task_i,
        task_j=args.task_j,
        seed=args.seed,
        cfg=cfg,
        device=args.device,
        checkpoint_path=ckpt_path,
        test_data=test,
        predictions_path=pred_path,
    )

    log.info(
        "best_val_loss=%.4f best_epoch=%d ckpt=%s pred=%s",
        art.outcome.best_val_loss,
        art.outcome.best_epoch,
        art.checkpoint_path,
        art.predictions_path,
    )
    wandb_logging.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
