"""Training loops + checkpoint/prediction IO for STL, pairwise MTL, all-task MTL."""

from transfermtl.training.all_task_mtl import AllTaskArtifacts, train_all_task_mtl
from transfermtl.training.checkpoint import (
    CheckpointBundle,
    load_checkpoint,
    save_checkpoint,
)
from transfermtl.training.data import cache_path_for, load_pyg_dataset, smiles_hash
from transfermtl.training.loops import (
    TrainConfig,
    TrainOutcome,
    make_optimizer,
    make_scheduler,
    train_loop,
)
from transfermtl.training.pairwise_mtl import MTLArtifacts, train_pairwise_mtl
from transfermtl.training.predict import predict_dataset, save_predictions
from transfermtl.training.stl import STLArtifacts, train_stl

__all__ = [
    "AllTaskArtifacts",
    "CheckpointBundle",
    "MTLArtifacts",
    "STLArtifacts",
    "TrainConfig",
    "TrainOutcome",
    "cache_path_for",
    "load_checkpoint",
    "load_pyg_dataset",
    "make_optimizer",
    "make_scheduler",
    "predict_dataset",
    "save_checkpoint",
    "save_predictions",
    "smiles_hash",
    "train_all_task_mtl",
    "train_loop",
    "train_pairwise_mtl",
    "train_stl",
]
