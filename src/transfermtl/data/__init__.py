"""data/ — preprocessing pipeline (plan §2.2).

Public surface:
- standardize_smiles
- compute_scaffold
- morgan_fingerprint
- featurize_smiles
- scaffold_stratified_split
- prepare_dataset (top-level entry point invoked by scripts/prepare_dataset.py)
"""

from transfermtl.data import datasets  # noqa: F401  (registers loaders)
from transfermtl.data.featurize import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    featurize_smiles,
)
from transfermtl.data.fingerprints import morgan_fingerprint
from transfermtl.data.manifest import write_manifest
from transfermtl.data.scaffolds import compute_scaffold
from transfermtl.data.splits import scaffold_stratified_split
from transfermtl.data.standardize import standardize_smiles

__all__ = [
    "ATOM_FEATURE_DIM",
    "BOND_FEATURE_DIM",
    "compute_scaffold",
    "featurize_smiles",
    "morgan_fingerprint",
    "scaffold_stratified_split",
    "standardize_smiles",
    "write_manifest",
]
