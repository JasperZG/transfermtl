"""Morgan (ECFP-style) fingerprints with disk cache keyed by scaffold."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

CACHE_DIR = Path("outputs/cache/scaffold_fps")


def morgan_fingerprint(smi: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Return Morgan FP as uint8 bit vector of shape (n_bits,)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(n_bits, dtype=np.uint8)
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.uint8)
    from rdkit import DataStructs

    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def cache_scaffold_fingerprints(
    dataset: str,
    scaffolds: list[str],
    radius: int = 2,
    n_bits: int = 2048,
    force: bool = False,
) -> dict[str, np.ndarray]:
    """Compute and disk-cache Morgan FPs for the unique scaffolds in `scaffolds`.

    Stored at outputs/cache/scaffold_fps/{dataset}.parquet with two columns:
    `scaffold` (str) and `fp` (list[uint8]). Returns a dict mapping scaffold
    SMILES -> bit vector.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{dataset}.parquet"

    if path.exists() and not force:
        df = pd.read_parquet(path)
        return {row.scaffold: np.asarray(row.fp, dtype=np.uint8) for row in df.itertuples()}

    unique = sorted(set(scaffolds))
    rows = []
    for scaff in unique:
        fp = morgan_fingerprint(scaff, radius=radius, n_bits=n_bits)
        rows.append({"scaffold": scaff, "fp": fp.tolist()})
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    return {row.scaffold: np.asarray(row.fp, dtype=np.uint8) for row in df.itertuples()}
