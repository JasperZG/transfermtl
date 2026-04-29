"""Atom (74-d) + bond (12-d) featurization producing PyG Data objects.

The 74-d atom recipe (concatenated, in this order):
    25  symbol one-hot (24 elements + "other")
     8  degree one-hot (0-6 + "other")
     6  formal charge one-hot (-2..2 + "other")
     4  chirality one-hot (UNSPECIFIED, CW, CCW, OTHER)
     6  num H one-hot (0-4 + "other")
     7  hybridization one-hot (SP, SP2, SP3, SP3D, SP3D2, UNSPECIFIED, OTHER)
     5  num radical electrons one-hot (0-3 + "other")
     6  ring-size membership multi-hot (3, 4, 5, 6, 7, 8)
     3  binary: is_aromatic, is_in_ring, is_chiral_center
     4  scalars: mass/100, atomic_num/100, total_valence/8, num_h_total/8

The 12-d bond recipe:
     5  bond type one-hot (SINGLE, DOUBLE, TRIPLE, AROMATIC, OTHER)
     4  bond stereo one-hot (NONE, ANY, Z, E)
     3  binary: is_in_ring, is_conjugated, is_aromatic

Both dimensions match plan §2.21's pre-committed encoder hparams.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data

ATOM_FEATURE_DIM = 74
BOND_FEATURE_DIM = 12

CACHE_DIR = Path("outputs/cache/featurized")

_SYMBOL_LIST = [
    "C",
    "N",
    "O",
    "F",
    "P",
    "S",
    "Cl",
    "Br",
    "I",
    "Si",
    "B",
    "H",
    "Se",
    "Na",
    "K",
    "Mg",
    "Ca",
    "Fe",
    "Zn",
    "Cu",
    "Mn",
    "As",
    "Al",
    "Li",
]
_DEGREE_LIST = [0, 1, 2, 3, 4, 5, 6]
_FORMAL_CHARGE_LIST = [-2, -1, 0, 1, 2]
_NUM_H_LIST = [0, 1, 2, 3, 4]
_NUM_RAD_LIST = [0, 1, 2, 3]
_RING_SIZES = [3, 4, 5, 6, 7, 8]

_HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
    Chem.rdchem.HybridizationType.UNSPECIFIED,
]

_CHIRALITY = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
]

_BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
_BOND_STEREO = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
]


def _one_hot_with_other(value: Any, choices: list[Any]) -> list[float]:
    vec = [0.0] * (len(choices) + 1)
    try:
        idx = choices.index(value)
    except ValueError:
        idx = len(choices)
    vec[idx] = 1.0
    return vec


def _one_hot_exact(value: Any, choices: list[Any]) -> list[float]:
    vec = [0.0] * len(choices)
    try:
        idx = choices.index(value)
    except ValueError:
        idx = len(choices) - 1
    vec[idx] = 1.0
    return vec


def _atom_features(atom: Chem.Atom) -> list[float]:
    feats: list[float] = []
    feats += _one_hot_with_other(atom.GetSymbol(), _SYMBOL_LIST)  # 25
    feats += _one_hot_with_other(atom.GetTotalDegree(), _DEGREE_LIST)  # 8
    feats += _one_hot_with_other(atom.GetFormalCharge(), _FORMAL_CHARGE_LIST)  # 6
    feats += _one_hot_with_other(atom.GetChiralTag(), _CHIRALITY)  # 4
    feats += _one_hot_with_other(atom.GetTotalNumHs(), _NUM_H_LIST)  # 6
    feats += _one_hot_with_other(atom.GetHybridization(), _HYBRIDIZATIONS)  # 7
    feats += _one_hot_with_other(atom.GetNumRadicalElectrons(), _NUM_RAD_LIST)  # 5
    # ring-size multi-hot
    is_in_ring_size = [1.0 if atom.IsInRingSize(s) else 0.0 for s in _RING_SIZES]
    feats += is_in_ring_size  # 6
    feats += [
        1.0 if atom.GetIsAromatic() else 0.0,
        1.0 if atom.IsInRing() else 0.0,
        1.0 if atom.HasProp("_ChiralityPossible") else 0.0,
    ]  # 3
    feats += [
        atom.GetMass() / 100.0,
        atom.GetAtomicNum() / 100.0,
        atom.GetTotalValence() / 8.0,
        atom.GetTotalNumHs() / 8.0,
    ]  # 4
    return feats


def _bond_features(bond: Chem.Bond) -> list[float]:
    feats: list[float] = []
    feats += _one_hot_with_other(bond.GetBondType(), _BOND_TYPES)  # 5
    feats += _one_hot_exact(bond.GetStereo(), _BOND_STEREO)  # 4
    feats += [
        1.0 if bond.IsInRing() else 0.0,
        1.0 if bond.GetIsConjugated() else 0.0,
        1.0 if bond.GetIsAromatic() else 0.0,
    ]  # 3
    return feats


def featurize_smiles(smi: str, y: float | None = None) -> Data | None:
    """Return a PyG Data object for `smi`, or None if RDKit cannot parse it."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    atom_feats = np.array([_atom_features(a) for a in mol.GetAtoms()], dtype=np.float32)
    assert (
        atom_feats.shape[1] == ATOM_FEATURE_DIM
    ), f"atom feature dim mismatch: got {atom_feats.shape[1]}, expected {ATOM_FEATURE_DIM}"

    edge_src: list[int] = []
    edge_dst: list[int] = []
    edge_attr: list[list[float]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bf = _bond_features(bond)
        assert len(bf) == BOND_FEATURE_DIM
        edge_src.extend([i, j])
        edge_dst.extend([j, i])
        edge_attr.extend([bf, bf])

    if not edge_src:
        # isolated atom (e.g. methane has bonds; truly atomic species like 'O' have none).
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr_t = torch.zeros((0, BOND_FEATURE_DIM), dtype=torch.float32)
    else:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)

    data = Data(
        x=torch.from_numpy(atom_feats),
        edge_index=edge_index,
        edge_attr=edge_attr_t,
        smiles=smi,
    )
    if y is not None:
        data.y = torch.tensor([y], dtype=torch.float32)
    return data


def _smi_hash(smi: str) -> str:
    return hashlib.sha256(smi.encode()).hexdigest()[:16]


def cache_path_for(smi: str) -> Path:
    """Per-molecule cache path: outputs/cache/featurized/{16-char sha}.pt."""
    return CACHE_DIR / f"{_smi_hash(smi)}.pt"


def featurize_and_cache(smi: str, y: float | None = None, force: bool = False) -> Path | None:
    """Featurize `smi` and write to per-molecule cache; returns the path.

    Returns None if RDKit cannot parse the SMILES.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path_for(smi)
    if path.exists() and not force:
        return path
    data = featurize_smiles(smi, y=y)
    if data is None:
        return None
    torch.save(data, path)
    return path
