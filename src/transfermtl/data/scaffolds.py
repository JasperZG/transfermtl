"""Bemis-Murcko scaffolds with `<EMPTY>` bucket for acyclic molecules."""

from __future__ import annotations

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

EMPTY_SCAFFOLD = "<EMPTY>"


def compute_scaffold(smi: str, include_chirality: bool = False) -> str:
    """Return the Murcko scaffold SMILES for `smi`.

    Acyclic molecules (no rings) bucket under the literal sentinel
    `EMPTY_SCAFFOLD` so the scaffold-stratified splitter still treats them
    as a single group rather than 1-mol groups.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return EMPTY_SCAFFOLD
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=mol,
        includeChirality=include_chirality,
    )
    if not scaffold:
        return EMPTY_SCAFFOLD
    return scaffold
