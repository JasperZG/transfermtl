"""SMILES standardization: cleanup, largest-fragment, canonical form.

Returns None on RDKit parse failure so callers can drop bad rows.
"""

from __future__ import annotations

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize.rdMolStandardize import (
    Cleanup,
    LargestFragmentChooser,
)

# RDKit warnings about valence are noisy on real datasets; we drop bad rows
# explicitly when parsing fails, so silence the C++ logger.
RDLogger.DisableLog("rdApp.*")


_LARGEST_FRAGMENT = LargestFragmentChooser()


def standardize_smiles(smi: str) -> str | None:
    """Normalize a SMILES string.

    Steps: parse -> RDKit MolStandardize Cleanup -> LargestFragmentChooser ->
    canonical SMILES. Returns None if any stage fails.
    """
    if not isinstance(smi, str) or not smi.strip():
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        mol = Cleanup(mol)
        mol = _LARGEST_FRAGMENT.choose(mol)
    except (RuntimeError, ValueError):
        return None
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    canonical = Chem.MolToSmiles(mol, canonical=True)
    return canonical or None
