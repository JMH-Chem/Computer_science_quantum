"""Detect Abelian symmetry of actual orbitals without rotating saved states."""
from __future__ import annotations

import numpy as np
from pyscf import symm


def orbital_symmetry_labels(mol, coefficients, *, return_molecule=False):
    """Return XOR-compatible orbital irreps and the detected Abelian group.

    Orbital rotation would require transforming every dependent amplitude.
    Mixed orbitals therefore raise an error, allowing callers to use a full
    small-matrix calculation or report an unavailable symmetry search. With
    return_molecule=True, also return the symmetry-built copy for response
    solvers that accept a molecule with explicit symmetry metadata.
    """
    molecule = mol.copy()
    molecule.verbose = 0
    molecule.symmetry, molecule.symmetry_subgroup = True, None
    molecule.build(False, False)
    subgroup = {"Dooh": "D2h", "Coov": "C2v", "SO3": "D2h"}.get(molecule.groupname)
    if subgroup:
        molecule.symmetry_subgroup = subgroup
        molecule.build(False, False)
    if molecule.groupname not in {"D2h", "D2", "C2v", "C2h", "C2", "Cs", "Ci", "C1"}:
        raise ValueError(f"No supported Abelian subgroup for {molecule.groupname}.")
    if not np.allclose(molecule.atom_coords(), mol.atom_coords(), atol=1e-13, rtol=0):
        raise ValueError("Symmetry detection changed the recorded nuclear geometry.")
    labels = symm.label_orb_symm(molecule, molecule.irrep_id, molecule.symm_orb,
                                np.asarray(coefficients), check=True, tol=1e-10)
    if return_molecule:
        return labels, molecule.groupname, molecule
    return labels, molecule.groupname
