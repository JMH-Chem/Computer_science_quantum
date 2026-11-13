"""Search every detected response symmetry sector before selecting low roots.

Convergence of a few Koopmans guesses does not establish that lower roots in
other invariant sectors were found. Keep the actual SCF orbitals unchanged and
check every returned vector against the original, unprojected operator.
"""
from __future__ import annotations

import copy
import math

import numpy as np
from pyscf import lib, symm
from pyscf.tdscf import rhf, rks

from .symmetry import orbital_symmetry_labels


def solve_response(td, requested: int, emit) -> dict:
    mf = td._scf
    occupied = np.flatnonzero(mf.mo_occ == 2)
    virtual = np.flatnonzero(mf.mo_occ == 0)
    pairs = len(occupied) * len(virtual)
    casida = isinstance(td, rks.CasidaTDDFT)
    tda = isinstance(td, rhf.TDA) and not casida
    # PySCF's default 1e-3 positivity filter applies to squared energies in
    # Casida: it can discard a physical excitation below 0.86 eV even with a
    # complete trial basis. Use an explicit numerical floor in the operator's
    # native units, independently of the requested residual tolerance.
    td.positive_eig_threshold = float(np.finfo(float).eps)
    # The real TDHF/TDDFT solver instead diagonalizes a squared-energy problem
    # internally and accepts its strictly positive eigenvalues; it does not
    # consult positive_eig_threshold. Record the filter actually applied.
    real_tdhf = not tda and not casida and np.asarray(mf.mo_coeff).dtype == np.double
    squared_filter = casida or real_tdhf
    positive_threshold = 0.0 if real_tdhf else td.positive_eig_threshold
    gaps = (mf.mo_energy[virtual] - mf.mo_energy[occupied, None]).ravel()
    search = {
        "strategy": "abelian_symmetry_blocks", "group": None, "blocks": [], "complete": False,
        "scope": "Lowest requested positive roots are targeted in every detected Abelian symmetry sector and merged by energy. Iterative convergence is not a mathematical proof of global spectral completeness.",
        "initial_guesses": "Complete coordinate basis and full sector root count for sectors with at most 64 occupied-virtual pairs; lowest orbital-gap guesses for larger sectors.",
        "residual_tolerance": float(td.conv_tol),
        "residual_operator": "Full unprojected Casida operator (squared excitation energy)" if casida
        else "Full unprojected TDA operator" if tda else "Full unprojected TDHF/TDDFT operator",
        "positive_root_filter": {
            "operator_eigenvalue_threshold": positive_threshold,
            "operator_eigenvalue_unit": "hartree_squared" if squared_filter else "hartree",
            "excitation_energy_threshold_hartree": math.sqrt(positive_threshold) if squared_filter else positive_threshold,
            "definition": "Strictly greater than the stated numerical positivity cutoff; independent of residual convergence tolerance.",
        },
    }
    try:
        labels, group, molecule = orbital_symmetry_labels(mf.mol, mf.mo_coeff, return_molecule=True)
        pair_labels = (labels[occupied, None] ^ labels[virtual]).ravel()
        search["group"] = group
        symmetric_mf = copy.copy(mf)
        symmetric_mf.mol = molecule
        symmetric_mf.mo_coeff = lib.tag_array(np.asarray(mf.mo_coeff), orbsym=labels)
        blocks = [(int(label), np.flatnonzero(pair_labels == label)) for label in np.unique(pair_labels)]
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        search["reason"] = str(exc)
        symmetric_mf = mf
        if pairs <= 256:
            search["strategy"] = "full_coordinate_fallback"
            search["scope"] = "The complete small occupied-virtual coordinate basis is supplied, without assigning mixed orbitals to symmetry sectors."
            blocks = [(None, np.arange(pairs))]
        else:
            search["strategy"] = "unavailable"
            search["scope"] = "Actual orbitals could not be assigned pure symmetry sectors; no partial root list is certified."
            blocks = []

    full_operator, _ = td.gen_vind() if blocks else (None, None)
    candidates = []
    for label, indices in blocks:
        count = min(requested, len(indices))
        name = None if label is None else symm.irrep_id2name(search["group"], label)
        block = {"irrep": name, "dimension": len(indices), "requested_roots": count,
                 "converged_roots": 0, "complete": False}
        search["blocks"].append(block)
        emit("stage", {"name": "response_sector", "message": f"Searching {count} lowest response roots in {name or 'the full'} sector.",
                       "symmetry": name, "symmetry_group": search["group"]})
        response = copy.copy(td)
        response._scf, response.mol = symmetric_mf, symmetric_mf.mol
        response.wfnsym, response.nstates, response.chkfile = label, count, None
        try:
            guesses = None
            if len(indices) <= 64 or label is None:
                guesses = np.zeros((len(indices), pairs if tda or casida else 2 * pairs))
                guesses[np.arange(len(indices)), indices] = 1
                # PySCF caps each added trial block using nstates. Supplying an
                # identity alone can silently discard its later coordinates.
                response.nstates = len(indices)
            block["solver_roots"] = response.nstates
            response.kernel(x0=guesses)
            energies = np.atleast_1d(response.e)[:count]
            flags = np.atleast_1d(response.converged)[:count]
            for index, (energy, pair) in enumerate(zip(energies, response.xy[:count], strict=True)):
                x, y = pair
                if tda:
                    vector = np.asarray(x).ravel()
                    eigenvalue = energy
                elif casida:
                    if np.any(gaps <= 0):
                        raise ValueError("Casida response requires positive occupied-virtual gaps.")
                    vector = (np.asarray(x) + np.asarray(y)).ravel() / np.sqrt(gaps)
                    eigenvalue = energy**2
                else:
                    vector = np.concatenate([np.asarray(x).ravel(), np.asarray(y).ravel()])
                    eigenvalue = energy
                norm = float(np.linalg.norm(vector))
                residual = None
                if norm > 0 and np.isfinite(vector).all() and np.isfinite(eigenvalue):
                    normalized = vector / norm
                    residual = float(np.linalg.norm(np.asarray(full_operator(normalized[None, :]))[0]
                                                    - eigenvalue * normalized))
                    if not math.isfinite(residual):
                        residual = None
                valid = bool(index < len(flags) and flags[index] and np.isfinite(energy) and energy > 0
                             and residual is not None and residual < td.conv_tol)
                block["converged_roots"] += int(valid)
                candidates.append({"energy": float(energy), "xy": pair, "converged": valid,
                                   "residual_norm": residual, "symmetry": name})
            block["complete"] = len(energies) == count and block["converged_roots"] == count
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            block["error"] = str(exc)
    search["complete"] = bool(blocks) and all(block["complete"] for block in search["blocks"])
    candidates.sort(key=lambda item: item["energy"] if math.isfinite(item["energy"]) else math.inf)
    selected = candidates[:requested]
    td.e = np.asarray([item["energy"] for item in selected])
    td.xy = [item["xy"] for item in selected]
    td.converged = np.asarray([item["converged"] for item in selected], dtype=bool)
    search["selected_states"] = [{"symmetry": item["symmetry"], "residual_norm": item["residual_norm"]}
                                  for item in selected]
    if td.chkfile:
        lib.chkfile.save(td.chkfile, "tddft/e", td.e)
        lib.chkfile.save(td.chkfile, "tddft/xy", td.xy)
    return search
