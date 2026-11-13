"""Closed-shell correlation methods on a recorded RHF reference.

The reference orbitals are deliberately retained by the caller. A correlated
total energy does not turn the RHF density or canonical orbital energies into
correlated observables. Full/active CI spaces are bounded before transformation.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from pyscf import cc, ci, fci, lib, mcscf, mp, mrpt, symm
from scipy.constants import physical_constants

from .symmetry import orbital_symmetry_labels

METHODS = frozenset({"MP2", "CCSD", "CCSD(T)", "CISD", "FCI", "CASCI", "CASSCF", "NEVPT2"})
FROZEN_METHODS = frozenset({"MP2", "CCSD", "CCSD(T)", "CISD"})
MAX_CI_DETERMINANTS = 100_000
MAX_FCI_ORBITALS = 12
HARTREE_EV = physical_constants["Hartree energy in eV"][0]


def _finite(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def validate_space(mf, spec: dict) -> dict:
    """Check actual orbital/electron dimensions, including ECP electron counts."""
    method = spec["method"]
    if method not in METHODS:
        raise ValueError(f"Unknown correlated method: {method}")
    if mf.mol.spin or not spec.get("restricted", True) or np.ndim(mf.mo_coeff) != 2:
        raise ValueError("Correlated methods currently require a closed-shell RHF reference.")
    nmo = int(mf.mo_coeff.shape[1])
    occupied = int(mf.mol.nelectron // 2)
    frozen = int(spec.get("frozen_orbitals", 0))
    if frozen < 0 or (frozen and method not in FROZEN_METHODS):
        raise ValueError("Frozen occupied orbitals are supported only by MP2, CCSD, CCSD(T), and CISD.")
    if method in FROZEN_METHODS and (frozen >= occupied or occupied >= nmo):
        raise ValueError("At least one correlated occupied and one virtual orbital are required.")
    active = None
    determinants = None
    if method in {"CASCI", "CASSCF", "NEVPT2"}:
        electrons = int(spec.get("active_electrons", 2))
        orbitals = int(spec.get("active_orbitals", 2))
        if (electrons < 2 or electrons % 2 or electrons > mf.mol.nelectron
                or orbitals < 1 or orbitals > MAX_FCI_ORBITALS or electrons > 2 * orbitals):
            raise ValueError("Use an even active electron count and 1–12 active orbitals consistent with the molecule.")
        core = (int(mf.mol.nelectron) - electrons) // 2
        if core + orbitals > nmo:
            raise ValueError("Core plus active orbitals exceed the actual molecular orbital count.")
        determinants = math.comb(orbitals, electrons // 2) ** 2
        active = {"electrons": electrons, "orbitals": orbitals, "core_orbitals": core,
                  "initial_orbital_indices_1based": list(range(core + 1, core + orbitals + 1)),
                  "selection": "Contiguous canonical RHF orbitals after the doubly occupied core; CASSCF optimizes orbitals."}
        if method == "NEVPT2" and orbitals > 8:
            raise ValueError("NEVPT2 is limited to 8 active orbitals because four-particle density matrices scale as n_active^8.")
    elif method == "FCI":
        if nmo > MAX_FCI_ORBITALS:
            raise ValueError(f"Full CI is limited to {MAX_FCI_ORBITALS} molecular orbitals; choose a smaller basis or CASCI/CASSCF.")
        determinants = math.comb(nmo, occupied) ** 2
    if determinants is not None and determinants > MAX_CI_DETERMINANTS:
        raise ValueError(f"CI space has {determinants:,} determinants; the local limit is {MAX_CI_DETERMINANTS:,}.")
    # A conservative preflight estimate, not a claim of a hard RAM limit.
    integral_mb = 8 * (2 * nmo**4 + 4 * nmo**2) / 1024**2
    if method == "NEVPT2":
        integral_mb += 3 * 8 * active["orbitals"]**8 / 1024**2
    budget_mb = float(spec.get("memory_mb", 2000))
    if integral_mb > budget_mb * 0.6:
        raise ValueError(
            f"Estimated dense integral workspace ({integral_mb:.0f} MiB) exceeds 60% of the memory budget. "
            "Select a smaller basis or explicitly raise the memory budget; actual memory can be larger."
        )
    return {"frozen": frozen, "active": active, "determinants": determinants,
            "integral_memory_estimate_mib": integral_mb}


def _packed_eom_irreps(orbital_irreps, nocc: int) -> np.ndarray:
    """Irreps in PySCF's singlet R1 + lower-triangular (ia,jb) R2 order."""
    orbital_irreps = np.asarray(orbital_irreps, dtype=int)
    singles = (orbital_irreps[:nocc, None] ^ orbital_irreps[None, nocc:]).ravel()
    i, j = np.tril_indices(len(singles))
    return np.concatenate((singles, singles[i] ^ singles[j]))


def _eom_symmetry_labels(solver, response):
    """Label the actual orbitals without rotating them or the CC amplitudes."""
    coefficients = np.asarray(solver.mo_coeff)[:, solver.get_frozen_mask()]
    labels, group = orbital_symmetry_labels(solver.mol, coefficients)
    packed = _packed_eom_irreps(labels, response.nocc)
    if packed.size != response.vector_size():
        raise ValueError("Excitation symmetry labels do not match the EOM vector dimension.")
    return packed, group


def _eom_block_roots(matvec, diagonal, indices, requested, response):
    """Target the lowest real roots inside one invariant symmetry sector.

    Explicit initial guesses passed to EOMEESinglet.kernel enable overlap/root
    following. Use the nonsymmetric Davidson routine directly so the selection
    remains by energy. An extra dense guess avoids exactly zero components within a
    sector; no published energy or molecule-specific guess enters this search.
    """
    diagonal = np.asarray(diagonal)
    indices = np.asarray(indices, dtype=int)
    dimension, count = diagonal.size, min(int(requested), len(indices))
    if not count:
        return np.zeros(0, dtype=bool), np.zeros(0), np.empty((0, dimension))

    def projected(vectors):
        lifted = np.zeros((len(vectors), dimension))
        lifted[:, indices] = np.asarray(vectors)
        return [np.asarray(value)[indices] for value in matvec(lifted)]

    block_diagonal = diagonal[indices]
    generator = np.random.default_rng(0xE0CC)
    guesses = []
    for index in np.argsort(block_diagonal)[:count]:
        guess = np.zeros(len(indices))
        guess[index] = 1.0
        guesses.append(guess)
    if count < len(indices):
        dense = generator.standard_normal(len(indices))
        guesses.append(dense / np.linalg.norm(dense))

    def precondition(residual, energy, _):
        denominator = energy - block_diagonal
        denominator = np.where(abs(denominator) < 1e-12, 1e-12, denominator)
        return residual / denominator

    def real_roots(values, vectors, nroots, env):
        eligible = np.flatnonzero(abs(values.imag) < 1e-8)
        return lib.linalg_helper._eigs_cmplx2real(values, vectors, eligible, True)

    flags, energies, vectors = lib.davidson_nosym1(
        projected, guesses, precondition, tol=response.conv_tol,
        tol_residual=math.sqrt(response.conv_tol), max_cycle=response.max_cycle,
        max_space=response.max_space, max_memory=response.max_memory,
        nroots=count, pick=real_roots, verbose=response.verbose,
    )
    full_vectors = np.zeros((len(vectors), dimension))
    full_vectors[:, indices] = np.asarray(vectors).real
    return np.asarray(flags, dtype=bool), np.asarray(energies).real, full_vectors


def run_eom(solver, spec: dict, emit, workdir: Path) -> dict:
    """Singlet EOM-EE-CCSD right roots; no transition moments are inferred."""
    if not solver.converged:
        raise ValueError("EOM-EE-CCSD requires converged ground-state CCSD amplitudes.")
    response = solver.EOMEESinglet()
    requested = int(spec.get("nstates", 5))
    dimension = int(response.vector_size())
    if requested < 1 or requested > dimension:
        raise ValueError(f"Requested {requested} EOM roots, but the singlet excitation-vector dimension is {dimension}.")
    response.conv_tol = float(spec.get("conv_tol", 1e-9))
    response.max_cycle = int(spec.get("max_cycle", 80))
    response.max_memory = int(spec.get("memory_mb", 2000))
    emit("stage", {"name": "eomccsd", "message": "Solving singlet EOM-EE-CCSD roots above the ground-state CCSD energy."})
    intermediates = response.make_imds()
    matvec, diagonal = response.gen_matvec(intermediates)
    # PySCF's Davidson solver defaults to sqrt(energy tolerance) for its
    # eigenvector residual. Evaluate and record that residual independently.
    residual_tolerance = math.sqrt(response.conv_tol)
    search = {"strategy": "abelian_symmetry_blocks", "group": None, "blocks": [], "complete": False,
              "scope": "Lowest requested real roots targeted independently in every detected Abelian symmetry sector, then merged by energy. Iterative residual convergence is not a mathematical proof of global spectral completeness.",
              "initial_guesses": "Lowest diagonal coordinate vectors plus one deterministic dense vector within each searched sector."}
    candidates = []
    try:
        labels, group = _eom_symmetry_labels(solver, response)
        search["group"] = group
        blocks = [(symm.irrep_id2name(group, int(label)), np.flatnonzero(labels == label))
                  for label in np.unique(labels)]
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        search["reason"] = str(exc)
        if dimension <= 256:
            # A complete small Jacobian can be diagonalized without assuming
            # pure orbital irreps; do not rotate MOs while keeping old t1/t2.
            search["strategy"] = "full_dense_fallback"
            search["scope"] = "All eigenvalues of the complete small nonsymmetric EOM Jacobian are evaluated; requested lowest roots are selected by real energy. Symmetry labels are unavailable."
            blocks = [(None, np.arange(dimension))]
        else:
            search["strategy"] = "unavailable"
            search["scope"] = "Actual orbitals could not be assigned pure symmetry sectors; no partial set is certified. The converged CCSD ground-state result is retained."
            blocks = []

    for label, indices in blocks:
        count = min(requested, len(indices))
        block = {"irrep": label, "dimension": len(indices), "requested_roots": count,
                 "converged_roots": 0, "complete": False}
        search["blocks"].append(block)
        emit("stage", {"name": "eom_sector", "message": f"Searching {count} lowest EOM roots in {label or 'the full'} sector.",
                       "symmetry": label, "symmetry_group": search["group"]})
        try:
            if search["strategy"] == "full_dense_fallback":
                matrix = np.column_stack(matvec(np.eye(dimension)))
                values, eigenvectors = np.linalg.eig(matrix)
                order = np.argsort(values.real)[:count]
                energies = values[order].real
                vectors = eigenvectors[:, order].T.real
                flags = abs(values[order].imag) < 1e-8
            else:
                flags, energies, vectors = _eom_block_roots(matvec, diagonal, indices, count, response)
            for index, energy in enumerate(energies):
                vector = np.asarray(vectors[index])
                norm = float(np.linalg.norm(vector))
                if norm > 0 and np.isfinite(vector).all():
                    vector = vector / norm
                    residual = _finite(np.linalg.norm(np.asarray(matvec([vector]))[0] - energy * vector))
                else:
                    residual = None
                valid = bool(index < len(flags) and flags[index] and math.isfinite(float(energy)) and energy > 0
                             and residual is not None and residual < residual_tolerance)
                block["converged_roots"] += int(valid)
                candidates.append({"energy": float(energy), "vector": vector, "converged": valid,
                                   "residual_norm": residual, "symmetry": label})
            block["complete"] = len(energies) == count and block["converged_roots"] == count
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            block["error"] = str(exc)
    search["complete"] = bool(search["blocks"]) and all(block["complete"] for block in search["blocks"])
    candidates.sort(key=lambda item: item["energy"] if math.isfinite(item["energy"]) else math.inf)
    selected = candidates[:requested]
    states = []
    for index, candidate in enumerate(selected):
        energy = candidate["energy"]
        states.append({"state": index + 1, "energy_ev": _finite(energy * HARTREE_EV),
                       "converged": candidate["converged"], "residual_norm": candidate["residual_norm"],
                       "symmetry": candidate["symmetry"], "symmetry_group": search["group"]})
    energies = np.asarray([c["energy"] for c in selected])
    vectors = np.asarray([c["vector"] for c in selected]).reshape((-1, dimension))
    converged = search["complete"] and len(states) == requested and all(state["converged"] for state in states)
    np.savez_compressed(Path(workdir) / "eom-wavefunction.npz", energies_hartree=energies, right_vectors=vectors)
    emit("stage", {"name": "eom_complete", "converged": converged, "states": states})
    return {"reference": "CCSD", "spin": "singlet", "converged": converged, "states": states,
            "requested_states": requested, "residual_tolerance": residual_tolerance, "root_search": search,
            "energy_tolerance_hartree": response.conv_tol, "max_cycle": response.max_cycle,
            "wavefunction_artifact": "eom-wavefunction.npz",
            "notes": ["Vertical singlet excitation energies are differences from the CCSD ground state.",
                      "All searched symmetry sectors must converge; a converged subset is not reported as a successful lowest-state calculation.",
                      "Only right EOM eigenvectors are evaluated; oscillator strengths, absorption spectra, and NTOs are not computed."]}


def run_correlation(mf, spec: dict, emit, workdir: Path) -> dict:
    """Run exactly the requested method; keep nonconverged results distinct."""
    if not mf.converged:
        raise ValueError("Correlation requires a converged RHF reference.")
    space = validate_space(mf, spec)
    method, frozen = spec["method"], space["frozen"]
    tolerance, cycles = float(spec.get("conv_tol", 1e-9)), int(spec.get("max_cycle", 80))
    reference = float(mf.e_tot)
    records, warnings = [], []
    diagnostics = {"integral_memory_estimate_mib": space["integral_memory_estimate_mib"],
                   "memory_estimate_note": "Conservative preflight estimate, not a hard process memory limit."}
    if space["determinants"] is not None:
        diagnostics["determinants"] = space["determinants"]
        diagnostics["determinant_limit"] = MAX_CI_DETERMINANTS
    emit("stage", {"name": "correlation", "method": method,
                   "message": f"Computing {method} on the converged RHF reference."})

    def record(iteration, energy, *, delta=None, norm=None, stage="iteration"):
        item = {"iteration": int(iteration), "energy_hartree": _finite(energy),
                "delta_energy": _finite(delta), "residual_norm": _finite(norm), "stage": stage}
        records.append(item)
        emit("correlation_iteration", {"method": method, **item})

    occupations = None
    occupation_space = None
    eom = None
    wavefunction_arrays = {}
    total, converged = None, False
    if method == "MP2":
        solver = mp.MP2(mf, frozen=frozen or None)
        solver.max_memory = int(spec.get("memory_mb", 2000))
        occupied_energies = np.asarray(mf.mo_energy)[frozen:int(mf.mol.nelectron // 2)]
        virtual_energies = np.asarray(mf.mo_energy)[int(mf.mol.nelectron // 2):]
        gap = float(virtual_energies.min() - occupied_energies.max())
        if gap <= 1e-8:
            raise ValueError("The RHF occupied–virtual gap is too small or nonpositive for the supported MP2 path.")
        if gap < 0.05:
            warnings.append("The RHF gap is small; MP2 may be unreliable for near-degenerate or stretched bonds.")
        correction, t2 = solver.kernel()
        total = reference + float(correction)
        converged = math.isfinite(total) and bool(np.isfinite(t2).all())
        diagnostics.update(opposite_spin_energy_hartree=_finite(solver.e_corr_os),
                           same_spin_energy_hartree=_finite(solver.e_corr_ss),
                           reference_gap_hartree=gap, convergence_test="Noniterative canonical MP2 with finite amplitudes.")
        wavefunction_arrays["t2"] = t2
        record(1, total, delta=correction, stage="noniterative_complete")
    elif method in {"CCSD", "CCSD(T)"}:
        solver = cc.CCSD(mf, frozen=frozen or None)
        solver.max_cycle, solver.conv_tol = cycles, tolerance
        solver.conv_tol_normt = min(float(solver.conv_tol_normt), math.sqrt(tolerance))
        solver.max_memory = int(spec.get("memory_mb", 2000))

        def callback(env):
            # PySCF invokes this before DIIS. Report the measured candidate and
            # explicitly distinguish it from the final converged CC energy.
            correction = float(solver.energy(env["t1new"], env["t2new"], env["eris"]))
            delta = correction - float(env["eccsd"])
            norm = float(np.linalg.norm(solver.amplitudes_to_vector(env["t1new"], env["t2new"])
                                        - solver.amplitudes_to_vector(env["t1"], env["t2"])))
            record(int(env["istep"]) + 1, reference + correction, delta=delta, norm=norm,
                   stage="pre_diis_amplitude_update")

        solver.callback = callback
        correction, t1, t2 = solver.kernel()
        total = reference + float(correction)
        converged = bool(solver.converged) and math.isfinite(total)
        wavefunction_arrays.update(t1=t1, t2=t2)
        diagnostics.update(ccsd_energy_hartree=_finite(total),
                           t1_diagnostic=_finite(np.linalg.norm(t1) / math.sqrt(mf.mol.nelectron - 2 * frozen)),
                           energy_tolerance_hartree=tolerance, amplitude_tolerance=solver.conv_tol_normt,
                           convergence_test="CC energy change and amplitude-update norm; (T) is noniterative.")
        if method == "CCSD(T)":
            triples = None
            if converged:
                emit("stage", {"name": "triples", "message": "Evaluating perturbative triples after converged CCSD."})
                triples = float(solver.ccsd_t())
                total += triples
                converged = math.isfinite(total) and math.isfinite(triples)
            else:
                warnings.append("CCSD did not converge; the requested perturbative triples correction was not evaluated.")
            diagnostics["triples_energy_hartree"] = _finite(triples)
        if spec.get("task") == "eomccsd":
            if converged:
                eom = run_eom(solver, spec, emit, workdir)
                if not eom["converged"]:
                    warnings.append("EOM-EE-CCSD did not converge all requested positive-energy roots; ground-state CCSD data are retained.")
            else:
                warnings.append("Ground-state CCSD did not converge; EOM-EE-CCSD was not run.")
    elif method == "CISD":
        solver = ci.CISD(mf, frozen=frozen or None)
        solver.max_cycle, solver.conv_tol = cycles, tolerance
        solver.max_memory = int(spec.get("memory_mb", 2000))
        correction, vector = solver.kernel()
        total = reference + float(correction)
        converged = bool(solver.converged) and math.isfinite(total)
        diagnostics.update(reference_configuration_weight=_finite(vector[0] ** 2),
                           energy_tolerance_hartree=tolerance,
                           convergence_test="PySCF CISD Davidson convergence.")
        wavefunction_arrays["ci"] = vector
        if converged:
            occupations = np.linalg.eigvalsh(solver.make_rdm1())[::-1]
            occupation_space = "all"
        warnings.append("CISD truncation is not size extensive; separated-fragment energies require care.")
    elif method == "FCI":
        solver = fci.FCI(mf, singlet=True)
        solver.max_cycle, solver.conv_tol = cycles, tolerance
        solver.max_memory = int(spec.get("memory_mb", 2000))
        total, vector = solver.kernel()
        total = float(total)
        converged = bool(np.all(solver.converged)) and math.isfinite(total)
        wavefunction_arrays["ci"] = vector
        diagnostics.update(energy_tolerance_hartree=tolerance, spin_square=_finite(solver.spin_square(
            vector, mf.mo_coeff.shape[1], mf.mol.nelec)[0]), convergence_test="PySCF singlet FCI convergence.")
        if converged:
            occupations = np.linalg.eigvalsh(solver.make_rdm1(vector, mf.mo_coeff.shape[1], mf.mol.nelec))[::-1]
            occupation_space = "all"
        warnings.append("Full CI is exact only within the selected finite orbital basis and electronic Hamiltonian.")
    else:
        active = space["active"]
        factory = mcscf.CASSCF if method == "CASSCF" else mcscf.CASCI
        solver = factory(mf, active["orbitals"], active["electrons"])
        solver.fcisolver = fci.direct_spin0.FCI(mf.mol)
        solver.fcisolver.spin = 0
        solver.fcisolver.max_cycle, solver.fcisolver.conv_tol = cycles, tolerance
        solver.fcisolver.max_memory = int(spec.get("memory_mb", 2000))
        solver.max_memory = int(spec.get("memory_mb", 2000))
        solver.chkfile = str(Path(workdir) / "correlated-checkpoint.chk")
        if method == "CASSCF":
            solver.max_cycle_macro, solver.conv_tol = cycles, tolerance
            solver.conv_tol_grad = math.sqrt(tolerance)
            previous = [None]

            def callback(env):
                # Micro steps retain the previous macro energy. Keep each
                # distinct measured macro energy once rather than fabricating
                # a smooth path through the orbital optimizer.
                energy = _finite(env.get("e_tot"))
                if energy is not None and energy != previous[0]:
                    record(env.get("imacro", len(records) + 1), energy,
                           delta=None if previous[0] is None else energy - previous[0],
                           norm=env.get("norm_gorb0"), stage="casscf_macro_energy")
                    previous[0] = energy

            solver.callback = callback
            diagnostics["orbital_gradient_tolerance"] = solver.conv_tol_grad
        solver.kernel()
        total = float(solver.e_tot)
        converged = bool(solver.converged) and bool(np.all(solver.fcisolver.converged)) and math.isfinite(total)
        wavefunction_arrays.update(ci=solver.ci, optimized_mo_coeff=solver.mo_coeff)
        diagnostics.update(active_energy_hartree=_finite(solver.e_cas),
                           energy_tolerance_hartree=tolerance,
                           spin_square=_finite(solver.fcisolver.spin_square(solver.ci, solver.ncas, solver.nelecas)[0]),
                           convergence_test="Active CI convergence and CASSCF orbital convergence when requested.")
        if converged:
            density = solver.fcisolver.make_rdm1(solver.ci, solver.ncas, solver.nelecas)
            occupations = np.linalg.eigvalsh(density)[::-1]
            occupation_space = "active"
        warnings.append("The selected active space is an explicit modeling choice; orbitals outside it are not fully correlated.")
        if method == "CASSCF":
            warnings.append("The CASSCF solution may be a local stationary point; alternate active spaces and initial orbitals need comparison.")
        if method == "NEVPT2":
            diagnostics.update(perturbation_reference="CASCI", casci_energy_hartree=_finite(total),
                               contraction="strongly contracted NEVPT2", nevpt2_correction_hartree=None)
            if converged:
                emit("stage", {"name": "nevpt2", "message": "Evaluating strongly contracted NEVPT2 on the singlet CASCI reference."})
                correction = float(mrpt.NEVPT(solver, root=0).kernel())
                total += correction
                converged = math.isfinite(total) and math.isfinite(correction)
                diagnostics["nevpt2_correction_hartree"] = _finite(correction)
                # NEVPT may canonicalize the CASCI reference again. Store a
                # consistent final CI/orbital pair if that rotated the basis.
                wavefunction_arrays.update(ci=solver.ci, optimized_mo_coeff=solver.mo_coeff)
            else:
                warnings.append("CASCI did not converge; the NEVPT2 correction was not evaluated.")
            warnings.append("NEVPT2 uses a CASCI reference in the selected canonical RHF active space; CASSCF orbital optimization is not performed.")

    if not converged:
        warnings.append(f"{method} did not converge. The retained energy and amplitudes are intermediate results.")
    if occupations is not None:
        occupations = np.asarray(occupations)
        if not np.isfinite(occupations).all():
            occupations = None
    np.savez_compressed(Path(workdir) / "correlation-wavefunction.npz", **wavefunction_arrays)
    emit("stage", {"name": "correlation_complete", "method": method, "converged": converged,
                   "energy_hartree": _finite(total), "reference_energy_hartree": reference})
    return {
        "method": method, "reference": "RHF", "converged": converged,
        "reference_energy_hartree": reference, "total_energy_hartree": _finite(total),
        "correlation_energy_hartree": _finite(None if total is None else total - reference),
        "frozen_orbitals": frozen, "active_space": space["active"], "diagnostics": diagnostics,
        "natural_occupations": occupations.tolist() if occupations is not None else None,
        "natural_occupations_space": occupation_space, "iterations": records,
        "wavefunction_artifact": "correlation-wavefunction.npz", "warnings": warnings,
        "wavefunction_contents": {
            "MP2": "MP2 doubles amplitudes in the RHF orbital basis.",
            "CCSD": "CCSD singles and doubles amplitudes in the RHF orbital basis.",
            "CCSD(T)": "CCSD singles and doubles amplitudes; (T) contributes an energy correction, not stored triples amplitudes.",
            "CISD": "CISD coefficients in the RHF orbital basis.",
            "FCI": "Singlet full-CI coefficients in the RHF orbital basis.",
            "CASCI": "CASCI coefficients and canonicalized molecular orbitals.",
            "CASSCF": "Active-space CI coefficients and optimized molecular orbitals.",
            "NEVPT2": "CASCI reference coefficients and orbitals; the NEVPT2 correction is recorded as an energy, not a full perturbed wavefunction.",
        }[method],
        "eom": eom,
        "orbital_and_density_note": "Displayed canonical orbitals, density, dipole, and populations are the RHF reference; correlated wavefunction arrays are stored separately.",
    }
