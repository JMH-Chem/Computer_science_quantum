"""Closed-shell correlation methods on a recorded RHF reference.

The reference orbitals are deliberately retained by the caller. A correlated
total energy does not turn the RHF density or canonical orbital energies into
correlated observables. Full/active CI spaces are bounded before transformation.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from pyscf import cc, ci, fci, mcscf, mp, mrpt
from scipy.constants import physical_constants

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
    energies, vectors = response.kernel(nroots=requested, imds=intermediates)
    energies = np.atleast_1d(np.asarray(energies, dtype=float))
    vectors = np.atleast_2d(np.asarray(vectors))
    flags = np.atleast_1d(response.converged)
    matvec, _ = response.gen_matvec(intermediates)
    # PySCF's Davidson solver defaults to sqrt(energy tolerance) for its
    # eigenvector residual. Evaluate and record that residual independently.
    residual_tolerance = math.sqrt(response.conv_tol)
    states = []
    for index, energy in enumerate(energies):
        vector = vectors[index]
        norm = float(np.linalg.norm(vector))
        if norm > 0 and np.isfinite(vector).all():
            normalized = vector / norm
            residual = _finite(np.linalg.norm(np.asarray(matvec([normalized]))[0] - energy * normalized))
        else:
            residual = None
        valid = bool(index < len(flags) and flags[index] and math.isfinite(float(energy)) and energy > 0
                     and residual is not None and residual < residual_tolerance)
        states.append({"state": index + 1, "energy_ev": _finite(energy * HARTREE_EV),
                       "converged": valid, "residual_norm": residual})
    converged = len(states) == requested and all(state["converged"] for state in states)
    np.savez_compressed(Path(workdir) / "eom-wavefunction.npz", energies_hartree=energies, right_vectors=vectors)
    emit("stage", {"name": "eom_complete", "converged": converged, "states": states})
    return {"reference": "CCSD", "spin": "singlet", "converged": converged, "states": states,
            "requested_states": requested, "residual_tolerance": residual_tolerance,
            "energy_tolerance_hartree": response.conv_tol, "max_cycle": response.max_cycle,
            "wavefunction_artifact": "eom-wavefunction.npz",
            "notes": ["Vertical singlet excitation energies are differences from the CCSD ground state.",
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
