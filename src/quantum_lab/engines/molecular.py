"""Molecular PySCF adapter with actual iteration events and reproducible artifacts.

Coordinates are Angstrom; energies in the public result are Hartree unless the
field says otherwise. Numerical SCF iterations are not physical time evolution.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path

import basis_set_exchange as bse
import numpy as np
import pyscf
from pyscf import dft, gto, lib, scf
from pyscf.gto.basis import bse as pyscf_bse
from pyscf.tools import cubegen
from scipy.constants import physical_constants

from .correlated import METHODS as CORRELATED_METHODS
from .correlated import run_correlation
from .properties import analyze_properties, run_frequencies

HARTREE_EV = physical_constants["Hartree energy in eV"][0]
Emit = Callable[[str, dict], None]


def _number(value):
    if value is None:
        return None
    n = float(value)
    return n if np.isfinite(n) else None


def _json_write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _atoms(mol: gto.Mole) -> list[dict]:
    return [
        {"symbol": mol.atom_symbol(i), "position": pos.tolist()}
        for i, pos in enumerate(mol.atom_coords(unit="Angstrom"))
    ]


def resolve_basis(spec: dict, workdir: Path) -> tuple[dict, dict, dict]:
    """Resolve BSE data, preserving definitions rather than only basis aliases.

    PySCF's BSE converters are used against the project's pinned PySCF version;
    they retain contractions and correctly distinguish the orbital/ECP sections.
    An ECP is never added silently to a request that did not select one.
    """
    symbols = sorted({a["symbol"] for a in spec["atoms"]})
    selected = spec.get("basis", "sto-3g")
    selected_ecp = spec.get("ecp")
    orbital_basis, ecps, definitions, citations = {}, {}, {}, []
    for symbol in symbols:
        name = selected.get(symbol) if isinstance(selected, dict) else selected
        if not name:
            raise ValueError(f"No orbital basis selected for {symbol}.")
        try:
            data = bse.get_basis(name, elements=[symbol])
        except (KeyError, RuntimeError) as exc:
            raise ValueError(f"Basis {name!r} has no available definition for {symbol}: {exc}") from exc
        converted, _ = pyscf_bse._orbital_basis(data)
        if symbol not in converted:
            raise ValueError(f"{name!r} is not an orbital basis for {symbol}.")
        orbital_basis[symbol] = converted[symbol]
        atomic_data = data["elements"][str(gto.charge(symbol))]
        required_core = int(atomic_data.get("ecp_electrons", 0))
        ecp_name = selected_ecp.get(symbol) if isinstance(selected_ecp, dict) else selected_ecp
        ecp_data = None
        actual_core = 0
        if ecp_name:
            ecp_data = bse.get_basis(ecp_name, elements=[symbol])
            converted_ecp = pyscf_bse._ecp_basis(ecp_data)
            if symbol in converted_ecp:
                ecps[symbol] = converted_ecp[symbol]
                actual_core = int(converted_ecp[symbol][0])
        if actual_core != required_core:
            raise ValueError(
                f"Incompatible basis/ECP for {symbol}: orbital basis {name} expects "
                f"{required_core} removed core electrons, selected ECP provides {actual_core}. "
                "Select the matching ECP explicitly, or use an all-electron basis."
            )
        definitions[symbol] = {"basis": data, "ecp": ecp_data, "ecp_core_electrons": actual_core}
        citations.append(bse.get_references(name, elements=[symbol], fmt="txt"))
        if ecp_name and ecp_name != name:
            citations.append(bse.get_references(ecp_name, elements=[symbol], fmt="txt"))
    provenance = {
        "provider": "Basis Set Exchange",
        "version": version("basis-set-exchange"),
        "source": "https://www.basissetexchange.org/",
        "elements": definitions,
        "resolved_pyscf_basis": orbital_basis,
        "resolved_pyscf_ecp": ecps,
    }
    _json_write(workdir / "basis.json", provenance)
    (workdir / "basis-references.txt").write_text("\n\n".join(dict.fromkeys(citations)), encoding="utf-8")
    return orbital_basis, ecps, provenance


def make_molecule(spec: dict, workdir: Path) -> tuple[gto.Mole, dict]:
    basis, ecps, provenance = resolve_basis(spec, workdir)
    mol = gto.M(
        atom=[(a["symbol"], a["position"]) for a in spec["atoms"]],
        unit="Angstrom", basis=basis, ecp=ecps, charge=int(spec.get("charge", 0)),
        spin=int(spec.get("spin", 0)), verbose=4, max_memory=int(spec.get("memory_mb", 2000)),
    )
    if spec.get("restricted", True) and mol.spin != 0:
        raise ValueError("Restricted v1 calculations require spin=0; use unrestricted HF/DFT for open shells.")
    overlap = mol.intor_symmetric("int1e_ovlp")
    eig = np.linalg.eigvalsh(overlap)
    if float(eig[0]) < 1e-8:
        raise ValueError(
            f"Near-linearly-dependent basis (smallest overlap eigenvalue={eig[0]:.3g}). "
            "Change the basis or molecular geometry; no basis functions were silently removed."
        )
    return mol, provenance


def _mean_field(mol: gto.Mole, spec: dict, workdir: Path, callback):
    restricted = spec.get("restricted", True)
    if spec.get("method", "HF") == "DFT":
        xc = spec.get("functional", "PBE")
        dft.libxc.parse_xc(xc)
        if dft.libxc.needs_laplacian(xc):
            raise ValueError("Laplacian-dependent functionals are not supported by this molecular adapter.")
        dft.libxc.test_deriv_order(xc, 2, raise_error=True)
        if spec.get("task") in {"optimize", "tda", "tddft", "frequencies"} and dft.libxc.is_nlc(xc):
            raise ValueError("Nonlocal-correlation functionals are not validated for this task in v1.")
        mf = dft.RKS(mol) if restricted else dft.UKS(mol)
        mf.xc = xc
        mf.grids.level = int(spec.get("grid_level", 2))
    else:
        mf = scf.RHF(mol) if restricted else scf.UHF(mol)
    mf.conv_tol = float(spec.get("conv_tol", 1e-9))
    mf.max_cycle = int(spec.get("max_cycle", 80))
    mf.max_memory = int(spec.get("memory_mb", 2000))
    mf.chkfile = str(workdir / "checkpoint.chk")
    mf.callback = callback
    return mf


class _OptimizationSCFFailure(RuntimeError):
    def __init__(self, mol):
        self.mol = mol.copy()
        super().__init__("SCF failed to converge during geometry optimization.")


def run(spec: dict, emit: Emit, workdir: Path) -> dict:
    """Execute molecular reference, correlation and singlet response calculations."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    lib.num_threads(int(spec.get("threads", 2)))
    _json_write(workdir / "engine-spec.json", spec)
    task = spec.get("task", "single_point")
    method = spec.get("method", "HF")
    correlated = method in CORRELATED_METHODS
    if method not in {"HF", "DFT"} | CORRELATED_METHODS:
        raise ValueError(f"Unsupported molecular method: {method}")
    if task not in {"single_point", "optimize", "tda", "tddft", "tdhf", "frequencies", "eomccsd"}:
        raise ValueError(f"Unsupported molecular task: {task}")
    if correlated and (task not in {"single_point", "eomccsd"}
                       or not spec.get("restricted", True) or spec.get("spin", 0)):
        raise ValueError("Correlated methods support closed-shell RHF single-point calculations.")
    if task == "eomccsd" and method != "CCSD":
        raise ValueError("EOM-EE-CCSD requires method=CCSD.")
    if task in {"tda", "tddft", "tdhf"} and (
        not spec.get("restricted", True) or spec.get("spin", 0) != 0
        or method not in ({"HF", "DFT"} if task == "tda" else {"HF"} if task == "tdhf" else {"DFT"})
    ):
        raise ValueError("Singlet response requires closed-shell RHF for TDHF, RKS for TDDFT, or either for TDA.")
    emit("stage", {"name": "basis", "message": "Resolving basis definitions and building molecular integrals."})
    mol, basis_provenance = make_molecule(spec, workdir)
    context = {"optimization_step": None}
    iteration_count = 0

    def on_scf(env):
        nonlocal iteration_count
        iteration_count += 1
        event = {
            "iteration": int(env["cycle"]) + 1,
            "energy_hartree": _number(env.get("e_tot")),
            "delta_energy": _number(env.get("e_tot", 0) - env.get("last_hf_e", 0)),
            "gradient_norm": _number(env.get("norm_gorb")),
            "density_change": _number(env.get("norm_ddm")),
        }
        if context["optimization_step"] is not None:
            event["optimization_step"] = context["optimization_step"]
        emit("scf_iteration", event)

    emit("stage", {"name": "scf", "message": "Solving the self-consistent electronic ground state."})
    mf = _mean_field(mol, spec, workdir, on_scf)
    mf.kernel()
    warnings = []
    optimization = []
    geometry_converged = None
    if not mf.converged:
        warnings.append("SCF did not converge. Values are intermediate; downstream calculations were not run.")
    elif task == "optimize":
        from pyscf.geomopt import geometric_solver

        emit("stage", {"name": "optimization", "message": "Optimizing nuclear coordinates; steps are not time."})
        context["optimization_step"] = 1

        def on_geometry(env):
            step = int(env["self"].cycle)
            item = {
                "step": step,
                "energy_hartree": _number(env["energy"]),
                "atoms": _atoms(env["mol"]),
                "gradient_norm": _number(np.linalg.norm(env["gradients"])),
                "scf_converged": bool(env["g_scanner"].converged),
            }
            optimization.append(item)
            emit("optimization_step", item)
            context["optimization_step"] = step + 1
            if not env["g_scanner"].converged:
                raise _OptimizationSCFFailure(env["mol"])

        try:
            geometry_converged, optimized_mol = geometric_solver.kernel(
                mf, callback=on_geometry, maxsteps=int(spec.get("max_steps", 50)),
                assert_convergence=True,
            )
        except _OptimizationSCFFailure as exc:
            geometry_converged, optimized_mol = False, exc.mol
            warnings.append(str(exc))
        if not geometry_converged:
            warnings.append("Geometry optimization did not converge; the last evaluated geometry is retained.")
        # A new final SCF aligns the checkpoint, energy and orbital arrays with
        # exactly the returned geometry (the optimization uses a copied scanner).
        context["optimization_step"] = len(optimization) + 1
        emit("stage", {"name": "final_scf", "message": "Recomputing the electronic state at the final geometry."})
        mf = _mean_field(optimized_mol, spec, workdir, on_scf)
        mf.kernel()
        mol = optimized_mol

    arrays = {
        "mo_coeff": np.asarray(mf.mo_coeff), "mo_occ": np.asarray(mf.mo_occ),
        "mo_energy": np.asarray(mf.mo_energy), "density": np.asarray(mf.make_rdm1()),
    }
    excitations = []
    excited_converged = None
    response_settings = None
    if mf.converged and task in {"tda", "tddft", "tdhf"}:
        emit("stage", {"name": task, "message": "Solving frequency-domain linear response, not a time trajectory."})
        td = mf.TDA() if task == "tda" else mf.TDHF() if task == "tdhf" else mf.TDDFT()
        td.singlet = True
        occupied = int(np.count_nonzero(mf.mo_occ > 0))
        max_states = occupied * (mol.nao_nr() - occupied)
        requested_states = int(spec.get("nstates", 5))
        if requested_states > max_states:
            raise ValueError(f"Requested {requested_states} states, but this basis has only {max_states} occupied-virtual pairs.")
        td.nstates = requested_states
        td.max_cycle = int(spec.get("max_cycle", 80))
        td.conv_tol = float(spec.get("conv_tol", 1e-9))
        # PySCF's response solver tests correction *squared norms* against
        # lindep. The default 1e-12 can discard corrections at norm 1e-6,
        # before a requested residual tolerance of 1e-9 is reached. Keep the
        # user's residual tolerance and iteration limit. A squared-norm cutoff
        # below floating-point epsilon can admit nearly dependent corrections
        # and destroy Davidson subspace orthogonality (notably at 1e-22).
        # Bound this supported setting by machine precision; convergence still
        # requires the user's actual residual tolerance and may honestly fail.
        td.lindep = min(float(td.lindep), max(td.conv_tol**2, np.finfo(float).eps))
        response_settings = {
            "conv_tol": td.conv_tol, "lindep": td.lindep,
            "max_cycle": td.max_cycle,
            "convergence_test": "response eigenvector residual norm",
        }
        emit("stage", {
            "name": "response_solver", "message": "Response solver preserves the requested residual tolerance.",
            **response_settings,
        })
        td.kernel()
        roots_ok = np.asarray(td.converged, dtype=bool)
        excitation_energies = np.asarray(td.e, dtype=float)
        excited_converged = bool(
            len(excitation_energies) == requested_states
            and roots_ok.size == requested_states and roots_ok.all()
            and np.isfinite(excitation_energies).all() and (excitation_energies > 0).all()
        )
        strengths = np.asarray(td.oscillator_strength(gauge="length"))
        for i, energy in enumerate(excitation_energies):
            item = {
                "state": i + 1,
                "energy_ev": _number(energy * HARTREE_EV),
                "oscillator_strength": _number(strengths[i]),
                "converged": bool(roots_ok[i]) if i < len(roots_ok) else False,
                "nto_weights": [],
            }
            if item["converged"] and energy > 0:
                weights, coefficients = td.get_nto(state=i + 1, verbose=0)
                item["nto_weights"] = np.asarray(weights).tolist()
                arrays[f"nto_hole_{i}"] = np.asarray(coefficients[:, :occupied])
                arrays[f"nto_particle_{i}"] = np.asarray(coefficients[:, occupied:])
                arrays[f"nto_weights_{i}"] = np.asarray(weights)
            excitations.append(item)
        if not excited_converged:
            warnings.append("Excited-state solver did not converge all requested positive-energy roots.")
        arrays["excitation_energies_hartree"] = excitation_energies
        arrays["oscillator_strengths"] = strengths

    correlation, eom = None, None
    if mf.converged and correlated:
        correlation = run_correlation(mf, spec, emit, workdir)
        eom = correlation.pop("eom", None)
        warnings.extend(correlation["warnings"])
    if correlated:
        warnings.append("Orbitals, density, dipole and populations describe the RHF reference, not the correlated wavefunction.")
    properties, vibrations = None, None
    if mf.converged:
        properties = analyze_properties(mf)
        if correlated:
            properties["reference_model"] = "RHF reference (not correlated)"
        warnings.extend(properties["warnings"])
        if task == "frequencies":
            vibrations = run_frequencies(mf, spec, emit, workdir)
            warnings.extend(vibrations["warnings"])

    np.savez_compressed(workdir / "orbitals.npz", **arrays)
    scf.chkfile.save_mol(mol, str(workdir / "checkpoint.chk"))
    atoms = _atoms(mol)
    xyz = f"{len(atoms)}\nQuantum Lab geometry; Angstrom\n" + "\n".join(
        f"{a['symbol']} " + " ".join(f"{v:.12f}" for v in a["position"]) for a in atoms
    ) + "\n"
    (workdir / "geometry.xyz").write_text(xyz, encoding="utf-8")
    converged = bool(mf.converged)
    if task == "optimize":
        converged = converged and bool(geometry_converged)
    if task in {"tda", "tddft", "tdhf"}:
        converged = converged and bool(excited_converged)
    if correlated:
        converged = converged and bool(correlation and correlation["converged"])
    if task == "eomccsd":
        converged = converged and bool(eom and eom["converged"])
    if task == "frequencies":
        converged = converged and bool(vibrations and vibrations["converged"])
    for warning in warnings:
        emit("warning", {"message": warning})
    total_dm = arrays["density"].sum(axis=0) if arrays["density"].ndim == 3 else arrays["density"]
    dipole = mf.dip_moment(mol, arrays["density"], unit="Debye", verbose=0)
    result = {
        "converged": converged,
        "scf_converged": bool(mf.converged),
        "geometry_converged": geometry_converged,
        "excited_converged": excited_converged,
        "energy_hartree": (correlation["total_energy_hartree"] if correlation else None)
        if correlated else _number(mf.e_tot),
        "reference_energy_hartree": _number(mf.e_tot) if correlated else None,
        "correlation_energy_hartree": correlation["correlation_energy_hartree"] if correlation else None,
        "correlation": correlation,
        "eom": eom,
        "properties": properties, "vibrations": vibrations,
        "orbital_reference": "RHF reference (not correlated)" if correlated else method,
        "density_reference": "RHF reference (not correlated)" if correlated else method,
        "dipole_reference": "RHF reference (not correlated)" if correlated else method,
        "atoms": atoms, "lattice": None,
        "engine": {"name": "PySCF molecular", "version": pyscf.__version__},
        "method": spec.get("method", "HF"), "task": task,
        "orbital_energies_hartree": arrays["mo_energy"].tolist(),
        "occupations": arrays["mo_occ"].tolist(),
        "dipole_debye": np.asarray(dipole).tolist(),
        "electron_count": int(mol.nelectron), "basis_functions": int(mol.nao_nr()),
        "integrated_electrons_ao": float(np.einsum("ij,ji->", total_dm, mol.intor_symmetric("int1e_ovlp"))),
        "iteration_count": iteration_count,
        "optimization": optimization, "excitations": excitations,
        "warnings": warnings,
        "provenance": {
            "basis_provider": basis_provenance["provider"],
            "basis_provider_version": basis_provenance["version"],
            "basis_definition_artifact": "basis.json", "basis_citations_artifact": "basis-references.txt",
            "libxc_version": dft.libxc.__version__,
            "functional_references": dft.libxc.xc_reference(spec.get("functional", "PBE"))
            if spec.get("method") == "DFT" else [],
            "numpy_version": np.__version__,
            "nto_convention": "PySCF X-only transition amplitudes normalized; Y amplitudes omitted; leading pair export.",
            "oscillator_strength_gauge": "length",
            "response_solver": response_settings,
            "reference_method": "RHF" if correlated else method,
            "time_dynamics": False,
        },
    }
    _json_write(workdir / "molecular-result.json", result)
    return result


def generate_volume(parent_dir: Path, request: dict, output_dir: Path) -> dict:
    """Generate a CUBE lazily from a converged job's retained scientific arrays."""
    parent_dir, output_dir = Path(parent_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_result_path = parent_dir / "molecular-result.json"
    result = json.loads(source_result_path.read_text(encoding="utf-8"))
    if not result.get("converged"):
        raise ValueError("Volume export requires a converged parent calculation.")
    spec = json.loads((parent_dir / "engine-spec.json").read_text(encoding="utf-8"))
    lib.num_threads(int(spec.get("threads", 2)))
    mol = scf.chkfile.load_mol(str(parent_dir / "checkpoint.chk"))
    mol.verbose = 0
    grid = int(request.get("grid", 40))
    if not 16 <= grid <= 100:
        raise ValueError("CUBE grid must be between 16 and 100 points per axis.")
    kind, index, spin = request.get("kind", "density"), int(request.get("index", 0)), int(request.get("spin", 0))
    if index < 0 or spin not in (0, 1):
        raise ValueError("Orbital/state index must be nonnegative and spin must be 0 (alpha) or 1 (beta).")
    name = f"{kind}-{index}-spin{spin}-{grid}.cube"
    filename = output_dir / name
    # PySCF's default 3-bohr margin visibly truncates even H2/STO-3G tails.
    # Retain a larger documented box; grid resolution still needs convergence.
    margin = 6.0
    meta = {"kind": kind, "index": index, "spin": spin, "grid": grid, "margin_bohr": margin,
            "reference": result.get("density_reference" if kind == "density" else "orbital_reference", spec.get("method"))}
    with np.load(parent_dir / "orbitals.npz", allow_pickle=False) as data:
        if kind == "density":
            density = data["density"]
            if density.ndim == 3:
                density = density.sum(axis=0)
            field = cubegen.density(mol, str(filename), density, nx=grid, ny=grid, nz=grid, margin=margin)
            meta["density_kind"] = ("RHF reference electron density, not correlated density"
                                    if result.get("correlation") else "total electron density, alpha + beta")
            # Trapezoid integration uses CUBE's actual volume and endpoint grid.
            cube = cubegen.Cube(mol, nx=grid, ny=grid, nz=grid, margin=margin)
            integral = field
            for _ in range(3):
                integral = np.trapezoid(integral, dx=1.0 / (grid - 1), axis=0)
            meta["grid_integrated_electrons"] = float(integral * abs(np.linalg.det(cube.box)))
            meta["expected_electrons"] = int(mol.nelectron)
            meta["integration_note"] = "Finite 6-bohr-margin box/grid quadrature; assess grid convergence and tail truncation."
        elif kind == "orbital":
            coeff = data["mo_coeff"]
            if coeff.ndim == 3:
                coeff = coeff[spin]
            elif spin:
                raise ValueError("Restricted orbitals have a single shared spin channel; select spin=0.")
            if index >= coeff.shape[1]:
                raise ValueError(f"Orbital index {index} exceeds the {coeff.shape[1]} available orbitals.")
            field = cubegen.orbital(mol, str(filename), coeff[:, index], nx=grid, ny=grid, nz=grid, margin=margin)
            meta["phase_note"] = "Positive/negative colors show orbital phase, not electric charge."
        elif kind in {"nto_hole", "nto_particle"}:
            if spin:
                raise ValueError("Singlet NTO volumes use spin=0.")
            key = f"{kind}_{index}"
            if key not in data:
                raise ValueError(f"No converged NTOs for zero-based excited-state index {index}.")
            weights = data[f"nto_weights_{index}"]
            field = cubegen.orbital(mol, str(filename), data[key][:, 0], nx=grid, ny=grid, nz=grid, margin=margin)
            meta.update({"state": index + 1, "pair": 1, "weight": float(weights[0]),
                         "convention": result["provenance"]["nto_convention"]})
        else:
            raise ValueError(f"Unsupported molecular volume kind: {kind}")
    meta["value_min"] = float(np.min(field))
    meta["value_max"] = float(np.max(field))
    if result.get("correlation"):
        # CUBE files can be shared without their JSON sidecar: retain the
        # reference distinction in the standalone artifact too.
        lines = filename.read_text(encoding="utf-8").splitlines()
        lines[0] = f"Quantum Lab: RHF reference {kind}; not a correlated density/orbital"
        filename.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _json_write(output_dir / (name + ".json"), meta)
    return {
        "converged": True, "artifact": name, "volume": meta,
        "energy_hartree": result["energy_hartree"], "atoms": result["atoms"], "lattice": None,
        "engine": result["engine"],
    }
