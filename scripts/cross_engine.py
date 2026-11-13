#!/usr/bin/env python3
"""Optional, isolated Psi4 comparison; never imports Psi4 into the application.

    uv run python scripts/cross_engine.py --psi4-python /path/to/psi4/bin/python

Each run preserves its input, BSE definitions, engine outputs, settings and logs.
Missing engines, nonconvergence and incompatible inputs cannot count as a pass.
This is a small molecular verification suite, not validation of every method or
proof of agreement with experiment. DFT quadratures are explicitly different.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

SCHEMA = 1
H2 = [("H", [0, 0, 0]), ("H", [0, 0, 0.74])]
WATER = [("O", [0, 0, 0]), ("H", [0, -0.757, 0.587]), ("H", [0, 0.757, 0.587])]
FORMALDEHYDE = [("C", [0, 0, 0]), ("O", [0, 0, 1.208]),
                ("H", [0, 0.943, -0.588]), ("H", [0, -0.943, -0.588])]


def cases() -> dict[str, dict]:
    """Acceptance criteria are declared before any numerical calculation."""
    def case(atoms, basis, method="HF", frozen=0, grid=None):
        spec = {"domain": "molecule", "task": "single_point", "method": method,
                "basis": basis, "atoms": [{"symbol": s, "position": p} for s, p in atoms],
                "charge": 0, "spin": 0, "restricted": True, "ecp": None,
                "threads": 1, "memory_mb": 1500, "conv_tol": 1e-11, "max_cycle": 100,
                "frozen_orbitals": frozen, "functional": "PBE", "grid_level": grid or 5}
        criteria = {"energy_hartree": 5e-6 if method == "DFT" else 1e-8 if method == "HF" else 1e-7,
                    "nuclear_repulsion_hartree": 1e-10}
        if method in {"MP2", "CCSD"}:
            criteria.update(reference_energy_hartree=1e-8, correlation_energy_hartree=1e-7)
        return {"spec": spec, "criteria": criteria,
                "comparison": "different_quadratures" if method == "DFT" else "matched_hamiltonian",
                "psi4_grid": {"radial_points": 99 if grid == 5 else 175,
                              "spherical_points": 590 if grid == 5 else 974,
                              "pruning_scheme": "FLAT"} if method == "DFT" else None}

    return {
        "h2_rhf_sto3g": case(H2, "sto-3g"),
        "water_rhf_def2svp": case(WATER, "def2-svp"),
        "formaldehyde_rhf_def2svp": case(FORMALDEHYDE, "def2-svp"),
        "water_mp2_def2svp": case(WATER, "def2-svp", "MP2"),
        "water_mp2_frozen1_def2svp": case(WATER, "def2-svp", "MP2", frozen=1),
        "water_ccsd_sto3g": case(WATER, "sto-3g", "CCSD"),
        "water_pbe_grid5": case(WATER, "def2-svp", "DFT", grid=5),
        "water_pbe_grid6": case(WATER, "def2-svp", "DFT", grid=6),
    }


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_case(name: str, definition: dict, folder: Path, runner_sha256: str | None = None) -> dict:
    import basis_set_exchange as bse
    from pyscf import gto
    from pyscf.lib import param

    from quantum_lab.models import JobSpec

    JobSpec.model_validate(definition["spec"])
    folder.mkdir(parents=True, exist_ok=False)
    spec = definition["spec"]
    symbols = sorted({a["symbol"] for a in spec["atoms"]})
    basis = bse.get_basis(spec["basis"], elements=symbols)
    if any(data.get("ecp_electrons", 0) for data in basis["elements"].values()):
        raise ValueError("This suite requires all-electron basis definitions.")
    write_json(folder / "shared-basis.json", basis)
    (folder / "shared-basis.gbs").write_text(bse.write_formatted_basis_str(basis, fmt="psi4"), encoding="utf-8")
    (folder / "basis-references.txt").write_text(
        bse.get_references(spec["basis"], elements=symbols, fmt="txt"), encoding="utf-8")
    payload = {"schema_version": SCHEMA, "case": name, **definition,
               "expected_electrons": sum(gto.charge(a["symbol"]) for a in spec["atoms"]) - spec["charge"],
               "angstrom_per_bohr": param.BOHR,
               "geometry_bohr": [[float(v) / param.BOHR for v in a["position"]] for a in spec["atoms"]],
               "coordinate_note": "Psi4 receives the PySCF-converted Bohr geometry explicitly; neither engine reorients it.",
               "runner_sha256": runner_sha256 or file_digest(Path(__file__)),
               "basis_provider": {"name": "Basis Set Exchange", "version": importlib.metadata.version("basis-set-exchange")},
               "basis_sha256": file_digest(folder / "shared-basis.json"),
               "gbs_sha256": file_digest(folder / "shared-basis.gbs"),
               "common_settings": {"spherical": True, "ecp": False, "density_fitting": False,
                                   "charge": spec["charge"], "spin": spec["spin"],
                                   "frozen_orbitals": spec["frozen_orbitals"],
                                   "reference": "RKS" if spec["method"] == "DFT" else "RHF"}}
    payload["input_sha256"] = digest(payload)
    write_json(folder / "input.json", payload)
    return payload


def read_input(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    unsigned = {k: v for k, v in payload.items() if k != "input_sha256"}
    if payload.get("schema_version") != SCHEMA or digest(unsigned) != payload.get("input_sha256"):
        raise ValueError("Input provenance checksum mismatch.")
    for name, key in [("shared-basis.json", "basis_sha256"), ("shared-basis.gbs", "gbs_sha256")]:
        if file_digest(path.parent / name) != payload[key]:
            raise ValueError(f"Basis provenance checksum mismatch: {name}")
    return payload


def app_worker(payload: dict, input_path: Path, output: Path) -> dict:
    import pyscf
    from pyscf import scf

    import quantum_lab
    from quantum_lab import __version__
    from quantum_lab.engines.molecular import run
    from quantum_lab.provenance import execution_environment

    spec = payload["spec"]
    environment = execution_environment()
    imported_modules = {name: str(Path(module.__file__).resolve()) for name, module in sys.modules.items()
                        if (name == "quantum_lab" or name.startswith("quantum_lab."))
                        and getattr(module, "__file__", None)}
    source_record = {"run_environment": environment,
                     "package_path": str(Path(quantum_lab.__file__).resolve().parent),
                     "imported_module_paths": imported_modules}
    write_json(output / "environment.json", source_record)
    with (output / "events.jsonl").open("w", encoding="utf-8") as event_file:
        def emit(kind, data):
            event_file.write(json.dumps({"type": kind, "data": data}, allow_nan=False) + "\n")
        started = time.perf_counter()
        result = run(spec, emit, output)
        elapsed = time.perf_counter() - started
    final_environment = execution_environment()
    source_record["source_files_unchanged"] = environment["source_sha256"] == final_environment["source_sha256"]
    source_record["finish_source_sha256"] = final_environment["source_sha256"]
    write_json(output / "environment.json", source_record)
    if not source_record["source_files_unchanged"]:
        raise ValueError("Application source files changed during calculation; the run is not reproducible.")
    saved = json.loads((output / "basis.json").read_text())
    shared = json.loads((input_path.parent / "shared-basis.json").read_text())
    actual = {}
    for entry in saved["elements"].values():
        if entry["ecp"] is not None or entry["ecp_core_electrons"]:
            raise ValueError("Unexpected ECP in application calculation.")
        actual.update(entry["basis"]["elements"])
    if actual != shared["elements"]:
        raise ValueError("The application's resolved basis differs from the shared BSE definitions.")
    mol = scf.chkfile.load_mol(str(output / "checkpoint.chk"))
    if mol.cart:
        raise ValueError("Application used Cartesian instead of spherical basis functions.")
    write_json(output / "full-result.json", result)
    return {"engine": "Quantum Lab / PySCF", "version": pyscf.__version__, "app_version": __version__,
            "libxc_version": result["provenance"]["libxc_version"],
            "converged": result["converged"], "scf_converged": result["scf_converged"],
            "energy_hartree": result["energy_hartree"],
            "reference_energy_hartree": result.get("reference_energy_hartree"),
            "correlation_energy_hartree": result.get("correlation_energy_hartree"),
            "nuclear_repulsion_hartree": float(mol.energy_nuc()),
            "geometry_bohr": mol.atom_coords().tolist(),
            "basis_functions": result["basis_functions"], "electron_count": result["electron_count"],
            "frozen_orbitals": (result.get("correlation") or {}).get("frozen_orbitals", 0),
            "wall_seconds": elapsed,
            **source_record,
            "settings": {"spec": spec, "spherical": True, "density_fitting": False,
                         "dft_grid": {"level": spec["grid_level"], "radial": "Treutler",
                                      "partition": "original Becke", "pruning": "NWChem"}
                         if spec["method"] == "DFT" else None}}


def psi4_worker(payload: dict, input_path: Path, output: Path) -> dict:
    import psi4

    spec = payload["spec"]
    psi4.core.clean()
    psi4.core.clean_options()
    psi4.set_num_threads(spec["threads"])
    psi4.set_memory(f"{spec['memory_mb']} MB")
    psi4.core.set_output_file(str(output / "psi4.log"), False)
    scratch = output / "scratch"
    scratch.mkdir()
    psi4.core.IOManager.shared_object().set_default_path(str(scratch))
    lines = [f"{spec['charge']} {spec['spin'] + 1}", "units bohr", "symmetry c1", "no_reorient", "no_com"]
    for atom, position in zip(spec["atoms"], payload["geometry_bohr"], strict=True):
        lines.append(atom["symbol"] + " " + " ".join(f"{x:.17g}" for x in position))
    geometry = "\n".join(lines)
    (output / "geometry.psi4").write_text(geometry + "\n", encoding="utf-8")
    molecule = psi4.geometry(geometry)
    psi4.basis_helper((input_path.parent / "shared-basis.gbs").read_text(), name="QLAB_SHARED")
    options = {"basis": "QLAB_SHARED", "puream": True, "reference": "RHF", "scf_type": "PK",
               "mp2_type": "CONV", "cc_type": "CONV", "freeze_core": False,
               "frozen_docc": [spec["frozen_orbitals"]], "e_convergence": spec["conv_tol"],
               "d_convergence": 1e-10, "r_convergence": 1e-9,
               "maxiter": spec["max_cycle"], "fail_on_maxiter": True}
    if spec["method"] == "DFT":
        grid = payload["psi4_grid"]
        options.update(dft_radial_points=grid["radial_points"], dft_spherical_points=grid["spherical_points"],
                       dft_pruning_scheme=grid["pruning_scheme"])
    psi4.set_options(options)
    write_json(output / "options.json", options)
    started = time.perf_counter()
    energy, wavefunction = psi4.energy("pbe" if spec["method"] == "DFT" else spec["method"].lower(),
                                      molecule=molecule, return_wfn=True)
    elapsed = time.perf_counter() - started
    correlated = spec["method"] in {"MP2", "CCSD"}
    reference = float(psi4.variable("SCF TOTAL ENERGY")) if correlated else None
    correlation = float(psi4.variable(f"{spec['method']} CORRELATION ENERGY")) if correlated else None
    write_json(output / "scalar-variables.json", {key: float(value) for key, value in psi4.core.variables().items()
                                                if isinstance(value, (int, float)) and math.isfinite(float(value))})
    measurement = {"engine": "Psi4", "version": psi4.__version__,
                   "converged": True, "scf_converged": True,
                   "convergence_evidence": "Psi4 energy() returned with FAIL_ON_MAXITER enabled; exceptions fail the case.",
                   "energy_hartree": float(energy), "reference_energy_hartree": reference,
                   "correlation_energy_hartree": correlation,
                   "nuclear_repulsion_hartree": float(molecule.nuclear_repulsion_energy()),
                   "geometry_bohr": molecule.geometry().to_array().tolist(),
                   "basis_functions": int(wavefunction.basisset().nbf()),
                   "electron_count": int(wavefunction.nalpha() + wavefunction.nbeta()),
                   "frozen_orbitals": int(sum(wavefunction.frzcpi())) if correlated else 0,
                   "wall_seconds": elapsed, "settings": options}
    psi4.core.clean()
    return measurement


def worker(engine: str, input_path: Path, output: Path) -> int:
    output.mkdir(parents=True, exist_ok=False)
    try:
        payload = read_input(input_path)
        runner_start = file_digest(Path(__file__))
        if runner_start != payload["runner_sha256"]:
            raise ValueError("Worker runner differs from the runner snapshotted before this suite.")
        measurement = (app_worker if engine == "app" else psi4_worker)(payload, input_path, output)
        runner_provenance = verify_file_snapshot(Path(__file__), runner_start)
        if not runner_provenance["unchanged"]:
            raise ValueError("Worker runner source changed during calculation.")
        measurement.update(status="completed", input_sha256=payload["input_sha256"],
                           basis_sha256=payload["basis_sha256"], gbs_sha256=payload["gbs_sha256"],
                           python=platform.python_version(), runner_provenance=runner_provenance,
                           process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        write_json(output / "measurement.json", measurement)
        return 0
    except Exception as exc:
        traceback.print_exc()
        write_json(output / "measurement.json", {"status": "error", "engine": engine,
                                                "error": f"{type(exc).__name__}: {exc}"})
        return 1


def launch_worker(python: str, engine: str, input_path: Path, output: Path, timeout: float) -> dict:
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    command = [python, str(Path(__file__).resolve()), "--worker", engine,
               "--input", str(input_path), "--output", str(output)]
    log = output.parent / f"{engine}.stdout.log"
    started = time.perf_counter()
    try:
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                    env=environment, cwd=input_path.parent, timeout=timeout, check=False)
        path = output / "measurement.json"
        measurement = json.loads(path.read_text()) if path.is_file() else {"status": "error", "error": "No measurement produced."}
        measurement["process_exit_code"] = result.returncode
        if result.returncode != 0:
            measurement["status"] = "error"
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        measurement = {"status": "missing_engine" if isinstance(exc, FileNotFoundError) else "error",
                       "error": f"{type(exc).__name__}: {exc}"}
    measurement["process_wall_seconds"] = time.perf_counter() - started
    write_json(output.parent / f"{engine}-execution.json", measurement)
    return measurement


def finite_number(value) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def verify_file_snapshot(path: Path, initial_sha256: str) -> dict:
    try:
        final_sha256 = file_digest(path)
    except OSError:
        final_sha256 = None
    return {"start_sha256": initial_sha256, "finish_sha256": final_sha256,
            "unchanged": initial_sha256 == final_sha256}


def compare(payload: dict, app: dict, psi4: dict) -> dict:
    failures, differences = [], {}
    for engine, measured in [("app", app), ("psi4", psi4)]:
        if measured.get("status") != "completed":
            failures.append(f"{engine}: {measured.get('status', 'missing result')} ({measured.get('error', '')})")
            continue
        if measured.get("converged") is not True or measured.get("scf_converged") is not True:
            failures.append(f"{engine}: calculation or SCF did not converge")
        for key in ("input_sha256", "basis_sha256", "gbs_sha256"):
            if measured.get(key) != payload[key]:
                failures.append(f"{engine}: {key} mismatch")
        if measured.get("electron_count") != payload["expected_electrons"]:
            failures.append(f"{engine}: electron count mismatch")
        if measured.get("frozen_orbitals") != payload["spec"]["frozen_orbitals"]:
            failures.append(f"{engine}: frozen occupied orbitals mismatch")
        actual = measured.get("geometry_bohr", [])
        expected = payload["geometry_bohr"]
        valid_geometry = isinstance(actual, list) and len(actual) == len(expected) and all(
            isinstance(row, list) and len(row) == 3 and all(finite_number(x) and abs(x-y) <= 1e-12
                                                          for x, y in zip(row, ref, strict=True))
            for row, ref in zip(actual, expected, strict=True))
        if not valid_geometry:
            failures.append(f"{engine}: nuclear geometry mismatch")
    if not (isinstance(app.get("basis_functions"), int) and app["basis_functions"] > 0
            and app["basis_functions"] == psi4.get("basis_functions")):
        failures.append("basis function count mismatch or missing")
    for key, tolerance in payload["criteria"].items():
        a, b = app.get(key), psi4.get(key)
        if not finite_number(a) or not finite_number(b):
            failures.append(f"{key}: missing or nonfinite value")
            continue
        delta = a - b
        differences[key] = {"app_minus_psi4": delta, "absolute_difference": abs(delta),
                            "tolerance": tolerance, "passed": abs(delta) <= tolerance}
        if abs(delta) > tolerance:
            failures.append(f"{key}: difference exceeds predeclared tolerance")
    return {"case": payload["case"], "status": "passed" if not failures else "failed",
            "comparison": payload["comparison"], "differences": differences, "failures": failures}


def grid_refinement(reports: list[dict]) -> dict:
    indexed = {r["case"]: r for r in reports}
    keys = ("water_pbe_grid5", "water_pbe_grid6")
    if not any(k in indexed for k in keys):
        return {"status": "not_requested"}
    if not all(k in indexed for k in keys):
        return {"status": "not_evaluated", "note": "Run both PBE cases to evaluate quadrature refinement."}
    criterion = 2e-6
    checks = {}
    for engine in ("app", "psi4"):
        values = [indexed[k][engine].get("energy_hartree") for k in keys]
        valid = all(indexed[k][engine].get("status") == "completed"
                    and indexed[k][engine].get("converged") is True for k in keys)
        delta = abs(values[1] - values[0]) if valid and all(map(finite_number, values)) else None
        checks[engine] = {"absolute_energy_change_hartree": delta,
                          "tolerance_hartree": criterion, "passed": delta is not None and delta <= criterion}
    return {"status": "passed" if all(v["passed"] for v in checks.values()) else "failed",
            "checks": checks, "note": "Different atom-centered quadratures; refinement is evidence for these geometries only."}


def overall_status(reports: list[dict], refinement: dict, runner_provenance: dict) -> str:
    if not runner_provenance["unchanged"] or any(r["status"] != "passed" for r in reports):
        return "failed"
    if refinement["status"] == "failed":
        return "failed"
    if refinement["status"] == "not_evaluated":
        return "incomplete"
    return "passed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--psi4-python", help="Python executable in a separate environment containing Psi4")
    parser.add_argument("--output", default=".quantum-lab/cross-engine")
    parser.add_argument("--case", choices=list(cases()), action="append")
    parser.add_argument("--timeout", type=float, default=600, help="Seconds allowed per engine/case")
    parser.add_argument("--worker", choices=["app", "psi4"], help=argparse.SUPPRESS)
    parser.add_argument("--input", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if not args.input:
            parser.error("worker requires --input")
        return worker(args.worker, Path(args.input).resolve(), Path(args.output).resolve())
    if not args.psi4_python:
        parser.error("--psi4-python is required; Psi4 is an optional independent validation environment")
    if args.timeout <= 0 or not math.isfinite(args.timeout):
        parser.error("--timeout must be a positive finite number")
    selected = {key: value for key, value in cases().items() if not args.case or key in args.case}
    output = Path(args.output).resolve()
    root = output / "runs" / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    root.mkdir(parents=True, exist_ok=False)
    runner_start = file_digest(Path(__file__))
    write_json(root / "runner-start.json", {"sha256": runner_start})
    reports = []
    for name, definition in selected.items():
        directory = root / name
        payload = prepare_case(name, definition, directory, runner_start)
        app = launch_worker(sys.executable, "app", directory / "input.json", directory / "app", args.timeout)
        psi4 = launch_worker(args.psi4_python, "psi4", directory / "input.json", directory / "psi4", args.timeout)
        comparison = compare(payload, app, psi4)
        report = {**comparison, "input": payload, "app": app, "psi4": psi4}
        write_json(directory / "comparison.json", report)
        reports.append(report)
        print(name, comparison["status"].upper(), flush=True)
    refinement = grid_refinement(reports)
    runner_provenance = verify_file_snapshot(Path(__file__), runner_start)
    status = overall_status(reports, refinement, runner_provenance)
    report = {"schema_version": SCHEMA, "created_utc": datetime.now(UTC).isoformat(),
              "status": status, "platform": platform.platform(),
              "runner_sha256": runner_start, "runner_provenance": runner_provenance, "run_directory": str(root),
              "scope": "Independent-engine verification for these small, closed-shell molecular cases only; not experimental accuracy or all-method certification.",
              "timing_note": "App wall_seconds includes BSE resolution, properties and artifact writing through the app adapter; Psi4 wall_seconds measures energy(). These are different workflows, not a speed comparison. process_wall_seconds also includes imports; peak RSS includes interpreter and imports.",
              "dft_note": "PBE grids differ between engines. Separate 5e-6 Eh cross-engine and 2e-6 Eh within-engine refinement criteria are predeclared.",
              "cases": reports, "dft_grid_refinement": refinement}
    write_json(root / "report.json", report)
    write_json(output / "report.json", report)
    print(f"Report: {output / 'report.json'}", flush=True)
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
