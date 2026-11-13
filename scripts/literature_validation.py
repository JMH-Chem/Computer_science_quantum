#!/usr/bin/env python3
"""Reproduce or collect declared literature comparisons without conflating accuracy with convergence.

  uv run python scripts/literature_validation.py --run --case water-eomccsd
  uv run python scripts/literature_validation.py --collect collected-results.json

The collection manifest maps input filename stems to existing public-CLI result.json
files. Relative result paths are resolved relative to the manifest. Old runs keep
their original source hashes; collecting does not rewrite their provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

REPOSITORY = Path(__file__).resolve().parents[1]
INPUTS = REPOSITORY / "validation" / "literature"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def normalized_spec(value: dict) -> dict:
    from quantum_lab.models import JobSpec
    return JobSpec.model_validate(value).model_dump(mode="json")


def registered_inputs(directory: Path = INPUTS) -> dict[str, Path]:
    result = {}
    for path in sorted(directory.glob("*.json")):
        data = read_json(path)
        if isinstance(data, dict) and all(key in data for key in ("domain", "atoms", "method", "task")):
            normalized_spec(data)
            result[path.stem] = path
    return result


def finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def same_geometry(result: dict, spec: dict, tolerance: float = 1e-10) -> bool:
    """Allow only Å/Bohr round-trip noise, not geometry relaxation."""
    atoms = result.get("atoms")
    if not isinstance(atoms, list) or len(atoms) != len(spec["atoms"]):
        return False
    for actual, expected in zip(atoms, spec["atoms"], strict=True):
        if actual.get("symbol") != expected["symbol"]:
            return False
        coordinates = actual.get("position")
        if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 3 or not all(
                finite(x) and abs(x - y) <= tolerance for x, y in zip(coordinates, expected["position"], strict=True)):
            return False
    a, b = result.get("lattice"), spec["lattice"]
    if a is None or b is None:
        return a is b
    return isinstance(a, list) and len(a) == 3 and all(
        isinstance(row, list) and len(row) == 3 and all(finite(x) and abs(x - y) <= tolerance
                                                     for x, y in zip(row, ref, strict=True))
        for row, ref in zip(a, b, strict=True))


def completed_result(result: dict) -> bool:
    if result.get("converged") is not True or result.get("scf_converged") is not True:
        return False
    if not finite(result.get("energy_hartree")):
        return False
    correlation = result.get("correlation")
    if correlation and correlation.get("converged") is not True:
        return False
    if result.get("task") == "eomccsd":
        eom = result.get("eom") or {}
        states = eom.get("states", [])
        return bool(eom.get("converged") is True and states
                    and len(states) == eom.get("requested_states")
                    and all(s.get("converged") is True and finite(s.get("energy_ev")) for s in states))
    if result.get("task") in {"tda", "tddft", "tdhf"}:
        states = result.get("excitations", [])
        return bool(result.get("excited_converged") is True and states
                    and all(s.get("converged") is True and finite(s.get("energy_ev")) for s in states))
    return True


def snapshot_artifacts(job: Path, destination: Path, events: list[dict]) -> dict:
    copied = {}
    for path in sorted(job.iterdir()):
        if path.is_file() and path.suffix in {".json", ".log", ".txt", ".xyz", ".cif"}:
            before = sha256(path)
            shutil.copyfile(path, destination / path.name)
            if before != sha256(path) or before != sha256(destination / path.name):
                raise ValueError("A source artifact changed while collecting.")
            copied[path.name] = before
    with (destination / "events.jsonl").open("w", encoding="utf-8") as stream:
        for event in events:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
    return {"copied_artifact_sha256": copied, "events_sha256": sha256(destination / "events.jsonl"),
            "event_count": len(events)}


def collect_case(name: str, expected: dict, source: Path, destination: Path) -> dict:
    """Read a completed durable CLI job and copy auditable text artifacts.

    The SQLite record is opened read-only. Its spec/result must agree with the
    files, and result provenance must equal the worker's environment snapshot.
    """
    destination.mkdir(parents=True, exist_ok=False)
    record = {"case": name, "source_result": str(source), "status": "invalid"}
    try:
        source = source.resolve()
        if source.name != "result.json" or source.parent.parent.name != "jobs":
            raise ValueError("Collect a public-CLI jobs/<id>/result.json, not an arbitrary energy file.")
        job = source.parent
        actual = normalized_spec(read_json(job / "input.json"))
        if canonical(actual) != canonical(normalized_spec(expected)):
            raise ValueError("Stored input differs from the declared literature case.")
        database = job.parent.parent / "jobs.sqlite3"
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute("SELECT * FROM jobs WHERE id=?", (job.name,)).fetchone()
            if row is None:
                raise ValueError("No matching durable job record.")
            if canonical(normalized_spec(json.loads(row["spec"]))) != canonical(actual):
                raise ValueError("Database input and saved input disagree.")
            events = [{**dict(event), "data": json.loads(event["data"])} for event in connection.execute(
                "SELECT * FROM events WHERE job_id=? ORDER BY seq", (job.name,))]
            status = row["status"]
            stored_result = json.loads(row["result"]) if row["result"] else None
            stored_error = row["error"]
        if status not in {"succeeded", "unconverged", "failed", "cancelled", "interrupted"}:
            record.update(status="incomplete", job_status=status, spec=actual,
                          error="Job has not reached a terminal state; partial calculation is not a literature result.")
            write_json(destination / "collection.json", record)
            return record
        if status != "succeeded" and not source.is_file() and stored_result is None:
            record.update(status="failed", job_status=status, spec=actual, error=stored_error,
                          **snapshot_artifacts(job, destination, events))
            write_json(destination / "collection.json", record)
            return record
        required = [job / "input.json", source, job / "environment.json"]
        checksums = {p.name: sha256(p) for p in required}
        result = read_json(source)
        environment = read_json(job / "environment.json")
        if result.get("run_environment") != environment or not environment.get("source_sha256"):
            raise ValueError("Result and worker environment/source provenance differ or are missing.")
        if stored_result != result:
            raise ValueError("Database result and saved result disagree.")
        if result.get("method") != actual["method"] or result.get("task") != actual["task"]:
            raise ValueError("Result method/task differs from the input.")
        if not same_geometry(result, actual):
            raise ValueError("Final coordinates differ; these are fixed-geometry comparisons.")
        if result.get("correlation") and result["correlation"].get("frozen_orbitals") != actual["frozen_orbitals"]:
            raise ValueError("Recorded frozen occupied orbital count differs from the input.")
        copied = snapshot_artifacts(job, destination, events)
        if any(sha256(job / filename) != checksum for filename, checksum in checksums.items()):
            raise ValueError("A required source artifact changed while collecting.")
        record.update(status="completed" if status == "succeeded" and completed_result(result) else "failed",
                      job_status=status, spec=actual, result=result, environment=environment,
                      normalized_input_sha256=hashlib.sha256(canonical(actual).encode()).hexdigest(),
                      source_sha256=checksums, **copied)
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    write_json(destination / "collection.json", record)
    return record


def run_case(name: str, input_file: Path, destination: Path, timeout: float) -> dict:
    destination.mkdir(parents=True)
    source = destination / "requested-input.json"
    shutil.copyfile(input_file, source)
    command = [sys.executable, "-m", "quantum_lab.cli", "run", str(source), "--json-events",
               "--data-dir", str(destination / "job-data")]
    with (destination / "cli.jsonl").open("w", encoding="utf-8") as stdout, \
            (destination / "cli.stderr.log").open("w", encoding="utf-8") as stderr:
        try:
            process = subprocess.Popen(command, cwd=destination, stdout=stdout, stderr=stderr,
                                       env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
                                       start_new_session=True)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # The CLI catches SIGINT and cancels its independently grouped
                # numerical worker. Killing only the CLI would orphan that job.
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                return {"case": name, "status": "failed", "process_exit_code": process.returncode,
                        "error": "CLI timeout; SIGINT requested cancellation. Partial files and logs are retained."}
        except OSError as exc:
            return {"case": name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    candidates = sorted((destination / "job-data" / "jobs").glob("*/result.json"))
    if len(candidates) != 1:
        return {"case": name, "status": "failed", "process_exit_code": process.returncode,
                "error": "CLI did not produce exactly one saved result. Partial job files and logs are preserved."}
    record = collect_case(name, read_json(source), candidates[0], destination / "collected")
    record["process_exit_code"] = process.returncode
    if process.returncode != 0:
        record["status"] = "failed"
    return record


def difference(actual: float, reference: float, unit: str) -> dict:
    if not finite(actual) or not finite(reference):
        raise ValueError("Nonfinite comparison value.")
    return {"calculated": actual, "reference": reference, "signed_deviation": actual - reference,
            "absolute_deviation": abs(actual - reference), "unit": unit}


def water_eom(record: dict | None, criteria: dict) -> dict:
    if not record or record.get("status") != "completed":
        return {"status": "incomplete", "comparison_kind": criteria["comparison_kind"]}
    matched_conditions = {"domain": "molecule", "task": "eomccsd", "method": "CCSD",
                          "basis": "aug-cc-pvtz", "frozen_orbitals": 1, "ecp": None,
                          "restricted": True, "charge": 0, "spin": 0}
    if any(record.get("spec", {}).get(key) != value for key, value in matched_conditions.items()):
        return {"status": "incompatible", "comparison_kind": criteria["comparison_kind"],
                "reason": "The published water comparison requires closed-shell frozen-core CCSD/aug-cc-pVTZ."}
    eom = record["result"].get("eom") or {}
    states = eom.get("states", [])
    report = {"comparison_kind": criteria["comparison_kind"], "criteria": criteria,
              "matched_conditions": matched_conditions,
              "calculated_energies_ev": [state.get("energy_ev") for state in states]}
    search = eom.get("root_search") or {}
    assigned = (search.get("strategy") == "abelian_symmetry_blocks" and search.get("group") == "C2v"
                and search.get("complete") is True
                and all(s.get("symmetry_group") == "C2v" for s in states))
    by_symmetry = {label: sorted([s for s in states if s.get("symmetry") == label],
                                 key=lambda s: s["energy_ev"]) for label in ("B1", "A2", "A1")}
    if not assigned or any(len(by_symmetry[label]) != 1 for label in by_symmetry):
        return {**report, "status": "not_assigned",
                "reason": "Require complete physical C2v sector search and one B1/A2/A1 root; converged root count alone does not prove the lowest roots were found."}
    rows = []
    for label, reference, state_name in zip(("B1", "A2", "A1"), criteria["reference_energies_ev"],
                                           criteria["reference_states"], strict=True):
        row = difference(by_symmetry[label][0]["energy_ev"], reference, "eV")
        row.update(state=state_name, symmetry=label,
                   tolerance=criteria["absolute_tolerance_ev"],
                   passed=row["absolute_deviation"] <= criteria["absolute_tolerance_ev"])
        rows.append(row)
    return {**report, "status": "passed" if all(row["passed"] for row in rows) else "failed", "states": rows}


def dimer_comparison(records: dict, criteria: dict) -> dict:
    names = ("water-dimer", "water-fragment-a", "water-fragment-b")
    if not all(records.get(name, {}).get("status") == "completed" for name in names):
        return {"status": "incomplete", "comparison_kind": criteria["comparison_kind"]}
    dimer, a, b = [records[name] for name in names]
    specs = [item["spec"] for item in (dimer, a, b)]
    keys = ("method", "basis", "ecp", "restricted", "functional", "conv_tol")
    if any(spec["task"] != "single_point" for spec in specs) or any(
            specs[0][key] != spec[key] for spec in specs[1:] for key in keys):
        return {"status": "incompatible", "reason": "Fragment method, basis or fixed-geometry settings differ."}
    if specs[0]["atoms"] != specs[1]["atoms"] + specs[2]["atoms"] or (
            specs[0]["charge"] != specs[1]["charge"] + specs[2]["charge"] or
            specs[0]["frozen_orbitals"] != specs[1]["frozen_orbitals"] + specs[2]["frozen_orbitals"]):
        return {"status": "incompatible", "reason": "Fragment geometry, total charge or frozen-core definition differs."}
    basis_hashes = [item.get("copied_artifact_sha256", {}).get("basis.json") for item in (dimer, a, b)]
    if not basis_hashes[0] or len(set(basis_hashes)) != 1:
        return {"status": "incompatible", "reason": "Resolved BSE basis artifacts differ or are missing."}
    from scipy.constants import Avogadro, physical_constants
    factor = physical_constants["Hartree energy"][0] * Avogadro / 4184
    energies = [item["result"]["energy_hartree"] for item in (dimer, a, b)]
    interaction = (energies[0] - energies[1] - energies[2]) * factor
    return {"status": "assessment_only", "comparison_kind": criteria["comparison_kind"],
            **difference(interaction, criteria["reference_energy_kcal_mol"], "kcal/mol"),
            "component_energies_hartree": dict(zip(names, energies, strict=True)),
            "hartree_to_kcal_mol": factor,
            "conversion": "scipy.constants Hartree energy (J) * Avogadro /4184; thermochemical calorie",
            "note": criteria["reason"], "counterpoise_corrected": False, "zpe_or_thermal_correction": False}


def formaldehyde_comparison(record: dict | None, criteria: dict) -> dict:
    if not record or record.get("status") != "completed":
        return {"status": "incomplete", "comparison_kind": criteria["comparison_kind"]}
    states = record["result"].get("excitations", [])
    if not states or not all(finite(s.get("energy_ev")) for s in states):
        return {"status": "incomplete", "reason": "Missing finite excitation energies."}
    lowest = min(states, key=lambda state: state["energy_ev"])
    search = record["result"].get("provenance", {}).get("response_solver", {}).get("root_search") or {}
    if (search.get("complete") is not True or search.get("group") != "C2v"
            or lowest.get("state") != 1 or lowest.get("symmetry") != "A2"
            or lowest.get("symmetry_group") != "C2v"):
        return {"status": "not_assigned", "comparison_kind": criteria["comparison_kind"],
                "reference_state": criteria["reference_state"],
                "calculated_symmetry": lowest.get("symmetry"),
                "reason": "Require a complete physical C2v root search and a globally lowest singlet A2 state before comparing to the published first A2 excitation."}
    return {"status": "assessment_only", "comparison_kind": criteria["comparison_kind"],
            **difference(lowest["energy_ev"], criteria["reference_energy_ev"], "eV"),
            "reference_state": criteria["reference_state"], "calculated_state": lowest.get("state"),
            "symmetry": "A2", "symmetry_group": "C2v", "root_search_complete": True,
            "state_assignment": "Globally lowest singlet A2 root from complete C2v sector search. Symmetry/order support the published state assignment; this check does not independently classify NTO n→pi* orbital character.",
            "note": criteria["reason"]}


def diamond_comparison(records: dict, reference: dict | None, criteria: dict | None = None) -> dict:
    rows = []
    for name in ("diamond-k2", "diamond-k3", "diamond-mesh33", "diamond-k4"):
        record = records.get(name)
        if not record or record.get("status") != "completed":
            continue
        bands = record["result"].get("bands") or {}
        value = bands.get("gap_ev")
        if not finite(value):
            continue
        row = {"case": name, "sampled_ks_gap_ev": value, "kmesh": record["spec"]["kmesh"],
               "mesh": record["spec"]["mesh"], "energy_hartree": record["result"]["energy_hartree"]}
        if reference:
            row["paper_deviation"] = difference(value, reference["reference"]["value_eV"], "eV")
        rows.append(row)
    changes = []
    diagnostic = (criteria or {}).get("successive_gap_diagnostic", {})
    for left, right, varied in (("diamond-k2", "diamond-k3", "kmesh"),
                                ("diamond-k3", "diamond-mesh33", "mesh"),
                                ("diamond-mesh33", "diamond-k4", "kmesh")):
        matching = {row["case"]: row for row in rows}
        if left not in matching or right not in matching:
            continue
        a, b = records[left]["spec"], records[right]["spec"]
        same = ({key: value for key, value in a.items() if key not in {"name", varied}}
                == {key: value for key, value in b.items() if key not in {"name", varied}})
        change = {"from": left, "to": right, "varied": varied, "conditions_match": same,
                  "gap_change_ev": matching[right]["sampled_ks_gap_ev"] - matching[left]["sampled_ks_gap_ev"]
                  if same else None}
        # Apply the prospective criterion only to the declared extension, not
        # retroactively to earlier observations that motivated this extra run.
        if (same and diagnostic.get("from") == left and diagnostic.get("to") == right
                and diagnostic.get("varied") == varied):
            threshold = diagnostic.get("absolute_change_threshold_ev")
            if finite(threshold) and threshold > 0:
                change["absolute_gap_change_ev"] = abs(change["gap_change_ev"])
                change["diagnostic_threshold_ev"] = threshold
                change["meets_gap_change_diagnostic"] = abs(change["gap_change_ev"]) <= threshold
                change["diagnostic_scope"] = diagnostic["interpretation"]
        changes.append(change)
    return {"status": "assessment_only" if rows else "incomplete", "cases": rows, "refinements": changes,
            "note": "Gaussian/GTH and plane-wave/SG15 Hamiltonians differ. Sampled KS gaps are not optical excitation energies; no literature pass/fail is inferred.",
            "reference": reference, "criteria": criteria}


def assess(records: dict, criteria: dict, materials: dict | None = None) -> dict:
    return {"water_eomccsd": water_eom(records.get("water-eomccsd"), criteria["water_eomccsd"]),
            "formaldehyde_tddft": formaldehyde_comparison(records.get("formaldehyde-tddft"), criteria["formaldehyde_tddft"]),
            "water_dimer": dimer_comparison(records, criteria["water_dimer"]),
            "diamond": diamond_comparison(records, materials, criteria.get("diamond"))}


def main() -> int:
    registry = registered_inputs()
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true", help="Explicitly execute the selected scientific calculations.")
    mode.add_argument("--collect", type=Path, help="JSON object mapping case names to existing result.json paths.")
    parser.add_argument("--case", choices=list(registry), action="append")
    parser.add_argument("--output", type=Path, default=Path(".quantum-lab/literature-validation"))
    parser.add_argument("--timeout", type=float, default=3600, help="Maximum seconds per CLI calculation.")
    args = parser.parse_args()
    if not finite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be positive and finite")
    selected = {name: path for name, path in registry.items() if not args.case or name in args.case}
    mapping = read_json(args.collect) if args.collect else {}
    if not isinstance(mapping, dict) or any(name not in registry or not isinstance(value, str)
                                           for name, value in mapping.items()):
        parser.error("Collection manifest must map registered case names to result.json paths.")
    output = args.output.resolve()
    root = output / "runs" / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    root.mkdir(parents=True, exist_ok=False)
    settings = root / "declared-inputs"
    settings.mkdir()
    snapshots = {}
    for path in INPUTS.glob("*.json"):
        snapshots[path.name] = sha256(path)
        shutil.copyfile(path, settings / path.name)
    runner_hash = sha256(Path(__file__))
    records = {}
    for name, path in selected.items():
        if args.run:
            record = run_case(name, settings / path.name, root / name, args.timeout)
        elif name in mapping:
            source = Path(mapping[name])
            if not source.is_absolute():
                source = args.collect.resolve().parent / source
            record = collect_case(name, read_json(settings / path.name), source, root / name)
        else:
            record = {"case": name, "status": "not_collected", "reason": "No result path in collection manifest."}
        records[name] = record
        print(name, record["status"].upper(), flush=True)
    unchanged = runner_hash == sha256(Path(__file__)) and all(
        (INPUTS / name).is_file() and sha256(INPUTS / name) == checksum for name, checksum in snapshots.items())
    computational = ("complete" if all(r["status"] == "completed" for r in records.values()) else
                     "incomplete" if all(r["status"] in {"completed", "not_collected", "incomplete"}
                                         for r in records.values()) else "failed")
    comparisons = assess(records, read_json(settings / "criteria.json"),
                         read_json(settings / "materials-reference.json") if (settings / "materials-reference.json").exists() else None)
    matched = comparisons["water_eomccsd"]["status"]
    report = {"created_utc": datetime.now(UTC).isoformat(), "run_directory": str(root),
              "computational_completion": computational, "matched_method_comparison": matched,
              "provenance_unchanged": unchanged, "runner_sha256": runner_hash,
              "declared_source_sha256": snapshots, "cases": records, "comparisons": comparisons,
              "scope": "Selected published comparisons only; assessment_only is not a universal accuracy pass. Historical runs retain their original source versions."}
    write_json(root / "report.json", report)
    write_json(output / "report.json", report)
    print(f"Report: {output / 'report.json'}", flush=True)
    # A selected non-matching-method study can complete without claiming a
    # literature reproduction. A selected water EOM case requires assignment.
    return 0 if computational == "complete" and unchanged and (
        "water-eomccsd" not in selected or matched == "passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
