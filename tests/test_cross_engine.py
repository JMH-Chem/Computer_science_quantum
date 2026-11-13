"""Cross-engine comparisons must fail closed on incompatible or partial data."""
from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

script = Path(__file__).resolve().parents[1] / "scripts" / "cross_engine.py"
module_spec = importlib.util.spec_from_file_location("cross_engine", script)
cross = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(cross)


@pytest.fixture
def matched():
    payload = {"case": "reference", "comparison": "matched_hamiltonian",
               "input_sha256": "input", "basis_sha256": "basis", "gbs_sha256": "gbs",
               "expected_electrons": 2, "spec": {"frozen_orbitals": 0},
               "geometry_bohr": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
               "criteria": {"energy_hartree": 1e-8, "nuclear_repulsion_hartree": 1e-10}}
    result = {"status": "completed", "converged": True, "scf_converged": True,
              "input_sha256": "input", "basis_sha256": "basis", "gbs_sha256": "gbs",
              "electron_count": 2, "frozen_orbitals": 0, "basis_functions": 2,
              "geometry_bohr": payload["geometry_bohr"], "energy_hartree": -1.1167,
              "nuclear_repulsion_hartree": 1 / 1.4}
    return payload, copy.deepcopy(result), copy.deepcopy(result)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), True])
def test_missing_or_nonfinite_energy_never_passes(matched, value):
    payload, app, psi4 = matched
    psi4["energy_hartree"] = value
    result = cross.compare(payload, app, psi4)
    assert result["status"] == "failed"
    assert "energy_hartree: missing or nonfinite value" in result["failures"]


@pytest.mark.parametrize("changes", [
    {"converged": False}, {"scf_converged": False}, {"basis_sha256": "different"},
    {"gbs_sha256": "different"}, {"input_sha256": "different"},
    {"frozen_orbitals": 1}, {"basis_functions": 3}, {"electron_count": 4},
    {"geometry_bohr": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.41]]},
    {"status": "error"},
])
def test_equal_energies_do_not_hide_incompatible_or_failed_calculations(matched, changes):
    payload, app, psi4 = matched
    psi4.update(changes)
    assert cross.compare(payload, app, psi4)["status"] == "failed"


def test_predeclared_tolerance_boundary_is_inclusive_and_does_not_adapt(matched):
    payload, app, psi4 = matched
    payload["criteria"]["energy_hartree"] = 0.125
    app["energy_hartree"], psi4["energy_hartree"] = -1.0, -1.125
    assert cross.compare(payload, app, psi4)["status"] == "passed"
    psi4["energy_hartree"] -= 0.001
    comparison = cross.compare(payload, app, psi4)
    assert comparison["status"] == "failed"
    assert comparison["differences"]["energy_hartree"]["tolerance"] == 0.125


def test_missing_engine_is_recorded_instead_of_skipped(tmp_path):
    result = cross.launch_worker(str(tmp_path / "uninstalled-python"), "psi4",
                                 tmp_path / "input.json", tmp_path / "psi4", 1)
    assert result["status"] == "missing_engine"
    assert json.loads((tmp_path / "psi4-execution.json").read_text())["status"] == "missing_engine"


def test_nonzero_process_exit_overrides_apparently_complete_measurement(tmp_path, monkeypatch):
    def fake_run(command, **kwargs):
        output = Path(command[-1])
        output.mkdir()
        cross.write_json(output / "measurement.json", {"status": "completed", "converged": True})
        return subprocess.CompletedProcess(command, 9)
    monkeypatch.setattr(cross.subprocess, "run", fake_run)
    result = cross.launch_worker("python", "psi4", tmp_path / "input.json", tmp_path / "psi4", 1)
    assert result["status"] == "error" and result["process_exit_code"] == 9


def test_timeout_preserves_an_error_record(tmp_path, monkeypatch):
    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])
    monkeypatch.setattr(cross.subprocess, "run", timeout)
    result = cross.launch_worker("python", "psi4", tmp_path / "input.json", tmp_path / "psi4", 0.1)
    assert result["status"] == "error" and "TimeoutExpired" in result["error"]
    assert (tmp_path / "psi4-execution.json").is_file()


def test_input_and_shared_basis_tampering_are_rejected(tmp_path):
    path = tmp_path / "case"
    payload = cross.prepare_case("h2_rhf_sto3g", cross.cases()["h2_rhf_sto3g"], path)
    assert cross.read_input(path / "input.json") == payload
    gbs = path / "shared-basis.gbs"
    original = gbs.read_text()
    gbs.write_text(original.replace("0.3425250914", "0.5425250914"))
    with pytest.raises(ValueError, match="Basis provenance"):
        cross.read_input(path / "input.json")
    gbs.write_text(original)
    payload["spec"]["charge"] = 1
    cross.write_json(path / "input.json", payload)
    with pytest.raises(ValueError, match="Input provenance"):
        cross.read_input(path / "input.json")


def test_all_declared_cases_use_public_application_input_constraints():
    from quantum_lab.models import JobSpec
    for definition in cross.cases().values():
        JobSpec.model_validate(definition["spec"])


def test_dft_cross_agreement_does_not_replace_within_engine_refinement():
    reports = []
    for name, energy in [("water_pbe_grid5", -76.0), ("water_pbe_grid6", -76.0001)]:
        measurement = {"status": "completed", "converged": True, "energy_hartree": energy}
        reports.append({"case": name, "status": "passed", "app": measurement, "psi4": measurement})
    result = cross.grid_refinement(reports)
    assert result["status"] == "failed"
    assert not result["checks"]["app"]["passed"]
    assert cross.grid_refinement(reports[:1])["status"] == "not_evaluated"


def test_partial_dft_suite_cannot_receive_overall_pass():
    measurement = {"status": "completed", "converged": True, "energy_hartree": -76.0}
    reports = [{"case": "water_pbe_grid5", "status": "passed", "app": measurement, "psi4": measurement}]
    refinement = cross.grid_refinement(reports)
    assert cross.overall_status(reports, refinement, {"unchanged": True}) == "incomplete"
    assert cross.overall_status(reports, refinement, {"unchanged": False}) == "failed"


@pytest.mark.parametrize("geometry", [None, {}, "0 0 0", [[0, 0]], [[0, 0, 0], [0, None, 1.4]]])
def test_malformed_geometry_is_a_failed_comparison_not_an_exception(matched, geometry):
    payload, app, psi4 = matched
    psi4["geometry_bohr"] = geometry
    result = cross.compare(payload, app, psi4)
    assert result["status"] == "failed"
    assert "psi4: nuclear geometry mismatch" in result["failures"]


def test_worker_rejects_runner_changed_during_calculation(tmp_path, monkeypatch):
    runner = tmp_path / "runner.py"
    runner.write_text("# initial source\n")
    monkeypatch.setattr(cross, "__file__", str(runner))
    path = tmp_path / "case"
    cross.prepare_case("h2_rhf_sto3g", cross.cases()["h2_rhf_sto3g"], path)
    def changed_runner(*args):
        runner.write_text("# different source\n")
        return {"converged": True}
    monkeypatch.setattr(cross, "app_worker", changed_runner)
    assert cross.worker("app", path / "input.json", path / "app") == 1
    measured = json.loads((path / "app" / "measurement.json").read_text())
    assert measured["status"] == "error"
    assert "runner source changed" in measured["error"]


@pytest.mark.science
@pytest.mark.parametrize("changed", [False, True], ids=["actual-source-record", "changed-source-rejected"])
def test_actual_app_worker_preserves_source_record_and_rejects_changes(tmp_path, monkeypatch, changed):
    import quantum_lab.provenance as provenance
    original = provenance.execution_environment
    calls = 0
    def environment_snapshot():
        nonlocal calls
        calls += 1
        result = original()
        if changed and calls > 1:
            result["source_sha256"]["engines/molecular.py"] = "modified-during-computation"
        return result
    monkeypatch.setattr(provenance, "execution_environment", environment_snapshot)
    path = tmp_path / "case"
    cross.prepare_case("h2_rhf_sto3g", cross.cases()["h2_rhf_sto3g"], path)
    assert cross.worker("app", path / "input.json", path / "app") == (1 if changed else 0)
    environment = json.loads((path / "app" / "environment.json").read_text())
    assert environment["source_files_unchanged"] is not changed
    assert environment["run_environment"]["packages"]["pyscf"]
    assert "engines/molecular.py" in environment["run_environment"]["source_sha256"]
    assert "quantum_lab.engines.molecular" in environment["imported_module_paths"]
    measured = json.loads((path / "app" / "measurement.json").read_text())
    if changed:
        assert measured["status"] == "error" and "source files changed" in measured["error"]
    else:
        assert measured["run_environment"] == environment["run_environment"]
        assert measured["runner_provenance"]["unchanged"] is True
