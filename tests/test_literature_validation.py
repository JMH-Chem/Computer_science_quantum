"""A converged calculation is not automatically a matching literature result."""
from __future__ import annotations

import copy
import importlib.util
import json
import signal
import subprocess
from pathlib import Path

import pytest

from quantum_lab.storage import Store, atomic_json

path = Path(__file__).resolve().parents[1] / "scripts" / "literature_validation.py"
module_spec = importlib.util.spec_from_file_location("literature_validation", path)
literature = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(literature)


@pytest.fixture
def saved_job(tmp_path):
    spec = literature.normalized_spec({"domain": "molecule", "method": "HF", "task": "single_point",
                                      "atoms": [{"symbol": "H", "position": [0, 0, 0]},
                                                {"symbol": "H", "position": [0, 0, 0.74]}]})
    store = Store(tmp_path / "data")
    job = store.create(spec)
    store.claim()
    environment = {"source_sha256": {"engines/molecular.py": "original-source"}}
    result = {"converged": True, "scf_converged": True, "method": "HF", "task": "single_point",
              "energy_hartree": -1.1167, "atoms": spec["atoms"], "lattice": None,
              "run_environment": environment}
    directory = store.job_dir(job["id"])
    atomic_json(directory / "input.json", spec)
    atomic_json(directory / "environment.json", environment)
    atomic_json(directory / "result.json", result)
    atomic_json(directory / "basis.json", {"definition": "fixture"})
    (directory / "engine.log").write_text("fixture engine log")
    store.transition(job["id"], "succeeded", result=result)
    return spec, result, store, job["id"], directory / "result.json"


def test_collect_preserves_real_job_contract_and_old_source_hash(saved_job, tmp_path):
    spec, _, _, _, source = saved_job
    record = literature.collect_case("h2", spec, source, tmp_path / "collection")
    assert record["status"] == "completed"
    assert record["environment"]["source_sha256"]["engines/molecular.py"] == "original-source"
    assert record["event_count"] == 3
    assert (tmp_path / "collection" / "engine.log").read_text() == "fixture engine log"
    assert record["events_sha256"] == literature.sha256(tmp_path / "collection" / "events.jsonl")


def test_input_condition_mismatch_is_rejected_even_with_converged_energy(saved_job, tmp_path):
    spec, _, _, _, source = saved_job
    spec["basis"] = "def2-svp"
    result = literature.collect_case("h2", spec, source, tmp_path / "collection")
    assert result["status"] == "invalid" and "declared literature case" in result["error"]


@pytest.mark.parametrize("artifact,field,value", [
    ("result.json", "energy_hartree", -50),
    ("environment.json", "source_sha256", {"engines/molecular.py": "replacement"}),
])
def test_artifact_tampering_does_not_pass_durable_record_checks(saved_job, tmp_path, artifact, field, value):
    spec, _, _, _, source = saved_job
    file = source.parent / artifact
    changed = json.loads(file.read_text())
    changed[field] = value
    atomic_json(file, changed)
    record = literature.collect_case("h2", spec, source, tmp_path / "collection")
    assert record["status"] == "invalid"


def test_unconverged_durable_job_stays_failed(saved_job, tmp_path):
    spec, result, store, job_id, source = saved_job
    result["converged"] = False
    atomic_json(source, result)
    with store.connection() as database:
        database.execute("UPDATE jobs SET status=?, result=? WHERE id=?",
                         ("unconverged", json.dumps(result), job_id))
    record = literature.collect_case("h2", spec, source, tmp_path / "collection")
    assert record["status"] == "failed" and record["job_status"] == "unconverged"


def test_failed_job_without_final_result_keeps_input_logs_and_events(saved_job, tmp_path):
    spec, _, store, job_id, source = saved_job
    source.unlink()
    with store.connection() as database:
        database.execute("UPDATE jobs SET status='failed', result=NULL, error='engine exception' WHERE id=?", (job_id,))
    record = literature.collect_case("h2", spec, source, tmp_path / "collection")
    assert record["status"] == "failed" and record["error"] == "engine exception"
    assert (tmp_path / "collection" / "engine.log").is_file()
    assert (tmp_path / "collection" / "input.json").is_file()
    assert record["event_count"] == 3
    assert "result" not in record


def test_unfinished_job_cannot_be_collected_as_a_completed_reference(saved_job, tmp_path):
    spec, _, store, job_id, source = saved_job
    with store.connection() as database:
        database.execute("UPDATE jobs SET status='running' WHERE id=?", (job_id,))
    record = literature.collect_case("h2", spec, source, tmp_path / "collection")
    assert record["status"] == "incomplete"


def test_round_trip_coordinate_noise_is_allowed_but_relaxation_is_not(saved_job):
    spec, result, *_ = saved_job
    result = copy.deepcopy(result)
    result["atoms"][1]["position"][2] += 1e-15
    assert literature.same_geometry(result, spec)
    result["atoms"][1]["position"][2] += 1e-5
    assert not literature.same_geometry(result, spec)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), True])
def test_nonfinite_or_missing_energy_is_not_completed(value):
    assert not literature.completed_result({"converged": True, "scf_converged": True, "energy_hartree": value})


@pytest.fixture
def eom_record():
    criteria = {"comparison_kind": "matched_method_published_table", "reference_energies_ev": [7.6, 9.36, 9.96],
                "reference_states": ["1B1", "1A2", "2A1"], "absolute_tolerance_ev": 0.01}
    states = [{"symmetry": label, "symmetry_group": "C2v", "energy_ev": energy}
              for label, energy in zip(("B1", "A2", "A1"), (7.5965, 9.3613, 9.9568), strict=True)]
    result = {"status": "completed", "spec": {"domain": "molecule", "task": "eomccsd", "method": "CCSD",
              "basis": "aug-cc-pvtz", "frozen_orbitals": 1, "ecp": None, "restricted": True, "charge": 0, "spin": 0},
              "result": {"eom": {"states": states,
              "root_search": {"strategy": "abelian_symmetry_blocks", "group": "C2v", "complete": True}}}}
    return criteria, result


def test_physically_assigned_states_match_rounded_publication(eom_record):
    criteria, result = eom_record
    comparison = literature.water_eom(result, criteria)
    assert comparison["status"] == "passed"
    assert [row["symmetry"] for row in comparison["states"]] == ["B1", "A2", "A1"]
    result["result"]["eom"]["states"][2]["energy_ev"] += 0.02
    assert literature.water_eom(result, criteria)["status"] == "failed"


def test_wrong_third_root_and_missing_symmetry_never_receive_an_assignment(eom_record):
    criteria, result = eom_record
    result["result"]["eom"]["states"][2].update(symmetry="B2", energy_ev=10.80567)
    assert literature.water_eom(result, criteria)["status"] == "not_assigned"
    result["result"]["eom"].pop("root_search")
    assert literature.water_eom(result, criteria)["status"] == "not_assigned"


def test_incomplete_symmetry_search_cannot_pass(eom_record):
    criteria, result = eom_record
    result["result"]["eom"]["root_search"]["complete"] = False
    assert literature.water_eom(result, criteria)["status"] == "not_assigned"


def test_changed_basis_does_not_become_matched_method_by_energy_coincidence(eom_record):
    criteria, result = eom_record
    result["spec"]["basis"] = "sto-3g"
    assert literature.water_eom(result, criteria)["status"] == "incompatible"


@pytest.fixture
def dimer_records():
    base = literature.normalized_spec({"method": "CCSD(T)", "task": "single_point", "basis": "aug-cc-pvdz",
                                      "atoms": [{"symbol": "H", "position": [0, 0, 0]},
                                                {"symbol": "H", "position": [0, 0, 0.74]}]})
    records = {}
    for name, energy in (("water-fragment-a", -1.0), ("water-fragment-b", -1.0), ("water-dimer", -2.01)):
        spec = copy.deepcopy(base)
        if name == "water-dimer":
            spec["atoms"] = copy.deepcopy(base["atoms"] + base["atoms"])
        records[name] = {"status": "completed", "spec": spec, "result": {"energy_hartree": energy},
                         "copied_artifact_sha256": {"basis.json": "shared-definition"}}
    criteria = {"comparison_kind": "finite_basis_uncorrected_vs_cbs_reference", "reference_energy_kcal_mol": -4.989,
                "reason": "Different basis and counterpoise definitions: not a numerical pass/fail."}
    return criteria, records


def test_fragment_difference_and_units_are_explicit_and_assessment_only(dimer_records):
    criteria, records = dimer_records
    result = literature.dimer_comparison(records, criteria)
    assert result["status"] == "assessment_only"
    assert result["calculated"] == pytest.approx(-6.2750947, abs=1e-6)
    assert result["signed_deviation"] == pytest.approx(result["calculated"] + 4.989)
    assert "passed" not in result and not result["counterpoise_corrected"]


@pytest.mark.parametrize("change", ["geometry", "frozen", "basis", "basis_definition"])
def test_incompatible_fragments_are_not_combined(dimer_records, change):
    criteria, records = dimer_records
    fragment = records["water-fragment-a"]
    if change == "geometry":
        fragment["spec"]["atoms"][0]["position"][0] += 0.1
    elif change == "frozen":
        fragment["spec"]["frozen_orbitals"] = 1
    elif change == "basis":
        fragment["spec"]["basis"] = "sto-3g"
    else:
        fragment["copied_artifact_sha256"]["basis.json"] = "different"
    assert literature.dimer_comparison(records, criteria)["status"] == "incompatible"


def test_missing_fragment_is_incomplete(dimer_records):
    criteria, records = dimer_records
    del records["water-fragment-a"]
    assert literature.dimer_comparison(records, criteria)["status"] == "incomplete"


def test_tddft_vs_tbe_is_never_a_matching_method_pass():
    record = {"status": "completed", "result": {
        "excitations": [{"state": 1, "energy_ev": 3.98, "symmetry": "A2", "symmetry_group": "C2v"}],
        "provenance": {"response_solver": {"root_search": {"complete": True, "group": "C2v"}}}}}
    criteria = {"comparison_kind": "approximate_method_vs_high_level_reference", "reference_energy_ev": 3.98,
                "reference_state": "1A2", "reason": "Different methods."}
    result = literature.formaldehyde_comparison(record, criteria)
    assert result["status"] == "assessment_only" and result["absolute_deviation"] == 0
    assert "passed" not in result


@pytest.mark.parametrize("symmetry,complete", [("B1", True), (None, True), ("A2", False)])
def test_unassigned_formaldehyde_state_does_not_receive_a_published_state_deviation(symmetry, complete):
    record = {"status": "completed", "result": {
        "excitations": [{"state": 1, "energy_ev": 3.98, "symmetry": symmetry, "symmetry_group": "C2v"}],
        "provenance": {"response_solver": {"root_search": {"complete": complete, "group": "C2v"}}}}}
    criteria = {"comparison_kind": "approximate_method_vs_high_level_reference", "reference_energy_ev": 3.98,
                "reference_state": "1A2", "reason": "Different methods."}
    result = literature.formaldehyde_comparison(record, criteria)
    assert result["status"] == "not_assigned"
    assert "signed_deviation" not in result and "absolute_deviation" not in result


def test_diamond_k4_input_changes_only_the_uniform_k_mesh():
    registry = literature.registered_inputs()
    previous = literature.read_json(registry["diamond-mesh33"])
    extension = literature.read_json(registry["diamond-k4"])
    assert extension["kmesh"] == [4, 4, 4]
    assert extension["mesh"] == [33, 33, 33]
    assert extension["memory_mb"] == 6000 and extension["threads"] == 1
    assert {k: v for k, v in previous.items() if k not in {"name", "kmesh"}} == {
        k: v for k, v in extension.items() if k not in {"name", "kmesh"}}


@pytest.fixture
def diamond_records():
    records = {}
    for name, gap in (("diamond-k2", 4.0), ("diamond-k3", 4.14),
                      ("diamond-mesh33", 4.15), ("diamond-k4", 4.17)):
        spec = literature.read_json(literature.INPUTS / f"{name}.json")
        records[name] = {"status": "completed", "spec": spec,
                         "result": {"energy_hartree": -11.2, "bands": {"gap_ev": gap}}}
    criteria = literature.read_json(literature.INPUTS / "criteria.json")["diamond"]
    return records, criteria


@pytest.mark.parametrize("change,meets", [(0.02, True), (-0.04, False)])
def test_diamond_refinement_diagnostic_is_not_a_literature_or_global_convergence_pass(
        diamond_records, change, meets):
    records, criteria = diamond_records
    records["diamond-k4"]["result"]["bands"]["gap_ev"] = 4.15 + change
    comparison = literature.diamond_comparison(records, {"reference": {"value_eV": 4.15 + change}}, criteria)
    assert comparison["status"] == "assessment_only"
    assert len(comparison["cases"]) == 4 and len(comparison["refinements"]) == 3
    extension = comparison["refinements"][-1]
    assert extension["from"] == "diamond-mesh33" and extension["to"] == "diamond-k4"
    assert extension["conditions_match"]
    assert extension["gap_change_ev"] == pytest.approx(change)
    assert extension["absolute_gap_change_ev"] == pytest.approx(abs(change))
    assert extension["meets_gap_change_diagnostic"] is meets
    assert extension["diagnostic_threshold_ev"] == 0.025
    assert "does not establish convergence" in extension["diagnostic_scope"]
    # Initial observations predate this extra-run criterion; no retroactive pass.
    assert all("meets_gap_change_diagnostic" not in row for row in comparison["refinements"][:-1])
    assert "passed" not in comparison and "converged" not in comparison
    assert comparison["cases"][-1]["paper_deviation"]["absolute_deviation"] == 0


@pytest.mark.parametrize("key,value", [("basis", "gth-szv"), ("unmatched_condition", "extra")])
def test_diamond_changed_controls_cannot_pass_the_successive_gap_diagnostic(diamond_records, key, value):
    records, criteria = diamond_records
    records["diamond-k4"]["spec"][key] = value
    comparison = literature.diamond_comparison(records, None, criteria)
    extension = comparison["refinements"][-1]
    assert extension["conditions_match"] is False and extension["gap_change_ev"] is None
    assert "meets_gap_change_diagnostic" not in extension


def test_timeout_asks_cli_to_cancel_its_independent_worker(tmp_path, monkeypatch):
    input_file = tmp_path / "input.json"
    input_file.write_text('{}')
    class TimedOutCLI:
        returncode = 130
        calls = 0
        received_signal = None

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired(["cli"], timeout)

        def send_signal(self, value):
            self.received_signal = value

    process = TimedOutCLI()
    monkeypatch.setattr(literature.subprocess, "Popen", lambda *args, **kwargs: process)
    result = literature.run_case("example", input_file, tmp_path / "run", 0.01)
    assert result["status"] == "failed" and process.received_signal == signal.SIGINT
    assert process.calls == 2
