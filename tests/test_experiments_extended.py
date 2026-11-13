"""Frozen multi-variable experiments, legacy migration, and real numerical samples."""
from __future__ import annotations

import copy
import json
import time
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pyscf import dft, gto, lib, scf

from quantum_lab.api import create_app
from quantum_lab.experiment_science import build_specs, freeze_definition, summarize, templates
from quantum_lab.storage import Store

TEMPLATES = {
    "water_basis_v1": ("basis", ["sto-3g", "def2-svp", "def2-tzvp"]),
    "water_grid_v1": ("grid_level", [1, 3, 5]),
    "water_functional_v1": ("functional", ["PBE", "PBE0", "B3LYP"]),
    "hydrogen_dissociation_v1": ("bond_distance", [0.5, 0.74, 1.0, 1.5, 2.0]),
    "diamond_kmesh_v1": ("kmesh", [[1, 1, 1], [2, 2, 2], [3, 3, 3]]),
    "diamond_mesh_v1": ("mesh", [[15, 15, 15], [21, 21, 21], [27, 27, 27]]),
    "water_correlation_v1": ("method", ["HF", "MP2", "CCSD"]),
}


def request(template_id):
    return {"name": "Controlled experiment", "template_id": template_id,
            "prediction": "unsure", "request_id": str(uuid4())}


def synthetic_result(spec, index=0):
    return {
        "converged": True, "scf_converged": True, "energy_hartree": -10 - index * 0.1,
        "atoms": copy.deepcopy(spec["atoms"]), "lattice": copy.deepcopy(spec["lattice"]),
        "method": spec["method"], "task": spec["task"],
        "basis_functions": 8, "iteration_count": 5, "elapsed_seconds": 0.5, "peak_memory_mb": 100,
        "engine": {"name": "PySCF PBC" if spec["domain"] == "crystal" else "PySCF molecular",
                   "version": "synthetic-test"},
        "provenance": {"basis_provider": "test-provider", "basis_provider_version": "1",
                       "libxc_version": "1", "numpy_version": "1"},
    }


def fixture(template_id):
    specs = build_specs(template_id, "Frozen comparison")
    record = {**request(template_id), "id": str(uuid4()), "created_at": "2026-11-14",
              "cancel_requested": False, "base_spec": copy.deepcopy(specs[0]),
              "bases": [spec["basis"] for spec in specs],
              "definition": freeze_definition(template_id, specs)}
    jobs = [{"id": str(uuid4()), "status": "succeeded", "spec": spec,
             "result": synthetic_result(spec, index), "error": None}
            for index, spec in enumerate(specs)]
    return record, jobs


@pytest.mark.parametrize("template_id", TEMPLATES)
def test_template_changes_exactly_the_named_variable_and_freezes_inputs(template_id):
    variable, values = TEMPLATES[template_id]
    specs = build_specs(template_id, "One variable")
    catalog = next(template for template in templates() if template["id"] == template_id)
    assert catalog["variable"] == variable and catalog["values"] == values
    assert len(specs) == len(catalog["labels"]) == len(values)
    fixed = []
    for spec, value in zip(specs, values, strict=True):
        spec = copy.deepcopy(spec)
        spec.pop("name")
        if variable == "bond_distance":
            np.testing.assert_allclose(spec["atoms"][0]["position"], [0, 0, 0])
            np.testing.assert_allclose(spec["atoms"][1]["position"], [0, 0, value])
            assert [atom["symbol"] for atom in spec["atoms"]] == ["H", "H"]
            spec["atoms"][1]["position"] = [0, 0, 0]
        else:
            assert spec.pop(variable) == value
        fixed.append(spec)
    assert all(spec == fixed[0] for spec in fixed)
    definition = freeze_definition(template_id, specs)
    old_atom = definition["expected_specs"][0]["atoms"][0]["position"][0]
    specs[0]["atoms"][0]["position"][0] += 5
    assert definition["expected_specs"][0]["atoms"][0]["position"][0] == old_atom
    assert definition["template_version"] == "1"


@pytest.mark.parametrize("template_id", TEMPLATES)
def test_each_variable_compares_only_its_frozen_inputs(template_id):
    record, jobs = fixture(template_id)
    report = summarize(record, jobs)
    assert report["status"] == "completed"
    assert report["analysis"]["comparable"]
    assert [row["label"] for row in report["rows"]] == record["definition"]["labels"]
    assert [row["value"] for row in report["rows"]] == record["definition"]["values"]
    assert report["rows"][0]["delta_previous_hartree"] is None
    assert report["rows"][1]["delta_previous_hartree"] == pytest.approx(-0.1)
    variable = TEMPLATES[template_id][0]
    if variable == "bond_distance":
        jobs[1]["spec"]["atoms"][1]["position"][2] += 0.1
    elif variable in {"kmesh", "mesh"}:
        jobs[1]["spec"][variable][0] += 1
    elif variable == "grid_level":
        jobs[1]["spec"][variable] = 4
    else:
        jobs[1]["spec"][variable] = "unplanned-setting"
    changed = summarize(record, jobs)
    assert changed["rows"][1]["comparable"] is False
    assert changed["rows"][1]["delta_previous_hartree"] is None
    assert changed["rows"][2]["delta_previous_hartree"] is None


@pytest.mark.parametrize("template_id", TEMPLATES)
def test_no_variable_allows_an_unrelated_fixed_condition_to_change(template_id):
    record, jobs = fixture(template_id)
    jobs[1]["spec"]["conv_tol"] = 1e-5
    assert summarize(record, jobs)["rows"][1]["comparable"] is False


def test_historical_definitions_and_wording_do_not_follow_current_catalog(monkeypatch):
    import quantum_lab.experiment_science as science

    record, jobs = fixture("water_grid_v1")
    original = summarize(record, jobs)
    monkeypatch.setattr(science, "_configs", lambda: [])
    restored = summarize(record, jobs)
    assert restored == original
    assert restored["definition"]["title"] == "물의 적분 격자를 촘촘하게 하면?"


def test_crystal_report_keeps_cell_energy_and_per_atom_energy_distinct():
    record, jobs = fixture("diamond_kmesh_v1")
    report = summarize(record, jobs)
    assert report["domain"] == "crystal"
    for row in report["rows"]:
        assert row["energy_per_atom_hartree"] == row["energy_hartree"] / 2
    assert report["rows"][1]["delta_previous_per_atom_hartree"] == pytest.approx(-0.05)
    assert report["definition"]["units"]["energy"] == "hartree"
    assert report["definition"]["units"]["energy_per_atom"] == "hartree/atom"
    assert any("단위격자" in message["text"] for message in report["analysis"]["messages"])
    jobs[1]["result"]["provenance"]["numpy_version"] = "different"
    changed = summarize(record, jobs)
    assert all(row["energy_per_atom_hartree"] is None for row in changed["rows"])


def test_geometry_curve_reports_sampled_minimum_not_dynamics_or_optimized_geometry():
    record, jobs = fixture("hydrogen_dissociation_v1")
    for job, energy in zip(jobs, [-1.0, -1.12, -1.07, -0.91, -0.78], strict=True):
        job["result"]["energy_hartree"] = energy
    report = summarize(record, jobs)
    assert report["analysis"]["energy_trend"] == "mixed"
    text = " ".join(message["text"] for message in report["analysis"]["messages"])
    assert "0.74 Å" in text and "평형 결합 길이를 계산한 것은 아닙니다" in text
    assert "실제 시간 운동이나 구조 최적화가 아닙니다" in text
    assert "다중참조" in text


def test_functional_and_correlation_comparisons_do_not_rank_accuracy_by_energy():
    for template_id in ("water_functional_v1", "water_correlation_v1"):
        record, jobs = fixture(template_id)
        text = " ".join(message["text"] for message in summarize(record, jobs)["analysis"]["messages"])
        assert "정확" in text
        assert "순위" in text
        assert "오차가 아닙니다" in text


def test_v02_database_migration_preserves_original_jobs_and_reads_old_defaults(tmp_path):
    store = Store(tmp_path)
    payload = request("water_basis_v1")
    specs = build_specs(payload["template_id"], payload["name"])
    # Remove a newly introduced model default to model a genuine older input.
    for spec in specs:
        spec.pop("active_electrons", None)
    group = store.create_experiment(payload, specs)
    original_jobs = []
    for index in range(3):
        job = store.claim()
        store.transition(job["id"], "succeeded", result=synthetic_result(job["spec"], index))
        original_jobs.append(store.get(job["id"]))
    with store.connection() as db:
        db.execute("ALTER TABLE experiments DROP COLUMN definition")
    restored = Store(tmp_path)
    record, jobs = restored.experiment_bundle(group["id"])
    assert jobs == original_jobs
    assert record["definition"] is None
    assert record["request_payload"] == payload
    report = summarize(record, jobs)
    assert report["status"] == "completed"
    assert "active_electrons" not in report["definition"]["expected_specs"][0]
    assert report["definition"]["values"] == ["sto-3g", "def2-svp", "def2-tzvp"]


@pytest.mark.parametrize("template_id", TEMPLATES)
def test_api_serializes_frozen_template_order_and_periodic_artifacts(tmp_path, template_id):
    # No worker is started in this API serialization test; numerical execution
    # is checked separately below and in the engine-specific science tests.
    app = create_app(tmp_path)
    app.state.store = store = Store(tmp_path)
    payload = request(template_id)
    group = store.create_experiment(payload, build_specs(template_id, payload["name"]))
    client = TestClient(app)
    detail = client.get(f"/api/experiments/{group['id']}").json()
    assert detail["variable"] == TEMPLATES[template_id][0]
    assert detail["values"] == TEMPLATES[template_id][1]
    assert detail["status"] == "queued"
    assert len(detail["jobs"]) == len(detail["values"])
    first_job = detail["jobs"][0]
    (store.job_dir(first_job["id"]) / "periodic_basis.json").write_text('{"test_fixture": true}')
    exported = client.get(f"/api/experiments/{group['id']}/export?format=json").json()
    assert exported["definition"] == detail["definition"]
    assert exported["job_artifacts"][first_job["id"]]["periodic_basis.json"] == {"test_fixture": True}
    csv = client.get(f"/api/experiments/{group['id']}/export?format=csv")
    assert csv.status_code == 200
    assert "energy_per_atom_hartree" in csv.text and "comparison_note" in csv.text


@pytest.mark.science
@pytest.mark.parametrize("template_id,index", [
    ("water_grid_v1", 0), ("water_grid_v1", 2),
    ("water_functional_v1", 1), ("water_functional_v1", 2),
    ("hydrogen_dissociation_v1", 0), ("hydrogen_dissociation_v1", 4),
])
def test_real_extended_molecular_samples_match_exact_direct_engine(tmp_path, template_id, index):
    from quantum_lab.engines.molecular import run

    lib.num_threads(1)
    spec = build_specs(template_id, "Numerical sample")[index]
    result = run(spec, lambda *_: None, tmp_path)
    assert result["converged"] and result["scf_converged"]
    saved = json.loads((tmp_path / "basis.json").read_text())
    mol = gto.M(atom=[(atom["symbol"], atom["position"]) for atom in spec["atoms"]],
                basis=saved["resolved_pyscf_basis"], ecp=saved["resolved_pyscf_ecp"],
                unit="Angstrom", charge=spec["charge"], spin=spec["spin"], verbose=0)
    direct = dft.RKS(mol) if spec["method"] == "DFT" else scf.RHF(mol)
    if spec["method"] == "DFT":
        direct.xc = spec["functional"]
        direct.grids.level = spec["grid_level"]
    direct.conv_tol, direct.max_cycle = spec["conv_tol"], spec["max_cycle"]
    direct.kernel()
    assert direct.converged
    assert abs(result["energy_hartree"] - direct.e_tot) < 1e-8
    record, jobs = fixture(template_id)
    jobs[index]["result"] = result
    # Replace the deliberately synthetic provenance in other rows with the
    # actual signature; their energies remain only summary fixtures here.
    for job in jobs:
        job["result"]["engine"] = result["engine"]
        job["result"]["provenance"] = result["provenance"]
    assert summarize(record, jobs)["rows"][index]["comparable"]


@pytest.mark.science
@pytest.mark.parametrize("template_id", [
    "hydrogen_dissociation_v1", "diamond_mesh_v1", "diamond_kmesh_v1", "water_correlation_v1",
])
def test_real_extended_experiment_completes_on_durable_queue(tmp_path, template_id):
    """Execute each control setting, including the largest periodic preset."""
    with TestClient(create_app(tmp_path)) as client:
        payload = request(template_id)
        response = client.post("/api/experiments", json=payload)
        assert response.status_code == 201, response.text
        experiment_id = response.json()["id"]
        deadline = time.monotonic() + 180
        while True:
            report = client.get(f"/api/experiments/{experiment_id}").json()
            if report["status"] not in {"queued", "running"}:
                break
            assert time.monotonic() < deadline, report
            time.sleep(0.1)
        assert report["status"] == "completed", report
        assert report["analysis"]["comparable"]
        assert report["total_jobs"] == len(TEMPLATES[template_id][1])
        for row, job in zip(report["rows"], report["jobs"], strict=True):
            assert row["energy_hartree"] == job["result"]["energy_hartree"]
            assert row["iteration_count"] > 0
            assert row["basis_functions"] > 0
            assert row["elapsed_seconds"] > 0
            assert row["energy_per_atom_hartree"] == pytest.approx(
                row["energy_hartree"] / len(job["spec"]["atoms"]), abs=1e-12)
        if template_id == "hydrogen_dissociation_v1":
            minimum = min(report["rows"], key=lambda row: row["energy_hartree"])
            assert minimum["value"] == 0.74
            assert report["analysis"]["energy_trend"] == "mixed"
        if template_id == "water_correlation_v1":
            assert [job["result"]["method"] for job in report["jobs"]] == ["HF", "MP2", "CCSD"]
            assert report["jobs"][1]["result"]["correlation"]["converged"]
        retried = client.post("/api/experiments", json=payload).json()
        assert retried["id"] == experiment_id
        assert len(client.get("/api/jobs").json()) == report["total_jobs"]
