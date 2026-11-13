"""Scientific integrity of the guided basis experiment and its real queued runs.

Synthetic rows exercise interpretation of inconvenient data; the integration
test separately executes all three jobs and rebuilds direct PySCF references
from their recorded, exact Basis Set Exchange definitions.
"""
from __future__ import annotations

import copy
import csv
import io
import json
import time
from uuid import uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pyscf import dft, gto, lib

from quantum_lab.api import create_app
from quantum_lab.experiment_science import build_specs, summarize, templates

BASES = ["sto-3g", "def2-svp", "def2-tzvp"]


def experiment_and_jobs(energies=(-75.0, -76.0, -75.5)):
    specs = build_specs("water_basis_v1", "Scientific comparison fixture")
    experiment = {
        "id": str(uuid4()), "name": "Scientific comparison fixture",
        "template_id": "water_basis_v1", "prediction": "unsure", "request_id": str(uuid4()),
        "created_at": "2026-11-14T00:00:00+00:00", "cancel_requested": False,
        "base_spec": copy.deepcopy(specs[0]), "bases": BASES.copy(),
    }
    jobs = []
    for index, (spec, energy) in enumerate(zip(specs, energies, strict=True)):
        jobs.append({
            "id": str(uuid4()), "name": spec["name"], "status": "succeeded", "spec": spec,
            "error": None, "created_at": experiment["created_at"], "updated_at": experiment["created_at"],
            "result": {
                "converged": True, "scf_converged": True, "energy_hartree": energy,
                "atoms": copy.deepcopy(spec["atoms"]), "lattice": None,
                "method": "DFT", "task": "single_point", "electron_count": 10,
                "engine": {"name": "PySCF molecular", "version": "2.15.0"},
                "provenance": {
                    "basis_provider": "Basis Set Exchange", "basis_provider_version": "0.12",
                    "libxc_version": "7.1.2", "numpy_version": "2.5.4",
                    "basis_definition_artifact": "basis.json", "basis_citations_artifact": "basis-references.txt",
                },
                "basis_functions": [7, 24, 43][index], "iteration_count": [8, 10, 12][index],
                "elapsed_seconds": [2.5, 2.0, 4.0][index], "peak_memory_mb": [130, 160, 190][index],
            },
        })
    return experiment, jobs


def test_template_varies_only_basis_and_preserves_independent_inputs():
    template = next(item for item in templates() if item["id"] == "water_basis_v1")
    specs = build_specs(template["id"], "Water basis experiment")
    assert [spec["basis"] for spec in specs] == BASES == template["bases"]
    constant_inputs = [{key: value for key, value in spec.items() if key not in {"basis", "name"}}
                       for spec in specs]
    assert constant_inputs[0] == constant_inputs[1] == constant_inputs[2]
    fixed = specs[0]
    assert fixed["method"] == "DFT" and fixed["functional"] == "PBE"
    assert fixed["grid_level"] == 3 and fixed["threads"] == 1
    assert fixed["charge"] == fixed["spin"] == 0 and fixed["restricted"]
    assert fixed["task"] == "single_point" and fixed["ecp"] is None
    assert [a["symbol"] for a in fixed["atoms"]] == ["O", "H", "H"]
    np.testing.assert_allclose([a["position"] for a in fixed["atoms"]],
                               [[0, 0, 0], [0, -0.757, 0.587], [0, 0.757, 0.587]])
    specs[0]["atoms"][0]["position"][0] = 3.0
    assert specs[1]["atoms"][0]["position"][0] == 0
    assert build_specs(template["id"], "New run")[0]["atoms"][0]["position"][0] == 0


@pytest.mark.parametrize("energies,trend", [
    ((-75, -76, -75.5), "mixed"),
    ((-75, -74, -73), "increased"),
    ((-75, -76, -77), "decreased"),
    ((-75, -75, -75), "unchanged"),
])
def test_comparison_preserves_signed_adjacent_differences_and_nonmonotonic_results(energies, trend):
    experiment, jobs = experiment_and_jobs(energies)
    report = summarize(experiment, jobs)
    assert report["status"] == "completed"
    assert report["analysis"]["comparable"] is True
    assert report["analysis"]["energy_trend"] == trend
    assert [row["basis"] for row in report["rows"]] == BASES
    assert [row["energy_hartree"] for row in report["rows"]] == list(energies)
    assert report["rows"][0]["delta_previous_hartree"] is None
    assert report["rows"][1]["delta_previous_hartree"] == pytest.approx(energies[1] - energies[0])
    assert report["rows"][2]["delta_previous_hartree"] == pytest.approx(energies[2] - energies[1])
    assert report["analysis"]["total_change_hartree"] == pytest.approx(energies[2] - energies[0])
    assert [row["elapsed_seconds"] for row in report["rows"]] == [2.5, 2.0, 4.0]
    assert report["analysis"].get("basis_converged") is not True


@pytest.mark.parametrize("status", ["queued", "running", "failed", "unconverged", "cancelled", "interrupted"])
def test_unfinished_or_failed_middle_never_becomes_zero_or_a_bridged_difference(status):
    experiment, jobs = experiment_and_jobs()
    jobs[1]["status"] = status
    jobs[1]["error"] = "Diagnostic fixture" if status in {"failed", "interrupted"} else None
    if status in {"queued", "running", "failed", "cancelled", "interrupted"}:
        jobs[1]["result"] = None
    else:
        jobs[1]["result"]["converged"] = False
        jobs[1]["result"]["scf_converged"] = False
    report = summarize(experiment, jobs)
    assert report["analysis"]["comparable"] is False
    assert report["rows"][1]["comparable"] is False
    assert report["rows"][1]["energy_hartree"] is None
    assert report["rows"][1]["delta_previous_hartree"] is None
    assert report["rows"][2]["delta_previous_hartree"] is None
    assert report["analysis"]["last_step_delta_hartree"] is None
    assert report["analysis"]["total_change_hartree"] is None


@pytest.mark.parametrize("energy", [None, float("nan"), float("inf"), -float("inf"), True])
def test_nonfinite_or_missing_energy_cannot_be_interpreted_as_a_converged_measurement(energy):
    experiment, jobs = experiment_and_jobs()
    jobs[1]["result"]["energy_hartree"] = energy
    report = summarize(experiment, jobs)
    assert report["analysis"]["comparable"] is False
    assert report["rows"][1]["comparable"] is False
    assert report["rows"][1]["energy_hartree"] is None
    assert report["rows"][1]["delta_previous_hartree"] is None
    assert report["rows"][2]["delta_previous_hartree"] is None


@pytest.mark.parametrize("field,value", [
    ("charge", 2), ("spin", 2), ("functional", "PBE0"), ("grid_level", 1),
    ("conv_tol", 1e-5), ("ecp", "lanl2dz"), ("restricted", False), ("method", "HF"),
    ("basis", "sto-3g"),
])
def test_changed_physical_or_numerical_conditions_cannot_enter_basis_only_comparison(field, value):
    experiment, jobs = experiment_and_jobs()
    jobs[1]["spec"][field] = value
    report = summarize(experiment, jobs)
    assert report["analysis"]["comparable"] is False
    assert report["rows"][1]["comparable"] is False
    assert report["rows"][1]["delta_previous_hartree"] is None
    assert report["rows"][2]["delta_previous_hartree"] is None


@pytest.mark.parametrize("target", ["spec", "result"])
def test_geometry_change_in_input_or_result_is_not_mislabeled_as_basis_effect(target):
    experiment, jobs = experiment_and_jobs()
    jobs[1][target]["atoms"][1]["position"][2] += 0.1
    report = summarize(experiment, jobs)
    assert report["analysis"]["comparable"] is False
    assert report["rows"][1]["comparable"] is False
    assert report["rows"][1]["delta_previous_hartree"] is None
    assert report["rows"][2]["delta_previous_hartree"] is None


@pytest.mark.parametrize("area,key", [
    ("engine", "version"), ("provenance", "libxc_version"),
    ("provenance", "basis_provider_version"), ("provenance", "basis_provider"),
    ("provenance", "numpy_version"),
])
def test_mixed_engine_or_basis_data_versions_do_not_claim_one_controlled_experiment(area, key):
    experiment, jobs = experiment_and_jobs()
    jobs[1]["result"][area][key] = "different"
    report = summarize(experiment, jobs)
    assert report["analysis"]["comparable"] is False
    assert report["rows"][1]["delta_previous_hartree"] is None
    assert report["rows"][2]["delta_previous_hartree"] is None


def test_success_label_without_engine_convergence_does_not_certify_result():
    experiment, jobs = experiment_and_jobs()
    jobs[1]["result"]["converged"] = False
    report = summarize(experiment, jobs)
    assert report["status"] == "completed_with_issues"
    assert report["rows"][1]["comparable"] is False
    assert report["rows"][1]["energy_hartree"] is None
    assert report["analysis"]["comparable"] is False


def test_missing_provenance_and_display_name_changes_have_distinct_meaning():
    experiment, jobs = experiment_and_jobs()
    jobs[1]["name"] = "User display label"
    jobs[1]["spec"]["name"] = "Another display label"
    assert summarize(experiment, jobs)["analysis"]["comparable"] is True
    jobs[1]["result"]["provenance"] = {}
    report = summarize(experiment, jobs)
    assert report["analysis"]["comparable"] is False
    assert report["rows"][1]["comparable"] is False
    assert report["rows"][1]["delta_previous_hartree"] is None
    assert report["rows"][2]["delta_previous_hartree"] is None


def test_learner_prediction_does_not_change_measured_energy_or_measured_trend():
    experiment, jobs = experiment_and_jobs((-75, -75 + 1e-8, -75 + 2e-8))
    reports = []
    for prediction in ("lower", "higher", "unsure"):
        experiment["prediction"] = prediction
        reports.append(summarize(experiment, jobs))
    assert reports[0]["rows"] == reports[1]["rows"] == reports[2]["rows"]
    assert all(report["analysis"]["energy_trend"] == "increased" for report in reports)
    assert all(report["analysis"].get("basis_converged") is not True for report in reports)


@pytest.mark.science
def test_real_queued_water_experiment_matches_exact_direct_references_and_exports(tmp_path):
    """A request traverses the actual durable scheduler and three engine workers."""
    data = tmp_path / "data"
    with TestClient(create_app(data)) as client:
        response = client.post("/api/experiments", json={
            "template_id": "water_basis_v1", "name": "Real water basis comparison",
            "prediction": "unsure", "request_id": str(uuid4()),
        })
        assert response.status_code == 201, response.text
        experiment_id = response.json()["id"]
        deadline = time.monotonic() + 90
        while True:
            response = client.get(f"/api/experiments/{experiment_id}")
            assert response.status_code == 200, response.text
            report = response.json()
            if report["status"] not in {"queued", "running"}:
                break
            assert time.monotonic() < deadline, report
            time.sleep(0.1)
        assert report["status"] == "completed", report
        assert report["successful_jobs"] == report["completed_jobs"] == report["total_jobs"] == 3
        assert report["analysis"]["comparable"] is True
        assert len({row["job_id"] for row in report["rows"]}) == 3
        assert [row["basis"] for row in report["rows"]] == BASES
        direct_energies = []
        lib.num_threads(1)
        for row in report["rows"]:
            job = client.get(f"/api/jobs/{row['job_id']}").json()
            spec = job["spec"]
            result = job["result"]
            assert job["status"] == "succeeded" and result["converged"] is True
            assert result["provenance"]["basis_provider"] == "Basis Set Exchange"
            definitions = json.loads((data / "jobs" / job["id"] / "basis.json").read_text())
            molecule = gto.M(
                atom=[(atom["symbol"], atom["position"]) for atom in spec["atoms"]],
                unit="Angstrom", basis=definitions["resolved_pyscf_basis"],
                ecp=definitions["resolved_pyscf_ecp"], charge=spec["charge"], spin=spec["spin"], verbose=0,
            )
            direct = dft.RKS(molecule)
            direct.xc = spec["functional"]
            direct.grids.level = spec["grid_level"]
            direct.conv_tol = spec["conv_tol"]
            direct.max_cycle = spec["max_cycle"]
            direct.kernel()
            assert direct.converged
            assert abs(row["energy_hartree"] - direct.e_tot) < 1e-8
            assert row["basis_functions"] == molecule.nao_nr()
            assert row["iteration_count"] == result["iteration_count"] > 0
            assert row["elapsed_seconds"] > 0 and row["peak_memory_mb"] > 0
            direct_energies.append(float(direct.e_tot))
        np.testing.assert_allclose([row["delta_previous_hartree"] for row in report["rows"][1:]],
                                   np.diff(direct_energies), atol=2e-8, rtol=0)
        exported = client.get(f"/api/experiments/{experiment_id}/export?format=json")
        assert exported.status_code == 200
        exported_text = exported.text
        assert "basis_provider_version" in exported_text and "grid_level" in exported_text
        assert all(row["job_id"] in exported_text for row in report["rows"])
        csv_response = client.get(f"/api/experiments/{experiment_id}/export?format=csv")
        assert csv_response.status_code == 200
        rows = list(csv.DictReader(io.StringIO(csv_response.text)))
        assert len(rows) == 3
        assert [row["basis"] for row in rows] == BASES
        assert rows[0]["delta_previous_hartree"] == ""
        np.testing.assert_allclose([float(row["energy_hartree"]) for row in rows], direct_energies,
                                   atol=1e-8, rtol=0)
