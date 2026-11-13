"""Portable exports retain scientific units, missing values and untrusted text safely."""
import csv
import hashlib
import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient

from quantum_lab.api import create_app
from quantum_lab.jobs import JobManager
from quantum_lab.models import JobSpec
from quantum_lab.reports import numeric_table


@pytest.fixture
def saved(tmp_path, monkeypatch):
    # Deliberately synthetic stored data: these tests exercise exports, not an engine.
    monkeypatch.setattr(JobManager, "start", lambda _: None)
    monkeypatch.setattr(JobManager, "stop", lambda _: None)
    app = create_app(tmp_path / "jobs")
    with TestClient(app) as client:
        store = app.state.store
        spec = JobSpec(name='<script>alert("input")</script>', atoms=[
            {"symbol": "H", "position": [0, 0, 0]}, {"symbol": "H", "position": [0, 0, .74]}]).model_dump(mode="json")
        job = store.create(spec)
        store.claim()
        for index, energy in enumerate([-1., -.9, -1.1]):
            store.emit(job["id"], "scf_iteration", {"iteration": index, "energy_hartree": energy})
        result = {
            "converged": True, "energy_hartree": -1.1, "atoms": spec["atoms"],
            "engine": {"name": "test-fixture", "version": "0"},
            "warnings": ['<img src=x onerror="alert(1)">'],
            "properties": {"mulliken_charges": [.1, -.1], "lowdin_charges": [.05, -.05], "reference_model": "RHF"},
            "vibrations": {"modes": [{"index": 0, "frequency_cm1": -30., "reduced_mass_amu": .5, "ir_intensity_km_mol": None}]},
            "sampled_dos": {"energy_ev": [-1., 0., 1.], "density_states_per_ev": [1., 2., 1.],
                            "occupied_density_states_per_ev": [1., 1., 0.]},
            "bands": {"distance": [0., 1.], "energies_ev": [[-1., 3.], [-.5, 4.]], "break_indices": [1]},
            "excitations": [{"state": 1, "energy_ev": 4., "oscillator_strength": 0., "converged": True}],
            "correlation": {"method": "MP2", "reference_energy_hartree": -1., "total_energy_hartree": -1.1,
                            "correlation_energy_hartree": -.1, "converged": True},
        }
        store.transition(job["id"], "succeeded", result=result)
        directory = store.job_dir(job["id"])
        (directory / "basis.json").write_text('{"basis":"test-only"}')
        (directory / "unfinished.tmp").write_text("must not export")
        secret = tmp_path / "outside.txt"
        secret.write_text("must not follow symlink")
        (directory / "outside-link.txt").symlink_to(secret)
        yield client, store, job["id"], result


def test_printable_report_escapes_input_and_preserves_signed_frequencies(saved):
    client, _, job_id, _ = saved
    response = client.get(f"/api/jobs/{job_id}/report")
    assert response.status_code == 200
    assert "<script>" not in response.text and "<img src=x" not in response.text
    assert "&lt;script&gt;" in response.text and "-30.0" in response.text
    assert "RHF" in response.text and "SCF" in response.text
    assert "default-src 'none'" in response.headers["content-security-policy"]
    assert "<polyline" in response.text


@pytest.mark.parametrize("kind,count", [("charges", 2), ("frequencies", 1), ("bands", 4),
                                       ("dos", 3), ("excitations", 1), ("correlation", 1)])
def test_numeric_csv_has_units_and_observed_rows(saved, kind, count):
    client, _, job_id, _ = saved
    response = client.get(f"/api/jobs/{job_id}/data?kind={kind}")
    assert response.status_code == 200
    rows = list(csv.DictReader(io.StringIO(response.text)))
    assert len(rows) == count
    if kind == "frequencies":
        assert rows[0]["frequency_cm1"] == "-30.0"
        assert rows[0]["ir_intensity_km_mol"] == ""
    if kind == "bands":
        assert [r["break_before"] for r in rows] == ["False", "False", "True", "True"]
    if kind == "dos":
        assert "density_states_per_ev_cell" in rows[0]


def test_missing_dataset_is_not_an_empty_successful_measurement(saved):
    client, _, job_id, _ = saved
    assert client.get(f"/api/jobs/{job_id}/data?kind=optimization").status_code == 422
    assert client.get(f"/api/jobs/{job_id}/data?kind=unknown").status_code == 422
    with pytest.raises(ValueError):
        numeric_table({"sampled_dos": {"energy_ev": [1]}}, "dos")


def test_zip_preserves_inputs_events_and_verifiable_artifacts(saved):
    client, store, job_id, _ = saved
    response = client.get(f"/api/jobs/{job_id}/bundle")
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert "artifacts/basis.json" in archive.namelist()
        assert not any("outside-link" in name or name.endswith(".tmp") for name in archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["job_id"] == job_id
        assert set(manifest["files"]) == set(archive.namelist()) - {"manifest.json"}
        for name, info in manifest["files"].items():
            content = archive.read(name)
            assert info["bytes"] == len(content)
            assert info["sha256"] == hashlib.sha256(content).hexdigest()
        assert json.loads(archive.read("input.json")) == store.get(job_id)["spec"]
        assert json.loads(archive.read("job.json"))["events"] == store.events(job_id)
        assert b"quantum-lab run input.json" in archive.read("README.txt")


def test_running_job_bundle_refused_and_failure_report_honest(saved):
    client, store, _, _ = saved
    job = store.create(JobSpec(atoms=[{"symbol": "He", "position": [0, 0, 0]}]).model_dump(mode="json"))
    assert client.get(f"/api/jobs/{job['id']}/bundle").status_code == 409
    store.claim()
    store.transition(job["id"], "failed", error="Convergence failure")
    response = client.get(f"/api/jobs/{job['id']}/report")
    assert response.status_code == 200 and "성공 결과가 아닙니다" in response.text
    assert "Convergence failure" in response.text


def test_oversized_bundle_reports_alternative(saved, monkeypatch):
    client, _, job_id, _ = saved
    monkeypatch.setattr("quantum_lab.reports.BUNDLE_LIMIT", 1)
    assert client.get(f"/api/jobs/{job_id}/bundle").status_code == 413


@pytest.mark.parametrize("changes", [
    {"method": "MP2", "restricted": False}, {"method": "CCSD", "task": "optimize"},
    {"method": "HF", "frozen_orbitals": 1}, {"method": "MP2", "frozen_orbitals": 1},
    {"method": "CASCI", "active_electrons": 3}, {"method": "CASCI", "active_electrons": 4},
    {"method": "CASSCF", "frozen_orbitals": 1}, {"method": "HF", "task": "tddft"},
    {"method": "DFT", "task": "tdhf"}, {"task": "eomccsd"},
    {"ir_intensities": True}, {"method": "MP2", "task": "frequencies"},
    {"task": "frequencies", "restricted": False}, {"task": "frequencies", "temperature_kelvin": 0},
])
def test_new_scientific_input_constraints(changes):
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        JobSpec(atoms=[{"symbol": "H", "position": [0, 0, 0]}, {"symbol": "H", "position": [0, 0, .74]}], **changes)
