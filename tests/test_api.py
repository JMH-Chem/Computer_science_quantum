import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from fastapi.testclient import TestClient

from quantum_lab.api import create_app
from quantum_lab.storage import TERMINAL


@pytest.fixture
def client(tmp_path, monkeypatch):
    (tmp_path / "api_test_worker.py").write_text('''
import sys,time
from quantum_lab.storage import Store,atomic_json
s=Store(sys.argv[1]);j=s.get(sys.argv[2]);d=s.job_dir(j['id'])
s.emit(j['id'],'scf_iteration',{'iteration':1,'energy_hartree':-1.0})
s.emit(j['id'],'scf_iteration',{'iteration':2,'energy_hartree':-0.9})
time.sleep(.1)
result={'converged':True,'energy_hartree':-1.1,'atoms':j['spec']['atoms'],'lattice':j['spec'].get('lattice')}
if j['spec']['task']=='volume':
    (d/'density.cube').write_text('test-only artifact')
    result={'converged':True,'artifact':'density.cube'}
atomic_json(d/'result.json',result)
''')
    monkeypatch.setenv("PYTHONPATH", str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    with TestClient(create_app(tmp_path / "data", worker_module="api_test_worker")) as c:
        yield c


def input_spec():
    return {"name": "H2", "domain": "molecule", "task": "single_point", "method": "HF", "basis": "sto-3g",
            "atoms": [{"symbol": "H", "position": [0, 0, 0]}, {"symbol": "H", "position": [0, 0, 0.74]}]}


def finish(client, job_id):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in TERMINAL:
            return job
        time.sleep(.02)
    pytest.fail("Worker did not complete")


def test_create_replay_export_and_volume_cache(client):
    response = client.post("/api/jobs", json=input_spec())
    assert response.status_code == 201, response.text
    job_id = response.json()["id"]
    assert finish(client, job_id)["status"] == "succeeded"
    history = client.get(f"/api/jobs/{job_id}/history").json()
    assert [e["seq"] for e in history] == list(range(1, len(history) + 1))
    sse = client.get(f"/api/jobs/{job_id}/events?after=1", headers={"Last-Event-ID": "2"})
    replay = [json.loads(line[6:]) for line in sse.text.splitlines() if line.startswith("data: ")]
    assert [e["seq"] for e in replay] == [e["seq"] for e in history if e["seq"] > 2]
    assert replay[-1]["data"]["status"] == "succeeded"
    assert client.get(f"/api/jobs/{job_id}/events", headers={"Last-Event-ID": "bad"}).status_code == 400
    for fmt in ("json", "csv", "xyz"):
        exported = client.get(f"/api/jobs/{job_id}/export?format={fmt}")
        assert exported.status_code == 200
        assert "attachment" in exported.headers["content-disposition"]
    assert "-0.9" in client.get(f"/api/jobs/{job_id}/export?format=csv").text
    assert client.get(f"/api/jobs/{job_id}/export?format=cif").status_code == 422
    volume = client.post(f"/api/jobs/{job_id}/volumes", json={"kind": "density", "grid": 40})
    assert volume.status_code == 201, volume.text
    child = volume.json()["id"]
    assert finish(client, child)["status"] == "succeeded"
    assert client.post(f"/api/jobs/{job_id}/volumes", json={"kind": "density", "grid": 40}).json()["id"] == child
    assert client.get(f"/api/jobs/{child}/artifacts/density.cube").status_code == 200


def test_invalid_structure_paths_and_cross_origin_mutation(client):
    assert client.post("/api/jobs", json={"atoms": []}).status_code == 422
    assert client.get("/api/jobs/not-a-uuid").status_code == 404
    assert client.post("/api/jobs", json=input_spec(), headers={"Origin": "https://unrelated.example"}).status_code == 403
    assert client.get("/api/health", headers={"Host": "unrelated.example"}).status_code == 400
    job_id = client.post("/api/jobs", json=input_spec()).json()["id"]
    finish(client, job_id)
    assert client.get(f"/api/jobs/{job_id}/artifacts/nonexistent").status_code == 404
    assert client.get(f"/api/jobs/{job_id}/artifacts/%2e%2e%2fjobs.sqlite3").status_code == 404
    bad = {**input_spec(), "task": "volume", "parent_job_id": job_id, "volume": {"kind": "density"}}
    assert client.post("/api/jobs", json=bad).status_code == 422


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_input_is_a_serializable_validation_error(client, token):
    content = '{"atoms":[{"symbol":"He","position":[' + token + ',0,0]}]}'
    response = client.post("/api/jobs", content=content, headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["type"] == "finite_number"
    assert error["input"] is None
    assert error["loc"] == ["body", "atoms", 0, "position", 0]
    json.dumps(response.json(), allow_nan=False)
    assert client.get("/api/jobs").json() == []


def test_model_validation_exception_context_is_serializable(client):
    response = client.post("/api/jobs", json={"atoms": [{"symbol": "He", "position": [0, 0, 0]}], "spin": 1})
    assert response.status_code == 422
    error = response.json()["detail"][0]
    assert error["type"] == "value_error"
    assert isinstance(error["ctx"]["error"], str)
    assert client.get("/api/jobs").json() == []


def test_concurrent_identical_volume_requests_create_one_child(client):
    job_id = client.post("/api/jobs", json=input_spec()).json()["id"]
    assert finish(client, job_id)["status"] == "succeeded"
    ready = Barrier(8)

    def request_volume(_):
        ready.wait(timeout=10)
        response = client.post(f"/api/jobs/{job_id}/volumes", json={"kind": "density", "grid": 40})
        assert response.status_code == 201, response.text
        return response.json()["id"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        identifiers = list(pool.map(request_volume, range(8)))
    assert len(set(identifiers)) == 1
    children = [job for job in client.get("/api/jobs").json() if job["spec"].get("parent_job_id") == job_id]
    assert len(children) == 1
    assert finish(client, identifiers[0])["status"] == "succeeded"


def test_cif_export_and_crystal_volume_restriction(client):
    spec = {**input_spec(), "domain": "crystal", "method": "DFT", "basis": "gth-szv", "functional": "PBE",
            "atoms": [{"symbol": "C", "position": [0, 0, 0]}, {"symbol": "C", "position": [.8917, .8917, .8917]}],
            "lattice": [[0, 1.7834, 1.7834], [1.7834, 0, 1.7834], [1.7834, 1.7834, 0]]}
    created = client.post("/api/jobs", json=spec)
    assert created.status_code == 201, created.text
    job_id = created.json()["id"]
    finish(client, job_id)
    cif = client.get(f"/api/jobs/{job_id}/export?format=cif")
    assert cif.status_code == 200
    assert "_cell_length_a" in cif.text
    assert client.post(f"/api/jobs/{job_id}/volumes", json={"kind": "orbital", "index": 0}).status_code == 422


@pytest.mark.science
def test_real_engine_process_and_cached_density_artifact(tmp_path):
    with TestClient(create_app(tmp_path / "real-data")) as c:
        spec = {**input_spec(), "threads": 1}
        created = c.post("/api/jobs", json=spec)
        assert created.status_code == 201, created.text
        job_id = created.json()["id"]
        calculation = finish(c, job_id)
        assert calculation["status"] == "succeeded", calculation["error"]
        assert abs(calculation["result"]["energy_hartree"] - (-1.1167593075)) < 1e-8
        request = {"kind": "density", "grid": 20}
        child = c.post(f"/api/jobs/{job_id}/volumes", json=request)
        assert child.status_code == 201, child.text
        child_id = child.json()["id"]
        volume = finish(c, child_id)
        assert volume["status"] == "succeeded", volume["error"]
        artifact = volume["result"]["artifact"]
        cube = c.get(f"/api/jobs/{child_id}/artifacts/{artifact}")
        assert cube.status_code == 200
        assert len(cube.content) > 1000
        assert c.post(f"/api/jobs/{job_id}/volumes", json=request).json()["id"] == child_id
