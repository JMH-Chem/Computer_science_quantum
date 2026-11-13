"""Experiment lifecycle tests use a deliberately synthetic worker, not fake science validation."""
import csv
import io
import json
import os
import sqlite3
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
from fastapi.testclient import TestClient

import quantum_lab.storage as storage
from quantum_lab.api import create_app
from quantum_lab.experiment_science import build_specs
from quantum_lab.storage import TERMINAL, Store


def payload(**changes):
    return {"name": "Water basis lifecycle test", "template_id": "water_basis_v1",
            "prediction": "lower", "request_id": str(uuid.uuid4()), **changes}


def synthetic_result(spec):
    basis = spec["basis"]
    energy, count = {"sto-3g": (-75., 7), "def2-svp": (-76., 24), "def2-tzvp": (-76.1, 43)}[basis]
    return {"converged": True, "scf_converged": True, "energy_hartree": energy,
            "basis_functions": count, "iteration_count": 5, "elapsed_seconds": .1,
            "peak_memory_mb": 128., "atoms": spec["atoms"], "lattice": spec.get("lattice"),
            "method": spec["method"], "task": spec["task"],
            "engine": {"name": "PySCF molecular", "version": "synthetic-test"},
            "provenance": {"basis_provider": "Basis Set Exchange", "basis_provider_version": "0.12",
                           "libxc_version": "7.1.2", "numpy_version": "2.5.4"}}


@pytest.fixture
def client(tmp_path, monkeypatch):
    worker = tmp_path / "experiment_test_worker.py"
    worker.write_text('''
import sys,time
from quantum_lab.storage import Store,atomic_json
s=Store(sys.argv[1]);j=s.get(sys.argv[2]);spec=j['spec'];d=s.job_dir(j['id'])
delay=0.9 if spec['name'].startswith('slow unrelated') else (0.05 if spec['basis']=='sto-3g' else 0.3)
s.emit(j['id'],'scf_iteration',{'iteration':1,'energy_hartree':-70.})
time.sleep(delay)
energy,count={'sto-3g':(-75.,7),'def2-svp':(-76.,24),'def2-tzvp':(-76.1,43)}[spec['basis']]
result={'converged':True,'scf_converged':True,'energy_hartree':energy,'basis_functions':count,
 'iteration_count':5,'elapsed_seconds':delay,'peak_memory_mb':128.,'atoms':spec['atoms'],
 'lattice':spec.get('lattice'),'method':spec['method'],'task':spec['task'],
 'engine':{'name':'PySCF molecular','version':'synthetic-test'},
 'provenance':{'basis_provider':'Basis Set Exchange','basis_provider_version':'0.12',
 'libxc_version':'7.1.2','numpy_version':'2.5.4'}}
atomic_json(d/'result.json',result)
atomic_json(d/'basis.json',{'test_only':True,'basis':spec['basis']})
(d/'basis-references.txt').write_text('Synthetic lifecycle test fixture')
''')
    monkeypatch.setenv("PYTHONPATH", str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    with TestClient(create_app(tmp_path / "data", worker_module="experiment_test_worker")) as result:
        yield result


def wait_for(fn, predicate, timeout=12):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = fn()
        if predicate(value):
            return value
        time.sleep(.02)
    pytest.fail(f"Condition did not become true; last value: {value}")


def wait_experiment(client, experiment_id):
    return wait_for(lambda: client.get(f"/api/experiments/{experiment_id}").json(),
                    lambda value: all(job["status"] in TERMINAL for job in value["jobs"]))


def test_create_order_results_export_and_persisted_history(client):
    templates = client.get("/api/experiments/templates")
    assert templates.status_code == 200
    assert templates.json()[0]["id"] == "water_basis_v1"
    response = client.post("/api/experiments", json=payload())
    assert response.status_code == 201, response.text
    group = wait_experiment(client, response.json()["id"])
    assert group["status"] == "completed"
    assert group["total_jobs"] == group["completed_jobs"] == group["successful_jobs"] == 3
    assert [j["spec"]["basis"] for j in group["jobs"]] == ["sto-3g", "def2-svp", "def2-tzvp"]
    assert len({j["id"] for j in group["jobs"]}) == 3
    assert group["rows"][0]["delta_previous_hartree"] is None
    assert group["rows"][1]["delta_previous_hartree"] == pytest.approx(-1)
    assert group["rows"][2]["delta_previous_hartree"] == pytest.approx(-.1)
    specs = [{k: v for k, v in j["spec"].items() if k not in {"name", "basis"}} for j in group["jobs"]]
    assert specs[0] == specs[1] == specs[2]
    assert specs[0]["grid_level"] == 3
    exported = client.get(f"/api/experiments/{group['id']}/export?format=json")
    assert exported.status_code == 200
    data = exported.json()
    assert data["export_format_version"] == 1
    assert len(data["job_artifacts"]) == len(data["events"]) == 3
    assert all(x["basis.json"]["test_only"] for x in data["job_artifacts"].values())
    rows = list(csv.DictReader(io.StringIO(client.get(f"/api/experiments/{group['id']}/export?format=csv").text)))
    assert len(rows) == 3 and rows[0]["delta_previous_hartree"] == ""
    assert rows[1]["delta_previous_hartree"] == "-1.0"
    assert client.get("/api/experiments").json()[0]["id"] == group["id"]
    assert "request_payload" not in group
    reopened = Store(client.app.state.store.root)
    saved, jobs = reopened.experiment_bundle(group["id"])
    assert saved["base_spec"] == group["base_spec"]
    assert [j["id"] for j in jobs] == [j["id"] for j in group["jobs"]]
    # Cancelling an already complete group must not label it cancelled.
    completed = client.post(f"/api/experiments/{group['id']}/cancel").json()
    assert completed["status"] == "completed" and not completed["cancel_requested"]


def test_concurrent_retry_reuses_group_and_rejects_changed_request(client):
    original = payload()
    barrier = Barrier(8)

    def create(_):
        barrier.wait(timeout=10)
        return client.post("/api/experiments", json=original)

    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(create, range(8)))
    assert all(r.status_code == 201 for r in responses)
    assert len({r.json()["id"] for r in responses}) == 1
    assert len(client.get("/api/jobs").json()) == 3
    assert len(client.get("/api/experiments").json()) == 1
    changed = client.post("/api/experiments", json={**original, "prediction": "higher"})
    assert changed.status_code == 409
    assert len(client.get("/api/jobs").json()) == 3


@pytest.mark.parametrize("status", ["running", "failed"])
@pytest.mark.parametrize("problem,reason", [("partial_json", "invalid_json"), ("read_error", "unreadable")])
def test_export_preserves_snapshot_when_optional_artifact_is_unavailable(tmp_path, monkeypatch, status, problem, reason):
    # Use persisted job states without starting a worker, so artifact races can
    # be reproduced deterministically for both active and terminated jobs.
    app = create_app(tmp_path)
    app.state.store = store = Store(tmp_path)
    group = store.create_experiment(payload(), build_specs("water_basis_v1", "artifact export"))
    first = store.claim()
    store.transition(first["id"], "succeeded", result=synthetic_result(first["spec"]))
    valid_basis = {"test_only": True, "basis": first["spec"]["basis"], "definitions": {"H": "preserved"}}
    storage.atomic_json(store.job_dir(first["id"]) / "basis.json", valid_basis)
    active = store.claim()
    store.emit(active["id"], "scf_iteration", {"iteration": 1, "energy_hartree": -70.})
    if status == "failed":
        store.transition(active["id"], "failed", error="Worker stopped while writing artifact")
    partial = store.job_dir(active["id"]) / "basis.json"
    partial.write_text('{"elements":', encoding="utf-8")
    if problem == "read_error":
        original_read = Path.read_text

        def unreadable(path, *args, **kwargs):
            if path == partial:
                raise OSError("injected artifact read failure")
            return original_read(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", unreadable)
    client = TestClient(app)
    snapshot = client.get(f"/api/experiments/{group['id']}").json()
    response = client.get(f"/api/experiments/{group['id']}/export?format=json")
    assert response.status_code == 200
    exported = response.json()
    assert {key: exported[key] for key in snapshot} == snapshot
    assert exported["job_artifacts"][first["id"]]["basis.json"] == valid_basis
    assert "basis.json" not in exported["job_artifacts"][active["id"]]
    warning = exported["unavailable_artifacts"][active["id"]]["basis.json"]
    assert warning["reason"] == reason and "omitted" in warning["message"]
    assert first["id"] not in exported["unavailable_artifacts"]
    assert exported["events"] == {job["id"]: store.events(job["id"]) for job in snapshot["jobs"]}


@pytest.mark.parametrize("changes", [
    {"template_id": "missing"}, {"prediction": "always correct"}, {"request_id": "not-uuid"},
    {"basis": "custom"}, {"name": "   "}, {"threads": 8},
])
def test_invalid_requests_create_no_jobs(client, changes):
    response = client.post("/api/experiments", json=payload(**changes))
    assert response.status_code == 422
    assert client.get("/api/experiments").json() == []
    assert client.get("/api/jobs").json() == []


def test_unknown_ids_and_cross_origin_create(client):
    for experiment_id in ("invalid", str(uuid.uuid4())):
        assert client.get(f"/api/experiments/{experiment_id}").status_code == 404
        assert client.post(f"/api/experiments/{experiment_id}/cancel").status_code == 404
        assert client.get(f"/api/experiments/{experiment_id}/export").status_code == 404
    assert client.post("/api/experiments", json=payload(), headers={"Origin": "https://example.com"}).status_code == 403
    group = client.post("/api/experiments", json=payload()).json()
    assert client.get(f"/api/experiments/{group['id']}/export?format=exe").status_code == 422


def test_cancel_pending_group_does_not_cancel_unrelated_running_job(client):
    spec = build_specs("water_basis_v1", "unrelated")[0]
    spec["name"] = "slow unrelated"
    unrelated = client.post("/api/jobs", json=spec).json()
    wait_for(lambda: client.get(f"/api/jobs/{unrelated['id']}").json(), lambda j: j["status"] == "running")
    group = client.post("/api/experiments", json=payload()).json()
    stopped = client.post(f"/api/experiments/{group['id']}/cancel").json()
    assert stopped["status"] == "cancelled"
    assert all(job["status"] == "cancelled" for job in stopped["jobs"])
    assert all(row["energy_hartree"] is None for row in stopped["rows"])
    done = wait_for(lambda: client.get(f"/api/jobs/{unrelated['id']}").json(), lambda j: j["status"] in TERMINAL)
    assert done["status"] == "succeeded"


def test_cancel_active_group_keeps_completed_child_and_continues_queue(client):
    group = client.post("/api/experiments", json=payload()).json()
    progressed = wait_for(lambda: client.get(f"/api/experiments/{group['id']}").json(),
                          lambda e: e["jobs"][0]["status"] == "succeeded" and e["jobs"][1]["status"] == "running")
    first_result = progressed["jobs"][0]["result"]
    stopped = client.post(f"/api/experiments/{group['id']}/cancel").json()
    assert stopped["cancel_requested"] and stopped["status"] == "cancelled"
    assert [job["status"] for job in stopped["jobs"]] == ["succeeded", "cancelled", "cancelled"]
    assert stopped["jobs"][0]["result"] == first_result
    assert stopped["rows"][1]["delta_previous_hartree"] is None
    spec = build_specs("water_basis_v1", "after cancellation")[0]
    job = client.post("/api/jobs", json=spec).json()
    completed = wait_for(lambda: client.get(f"/api/jobs/{job['id']}").json(), lambda j: j["status"] in TERMINAL)
    assert completed["status"] == "succeeded"


def test_file_preparation_failure_has_no_partial_group_or_jobs(tmp_path, monkeypatch):
    store = Store(tmp_path)
    specs = build_specs("water_basis_v1", "atomic")
    existing = store.create(specs[0])
    before = sorted((store.root / "jobs").iterdir())
    original = storage.atomic_json
    calls = 0

    def fail_second(path, data):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected disk failure")
        original(path, data)

    request = payload()
    monkeypatch.setattr(storage, "atomic_json", fail_second)
    with pytest.raises(OSError, match="injected"):
        store.create_experiment(request, specs)
    assert store.list_experiments() == []
    assert [j["id"] for j in store.list()] == [existing["id"]]
    assert sorted((store.root / "jobs").iterdir()) == before
    monkeypatch.setattr(storage, "atomic_json", original)
    assert store.create_experiment(request, specs)["request_id"] == request["request_id"]


def test_database_failure_rolls_back_prepared_inputs_and_all_members(tmp_path):
    store = Store(tmp_path)
    with store.connection() as db:
        db.executescript("""
            CREATE TRIGGER reject_second BEFORE INSERT ON experiment_jobs
            WHEN NEW.ordinal=1 BEGIN SELECT RAISE(ABORT, 'injected database failure'); END;
        """)
    with pytest.raises(sqlite3.IntegrityError, match="injected"):
        store.create_experiment(payload(), build_specs("water_basis_v1", "atomic"))
    assert store.list_experiments() == [] and store.list() == []
    assert list((store.root / "jobs").iterdir()) == []


def test_existing_database_additive_upgrade_recovery_and_fifo_order(tmp_path):
    # Construct the old v0.1 schema directly, preserving an unrelated saved job.
    with sqlite3.connect(tmp_path / "jobs.sqlite3") as db:
        db.executescript("""
            CREATE TABLE jobs(id TEXT PRIMARY KEY,name TEXT NOT NULL,status TEXT NOT NULL,
                created_at TEXT NOT NULL,updated_at TEXT NOT NULL,spec TEXT NOT NULL,result TEXT,error TEXT);
            CREATE TABLE events(job_id TEXT NOT NULL REFERENCES jobs(id),seq INTEGER NOT NULL,
                timestamp TEXT NOT NULL,type TEXT NOT NULL,data TEXT NOT NULL,PRIMARY KEY(job_id,seq));
        """)
        old_id = str(uuid.uuid4())
        spec = build_specs("water_basis_v1", "original saved calculation")[0]
        db.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,?,NULL)",
                   (old_id, spec["name"], "succeeded", "2026-01-01", "2026-01-01", json.dumps(spec),
                    json.dumps(synthetic_result(spec))))
    store = Store(tmp_path)
    assert store.get(old_id)["result"] == synthetic_result(spec)
    request = payload()
    experiment = store.create_experiment(request, build_specs("water_basis_v1", "new"))
    _, jobs = store.experiment_bundle(experiment["id"])
    assert store.claim()["id"] == jobs[0]["id"]
    store.transition(jobs[0]["id"], "succeeded", result=synthetic_result(jobs[0]["spec"]))
    assert store.claim()["id"] == jobs[1]["id"]
    restored = Store(tmp_path)
    restored.recover()
    saved, recovered_jobs = restored.experiment_bundle(experiment["id"])
    assert saved["request_payload"] == request
    assert [j["status"] for j in recovered_jobs] == ["succeeded", "interrupted", "queued"]
    assert recovered_jobs[0]["result"] == synthetic_result(jobs[0]["spec"])
    assert restored.claim()["id"] == jobs[2]["id"]
    assert restored.get(old_id)["result"] == synthetic_result(spec)
