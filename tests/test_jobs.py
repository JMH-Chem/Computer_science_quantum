import json
import os
import time
from itertools import pairwise

import pytest

from quantum_lab.jobs import JobManager
from quantum_lab.storage import TERMINAL, Store

FAKE_WORKER = '''
import os,sys,time
from quantum_lab.storage import Store,atomic_json
s=Store(sys.argv[1]); j=s.get(sys.argv[2]); d=s.job_dir(j['id'])
spec=j['spec']; name=spec['name']
(d/'pid.txt').write_text(str(os.getpid()))
for i in range(100 if name=='slow' else 3):
    s.emit(j['id'],'scf_iteration', {'iteration':i+1,'energy_hartree':[-1.,-.9,-1.1][i%3]})
    time.sleep(.05)
if name=='failure':
    atomic_json(d/'error.json',{'message':'diagnosed test failure'})
    sys.exit(1)
atomic_json(d/'result.json',{'converged':name!='unconverged','energy_hartree':-1.1})
'''


@pytest.fixture
def fake_worker(tmp_path, monkeypatch):
    (tmp_path / "test_quantum_worker.py").write_text(FAKE_WORKER)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    return "test_quantum_worker"


def wait_status(store, job_id, expected=TERMINAL, timeout=10):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        job = store.get(job_id)
        if job["status"] in expected:
            return job
        time.sleep(0.02)
    raise AssertionError(f"Job {job_id} did not reach {expected}: {store.get(job_id)}")


def spec(name="normal"):
    return {"name": name, "domain": "molecule", "task": "single_point", "threads": 1}


def test_event_order_finite_json_and_cancelled_state_is_final(tmp_path):
    store = Store(tmp_path)
    job = store.create(spec())
    store.claim()
    store.emit(job["id"], "scf_iteration", {"energy_hartree": -1.0, "gradient_norm": float("nan")})
    store.emit(job["id"], "scf_iteration", {"energy_hartree": -0.9})
    store.transition(job["id"], "cancelled")
    assert not store.transition(job["id"], "succeeded", result={"converged": True})
    assert store.emit(job["id"], "late", {}) is None
    events = store.events(job["id"])
    assert [e["seq"] for e in events] == list(range(1, 6))
    assert events[2]["data"]["gradient_norm"] is None
    assert events[3]["data"]["energy_hartree"] > events[2]["data"]["energy_hartree"]
    assert [e["seq"] for e in store.events(job["id"], 3)] == [4, 5]
    json.dumps(events, allow_nan=False)


def test_queue_is_sequential_and_distinguishes_failure_and_nonconvergence(tmp_path, fake_worker):
    store = Store(tmp_path / "data")
    manager = JobManager(store, fake_worker)
    jobs = [manager.submit(spec(name)) for name in ("normal", "unconverged", "failure")]
    manager.start()
    try:
        results = [wait_status(store, j["id"]) for j in jobs]
        assert [j["status"] for j in results] == ["succeeded", "unconverged", "failed"]
        for left, right in pairwise(jobs):
            terminal = store.events(left["id"])[-1]
            started = store.events(right["id"])[1]
            assert terminal["timestamp"] <= started["timestamp"]
        assert results[2]["error"] == "diagnosed test failure"
    finally:
        manager.stop()


def test_cancel_queued_and_running_preserves_events(tmp_path, fake_worker):
    store = Store(tmp_path / "data")
    manager = JobManager(store, fake_worker)
    first = manager.submit(spec("slow"))
    second = manager.submit(spec())
    assert manager.cancel(second["id"])["status"] == "cancelled"
    manager.start()
    try:
        wait_status(store, first["id"], {"running"})
        deadline = time.monotonic() + 5
        pidfile = store.job_dir(first["id"]) / "pid.txt"
        while not pidfile.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert pidfile.exists()
        pid = int(pidfile.read_text())
        assert manager.cancel(first["id"])["status"] == "cancelled"
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        assert store.events(first["id"])[-1]["data"]["status"] == "cancelled"
        assert not (store.job_dir(second["id"]) / "pid.txt").exists()
        following = manager.submit(spec())
        assert wait_status(store, following["id"])["status"] == "succeeded"
        assert store.get(first["id"])["status"] == "cancelled"
    finally:
        manager.stop()


def test_recovery_and_single_server_ownership(tmp_path, fake_worker):
    store = Store(tmp_path / "data")
    stale = store.create(spec())
    store.claim()
    queued = store.create(spec())
    manager = JobManager(store, fake_worker)
    manager.start()
    other = JobManager(Store(store.root), fake_worker)
    try:
        recovered = store.get(stale["id"])
        assert recovered["status"] == "interrupted"
        assert recovered["spec"] == stale["spec"]
        assert json.loads((store.job_dir(stale["id"]) / "input.json").read_text()) == stale["spec"]
        assert store.events(stale["id"])[-1]["data"]["status"] == "interrupted"
        with pytest.raises(RuntimeError, match="already owns"):
            other.start()
        assert wait_status(store, queued["id"])["status"] == "succeeded"
    finally:
        manager.stop()


def test_shutdown_interrupts_running_job(tmp_path, fake_worker):
    store = Store(tmp_path / "data")
    manager = JobManager(store, fake_worker)
    job = manager.submit(spec("slow"))
    manager.start()
    wait_status(store, job["id"], {"running"})
    manager.stop()
    assert store.get(job["id"])["status"] == "interrupted"
