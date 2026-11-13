"""Real command-line runs use the same durable isolated-worker lifecycle."""
from __future__ import annotations

import fcntl
import json
import os
import selectors
import signal
import subprocess
import sys

import pytest

from quantum_lab.jobs import JobManager
from quantum_lab.models import JobSpec
from quantum_lab.storage import Store


def cli(*args, cwd, input_text=None, env=None):
    return subprocess.run([sys.executable, "-m", "quantum_lab.cli", *args], cwd=cwd,
                          input=input_text, text=True, capture_output=True, timeout=60, env=env)


def hydrogen(**updates):
    return {"name": "CLI hydrogen", "atoms": [{"symbol": "H", "position": [0, 0, 0]},
                                               {"symbol": "H", "position": [0, 0, .74]}],
            "threads": 1, **updates}


def result_events(output):
    return [json.loads(line) for line in output.splitlines() if line]


def test_doctor_and_example_commands_are_read_only(tmp_path):
    doctor = cli("doctor", "--json", cwd=tmp_path)
    assert doctor.returncode == 0, doctor.stderr
    report = json.loads(doctor.stdout)
    for name in ("Python", "PySCF", "NumPy", "ASE", "geomeTRIC", "Basis Set Exchange", "Libxc"):
        assert report["versions"][name]
    assert report["ready_for_headless"]
    assert "not performed" in report["platform"]["wsl_validation"]
    assert report["resources"]["logical_cpus"] >= 1
    assert isinstance(report["frontend"]["built"], bool)
    assert not list(tmp_path.iterdir())
    listing = cli("examples", cwd=tmp_path)
    assert listing.returncode == 0 and "h2\t" in listing.stdout and "diamond\t" in listing.stdout
    example = cli("example", "h2", cwd=tmp_path)
    assert example.returncode == 0, example.stderr
    spec = JobSpec.model_validate_json(example.stdout)
    assert len(spec.atoms) == 2 and spec.method == "HF"
    assert not list(tmp_path.iterdir())
    invalid = cli("example", "unknown-example", cwd=tmp_path)
    assert invalid.returncode == 1 and "Unknown example" in invalid.stderr


@pytest.mark.science
def test_actual_hydrogen_cli_events_saved_result_and_repeat_preserves_jobs(tmp_path):
    example = cli("example", "h2", cwd=tmp_path)
    source = tmp_path / "input.json"
    settings = json.loads(example.stdout)
    settings["threads"] = 1
    source.write_text(json.dumps(settings))
    environment = {**os.environ, "QUANTUM_LAB_DATA": str(tmp_path / "unrelated-web-data")}
    first = cli("run", str(source), "--json-events", cwd=tmp_path, env=environment)
    assert first.returncode == 0, first.stderr
    events = result_events(first.stdout)
    submitted, final = events[0], events[-1]
    assert submitted["type"] == "submitted" and final["type"] == "result"
    assert submitted["job_id"] == final["job_id"]
    assert final["status"] == "succeeded" and final["exit_code"] == 0
    assert -1.12 < final["energy_hartree"] < -1.11
    assert any(entry["type"] == "scf_iteration" for entry in events)
    seq = [entry["seq"] for entry in events if "seq" in entry]
    assert seq == list(range(1, len(seq) + 1))
    store = Store(tmp_path / ".quantum-lab-cli")
    job = store.get(final["job_id"])
    assert job["status"] == "succeeded"
    result_file = store.job_dir(job["id"]) / "result.json"
    original = result_file.read_bytes()
    saved = json.loads(original)
    assert saved["converged"] and saved["energy_hartree"] == final["energy_hartree"]
    assert (store.job_dir(job["id"]) / "engine.log").is_file()
    assert not (tmp_path / "unrelated-web-data").exists()
    # stdin is useful with `quantum-lab example h2 | quantum-lab run -`.
    again = cli("run", "-", "--json-events", cwd=tmp_path, input_text=json.dumps(settings))
    assert again.returncode == 0, again.stderr
    repeated = result_events(again.stdout)[-1]
    assert repeated["job_id"] != final["job_id"]
    assert repeated["energy_hartree"] == pytest.approx(final["energy_hartree"], abs=1e-10)
    assert result_file.read_bytes() == original
    assert len(store.list()) == 2


@pytest.mark.parametrize("payload", ["{", json.dumps(hydrogen(spin=1)), json.dumps(hydrogen(
    task="volume", parent_job_id="00000000-0000-0000-0000-000000000001", volume={"kind": "density"},
))])
def test_invalid_inputs_do_not_create_job_store(tmp_path, payload):
    result = cli("run", "-", cwd=tmp_path, input_text=payload)
    assert result.returncode == 1
    assert "Input error" in result.stderr
    assert not (tmp_path / ".quantum-lab-cli").exists()


@pytest.mark.science
def test_engine_failure_has_nonzero_exit_and_saved_diagnosis(tmp_path):
    result = cli("run", "-", "--json-events", cwd=tmp_path,
                 input_text=json.dumps(hydrogen(basis="not-a-basis-set")))
    assert result.returncode == 1
    final = result_events(result.stdout)[-1]
    assert final["status"] == "failed" and final["error"]
    assert final["result_file"] is None
    store = Store(tmp_path / ".quantum-lab-cli")
    job = store.get(final["job_id"])
    assert job["status"] == "failed"
    assert (store.job_dir(job["id"]) / "error.json").is_file()


@pytest.mark.science
def test_unconverged_cli_has_distinct_exit_code(tmp_path):
    settings = {"atoms": [{"symbol": "O", "position": [0, 0, 0]},
                           {"symbol": "H", "position": [0, -.757, .587]},
                           {"symbol": "H", "position": [0, .757, .587]}],
                "threads": 1, "max_cycle": 1}
    result = cli("run", "-", "--json-events", cwd=tmp_path, input_text=json.dumps(settings))
    assert result.returncode == 2, result.stderr
    final = result_events(result.stdout)[-1]
    assert final["status"] == "unconverged" and final["exit_code"] == 2
    assert final["result_file"]
    store = Store(tmp_path / ".quantum-lab-cli")
    assert store.get(final["job_id"])["result"]["converged"] is False


def test_cli_respects_server_data_lock_and_does_not_enqueue(tmp_path):
    store = Store(tmp_path / "web-data")
    manager = JobManager(store)
    manager.start()
    try:
        result = cli("run", "-", "--data-dir", str(store.root), cwd=tmp_path,
                     input_text=json.dumps(hydrogen()))
        assert result.returncode == 1
        assert "already owns" in result.stderr
        assert store.list() == []
        assert manager._instance_lock is not None
    finally:
        manager.stop()


def test_cli_refuses_existing_unfinished_jobs_under_lock(tmp_path):
    store = Store(tmp_path / "existing-data")
    pending = store.create(JobSpec.model_validate(hydrogen()).model_dump(mode="json"))
    before = store.get(pending["id"])
    result = cli("run", "-", "--data-dir", str(store.root), cwd=tmp_path,
                 input_text=json.dumps(hydrogen()))
    assert result.returncode == 1 and "unfinished jobs" in result.stderr
    assert store.get(pending["id"]) == before
    assert len(store.list()) == 1
    # A rejected command must release its acquired directory lock.
    with (store.root / "server.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def test_keyboard_interrupt_cancels_only_owned_cli_job(tmp_path):
    source = tmp_path / "input.json"
    source.write_text(json.dumps(hydrogen()))
    process = subprocess.Popen([sys.executable, "-m", "quantum_lab.cli", "run", str(source), "--json-events"],
                               cwd=tmp_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            assert selector.select(timeout=30), "CLI did not submit a job"
            submitted = json.loads(process.stdout.readline())
        assert submitted["type"] == "submitted"
        process.send_signal(signal.SIGINT)
        remaining, error = process.communicate(timeout=15)
        assert process.returncode == 130, error
        final = result_events(remaining)[-1]
        assert final["status"] == "cancelled" and final["exit_code"] == 130
        store = Store(tmp_path / ".quantum-lab-cli")
        assert len(store.list()) == 1
        assert store.get(submitted["job_id"])["status"] == "cancelled"
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=15)
