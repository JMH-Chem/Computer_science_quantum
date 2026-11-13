"""Actual worker records run-time software before execution, including failed jobs."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from quantum_lab import __version__
from quantum_lab.models import JobSpec
from quantum_lab.provenance import execution_environment
from quantum_lab.storage import Store


def test_environment_matches_source_and_excludes_host_identifiers():
    record = execution_environment()
    assert record["quantum_lab_version"] == __version__
    assert record["packages"]["pyscf"]
    assert record["packages"]["scipy"]
    assert {"hostname", "username", "cwd", "environment"}.isdisjoint(record)
    package = Path(__file__).resolve().parents[1] / "src" / "quantum_lab"
    assert record["source_sha256"]["worker.py"] == hashlib.sha256((package / "worker.py").read_bytes()).hexdigest()
    assert all(not Path(name).is_absolute() for name in record["source_sha256"])


@pytest.mark.parametrize("basis,succeeded", [("sto-3g", True), ("nonexistent-basis-for-provenance", False)])
def test_real_worker_captures_environment_for_success_and_failure(tmp_path, basis, succeeded):
    store = Store(tmp_path)
    spec = JobSpec(atoms=[{"symbol": "H", "position": [0, 0, 0]}, {"symbol": "H", "position": [0, 0, .74]}],
                   basis=basis, threads=1).model_dump(mode="json")
    job = store.create(spec)
    store.claim()
    completed = subprocess.run([sys.executable, "-m", "quantum_lab.worker", str(tmp_path), job["id"]],
                               capture_output=True, text=True, timeout=30)
    assert (completed.returncode == 0) == succeeded
    environment = json.loads((store.job_dir(job["id"]) / "environment.json").read_text())
    assert environment["quantum_lab_version"] == __version__
    assert "engines/molecular.py" in environment["source_sha256"]
    if succeeded:
        result = json.loads((store.job_dir(job["id"]) / "result.json").read_text())
        assert result["run_environment"] == environment
        assert abs(result["energy_hartree"] - (-1.116759307508)) < 1e-9
    else:
        assert (store.job_dir(job["id"]) / "error.json").exists()
