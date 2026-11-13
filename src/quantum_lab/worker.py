"""Isolated numerical worker. The scheduler alone commits final job status."""
from __future__ import annotations

import importlib
import math
import os
import signal
import sys
import time
import traceback

from .storage import Store, atomic_json


def main() -> int:
    # Linux/WSL2: do not leave expensive orphan numerics after a server crash.
    expected_parent = int(os.environ.get("QUANTUM_LAB_PARENT_PID", os.getppid()))
    if sys.platform.startswith("linux"):
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "Could not set the worker parent-death signal.")
        if os.getppid() != expected_parent:
            return 1
    store = Store(sys.argv[1])
    job_id = sys.argv[2]
    job = store.get(job_id)
    directory = store.job_dir(job_id)
    spec = job["spec"]
    started = time.monotonic()
    try:
        # A queued cancellation that raced process creation must not start numerics.
        if job["status"] != "running":
            return 0
        adapter = "periodic" if spec["domain"] == "crystal" else "molecular"
        if spec["task"] == "educational_rhf":
            adapter = "educational"
        if spec["task"] == "volume":
            parent = store.get(spec["parent_job_id"])
            if parent["spec"]["task"] == "educational_rhf":
                adapter = "educational"
        module = importlib.import_module(f"quantum_lab.engines.{adapter}")
        def emit(kind, data):
            store.emit(job_id, kind, data)
        emit("stage", {"name": "initializing", "message": "Preparing the numerical engine."})
        if spec["task"] == "volume":
            parent_id = spec["parent_job_id"]
            parent = store.get(parent_id)
            if parent["status"] != "succeeded":
                raise ValueError("Volume generation requires a converged parent calculation.")
            result = module.generate_volume(store.job_dir(parent_id), spec["volume"], directory)
        else:
            result = module.run(spec, emit, directory)
        if not isinstance(result, dict) or not isinstance(result.get("converged"), bool):
            raise TypeError("The engine did not return a boolean convergence status.")
        if result.get("energy_hartree") is not None and not math.isfinite(float(result["energy_hartree"])):
            raise ValueError("The engine returned a non-finite energy.")
        result["elapsed_seconds"] = time.monotonic() - started
        import resource
        result["peak_memory_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        atomic_json(directory / "result.json", result)
        emit("stage", {"name": "completed", "message": "Engine returned; checking final convergence status."})
        return 0
    except Exception as exc:  # noqa: BLE001 - numerical process boundary, persisted and nonzero exit
        traceback.print_exc()
        atomic_json(directory / "error.json", {"type": type(exc).__name__, "message": str(exc)})
        store.emit(job_id, "warning", {"message": str(exc)})
        return 1


if __name__ == "__main__":
    sys.exit(main())
