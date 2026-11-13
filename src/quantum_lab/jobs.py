"""A local durable FIFO queue with exactly one owned calculation process."""
from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import threading

from .storage import Store


class JobManager:
    def __init__(self, store: Store, worker_module="quantum_lab.worker"):
        self.store = store
        self.worker_module = worker_module
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._process = None
        self._job_id = None
        self._log = None
        self._thread = None
        self._instance_lock = None

    def start(self):
        self._instance_lock = (self.store.root / "server.lock").open("a")
        try:
            fcntl.flock(self._instance_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._instance_lock.close()
            self._instance_lock = None
            raise RuntimeError("Another Quantum Lab server already owns this data directory.") from exc
        self.store.recover()
        self._thread = threading.Thread(target=self._run, name="quantum-job-scheduler", daemon=True)
        self._thread.start()

    def submit(self, spec: dict) -> dict:
        record = self.store.create(spec)
        self._wake.set()
        return record

    def submit_volume(self, spec: dict) -> dict:
        """Atomically reuse or enqueue a requested derived grid across HTTP threads."""
        with self._lock:
            for child in self.store.list():
                if (child["spec"].get("parent_job_id") == spec["parent_job_id"]
                        and child["spec"].get("volume") == spec["volume"]
                        and child["status"] in {"queued", "running", "succeeded"}):
                    return child
            return self.submit(spec)

    def cancel(self, job_id: str) -> dict:
        with self._lock:
            changed = self.store.transition(job_id, "cancelled", allowed=("queued", "running"))
            if changed and self._job_id == job_id:
                self._terminate()
        self._wake.set()
        return self.store.get(job_id)

    def _terminate(self):
        process = self._process
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=3)

    def stop(self):
        self._stopping.set()
        self._wake.set()
        with self._lock:
            if self._job_id:
                self.store.transition(self._job_id, "interrupted", error="Server stopped during calculation.")
                self._terminate()
                self._clear_process()
        if self._thread:
            self._thread.join(timeout=10)
        if self._instance_lock:
            fcntl.flock(self._instance_lock.fileno(), fcntl.LOCK_UN)
            self._instance_lock.close()
            self._instance_lock = None

    def _clear_process(self):
        if self._log:
            self._log.close()
        self._log = self._process = self._job_id = None

    def _launch(self, job: dict):
        spec = job["spec"]
        threads = str(max(1, min(int(spec.get("threads", 2)), os.cpu_count() or 1)))
        env = os.environ.copy()
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[name] = threads
        env["PYTHONUNBUFFERED"] = "1"
        env["QUANTUM_LAB_PARENT_PID"] = str(os.getpid())
        self._job_id = job["id"]
        try:
            self._log = (self.store.job_dir(job["id"]) / "engine.log").open("ab", buffering=0)
            self._process = subprocess.Popen(
                [sys.executable, "-m", self.worker_module, str(self.store.root), job["id"]],
                env=env, stdout=self._log, stderr=subprocess.STDOUT, start_new_session=True,
                # Do not leak the server's flock into its child.
                close_fds=True,
            )
        except OSError as exc:
            self.store.transition(job["id"], "failed", error=f"Cannot launch worker: {exc}")
            self._clear_process()

    def _finish(self):
        code = self._process.poll()
        if code is None:
            return
        job_id = self._job_id
        directory = self.store.job_dir(job_id)
        try:
            if code != 0:
                error_path = directory / "error.json"
                error = json.loads(error_path.read_text())["message"] if error_path.exists() else f"Worker exited with code {code}; see engine.log."
                self.store.transition(job_id, "failed", error=error)
            else:
                result = json.loads((directory / "result.json").read_text())
                if not isinstance(result.get("converged"), bool):
                    raise ValueError("Worker returned no valid convergence status.")
                status = "succeeded" if result["converged"] else "unconverged"
                self.store.transition(job_id, status, result=result)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.store.transition(job_id, "failed", error=f"Could not finalize worker result: {exc}")
        finally:
            self._clear_process()

    def _run(self):
        while not self._stopping.is_set():
            try:
                with self._lock:
                    if self._stopping.is_set():
                        break
                    if self._process is not None:
                        self._finish()
                    if self._process is None:
                        job = self.store.claim()
                        if job is not None:
                            self._launch(job)
            except Exception as exc:  # noqa: BLE001 - keep scheduler alive and retain diagnosis
                # Retain a diagnosis instead of silently losing the scheduler.
                with (self.store.root / "scheduler.log").open("a", encoding="utf-8") as out:
                    out.write(f"{type(exc).__name__}: {exc}\n")
            self._wake.wait(0.15)
            self._wake.clear()
