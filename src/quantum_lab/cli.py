"""Local browser service, isolated command-line jobs, and read-only diagnostics."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

from . import __version__


def _emit_json(value: dict) -> None:
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def _read_number(path: str) -> int | None:
    try:
        value = int(Path(path).read_text().strip())
        return value if 0 < value < 2**60 else None
    except (OSError, ValueError):
        return None


def diagnostics() -> dict:
    """Inspect installed packages and local limits without running a calculation."""
    versions = {"Quantum Lab": __version__, "Python": platform.python_version()}
    packages = {"PySCF": "pyscf", "NumPy": "numpy", "SciPy": "scipy", "ASE": "ase",
                "geomeTRIC": "geometric", "Basis Set Exchange": "basis-set-exchange",
                "FastAPI": "fastapi", "Uvicorn": "uvicorn"}
    missing = []
    for label, package in packages.items():
        try:
            versions[label] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[label] = None
            missing.append(package)
    try:
        from pyscf.dft import libxc
        versions["Libxc"] = libxc.__version__
    except (ImportError, OSError):
        versions["Libxc"] = None
        missing.append("Libxc runtime")
    try:
        physical_memory = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        physical_memory = None
    cgroup_memory = (_read_number("/sys/fs/cgroup/memory.max")
                     or _read_number("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
    limits = [value for value in (physical_memory, cgroup_memory) if value is not None and value > 0]
    try:
        affinity_cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity_cpus = None
    cpu_quota = None
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max" and int(period) > 0:
            cpu_quota = int(quota) / int(period)
    except (OSError, ValueError):
        pass
    frontend = Path(__file__).resolve().parent / "static"
    if not frontend.is_dir():
        frontend = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    release = platform.release()
    wsl = "microsoft" in (release + " " + platform.version()).lower()
    runtime = sys.version_info[:2] == (3, 12) and platform.system() == "Linux"
    return {
        "versions": versions, "missing_components": missing,
        "platform": {"system": platform.system(), "release": release, "machine": platform.machine(),
                     "wsl_detected": wsl, "wsl_validation": "not performed by this diagnostic"},
        "resources": {"logical_cpus": os.cpu_count(), "affinity_cpus": affinity_cpus,
                      "cgroup_cpu_quota": cpu_quota, "host_memory_bytes": physical_memory,
                      "cgroup_memory_limit_bytes": cgroup_memory,
                      "effective_memory_limit_bytes": min(limits) if limits else None},
        "frontend": {"built": (frontend / "index.html").is_file(), "path": str(frontend)},
        "ready_for_headless": runtime and not missing,
        "checks_performed": "Installed metadata and local runtime only; no SCF, network, or installation checks.",
    }


def _show_doctor(as_json: bool) -> int:
    report = diagnostics()
    if as_json:
        _emit_json(report)
    else:
        print("Quantum Lab local diagnostics")
        for name, version in report["versions"].items():
            print(f"  {name}: {version or 'not installed'}")
        host = report["platform"]
        resources = report["resources"]
        print(f"Platform: {host['system']} {host['release']} ({host['machine']})")
        print(f"WSL detected: {host['wsl_detected']}; this does not certify a WSL installation.")
        print(f"CPU: {resources['logical_cpus']} logical; {resources['affinity_cpus']} affinity; "
              f"cgroup quota {resources['cgroup_cpu_quota']}")
        memory = resources["effective_memory_limit_bytes"]
        print(f"Detected memory limit: {memory / 2**30:.2f} GiB" if memory else "Memory limit: unknown")
        print(f"Frontend build: {'present' if report['frontend']['built'] else 'missing'} "
              f"({report['frontend']['path']})")
        print(f"Headless prerequisites: {'present' if report['ready_for_headless'] else 'incomplete'}")
        print(report["checks_performed"])
    return 0 if report["ready_for_headless"] else 1


def _run_job(input_name: str, data_dir: str, json_events: bool) -> int:
    from pydantic import ValidationError

    from .jobs import JobManager
    from .models import JobSpec
    from .storage import TERMINAL, Store

    try:
        text = sys.stdin.read() if input_name == "-" else Path(input_name).read_text(encoding="utf-8")
        spec = JobSpec.model_validate_json(text)
        if spec.task == "volume":
            raise ValueError("Standalone volume jobs require a stored parent; use the API /volumes endpoint.")
    except (OSError, ValueError, ValidationError) as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 1

    class IdleCommandStore(Store):
        def recover(self):
            # JobManager calls this only after acquiring its exclusive data
            # directory lock. Refuse unfinished prior work before its scheduler
            # starts, so this command runs and cancels only the submitted job.
            unfinished = [job for job in self.list() if job["status"] not in TERMINAL]
            if unfinished:
                raise RuntimeError(
                    "The data directory contains unfinished jobs. Use a fresh --data-dir or "
                    "resume those jobs through the browser service before a command-line run."
                )

    store = None
    manager = None
    job_id = None
    cursor = 0

    def drain_events():
        nonlocal cursor
        for event in store.events(job_id, cursor):
            cursor = event["seq"]
            if json_events:
                _emit_json(event)
            else:
                data = event["data"]
                if event["type"] == "status":
                    print(f"Status: {data['status']}", flush=True)
                elif event["type"] in {"stage", "warning"}:
                    print(data.get("message", str(data)), flush=True)
                elif event["type"] in {"scf_iteration", "correlation_iteration"}:
                    energy = data.get("energy_hartree")
                    suffix = f"  {energy:.12f} Eh" if isinstance(energy, (float, int)) else ""
                    print(f"{event['type']} {data.get('iteration', '?')}{suffix}", flush=True)

    def finish(code: int) -> int:
        job = store.get(job_id)
        directory = store.job_dir(job_id)
        result_file = directory / "result.json"
        summary = {"type": "result", "job_id": job_id, "status": job["status"], "exit_code": code,
                   "energy_hartree": (job["result"] or {}).get("energy_hartree"),
                   "work_directory": str(directory),
                   "result_file": str(result_file) if result_file.is_file() else None,
                   "error": job.get("error")}
        if json_events:
            _emit_json(summary)
        else:
            print(f"Finished: {job['status']} (exit {code})", flush=True)
            if summary["energy_hartree"] is not None:
                print(f"Energy: {summary['energy_hartree']:.12f} Eh", flush=True)
            print(f"Saved job: {directory}", flush=True)
            if summary["result_file"]:
                print(f"Result: {summary['result_file']}", flush=True)
            if summary["error"]:
                print(f"Error: {summary['error']}", file=sys.stderr)
        return code

    try:
        store = IdleCommandStore(data_dir)
        manager = JobManager(store)
        manager.start()
        job = manager.submit(spec.model_dump(mode="json"))
        job_id = job["id"]
        if json_events:
            _emit_json({"type": "submitted", "job_id": job_id, "data_dir": str(store.root),
                        "result_path": str(store.job_dir(job_id) / "result.json")})
        else:
            print(f"Job: {job_id}\nData: {store.root}", flush=True)
        while True:
            drain_events()
            job = store.get(job_id)
            if job["status"] in TERMINAL:
                # A final status event may have committed after the prior read.
                drain_events()
                code = 0 if job["status"] == "succeeded" else 2 if job["status"] == "unconverged" else 1
                return finish(code)
            time.sleep(0.1)
    except KeyboardInterrupt:
        if manager is not None and job_id is not None:
            manager.cancel(job_id)
            drain_events()
            return finish(130)
        print("Interrupted before a job was submitted.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Run error: {exc}", file=sys.stderr)
        return 1
    finally:
        if manager is not None:
            manager.stop()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="quantum-lab", description="Local quantum chemistry workbench")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Start the API and built browser application")
    serve.add_argument("--host", choices=["127.0.0.1", "localhost", "::1"], default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--data-dir", default=os.environ.get("QUANTUM_LAB_DATA", ".quantum-lab"))
    run = sub.add_parser("run", help="Run a validated input JSON through an isolated local job queue")
    run.add_argument("input", help="JobSpec JSON file, or - to read standard input")
    run.add_argument("--data-dir", default=".quantum-lab-cli",
                     help="Idle job store (default: .quantum-lab-cli; independent of the web service)")
    run.add_argument("--json-events", action="store_true", help="Emit ordered events and summary as JSON lines")
    doctor = sub.add_parser("doctor", help="Inspect installed packages and runtime without running calculations")
    doctor.add_argument("--json", action="store_true", help="Print machine-readable JSON diagnostics")
    sub.add_parser("examples", help="List bundled example identifiers and titles")
    example = sub.add_parser("example", help="Print a bundled example as valid input JSON")
    example.add_argument("id", help="Identifier from quantum-lab examples")
    args = parser.parse_args(argv)
    if args.command == "serve":
        os.environ["QUANTUM_LAB_DATA"] = args.data_dir
        import uvicorn
        uvicorn.run("quantum_lab.api:app", host=args.host, port=args.port, workers=1, log_level="info")
        return 0
    if args.command == "run":
        return _run_job(args.input, args.data_dir, args.json_events)
    if args.command == "doctor":
        return _show_doctor(args.json)
    from .examples import examples
    records = examples()
    if args.command == "examples":
        for record in records:
            print(f"{record['id']}\t{record['title']}")
        return 0
    record = next((entry for entry in records if entry["id"] == args.id), None)
    if record is None:
        print(f"Unknown example: {args.id}. Use quantum-lab examples to list identifiers.", file=sys.stderr)
        return 1
    print(json.dumps(record["spec"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
