"""Reproduce measured example runs; JSON reports distinguish finite-grid demos from convergence.

    uv run python scripts/baseline.py --output .quantum-lab/baselines
    uv run python scripts/baseline.py --convergence --output .quantum-lab/convergence

Each example is executed in a fresh process so peak memory and runtime are attributable.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from quantum_lab.examples import examples


def cases(convergence=False):
    base = {x["id"]: x["spec"] for x in examples()}
    if not convergence:
        return base
    runs = {}
    for basis in ["sto-3g", "def2-svp", "def2-tzvp"]:
        for grid in [1, 3]:
            runs[f"water_{basis}_grid{grid}"] = dict(base["water"], basis=basis, grid_level=grid)
    for basis in ["gth-szv", "gth-dzvp"]:
        for k in [1, 2]:
            for mesh in [15, 21]:
                runs[f"diamond_{basis}_k{k}_mesh{mesh}"] = dict(base["diamond"], basis=basis, task="single_point",
                                                               kmesh=[k]*3, mesh=[mesh]*3)
    return runs


def execute(key, spec, folder):
    folder.mkdir(parents=True, exist_ok=False)
    spec = dict(spec, threads=1)
    (folder / "input.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False))
    if spec["domain"] == "crystal":
        from quantum_lab.engines.periodic import run
    elif spec["task"] == "educational_rhf":
        from quantum_lab.engines.educational import run
    else:
        from quantum_lab.engines.molecular import run
    events = []
    started = time.perf_counter()
    with (folder / "engine.log").open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        result = run(spec, lambda kind, data: events.append({"type": kind, "data": data}), folder)
    elapsed = time.perf_counter() - started
    (folder / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    report = {"case": key, "converged": result["converged"], "energy_hartree": result.get("energy_hartree"),
              "wall_seconds": elapsed, "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "scf_events": sum(e["type"] == "scf_iteration" for e in events),
              "spec": spec, "excitation_energies_ev": [x["energy_ev"] for x in result.get("excitations", [])],
              "sampled_ks_gap_ev": result.get("bands", {}).get("gap_ev")}
    (folder / "measurement.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    return result["converged"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=".quantum-lab/baselines")
    parser.add_argument("--convergence", action="store_true")
    parser.add_argument("--case", choices=list(cases()) + list(cases(True)))
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    selected = cases(args.convergence)
    if args.case:
        if args.case not in selected:
            parser.error("This case belongs to the convergence matrix; pass --convergence.")
        selected = {args.case: selected[args.case]}
    output = Path(args.output).resolve()
    if args.child:
        key, spec = next(iter(selected.items()))
        return 0 if execute(key, spec, output / key) else 1
    output.mkdir(parents=True, exist_ok=True)
    run_root = output / "runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    run_root.mkdir(parents=True)
    reports = []
    ok = True
    for key in selected:
        command = [sys.executable, str(Path(__file__).resolve()), "--child", "--case", key, "--output", str(run_root)]
        if args.convergence:
            command.append("--convergence")
        with (run_root / f"{key}.stdout.log").open("w") as log:
            child = subprocess.run(command, env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
                                   stdout=log, stderr=subprocess.STDOUT, check=False)
        ok = ok and child.returncode == 0
        measured = run_root / key / "measurement.json"
        reports.append(json.loads(measured.read_text()) if measured.exists() else {"case": key, "error": "execution failed"})
        print(key, "PASS" if child.returncode == 0 else "FAIL", flush=True)
    summary = {"platform": platform.platform(), "python": platform.python_version(), "run_directory": str(run_root),
               "versions": {x: importlib.metadata.version(x) for x in ["pyscf", "numpy", "scipy", "ase", "geometric", "basis-set-exchange"]},
               "note": "Finite-grid examples, not converged bulk reference values. RSS includes interpreter and engine imports. Same-engine reproduction is not independent physical accuracy validation.",
               "cases": reports}
    text = json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)
    (run_root / "report.json").write_text(text)
    (output / "report.json").write_text(text)
    print(f"Report: {output / 'report.json'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
