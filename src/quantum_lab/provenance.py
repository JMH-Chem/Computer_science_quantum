"""Record the running software before numerical work, without host identifiers."""
from __future__ import annotations

import hashlib
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import __version__


def execution_environment() -> dict:
    packages = {}
    for name in ("pyscf", "numpy", "scipy", "ase", "geometric", "basis-set-exchange"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    root = Path(__file__).resolve().parent
    sources = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(root.rglob("*.py")) if "__pycache__" not in path.parts and "static" not in path.relative_to(root).parts}
    return {
        "quantum_lab_version": __version__, "python_version": platform.python_version(),
        "python_implementation": sys.implementation.name,
        "operating_system": platform.system(), "architecture": platform.machine(),
        "packages": packages, "source_sha256": sources,
        "source_hash_scope": "Quantum Lab Python source files read at worker startup; dependencies recorded by version",
    }
