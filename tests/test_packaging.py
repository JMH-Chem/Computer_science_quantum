"""Packaging must include working-tree modules and the built local UI intact."""
from __future__ import annotations

import importlib.util
import os
import zipfile
from pathlib import Path

import pytest

script = Path(__file__).resolve().parents[1] / "scripts" / "package_release.py"
spec = importlib.util.spec_from_file_location("package_release", script)
packaging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packaging)


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "repository"
    source = root / "src" / "quantum_lab"
    source.mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname="quantum-lab"\nversion="0.3.0"\n')
    (source / "__init__.py").write_text('__version__ = "0.3.0"\n')
    # There is deliberately no Git repository: new modules cannot be selected
    # using git ls-files or lost because they are not yet tracked.
    (source / "untracked_new_engine.py").write_text('ANSWER = "included"\n')
    (source / "__pycache__").mkdir()
    (source / "__pycache__" / "untracked_new_engine.cpython-312.pyc").write_bytes(b"stale-cache")
    frontend = root / "frontend" / "dist"
    (frontend / "assets").mkdir(parents=True)
    (frontend / "index.html").write_text('<script src="/assets/app.js"></script>')
    (frontend / "assets" / "app.js").write_text('document.title = "Quantum Lab";')
    return root


def test_stage_includes_untracked_backend_and_ui_without_mutating_source(repository, tmp_path):
    staged = tmp_path / "stage"
    snapshot = packaging.stage_project(repository, staged)
    assert "quantum_lab/untracked_new_engine.py" in snapshot["wheel_entries"]
    assert "quantum_lab/static/index.html" in snapshot["wheel_entries"]
    assert "quantum_lab/static/assets/app.js" in snapshot["wheel_entries"]
    assert not any("__pycache__" in name for name in snapshot["wheel_entries"])
    assert not (repository / "src" / "quantum_lab" / "static").exists()
    assert (staged / "src" / "quantum_lab" / "static" / "assets" / "app.js").read_text().startswith("document")
    wheel = tmp_path / "package.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in snapshot["wheel_entries"]:
            archive.write(staged / "src" / name, name)
    packaging.validate_wheel(wheel, snapshot["wheel_entries"])


def test_packaging_rejects_missing_or_stale_frontend(repository, tmp_path):
    index = repository / "frontend" / "dist" / "index.html"
    index.unlink()
    with pytest.raises(ValueError, match="Missing browser"):
        packaging.stage_project(repository, tmp_path / "missing")
    index.write_text("<html></html>")
    source = repository / "frontend" / "src" / "App.tsx"
    source.parent.mkdir()
    source.write_text("export default function App() {}")
    modified = index.stat().st_mtime_ns + 1_000_000_000
    os.utime(source, ns=(modified, modified))
    with pytest.raises(ValueError, match="newer than its build"):
        packaging.stage_project(repository, tmp_path / "stale")


def test_wheel_verification_catches_missing_or_changed_new_module(repository, tmp_path):
    staged = tmp_path / "stage"
    snapshot = packaging.stage_project(repository, staged)
    wheel = tmp_path / "package.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("quantum_lab/__init__.py", '__version__ = "0.3.0"\n')
    with pytest.raises(ValueError, match="omits staged"):
        packaging.validate_wheel(wheel, snapshot["wheel_entries"])
    with zipfile.ZipFile(wheel, "w") as archive:
        for name in snapshot["wheel_entries"]:
            if name.endswith("untracked_new_engine.py"):
                archive.writestr(name, "WRONG = True")
            else:
                archive.write(staged / "src" / name, name)
    with pytest.raises(ValueError, match="differs from staged"):
        packaging.validate_wheel(wheel, snapshot["wheel_entries"])


def test_packaging_rejects_mismatched_application_versions(repository, tmp_path):
    (repository / "src" / "quantum_lab" / "__init__.py").write_text('__version__ = "0.2.0"')
    with pytest.raises(ValueError, match="version differ"):
        packaging.stage_project(repository, tmp_path / "stage")
