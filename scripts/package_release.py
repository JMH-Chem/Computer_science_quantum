#!/usr/bin/env python3
"""Build a local wheel with the existing browser build, without changing src/.

Run after ``npm --prefix frontend run build``. Source files are copied from the
working tree, including new untracked modules. Nothing is uploaded or published.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import uuid
import zipfile
from datetime import UTC, datetime
from pathlib import Path

IGNORED_DIRECTORIES = {"__pycache__", ".pytest_cache", ".ruff_cache", ".git", "node_modules"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_in(directory: Path, *, omit_static=False):
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if any(part in IGNORED_DIRECTORIES for part in relative.parts):
            continue
        if omit_static and relative.parts[0] == "static":
            continue
        if path.is_symlink():
            raise ValueError(f"Packaging requires ordinary source files, not a symbolic link: {path}")
        if path.is_file() and path.suffix not in {".pyc", ".pyo"}:
            yield path


def _source_version(source: Path) -> str | None:
    for item in ast.parse(source.read_text(encoding="utf-8")).body:
        if isinstance(item, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__version__"
                                               for t in item.targets):
            return ast.literal_eval(item.value)
    return None


def stage_project(repository: Path, stage: Path) -> dict:
    """Snapshot source and compiled UI, with hashes independent of Git status."""
    project_file = repository / "pyproject.toml"
    project = tomllib.loads(project_file.read_text(encoding="utf-8"))["project"]
    package = repository / "src" / "quantum_lab"
    frontend = repository / "frontend" / "dist"
    if not (frontend / "index.html").is_file():
        raise ValueError("Missing browser build; run npm --prefix frontend run build before packaging.")
    if _source_version(package / "__init__.py") != project["version"]:
        raise ValueError("Package source version and pyproject.toml version differ.")
    built_at = (frontend / "index.html").stat().st_mtime_ns
    browser_source = repository / "frontend" / "src"
    if browser_source.exists() and any(p.stat().st_mtime_ns > built_at for p in files_in(browser_source)):
        raise ValueError("Frontend source is newer than its build; rebuild the browser before packaging.")
    source_files, wheel_entries = {}, {}
    stage.mkdir(parents=True, exist_ok=True)
    project_checksum = sha256(project_file)
    shutil.copyfile(project_file, stage / "pyproject.toml")
    if sha256(stage / "pyproject.toml") != project_checksum:
        raise RuntimeError("Project metadata changed while packaging.")
    source_files["pyproject.toml"] = project_checksum

    def copy_group(source_root, destination_root, *, prefix, omit_static=False):
        for source in files_in(source_root, omit_static=omit_static):
            relative = source.relative_to(source_root)
            destination = destination_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            checksum = sha256(source)
            shutil.copyfile(source, destination)
            if sha256(destination) != checksum or sha256(source) != checksum:
                raise RuntimeError(f"Source changed while packaging: {source}")
            source_files[source.relative_to(repository).as_posix()] = checksum
            wheel_entries[(Path(prefix) / relative).as_posix()] = checksum

    copy_group(package, stage / "src" / "quantum_lab", prefix="quantum_lab", omit_static=True)
    copy_group(frontend, stage / "src" / "quantum_lab" / "static", prefix="quantum_lab/static")
    # Confirm the complete snapshot remained stable, not only each individual copy.
    current = {"pyproject.toml", *(p.relative_to(repository).as_posix()
                                  for p in files_in(package, omit_static=True)),
               *(p.relative_to(repository).as_posix() for p in files_in(frontend))}
    if current != set(source_files):
        raise RuntimeError("The source file inventory changed while packaging; repeat after edits finish.")
    for relative, expected in source_files.items():
        if sha256(repository / relative) != expected:
            raise RuntimeError(f"Source changed while packaging: {relative}")
    return {"name": project["name"], "version": project["version"],
            "source_files": source_files, "wheel_entries": wheel_entries}


def validate_wheel(wheel: Path, expected: dict[str, str]) -> None:
    """Verify that the wheel includes every snapshotted backend and UI file."""
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        missing = set(expected) - names
        if missing:
            raise ValueError(f"Wheel omits staged files: {', '.join(sorted(missing))}")
        if any("__pycache__" in Path(name).parts or name.endswith((".pyc", ".pyo")) for name in names):
            raise ValueError("Wheel contains Python cache files.")
        for name, checksum in expected.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != checksum:
                raise ValueError(f"Wheel content differs from staged source: {name}")


def build_release(repository: Path, output: Path, *, offline=False) -> Path:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to build this local release.")
    repository, output = repository.resolve(), output.resolve()
    lock_path = repository / "uv.lock"
    locked = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    project = tomllib.loads((repository / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    package = next((p for p in locked["package"] if p["name"] == project["name"]), None)
    if package is None or package["version"] != project["version"]:
        raise ValueError("uv.lock does not contain the current project version; update the lock before packaging.")
    environment = os.environ.copy()
    environment.setdefault("UV_CACHE_DIR", str(repository / ".cache" / "uv"))
    flag = ["--offline"] if offline else []
    with tempfile.TemporaryDirectory(prefix="quantum-lab-package-") as temporary:
        root = Path(temporary)
        stage, built = root / "stage", root / "built"
        snapshot = stage_project(repository, stage)
        subprocess.run([uv, "build", "--wheel", "--python", sys.executable,
                        "--no-create-gitignore", "--out-dir", str(built), *flag, str(stage)],
                       check=True, env=environment, cwd=root)
        wheels = list(built.glob("*.whl"))
        if len(wheels) != 1:
            raise ValueError("The build did not produce exactly one wheel.")
        wheel = wheels[0]
        validate_wheel(wheel, snapshot["wheel_entries"])
        requirements = built / "requirements.lock.txt"
        subprocess.run([uv, "export", "--project", str(repository), "--frozen", "--no-dev",
                        "--no-emit-project", "--no-header", "--output-file", str(requirements), *flag],
                       check=True, env=environment, stdout=subprocess.DEVNULL)
        stamp = datetime.now(UTC)
        # Each artifact directory is immutable; a later rebuild cannot silently
        # replace an earlier package carrying the same application version.
        release = output / f"quantum-lab-{snapshot['version']}-{stamp:%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
        release.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(wheel, release / wheel.name)
        shutil.copyfile(requirements, release / requirements.name)
        manifest = {
            "created_at": stamp.isoformat(), "name": snapshot["name"], "version": snapshot["version"],
            "wheel": {"filename": wheel.name, "sha256": sha256(wheel), "bytes": wheel.stat().st_size},
            "requirements": {"filename": requirements.name, "sha256": sha256(requirements)},
            "source_files": snapshot["source_files"],
            "locks": {"uv.lock": sha256(lock_path),
                      "frontend/package-lock.json": sha256(repository / "frontend" / "package-lock.json")},
            "uv_version": subprocess.check_output([uv, "--version"], text=True).strip(),
            "python_version": platform_version(),
            "notes": ["Local build only; no publishing or upload was performed.",
                      "The wheel bundles browser assets and requires the separately locked Python dependencies.",
                      "Source and dependency hashes support reproduction; byte-identical future builds are not asserted."],
        }
        (release / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
        return release


def platform_version() -> str:
    return ".".join(str(part) for part in sys.version_info[:3])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out-dir", type=Path, help="Parent for a new, uniquely named local release directory")
    parser.add_argument("--offline", action="store_true", help="Use only locally cached build requirements")
    args = parser.parse_args(argv)
    try:
        release = build_release(args.repository, args.out_dir or args.repository / "dist", offline=args.offline)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Packaging failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"release_directory": str(release), "manifest": str(release / "manifest.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
