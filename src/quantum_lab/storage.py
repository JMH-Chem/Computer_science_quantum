"""Durable jobs and ordered events; each operation owns its SQLite connection."""
from __future__ import annotations

import json
import math
import shutil
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

TERMINAL = frozenset({"succeeded", "unconverged", "failed", "cancelled", "interrupted"})


class ExperimentConflict(ValueError):
    """An idempotency key was reused for a different experiment request."""


def now() -> str:
    return datetime.now(UTC).isoformat()


def json_value(value):
    """Scientific scalar conversion, without emitting invalid JSON NaN tokens."""
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if hasattr(value, "tolist"):
        return json_value(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dumps(value) -> str:
    return json.dumps(json_value(value), ensure_ascii=False, allow_nan=False)


def atomic_json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(dumps(value), encoding="utf-8")
    temp.replace(path)


class Store:
    def __init__(self, directory: Path | str):
        self.root = Path(directory).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "jobs").mkdir(exist_ok=True)
        self.db_path = self.root / "jobs.sqlite3"
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, status TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    spec TEXT NOT NULL, result TEXT, error TEXT
                );
                CREATE TABLE IF NOT EXISTS events (
                    job_id TEXT NOT NULL REFERENCES jobs(id), seq INTEGER NOT NULL,
                    timestamp TEXT NOT NULL, type TEXT NOT NULL, data TEXT NOT NULL,
                    PRIMARY KEY(job_id, seq)
                );
                CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(status, created_at);
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, template_id TEXT NOT NULL,
                    prediction TEXT NOT NULL, request_id TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL, cancel_requested INTEGER NOT NULL DEFAULT 0,
                    base_spec TEXT NOT NULL, bases TEXT NOT NULL, request_payload TEXT NOT NULL,
                    definition TEXT
                );
                CREATE TABLE IF NOT EXISTS experiment_jobs (
                    experiment_id TEXT NOT NULL REFERENCES experiments(id),
                    ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
                    job_id TEXT NOT NULL UNIQUE REFERENCES jobs(id),
                    PRIMARY KEY(experiment_id, ordinal)
                );
            """)
            # v0.2 records remain intact. Their basis-only definition is read
            # from persisted inputs until explicitly exported in the new form.
            columns = {row["name"] for row in db.execute("PRAGMA table_info(experiments)")}
            if "definition" not in columns:
                db.execute("ALTER TABLE experiments ADD COLUMN definition TEXT")

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.db_path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def job_dir(self, job_id: str) -> Path:
        try:
            if str(uuid.UUID(job_id)) != job_id:
                raise ValueError("Noncanonical job ID")
        except (ValueError, AttributeError) as exc:
            raise KeyError(job_id) from exc
        return self.root / "jobs" / job_id

    @staticmethod
    def _record(row) -> dict | None:
        if row is None:
            return None
        result = dict(row)
        result["spec"] = json.loads(result["spec"])
        result["result"] = json.loads(result["result"]) if result["result"] else None
        return result

    @staticmethod
    def _event(db, job_id: str, kind: str, data: dict) -> dict:
        seq = db.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM events WHERE job_id=?", (job_id,)).fetchone()[0]
        stamp = now()
        db.execute("INSERT INTO events VALUES (?, ?, ?, ?, ?)", (job_id, seq, stamp, kind, dumps(data)))
        return {"job_id": job_id, "seq": seq, "timestamp": stamp, "type": kind, "data": json_value(data)}

    def create(self, spec: dict) -> dict:
        job_id = str(uuid.uuid4())
        workdir = self.job_dir(job_id)
        workdir.mkdir()
        atomic_json(workdir / "input.json", spec)
        stamp = now()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)",
                       (job_id, spec.get("name", "Calculation"), "queued", stamp, stamp, dumps(spec)))
            self._event(db, job_id, "status", {"status": "queued"})
        return self.get(job_id)

    def get(self, job_id: str) -> dict:
        self.job_dir(job_id)
        with self.connection() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        result = self._record(row)
        if result is None:
            raise KeyError(job_id)
        return result

    def list(self) -> list[dict]:
        with self.connection() as db:
            rows = db.execute("SELECT * FROM jobs ORDER BY created_at DESC, id DESC").fetchall()
        return [self._record(row) for row in rows]

    def claim(self) -> dict | None:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created_at, rowid LIMIT 1").fetchone()
            if row is None:
                return None
            db.execute("UPDATE jobs SET status='running', updated_at=? WHERE id=?", (now(), row["id"]))
            self._event(db, row["id"], "status", {"status": "running"})
        return self.get(row["id"])

    def transition(self, job_id: str, status: str, *, allowed=("running",), result=None, error=None) -> bool:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            if row[0] not in allowed:
                return False
            db.execute("UPDATE jobs SET status=?, updated_at=?, result=?, error=? WHERE id=?",
                       (status, now(), dumps(result) if result is not None else None, error, job_id))
            self._event(db, job_id, "status", {"status": status, "error": error})
        return True

    def emit(self, job_id: str, kind: str, data: dict) -> dict | None:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            # A cancelled process may emit once before it observes termination.
            if row[0] in TERMINAL:
                return None
            return self._event(db, job_id, kind, data)

    def events(self, job_id: str, after: int = 0) -> list[dict]:
        self.get(job_id)
        with self.connection() as db:
            rows = db.execute("SELECT * FROM events WHERE job_id=? AND seq>? ORDER BY seq", (job_id, after)).fetchall()
        return [{**dict(row), "data": json.loads(row["data"])} for row in rows]

    def recover(self):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("SELECT id FROM jobs WHERE status='running'").fetchall()
            for row in rows:
                message = "Server restarted before this calculation completed. Intermediate events are preserved."
                db.execute("UPDATE jobs SET status='interrupted', updated_at=?, error=? WHERE id=?",
                           (now(), message, row["id"]))
                self._event(db, row["id"], "status", {"status": "interrupted", "error": message})

    @staticmethod
    def _experiment_record(row) -> dict:
        record = dict(row)
        for field in ("base_spec", "bases", "request_payload"):
            record[field] = json.loads(record[field])
        record["definition"] = json.loads(record["definition"]) if record.get("definition") else None
        record["cancel_requested"] = bool(record["cancel_requested"])
        return record

    def create_experiment(self, payload: dict, specs: list[dict]) -> dict:
        """Publish the complete group only after all durable inputs exist.

        A process crash before the database commit can leave unreferenced files,
        but cannot expose an incomplete runnable group. Ordinary failures roll
        back the transaction and remove only the directories this call created.
        The unique request key is checked under the database write lock, making
        retries safe across concurrent requests and independent Store objects.
        """
        if not specs:
            raise ValueError("An experiment must contain at least one calculation.")
        prepared: list[Path] = []
        experiment_id = str(uuid.uuid4())
        try:
            with self.connection() as db:
                db.execute("BEGIN IMMEDIATE")
                existing = db.execute("SELECT * FROM experiments WHERE request_id=?",
                                      (payload["request_id"],)).fetchone()
                if existing is not None:
                    record = self._experiment_record(existing)
                    if record["request_payload"] != payload:
                        raise ExperimentConflict("같은 요청 ID에 다른 실험 조건을 사용할 수 없습니다.")
                    return record
                stamp = now()
                # Imported here to avoid coupling the Store module's import
                # order to the scientific summary module.
                from .experiment_science import freeze_definition
                definition = freeze_definition(payload["template_id"], specs)
                db.execute("""INSERT INTO experiments
                    (id, name, template_id, prediction, request_id, created_at, cancel_requested,
                     base_spec, bases, request_payload, definition)
                    VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)""", (
                    experiment_id, payload["name"], payload["template_id"], payload["prediction"],
                    payload["request_id"], stamp, dumps(specs[0]), dumps([s["basis"] for s in specs]),
                    dumps(payload), dumps(definition),
                ))
                for ordinal, spec in enumerate(specs):
                    job_id = str(uuid.uuid4())
                    directory = self.job_dir(job_id)
                    directory.mkdir()
                    prepared.append(directory)
                    atomic_json(directory / "input.json", spec)
                    db.execute("INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, NULL, NULL)", (
                        job_id, spec["name"], "queued", stamp, stamp, dumps(spec),
                    ))
                    db.execute("INSERT INTO experiment_jobs VALUES (?, ?, ?)",
                               (experiment_id, ordinal, job_id))
                    self._event(db, job_id, "status", {"status": "queued"})
        except BaseException:
            for directory in prepared:
                shutil.rmtree(directory)
            raise
        return self.get_experiment(experiment_id)

    def get_experiment(self, experiment_id: str) -> dict:
        record, _ = self.experiment_bundle(experiment_id)
        return record

    def experiment_bundle(self, experiment_id: str) -> tuple[dict, list[dict]]:
        # Reuse the canonical UUID check, without creating a filesystem path.
        self.job_dir(experiment_id)
        with self.connection() as db:
            db.execute("BEGIN")
            row = db.execute("SELECT * FROM experiments WHERE id=?", (experiment_id,)).fetchone()
            if row is None:
                raise KeyError(experiment_id)
            jobs = db.execute("""
                SELECT jobs.* FROM experiment_jobs AS members JOIN jobs ON jobs.id=members.job_id
                WHERE members.experiment_id=? ORDER BY members.ordinal
            """, (experiment_id,)).fetchall()
        return self._experiment_record(row), [self._record(job) for job in jobs]

    def list_experiments(self) -> list[dict]:
        with self.connection() as db:
            rows = db.execute("SELECT * FROM experiments ORDER BY created_at DESC, rowid DESC").fetchall()
        return [self._experiment_record(row) for row in rows]

    def cancel_experiment_jobs(self, experiment_id: str) -> list[str]:
        """Atomically cancel only this group's active jobs, preserving results."""
        self.job_dir(experiment_id)
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT id FROM experiments WHERE id=?", (experiment_id,)).fetchone() is None:
                raise KeyError(experiment_id)
            rows = db.execute("""
                SELECT jobs.id FROM experiment_jobs AS members JOIN jobs ON jobs.id=members.job_id
                WHERE members.experiment_id=? AND jobs.status IN ('queued', 'running')
                ORDER BY members.ordinal
            """, (experiment_id,)).fetchall()
            job_ids = [row["id"] for row in rows]
            if job_ids:
                stamp = now()
                db.execute("UPDATE experiments SET cancel_requested=1 WHERE id=?", (experiment_id,))
                for job_id in job_ids:
                    db.execute("UPDATE jobs SET status='cancelled', updated_at=? WHERE id=?", (stamp, job_id))
                    self._event(db, job_id, "status", {"status": "cancelled", "error": None})
        return job_ids
