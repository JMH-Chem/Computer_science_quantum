"""Durable jobs and ordered events; each operation owns its SQLite connection."""
from __future__ import annotations

import json
import math
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

TERMINAL = frozenset({"succeeded", "unconverged", "failed", "cancelled", "interrupted"})


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
            """)

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
            row = db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created_at, id LIMIT 1").fetchone()
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
