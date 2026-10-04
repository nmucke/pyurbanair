"""Transactional queue shared by local clients and the supervisor."""

from __future__ import annotations

import builtins
import hashlib
import json
import pathlib
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

from mcp_server.jobs.paths import ensure_private_directory

TERMINAL = frozenset({"succeeded", "failed", "cancelled", "interrupted"})
ACTIVE = frozenset({"preparing", "running", "finalizing", "cancelling"})
STATES = TERMINAL | ACTIVE | {"queued"}


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class Registry:
    def __init__(self, root: str | pathlib.Path) -> None:
        self.root = pathlib.Path(root).expanduser().resolve()
        ensure_private_directory(self.root)
        self.path = self.root / "jobs.sqlite3"
        with self.connection() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, idempotency_key TEXT UNIQUE NOT NULL,
                    digest TEXT NOT NULL, kind TEXT NOT NULL, parent_run_id TEXT,
                    state TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL,
                    payload TEXT NOT NULL, details TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS transitions (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
                    state TEXT NOT NULL, timestamp REAL NOT NULL, details TEXT NOT NULL
                );
            """
            )

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA journal_mode=WAL")
            with db:
                yield db
        finally:
            db.close()

    def submit(self, payload: dict[str, Any], key: str) -> dict[str, Any]:
        if not key or len(key) > 256:
            raise ValueError("idempotency_key must contain 1–256 characters")
        kind = payload.get("kind", "forward")
        if kind not in {"forward", "visualization"}:
            raise ValueError("Unknown job kind")
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM jobs WHERE idempotency_key=?", (key,)
            ).fetchone()
            if row is not None:
                if row["digest"] != digest:
                    raise ValueError(
                        "Idempotency key already used with different inputs"
                    )
                return self._decode(row)
            job_id = uuid.uuid4().hex
            now = time.time()
            db.execute(
                "INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    job_id,
                    key,
                    digest,
                    kind,
                    payload.get("parent_run_id"),
                    "queued",
                    now,
                    now,
                    encoded,
                    "{}",
                ),
            )
            db.execute(
                "INSERT INTO transitions(job_id,state,timestamp,details) VALUES (?,?,?,?)",
                (job_id, "queued", now, "{}"),
            )
        return self.get(job_id)

    def existing(self, payload: dict[str, Any], key: str) -> dict[str, Any] | None:
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        with self.connection() as db:
            row = db.execute(
                "SELECT * FROM jobs WHERE idempotency_key=?", (key,)
            ).fetchone()
        if row is None:
            return None
        if row["digest"] != digest:
            raise ValueError("Idempotency key already used with different inputs")
        return self._decode(row)

    def _decode(self, row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["payload"] = json.loads(result["payload"])
        result["details"] = json.loads(result["details"])
        result["run_id"] = result["id"]
        result["job_id"] = result["id"]
        result["run_root"] = str(self.root / "runs" / result["id"])
        return result

    def get(self, job_id: str) -> dict[str, Any]:
        with self.connection() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise ValueError(f"Unknown job: {job_id}")
        return self._decode(row)

    def list(
        self, state: str | None = None, offset: int = 0, limit: int = 50
    ) -> list[dict[str, Any]]:
        if offset < 0 or not 1 <= limit <= 200:
            raise ValueError("offset >= 0 and 1 <= limit <= 200 required")
        if state is not None and state not in STATES:
            raise ValueError("Unknown state")
        with self.connection() as db:
            if state:
                rows = db.execute(
                    "SELECT * FROM jobs WHERE state=? ORDER BY created DESC LIMIT ? OFFSET ?",
                    (state, limit, offset),
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT * FROM jobs ORDER BY created DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
        return [self._decode(row) for row in rows]

    def unfinished(self) -> builtins.list[dict[str, Any]]:
        with self.connection() as db:
            rows = db.execute(
                "SELECT * FROM jobs WHERE state NOT IN ('succeeded','failed','cancelled','interrupted') ORDER BY created"
            ).fetchall()
        return [self._decode(row) for row in rows]

    def update(
        self, job_id: str, state: str | None = None, **details: Any
    ) -> dict[str, Any]:
        if state is not None and state not in STATES:
            raise ValueError("Unknown state")
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise ValueError(f"Unknown job: {job_id}")
            old_state = row["state"]
            if old_state in TERMINAL:
                return self._decode(row)
            # A late heartbeat must never undo cancellation.
            new_state = state or old_state
            if old_state == "queued" and new_state == "cancelling":
                new_state = "cancelled"
            if old_state == "cancelling" and new_state not in TERMINAL:
                new_state = "cancelling"
            merged = {**json.loads(row["details"]), **details}
            now = time.time()
            encoded = json.dumps(merged, allow_nan=False)
            db.execute(
                "UPDATE jobs SET state=?,updated=?,details=? WHERE id=?",
                (new_state, now, encoded, job_id),
            )
            if old_state != new_state:
                db.execute(
                    "INSERT INTO transitions(job_id,state,timestamp,details) VALUES (?,?,?,?)",
                    (job_id, new_state, now, encoded),
                )
        return self.get(job_id)

    def cancel(self, job_id: str) -> dict[str, Any]:
        return self.update(job_id, "cancelling", cancellation_requested=time.time())

    def logs(self, job_id: str, cursor: int = 0, limit: int = 32768) -> dict[str, Any]:
        if cursor < 0 or not 1 <= limit <= 131072:
            raise ValueError("cursor >= 0 and 1 <= limit <= 131072 required")
        job = self.get(job_id)
        path = pathlib.Path(job["run_root"]) / "worker.log"
        data = b""
        if path.exists():
            with path.open("rb") as stream:
                stream.seek(cursor)
                data = stream.read(limit)
        return {
            "job_id": job_id,
            "text": data.decode("utf-8", errors="replace"),
            "cursor": cursor + len(data),
            "eof": not path.exists() or cursor + len(data) >= path.stat().st_size,
        }
