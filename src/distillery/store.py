"""SQLite index + content-addressed artifacts + resumable stages."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, heldout_sha256 TEXT);
CREATE TABLE IF NOT EXISTS seals (
    run_id TEXT NOT NULL, name TEXT NOT NULL, sha256 TEXT NOT NULL, PRIMARY KEY (run_id, name));
CREATE TABLE IF NOT EXISTS stages (
    run_id TEXT NOT NULL, stage TEXT NOT NULL, input_hash TEXT NOT NULL,
    status TEXT NOT NULL, output_sha256 TEXT, error TEXT, updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, input_hash));
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, name TEXT NOT NULL,
    data_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL, usd REAL NOT NULL,
    created_at TEXT NOT NULL, latency_s REAL);
CREATE TABLE IF NOT EXISTS spend (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, kind TEXT NOT NULL,
    model TEXT, usd REAL NOT NULL, input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL, day TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL, actor TEXT NOT NULL,
    action TEXT NOT NULL, run_id TEXT, detail_json TEXT NOT NULL);
"""

AUDIT_ACTORS = frozenset({"admin", "supervisor", "controller", "chaos", "watchdog"})


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stage_key(stage_name: str, inputs: dict[str, Any]) -> str:
    return sha256_hex(canonical_json({"stage": stage_name, "inputs": inputs}).encode("utf-8"))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class HeldoutIntegrityError(RuntimeError):
    pass


class Store:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.root / "index.sqlite", check_same_thread=False)
        with self._lock:
            self._db.executescript(_SCHEMA)
            self._migrate()
            self._db.commit()

    def _migrate(self) -> None:
        """Additive columns for stores created by older versions (CREATE IF NOT EXISTS keeps an
        existing table as it was)."""
        cols = {r[1] for r in self._db.execute("PRAGMA table_info(llm_calls)").fetchall()}
        if "latency_s" not in cols:  # A2: per-call latency, for the B4 teacher p50/p95
            self._db.execute("ALTER TABLE llm_calls ADD COLUMN latency_s REAL")

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ---- paths -------------------------------------------------------
    def run_dir(self, run_id: str) -> Path:
        if not _RUN_ID_RE.fullmatch(run_id):
            raise ValueError(f"invalid run_id {run_id!r}")
        return self.root / "runs" / run_id

    def _exec(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        with self._lock:
            self._db.execute(sql, params)
            self._db.commit()

    def _query(self, sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        with self._lock:
            return [tuple(r) for r in self._db.execute(sql, params).fetchall()]

    def create_run(self, run_id: str) -> None:
        self.run_dir(run_id)
        self._exec("INSERT OR IGNORE INTO runs(run_id, created_at) VALUES (?, ?)", (run_id, _now()))

    # ---- artifacts ---------------------------------------------------
    def put_artifact(self, run_id: str, data: bytes) -> str:
        digest = sha256_hex(data)
        path = self.run_dir(run_id) / "artifacts" / digest
        if not path.exists():
            atomic_write_bytes(path, data)
        return digest

    def get_artifact(self, run_id: str, digest: str) -> bytes:
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("invalid artifact digest")
        data = (self.run_dir(run_id) / "artifacts" / digest).read_bytes()
        if sha256_hex(data) != digest:
            raise ValueError(f"artifact {digest} is corrupt")
        return data

    def _write_manifest(self, run_id: str) -> None:
        rows = self._query(
            "SELECT stage, input_hash, output_sha256, updated_at FROM stages "
            "WHERE run_id=? AND status='complete' ORDER BY stage, input_hash",
            (run_id,),
        )
        heldout = self._query("SELECT heldout_sha256 FROM runs WHERE run_id=?", (run_id,))
        manifest: dict[str, Any] = {
            "run_id": run_id,
            "heldout_sha256": heldout[0][0] if heldout else None,
            "stages": [
                {"stage": r[0], "input_hash": r[1], "output_sha256": r[2], "updated_at": r[3]}
                for r in rows
            ],
        }
        stress = self._query("SELECT sha256 FROM seals WHERE run_id=? AND name='stress'", (run_id,))
        if stress:
            manifest["stress_sha256"] = stress[0][0]
        human = self._query("SELECT sha256 FROM seals WHERE run_id=? AND name='human'", (run_id,))
        if human:
            manifest["human_sha256"] = human[0][0]
        atomic_write_bytes(
            self.run_dir(run_id) / "manifest.json",
            json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8"),
        )

    # ---- resumable stages -------------------------------------------
    def get_or_run(
        self, run_id: str, stage: str, inputs: dict[str, Any], fn: Callable[[], Any]
    ) -> Any:
        """Run `fn` (returning JSON-serialisable data) unless already completed for these inputs."""
        self.create_run(run_id)
        key = stage_key(stage, inputs)
        rows = self._query(
            "SELECT status, output_sha256 FROM stages WHERE run_id=? AND input_hash=?",
            (run_id, key),
        )
        if rows and rows[0][0] == "complete":
            try:
                return json.loads(self.get_artifact(run_id, rows[0][1]))
            except (OSError, ValueError):
                pass  # artifact missing/corrupt: fall through and recompute
        self._exec(
            "INSERT OR REPLACE INTO stages(run_id, stage, input_hash, status, updated_at) "
            "VALUES (?, ?, ?, 'running', ?)",
            (run_id, stage, key, _now()),
        )
        try:
            result = fn()
            payload = canonical_json(result).encode("utf-8")
        except BaseException as e:
            self._exec(
                "UPDATE stages SET status='failed', error=?, updated_at=? "
                "WHERE run_id=? AND input_hash=?",
                (f"{type(e).__name__}: {e}"[:500], _now(), run_id, key),
            )
            raise
        digest = self.put_artifact(run_id, payload)
        self._exec(
            "UPDATE stages SET status='complete', output_sha256=?, error=NULL, updated_at=? "
            "WHERE run_id=? AND input_hash=?",
            (digest, _now(), run_id, key),
        )
        self._write_manifest(run_id)
        return json.loads(payload)

    # ---- held-out ----------------------------------------------------
    def _heldout_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "heldout" / "heldout.json"

    def seal_heldout(self, run_id: str, items: list[dict[str, Any]]) -> str:
        self.create_run(run_id)
        payload = canonical_json(items).encode("utf-8")
        digest = sha256_hex(payload)
        rows = self._query("SELECT heldout_sha256 FROM runs WHERE run_id=?", (run_id,))
        existing = rows[0][0]
        if existing is not None and existing != digest:
            raise HeldoutIntegrityError("held-out set already sealed with different content")
        atomic_write_bytes(self._heldout_path(run_id), payload)
        self._exec("UPDATE runs SET heldout_sha256=? WHERE run_id=?", (digest, run_id))
        self._write_manifest(run_id)
        return digest

    def load_heldout(self, run_id: str) -> list[dict[str, Any]]:
        """Only evaluator.py may call this (enforced by an AST scan in tests)."""
        rows = self._query("SELECT heldout_sha256 FROM runs WHERE run_id=?", (run_id,))
        if not rows or rows[0][0] is None:
            raise HeldoutIntegrityError("no held-out set sealed for this run")
        data = self._heldout_path(run_id).read_bytes()
        if sha256_hex(data) != rows[0][0]:
            raise HeldoutIntegrityError("held-out set does not match its sealed sha256")
        items: list[dict[str, Any]] = json.loads(data)
        return items

    # ---- stress set (sealed like the gate set, but never a gate input) -
    def _stress_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "heldout" / "stress.json"

    def seal_stress(self, run_id: str, items: list[dict[str, Any]]) -> str:
        self.create_run(run_id)
        payload = canonical_json(items).encode("utf-8")
        digest = sha256_hex(payload)
        rows = self._query("SELECT sha256 FROM seals WHERE run_id=? AND name='stress'", (run_id,))
        if rows and rows[0][0] != digest:
            raise HeldoutIntegrityError("stress set already sealed with different content")
        atomic_write_bytes(self._stress_path(run_id), payload)
        self._exec(
            "INSERT OR REPLACE INTO seals(run_id, name, sha256) VALUES (?, 'stress', ?)",
            (run_id, digest),
        )
        self._write_manifest(run_id)
        return digest

    def load_stress(self, run_id: str) -> list[dict[str, Any]] | None:
        """Only evaluator.py may call this (enforced by an AST scan in tests). None = no stress
        set was sealed for this run (e.g. runs made before the stress set existed)."""
        rows = self._query("SELECT sha256 FROM seals WHERE run_id=? AND name='stress'", (run_id,))
        if not rows:
            return None
        data = self._stress_path(run_id).read_bytes()
        if sha256_hex(data) != rows[0][0]:
            raise HeldoutIntegrityError("stress set does not match its sealed sha256")
        items: list[dict[str, Any]] = json.loads(data)
        return items

    # ---- human held-out set (sealed like stress; Gate B, never a gate-A input) -----
    def _human_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "heldout" / "human.json"

    def seal_human(self, run_id: str, items: list[dict[str, Any]]) -> str:
        self.create_run(run_id)
        payload = canonical_json(items).encode("utf-8")
        digest = sha256_hex(payload)
        rows = self._query("SELECT sha256 FROM seals WHERE run_id=? AND name='human'", (run_id,))
        if rows and rows[0][0] != digest:
            raise HeldoutIntegrityError("human set already sealed with different content")
        atomic_write_bytes(self._human_path(run_id), payload)
        self._exec(
            "INSERT OR REPLACE INTO seals(run_id, name, sha256) VALUES (?, 'human', ?)",
            (run_id, digest),
        )
        self._write_manifest(run_id)
        return digest

    def load_human(self, run_id: str) -> list[dict[str, Any]] | None:
        """Only evaluator.py may call this (enforced by an AST scan in tests). None = no human
        set was sealed for this run."""
        rows = self._query("SELECT sha256 FROM seals WHERE run_id=? AND name='human'", (run_id,))
        if not rows:
            return None
        data = self._human_path(run_id).read_bytes()
        if sha256_hex(data) != rows[0][0]:
            raise HeldoutIntegrityError("human set does not match its sealed sha256")
        items: list[dict[str, Any]] = json.loads(data)
        return items

    # ---- experiments / llm calls / spend ----------------------------
    def add_experiment(self, run_id: str, name: str, data: dict[str, Any]) -> None:
        self.create_run(run_id)
        self._exec(
            "INSERT INTO experiments(run_id, name, data_json, created_at) VALUES (?,?,?,?)",
            (run_id, name, canonical_json(data), _now()),
        )

    # ---- audit log of interventions (who started, cancelled, restarted, decided) ----
    def add_audit(
        self, actor: str, action: str, run_id: str | None, detail: dict[str, Any] | None = None
    ) -> None:
        if actor not in AUDIT_ACTORS:
            raise ValueError(f"unknown audit actor {actor!r}")
        self._exec(
            "INSERT INTO audit(at, actor, action, run_id, detail_json) VALUES (?,?,?,?,?)",
            (_now(), actor, action, run_id, canonical_json(detail or {})),
        )

    def running_stages(self, run_id: str) -> list[str]:
        """Stages whose newest attempt is still marked running (the chaos supervisor's probe of
        "where is the child right now"; a killed child leaves its stage here until the resume)."""
        rows = self._query(
            "SELECT stage FROM stages WHERE run_id=? AND status='running' ORDER BY updated_at DESC",
            (run_id,),
        )
        return [str(r[0]) for r in rows]

    def list_audit(self, run_id: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT at, actor, action, run_id, detail_json FROM audit"
        params: tuple[Any, ...] = ()
        if run_id is not None:
            sql, params = sql + " WHERE run_id=?", (run_id,)
        rows = self._query(sql + " ORDER BY id", params)
        return [
            {"at": r[0], "actor": r[1], "action": r[2], "run_id": r[3], "detail": json.loads(r[4])}
            for r in rows
        ]

    def list_experiments(self, run_id: str) -> list[tuple[str, dict[str, Any]]]:
        rows = self._query(
            "SELECT name, data_json FROM experiments WHERE run_id=? ORDER BY id", (run_id,)
        )
        return [(r[0], json.loads(r[1])) for r in rows]

    def record_llm_call(
        self,
        run_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        usd: float,
        latency_s: float | None = None,
    ) -> None:
        self.create_run(run_id)
        self._exec(
            "INSERT INTO llm_calls(run_id, model, input_tokens, output_tokens, usd, created_at, "
            "latency_s) VALUES (?,?,?,?,?,?,?)",
            (run_id, model, input_tokens, output_tokens, usd, _now(), latency_s),
        )

    def record_spend(
        self,
        run_id: str,
        kind: str,
        model: str | None,
        usd: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        day: str | None = None,
    ) -> None:
        self.create_run(run_id)
        self._exec(
            "INSERT INTO spend(run_id, kind, model, usd, input_tokens, output_tokens, day, "
            "created_at) VALUES (?,?,?,?,?,?,?,?)",
            (
                run_id,
                kind,
                model,
                usd,
                input_tokens,
                output_tokens,
                day or datetime.now(UTC).date().isoformat(),
                _now(),
            ),
        )

    def total_spend(self, run_id: str | None = None, exclude_kind: str | None = None) -> float:
        sql = "SELECT COALESCE(SUM(usd),0) FROM spend WHERE 1=1"
        params: list[Any] = []
        if run_id is not None:
            sql += " AND run_id=?"
            params.append(run_id)
        if exclude_kind is not None:
            sql += " AND kind<>?"
            params.append(exclude_kind)
        return float(self._query(sql, tuple(params))[0][0])

    def spend_on_day(self, day: str, kind: str) -> float:
        rows = self._query(
            "SELECT COALESCE(SUM(usd),0) FROM spend WHERE day=? AND kind=?", (day, kind)
        )
        return float(rows[0][0])
