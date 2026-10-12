"""Cross-process GLOBAL spend ledger: one SQLite file shared by every store copy and process.

Each ``.distillery/*`` directory is its own store, pipeline runs are child processes, and the
local playground is a long-lived server, so a per-store ``total_spend()`` snapshot cannot enforce
the project cap. This file is the single place that is read, fresh, before every project-cap check
and written on every spend.

Concurrency: a connection per operation, WAL, ``busy_timeout``, and every check-and-write inside
``BEGIN IMMEDIATE`` (one writer at a time, so a check cannot race another process's insert).

``spend`` rows are money that has been spent. ``reservations`` are money that is held back: with
``expires_at`` NULL a reservation is permanent (a committed allocation, e.g. the hosted demo);
otherwise it counts until it expires. ``total_committed()`` = all spend + active reservations.

Location: env ``DISTILLERY_GLOBAL_LEDGER``. Unset: ``~/.distillery/global_ledger.sqlite``.
``none`` / ``off`` / empty string: disabled (callers keep their local-store behaviour).
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import sqlite3
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

ENV_VAR = "DISTILLERY_GLOBAL_LEDGER"
DEFAULT_RELATIVE = Path(".distillery") / "global_ledger.sqlite"
KIND_SEED = "seed"
BUSY_TIMEOUT_S = 30.0
_OFF = frozenset({"none", "off"})
_EPS = 1e-12  # a reservation this close to fully spent counts as spent

_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS spend("
    "id INTEGER PRIMARY KEY, source_root TEXT, run_id TEXT, kind TEXT, model TEXT, usd REAL, "
    "input_tokens INTEGER, output_tokens INTEGER, created_at TEXT, note TEXT)",
    "CREATE TABLE IF NOT EXISTS reservations("
    "id TEXT PRIMARY KEY, holder TEXT, usd REAL, created_at TEXT, expires_at TEXT, note TEXT)",
)


class BudgetExceeded(RuntimeError):
    pass


class GlobalLedgerError(RuntimeError):
    """The shared ledger could not be read or written. Callers fail closed: a cap that cannot be
    checked is not a cap that holds."""


def resolve_path(env: Mapping[str, str] | None = None) -> Path | None:
    """The ledger file, or None when disabled. ``env`` wins when it carries the variable;
    otherwise ``os.environ`` is consulted (so a hand-built mapping cannot silently fall through
    to the home default while the process environment says otherwise)."""
    if env is not None and ENV_VAR in env:
        raw: str | None = env[ENV_VAR]
    else:
        raw = os.environ.get(ENV_VAR)
    if raw is None:
        return Path.home() / DEFAULT_RELATIVE
    value = raw.strip()
    if value == "" or value.lower() in _OFF:
        return None
    return Path(value).expanduser()


def _check_usd(usd: float, what: str = "usd") -> None:
    if not isinstance(usd, int | float) or not math.isfinite(usd) or usd < 0:
        raise ValueError(f"{what} must be a finite number >= 0, got {usd!r}")


@dataclass
class SeedResult:
    spend_inserted: list[str] = field(default_factory=list)
    spend_skipped: list[str] = field(default_factory=list)
    reservations_inserted: list[str] = field(default_factory=list)
    reservations_skipped: list[str] = field(default_factory=list)


class GlobalLedger:
    def __init__(
        self,
        path: str | Path,
        *,
        busy_timeout_s: float = BUSY_TIMEOUT_S,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.path = Path(path)
        self.busy_timeout_s = busy_timeout_s
        self._clock = clock or (lambda: datetime.now(UTC))
        self._init_schema()

    @classmethod
    def from_env(
        cls, env: Mapping[str, str] | None = None, *, clock: Callable[[], datetime] | None = None
    ) -> GlobalLedger | None:
        path = resolve_path(env)
        return None if path is None else cls(path, clock=clock)

    # ---- plumbing ---------------------------------------------------------
    def _stamp(self, when: datetime | None = None) -> str:
        """One fixed-width UTC format, so ``expires_at > now`` is a correct string comparison."""
        return (when or self._clock()).astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    def _connect(self) -> sqlite3.Connection:
        # isolation_level=None: no implicit transactions; BEGIN/COMMIT are issued explicitly.
        con = sqlite3.connect(self.path, timeout=self.busy_timeout_s, isolation_level=None)
        con.execute(f"PRAGMA busy_timeout={round(self.busy_timeout_s * 1000)}")
        return con

    def _init_schema(self) -> None:
        deadline = time.monotonic() + self.busy_timeout_s
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            while True:
                con = self._connect()
                try:
                    # Switching a fresh file to WAL can fail with "locked" without waiting when
                    # several processes initialise at once: retry until the deadline.
                    con.execute("PRAGMA journal_mode=WAL")
                    con.execute("BEGIN IMMEDIATE")
                    for ddl in _SCHEMA:
                        con.execute(ddl)
                    con.execute("COMMIT")
                    return
                except sqlite3.OperationalError as exc:
                    with contextlib.suppress(sqlite3.Error):
                        con.execute("ROLLBACK")
                    if "locked" not in str(exc) or time.monotonic() >= deadline:
                        raise
                    time.sleep(0.02)
                finally:
                    con.close()
        except (sqlite3.Error, OSError) as exc:
            raise GlobalLedgerError(f"global ledger {self.path}: {exc}") from exc

    @contextlib.contextmanager
    def _txn(self, *, write: bool) -> Iterator[sqlite3.Connection]:
        try:
            con = self._connect()
            try:
                con.execute("BEGIN IMMEDIATE" if write else "BEGIN")
                try:
                    yield con
                except BaseException:
                    with contextlib.suppress(sqlite3.Error):
                        con.execute("ROLLBACK")
                    raise
                con.execute("COMMIT")
            finally:
                con.close()
        except sqlite3.Error as exc:
            raise GlobalLedgerError(f"global ledger {self.path}: {exc}") from exc

    def _active_sql(self) -> tuple[str, tuple[str]]:
        return "(expires_at IS NULL OR expires_at > ?)", (self._stamp(),)

    def _totals(self, con: sqlite3.Connection) -> tuple[float, float]:
        spent = con.execute("SELECT COALESCE(SUM(usd),0) FROM spend").fetchone()[0]
        cond, params = self._active_sql()
        held = con.execute(
            f"SELECT COALESCE(SUM(usd),0) FROM reservations WHERE {cond}",  # noqa: S608
            params,
        ).fetchone()[0]
        return float(spent), float(held)

    # ---- reads ------------------------------------------------------------
    def total_spend(self) -> float:
        with self._txn(write=False) as con:
            return self._totals(con)[0]

    def total_reserved(self) -> float:
        with self._txn(write=False) as con:
            return self._totals(con)[1]

    def total_committed(self) -> float:
        """All spend plus all active (unexpired) reservations, read fresh from the file."""
        with self._txn(write=False) as con:
            spent, held = self._totals(con)
        return spent + held

    def summary(self) -> dict[str, Any]:
        with self._txn(write=False) as con:
            spent, held = self._totals(con)
            by_kind = [
                {"kind": k, "n": n, "usd": float(u)}
                for k, n, u in con.execute(
                    "SELECT kind, COUNT(*), SUM(usd) FROM spend GROUP BY kind ORDER BY kind"
                )
            ]
            by_run = [
                {"run_id": r, "n": n, "usd": float(u)}
                for r, n, u in con.execute(
                    "SELECT run_id, COUNT(*), SUM(usd) FROM spend GROUP BY run_id ORDER BY run_id"
                )
            ]
            cond, params = self._active_sql()
            reservations = [
                {
                    "id": i,
                    "holder": h,
                    "usd": float(u),
                    "created_at": c,
                    "expires_at": e,
                    "note": n,
                }
                for i, h, u, c, e, n in con.execute(
                    "SELECT id, holder, usd, created_at, expires_at, note FROM reservations "  # noqa: S608
                    f"WHERE {cond} ORDER BY created_at, id",
                    params,
                )
            ]
        return {
            "path": str(self.path),
            "spend_usd": spent,
            "reserved_usd": held,
            "committed_usd": spent + held,
            "by_kind": by_kind,
            "by_run": by_run,
            "reservations": reservations,
        }

    # ---- writes -----------------------------------------------------------
    def reserve(
        self,
        holder: str,
        usd: float,
        cap: float,
        ttl_s: float | None = None,
        note: str = "",
        *,
        reservation_id: str | None = None,
    ) -> str:
        """Atomically hold ``usd`` against ``cap``: committed + usd > cap raises BudgetExceeded and
        writes nothing. ``ttl_s`` None is a permanent reservation. Returns the reservation id."""
        _check_usd(usd)
        rid = reservation_id or uuid.uuid4().hex
        now = self._clock()
        expires = None if ttl_s is None else self._stamp(now + timedelta(seconds=ttl_s))
        with self._txn(write=True) as con:
            spent, held = self._totals(con)
            if spent + held + usd > cap:
                raise BudgetExceeded(
                    f"project cap ${cap:.2f}: committed ${spent + held:.4f} "
                    f"+ reserve ${usd:.4f} ({holder})"
                )
            con.execute(
                "INSERT INTO reservations(id, holder, usd, created_at, expires_at, note) "
                "VALUES (?,?,?,?,?,?)",
                (rid, str(holder), usd, self._stamp(now), expires, note),
            )
        return rid

    def release(self, reservation_id: str) -> bool:
        """Drop a reservation. False when it was already gone."""
        with self._txn(write=True) as con:
            cur = con.execute("DELETE FROM reservations WHERE id=?", (reservation_id,))
            return cur.rowcount > 0

    def record(
        self,
        source_root: str,
        run_id: str,
        kind: str,
        model: str | None,
        usd: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        note: str = "",
        reservation_id: str | None = None,
        *,
        release_remainder: bool = False,
    ) -> int:
        """Record spend that already happened (never refused: a cap cannot undo a cost). With a
        ``reservation_id`` the reservation shrinks by ``usd`` in the same transaction and is
        dropped when used up, or at once with ``release_remainder``. Returns the spend row id."""
        _check_usd(usd)
        with self._txn(write=True) as con:
            cur = con.execute(
                "INSERT INTO spend(source_root, run_id, kind, model, usd, input_tokens, "
                "output_tokens, created_at, note) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    source_root,
                    run_id,
                    kind,
                    model,
                    usd,
                    input_tokens,
                    output_tokens,
                    self._stamp(),
                    note,
                ),
            )
            row_id = int(cur.lastrowid or 0)
            if reservation_id is not None:
                self._settle(con, reservation_id, usd, release_remainder)
        return row_id

    @staticmethod
    def _settle(con: sqlite3.Connection, reservation_id: str, usd: float, drop: bool) -> None:
        row = con.execute("SELECT usd FROM reservations WHERE id=?", (reservation_id,)).fetchone()
        if row is None:  # released or expired meanwhile: the spend is recorded all the same
            return
        left = float(row[0]) - usd
        if drop or left <= _EPS:
            con.execute("DELETE FROM reservations WHERE id=?", (reservation_id,))
        else:
            con.execute("UPDATE reservations SET usd=? WHERE id=?", (left, reservation_id))

    def seed(
        self,
        spend: Sequence[Mapping[str, Any]],
        reservations: Sequence[Mapping[str, Any]],
    ) -> SeedResult:
        """Insert seed rows ONCE, all in one transaction. A spend row whose ``note`` is already in
        the ledger, and a reservation whose ``id`` already exists, is skipped, so running the same
        file twice changes nothing."""
        notes = [str(r.get("note") or "") for r in spend]
        if any(not n for n in notes):
            raise ValueError("every seed spend row needs a non-empty note (the idempotency key)")
        if len(set(notes)) != len(notes):
            raise ValueError("seed spend notes must be unique within the file")
        for r in spend:
            _check_usd(r["usd"], "seed spend usd")
        for r in reservations:
            _check_usd(r["usd"], "seed reservation usd")
        out = SeedResult()
        now = self._stamp()
        with self._txn(write=True) as con:
            for r in spend:
                note = str(r["note"])
                if con.execute("SELECT 1 FROM spend WHERE note=?", (note,)).fetchone():
                    out.spend_skipped.append(note)
                    continue
                con.execute(
                    "INSERT INTO spend(source_root, run_id, kind, model, usd, input_tokens, "
                    "output_tokens, created_at, note) VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        r.get("source_root"),
                        r.get("run_id"),
                        r.get("kind") or KIND_SEED,
                        r.get("model"),
                        r["usd"],
                        int(r.get("input_tokens") or 0),
                        int(r.get("output_tokens") or 0),
                        now,
                        note,
                    ),
                )
                out.spend_inserted.append(note)
            for r in reservations:
                rid = str(r["id"])
                if con.execute("SELECT 1 FROM reservations WHERE id=?", (rid,)).fetchone():
                    out.reservations_skipped.append(rid)
                    continue
                con.execute(
                    "INSERT INTO reservations(id, holder, usd, created_at, expires_at, note) "
                    "VALUES (?,?,?,?,?,?)",
                    (
                        rid,
                        str(r.get("holder") or rid),
                        r["usd"],
                        now,
                        r.get("expires_at"),
                        str(r.get("note") or ""),
                    ),
                )
                out.reservations_inserted.append(rid)
        return out


def load_seed_file(path: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse a seed file: ``{"spend": [...], "reservations": [...]}``. A spend row's optional
    ``evidence`` is appended to its note, which is what makes the row identifiable (and unique)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read seed file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("seed file must be a JSON object")
    spend: list[dict[str, Any]] = []
    for r in data.get("spend", []):
        if not isinstance(r, dict) or "usd" not in r or not r.get("note"):
            raise ValueError("each seed spend row needs usd and note")
        row = dict(r)
        evidence = row.pop("evidence", None)
        if evidence:
            row["note"] = f"{row['note']} | evidence: {evidence}"
        spend.append(row)
    reservations = [dict(r) for r in data.get("reservations", [])]
    for r in reservations:
        if "id" not in r or "usd" not in r:
            raise ValueError("each seed reservation needs id and usd")
    return spend, reservations
