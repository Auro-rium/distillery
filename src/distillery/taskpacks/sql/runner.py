"""Read-only, time- and row-bounded SQLite query execution (shared by verifier and executor)."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from distillery.taskpacks.sql.sqltext import first_keyword, split_statements

DbRef = str | Path | bytes
Scalar = None | int | float | str | bytes

ErrorKind = Literal["syntax", "runtime", "timeout", "forbidden", "row_cap", "multi_statement"]

DEFAULT_TIMEOUT_S = 2.0
DEFAULT_MAX_ROWS = 10_000
_PROGRESS_OPS = 1000

_ALLOWED_ACTIONS = frozenset(
    {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
)


@dataclass(frozen=True)
class ExecOutcome:
    """Result of running one SQL string. ``rows`` is only meaningful when ``ok``."""

    ok: bool
    columns: tuple[str, ...] = ()
    rows: tuple[tuple[Scalar, ...], ...] = ()
    error_kind: ErrorKind | None = None
    error: str = ""
    elapsed_s: float = field(default=0.0, compare=False)


def _authorizer(action: int, *_args: str | None) -> int:
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def open_readonly(db: DbRef) -> sqlite3.Connection:
    """Open ``db`` (path or serialized bytes) so that nothing can be written."""
    if isinstance(db, bytes):
        con = sqlite3.connect(":memory:", isolation_level=None)
        con.deserialize(db)
    else:
        uri = Path(db).resolve().as_uri() + "?mode=ro"
        con = sqlite3.connect(uri, uri=True, isolation_level=None)
    con.execute("PRAGMA query_only = ON")
    con.set_authorizer(_authorizer)
    return con


def _check_single_select(sql: str) -> tuple[ErrorKind, str] | None:
    stmts = split_statements(sql)
    if not stmts:
        return "syntax", "empty SQL"
    if len(stmts) > 1:
        return "multi_statement", "multiple statements are not allowed"
    if first_keyword(stmts[0]) not in ("SELECT", "WITH"):
        return "forbidden", "only SELECT/WITH queries are allowed"
    return None


def run_select(
    db: DbRef,
    sql: str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> ExecOutcome:
    """Run one SELECT/WITH statement read-only; never raises for bad SQL."""
    started = time.monotonic()
    problem = _check_single_select(sql)
    if problem is not None:
        return ExecOutcome(False, error_kind=problem[0], error=problem[1])
    con = open_readonly(db)
    deadline = started + timeout_s
    timed_out = False

    def _progress() -> int:
        nonlocal timed_out
        if time.monotonic() > deadline:
            timed_out = True
            return 1
        return 0

    con.set_progress_handler(_progress, _PROGRESS_OPS)
    try:
        cur = con.execute(sql)
        cols = tuple(d[0] for d in (cur.description or ()))
        fetched = cur.fetchmany(max_rows + 1)
        if len(fetched) > max_rows:
            return ExecOutcome(
                False, cols, error_kind="row_cap", error=f"more than {max_rows} rows",
                elapsed_s=time.monotonic() - started,
            )  # fmt: skip
        return ExecOutcome(
            True, cols, tuple(tuple(r) for r in fetched), elapsed_s=time.monotonic() - started
        )
    except sqlite3.DatabaseError as exc:  # OperationalError/ProgrammingError/IntegrityError...
        elapsed = time.monotonic() - started
        if timed_out:
            return ExecOutcome(
                False, error_kind="timeout", error="query timed out", elapsed_s=elapsed
            )
        msg = str(exc)
        if "not authorized" in msg or "readonly" in msg or "read-only" in msg:
            kind: ErrorKind = "forbidden"
        elif isinstance(exc, sqlite3.OperationalError) and (
            "syntax" in msg or "no such" in msg or "ambiguous" in msg
        ):
            kind = "syntax"
        else:
            kind = "runtime"
        return ExecOutcome(False, error_kind=kind, error=msg, elapsed_s=elapsed)
    finally:
        con.close()
