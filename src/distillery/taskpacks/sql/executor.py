"""Executor protocol: run SQL batches against a database reference.

A Sandbox-backed implementation lives elsewhere; ``LocalExecutor`` runs in-process and is used
by unit tests and the offline path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from distillery.taskpacks.sql.runner import (
    DEFAULT_MAX_ROWS,
    DEFAULT_TIMEOUT_S,
    DbRef,
    ExecOutcome,
    run_select,
)

__all__ = ["ExecOutcome", "Executor", "LocalExecutor"]


class Executor(Protocol):
    """Runs SQL strings read-only against the database named by ``db_ref``.

    Must return exactly one ``ExecOutcome`` per input SQL, in order, and must never raise for
    bad SQL (errors are reported in the outcome).
    """

    def run_batch(self, db_ref: str, sqls: Sequence[str]) -> list[ExecOutcome]: ...


class LocalExecutor:
    """In-process executor. ``db_ref`` is a key in ``databases`` (bytes) or else a file path."""

    def __init__(
        self,
        databases: Mapping[str, bytes] | None = None,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_rows: int = DEFAULT_MAX_ROWS,
    ) -> None:
        self._databases: dict[str, bytes] = dict(databases or {})
        self._timeout_s = timeout_s
        self._max_rows = max_rows

    def register(self, db_ref: str, data: bytes) -> None:
        self._databases[db_ref] = data

    def _resolve(self, db_ref: str) -> DbRef:
        data = self._databases.get(db_ref)
        return data if data is not None else Path(db_ref)

    def run_batch(self, db_ref: str, sqls: Sequence[str]) -> list[ExecOutcome]:
        db = self._resolve(db_ref)
        return [run_select(db, s, timeout_s=self._timeout_s, max_rows=self._max_rows) for s in sqls]
