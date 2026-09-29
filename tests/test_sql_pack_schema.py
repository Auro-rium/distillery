# ruff: noqa: S101, E501, S108, S608
from __future__ import annotations

import sqlite3

import pytest

from distillery.taskpacks.sql.schema import (
    TABLES,
    build_database,
    schema_ddl,
    write_database,
)


def _con(seed: int) -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.deserialize(build_database(seed))
    return con


@pytest.fixture(scope="module")
def con() -> sqlite3.Connection:
    return _con(0)


def test_deterministic_and_seed_sensitive() -> None:
    assert build_database(3) == build_database(3)
    assert build_database(3) != build_database(4)


def test_schema_ddl_stable_and_matches_tables(con: sqlite3.Connection) -> None:
    ddl = schema_ddl(0)
    assert ddl == schema_ddl(99)
    names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert names == set(TABLES) and 7 <= len(names) <= 9
    for t in TABLES:
        assert f"CREATE TABLE {t} (" in ddl


def test_row_counts_reasonable(con: sqlite3.Connection) -> None:
    total = sum(con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES)
    assert 3000 <= total <= 12000


def test_foreign_keys_consistent(con: sqlite3.Connection) -> None:
    assert con.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.parametrize(
    ("table", "column"),
    [
        ("accounts", "deleted_at"),
        ("accounts", "parent_account_id"),
        ("accounts", "sales_rep"),
        ("users", "last_login_at"),
        ("users", "deleted_at"),
        ("subscriptions", "end_date"),
        ("subscriptions", "discount_pct"),
        ("invoices", "amount_cents"),
        ("invoices", "subscription_id"),
        ("payments", "processor_ref"),
        ("support_tickets", "user_id"),
        ("support_tickets", "closed_at"),
        ("support_tickets", "csat_score"),
    ],
)
def test_null_traps_present(con: sqlite3.Connection, table: str, column: str) -> None:
    nulls, nonnull = con.execute(
        f"SELECT SUM({column} IS NULL), SUM({column} IS NOT NULL) FROM {table}"
    ).fetchone()
    assert nulls > 0 and nonnull > 0


def test_duplicate_names_exist(con: sqlite3.Connection) -> None:
    dup = con.execute(
        "SELECT COUNT(*) FROM (SELECT name FROM accounts GROUP BY name HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    assert dup > 0


def test_dates_are_iso_and_logical(con: sqlite3.Connection) -> None:
    assert (
        con.execute(
            "SELECT COUNT(*) FROM accounts WHERE date(created_at) IS NOT created_at"
        ).fetchone()[0]
        == 0
    )
    assert (
        con.execute("SELECT COUNT(*) FROM invoices WHERE due_date <= issued_date").fetchone()[0]
        == 0
    )
    assert (
        con.execute(
            "SELECT COUNT(*) FROM support_tickets WHERE closed_at IS NOT NULL AND closed_at < opened_at"
        ).fetchone()[0]
        == 0
    )
    assert (
        con.execute(
            "SELECT COUNT(*) FROM support_tickets WHERE status <> 'closed' AND closed_at IS NOT NULL"
        ).fetchone()[0]
        == 0
    )


def test_enums(con: sqlite3.Connection) -> None:
    got = {r[0] for r in con.execute("SELECT DISTINCT status FROM invoices")}
    assert got == {"paid", "open", "overdue", "void"}


def test_write_database_roundtrip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    p = write_database(1, tmp_path / "sub" / "db.sqlite")
    assert p.read_bytes() == build_database(1)
