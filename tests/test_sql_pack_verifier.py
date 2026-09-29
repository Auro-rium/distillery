# ruff: noqa: S101, E501, S108, S608, S311
from __future__ import annotations

import random
import time

import pytest

from distillery.taskpacks.sql.executor import LocalExecutor
from distillery.taskpacks.sql.schema import build_database
from distillery.taskpacks.sql.sqltext import (
    has_top_level_order_by,
    mask_sql,
    split_statements,
)
from distillery.taskpacks.sql.verifier import (
    compare_outcomes,
    corrupt_sql,
    execution_match,
)


@pytest.fixture(scope="module")
def db() -> bytes:
    return build_database(0)


def m(db: bytes, cand: str, gold: str, order: bool = False) -> bool:
    return execution_match(db, cand, gold, order).ok


GOLD = "SELECT account_id, name FROM accounts WHERE country = 'DE' AND status = 'active'"


def test_identical_and_alias_and_row_order(db: bytes) -> None:
    assert m(db, GOLD, GOLD)
    assert m(db, GOLD.replace("account_id, name", "account_id AS a, name AS b"), GOLD)
    assert m(db, GOLD + " ORDER BY account_id DESC", GOLD)


def test_order_sensitive_only_when_required(db: bytes) -> None:
    asc = "SELECT account_id FROM accounts WHERE country='DE' ORDER BY account_id"
    desc = asc + " DESC"
    assert m(db, desc, asc, order=False)
    r = execution_match(db, desc, asc, True)
    assert not r.ok and "order" in r.reason


def test_wrong_column_extra_and_missing_column(db: bytes) -> None:
    assert not m(
        db, "SELECT account_id, country FROM accounts WHERE country='DE' AND status='active'", GOLD
    )
    r = execution_match(
        db, "SELECT account_id FROM accounts WHERE country='DE' AND status='active'", GOLD, False
    )
    assert not r.ok and "column count" in r.reason
    assert not m(
        db, "SELECT account_id, name, 1 FROM accounts WHERE country='DE' AND status='active'", GOLD
    )


def test_extra_and_missing_row(db: bytes) -> None:
    r = execution_match(db, GOLD.replace("'active'", "'active' OR status = 'trial'"), GOLD, False)
    assert not r.ok and "row count" in r.reason
    r = execution_match(db, GOLD + " LIMIT 1", GOLD, False)
    assert not r.ok and "row count" in r.reason


def test_same_count_different_rows(db: bytes) -> None:
    a = "SELECT account_id FROM accounts WHERE account_id IN (1, 2)"
    b = "SELECT account_id FROM accounts WHERE account_id IN (1, 3)"
    r = execution_match(db, a, b, False)
    assert not r.ok and "differ" in r.reason


def test_duplicate_rows_matter(db: bytes) -> None:
    assert not m(db, "SELECT 1 UNION ALL SELECT 1", "SELECT 1 UNION SELECT 1")


def test_wrong_null_handling(db: bytes) -> None:
    gold = "SELECT COUNT(*) FROM invoices WHERE amount_cents IS NULL"
    assert not m(db, "SELECT COUNT(*) FROM invoices WHERE amount_cents = NULL", gold)
    assert not m(db, "SELECT COUNT(*) FROM invoices WHERE amount_cents IS NOT NULL", gold)
    assert not m(
        db,
        "SELECT AVG(COALESCE(amount_cents, 0)) FROM invoices",
        "SELECT AVG(amount_cents) FROM invoices",
    )
    assert not m(db, "SELECT NULL", "SELECT ''")
    assert m(db, "SELECT NULL", "SELECT NULL")


def test_numeric_normalisation(db: bytes) -> None:
    assert m(db, "SELECT 3", "SELECT 3.0")
    assert m(db, "SELECT 0.1 + 0.2", "SELECT 0.3")
    assert m(db, "SELECT 1.0000004", "SELECT 1.0000001")  # both round to 1.0 at 6 dp
    assert not m(db, "SELECT 1.00001", "SELECT 1.0")
    assert m(db, "SELECT 1 = 1", "SELECT 1")


def test_strings_not_normalised(db: bytes) -> None:
    assert not m(db, "SELECT 'Abc'", "SELECT 'abc'")
    assert not m(db, "SELECT 'a b'", "SELECT 'a  b'")
    assert not m(db, "SELECT 'a'", "SELECT 'a '")
    assert not m(db, "SELECT '1'", "SELECT 1")


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM accounts",
        "UPDATE accounts SET name = 'x'",
        "INSERT INTO plans VALUES (99,'x','free',0,NULL)",
        "DROP TABLE accounts",
        "CREATE TABLE z (a)",
        "PRAGMA writable_schema = 1",
        "ATTACH DATABASE '/tmp/zzz.db' AS z",
        "SELECT 1; DELETE FROM accounts",
        "SELECT 1; SELECT 2",
        "WITH x AS (SELECT 1) DELETE FROM accounts",
        "WITH x AS (SELECT 1) INSERT INTO plans VALUES (99,'x','free',0,NULL)",
        "",
        "   ;  ",
        "EXPLAIN SELECT 1",
        "VACUUM",
    ],
)
def test_rejects_writes_and_non_select(db: bytes, sql: str) -> None:
    r = execution_match(db, sql, "SELECT 1", False)
    assert not r.ok and "candidate error" in r.reason


def test_write_attempt_does_not_modify_db(db: bytes) -> None:
    execution_match(db, "WITH x AS (SELECT 1) DELETE FROM accounts", "SELECT 1", False)
    assert m(db, "SELECT COUNT(*) FROM accounts", "SELECT 160")


def test_allows_trailing_semicolon_and_comments(db: bytes) -> None:
    assert m(db, "-- hi\nSELECT 1; -- bye\n", "SELECT 1")
    assert m(db, "/* c */ WITH t AS (SELECT 1 AS x) SELECT x FROM t;", "SELECT 1")
    assert m(db, "SELECT ';'", "SELECT ';'")


def test_read_only_on_file_path(db: bytes, tmp_path) -> None:  # type: ignore[no-untyped-def]
    p = tmp_path / "x.db"
    p.write_bytes(db)
    assert execution_match(p, "SELECT COUNT(*) FROM plans", "SELECT 6", False).ok
    assert execution_match(str(p), "SELECT COUNT(*) FROM plans", "SELECT 6", False).ok
    r = execution_match(p, "DELETE FROM plans", "SELECT 1", False)
    assert not r.ok
    assert p.read_bytes() == db


def test_infinite_recursive_cte_times_out(db: bytes) -> None:
    inf = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT COUNT(*) FROM r"
    t0 = time.monotonic()
    r = execution_match(db, inf, "SELECT 1", False, timeout_s=0.3)
    assert not r.ok and "timeout" in r.reason
    assert time.monotonic() - t0 < 3


def test_row_cap(db: bytes) -> None:
    big = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM r WHERE n < 500) SELECT n FROM r"
    r = execution_match(db, big, "SELECT 1", False, max_rows=100)
    assert not r.ok and "row_cap" in r.reason


def test_syntax_and_missing_table_errors(db: bytes) -> None:
    for bad in ("SELEC 1", "SELECT * FROM nope", "SELECT nocol FROM accounts"):
        r = execution_match(db, bad, "SELECT 1", False)
        assert not r.ok and "candidate error" in r.reason


def test_bad_gold_reported(db: bytes) -> None:
    r = execution_match(db, "SELECT 1", "SELECT * FROM nope", False)
    assert not r.ok and "gold error" in r.reason


def test_compare_outcomes_with_any_executor(db: bytes) -> None:
    ex = LocalExecutor({"d": db})
    a, b = ex.run_batch("d", ["SELECT 1", "SELECT 1.0"])
    assert compare_outcomes(a, b, False).ok


# ---- sqltext ---------------------------------------------------------------------------------


def test_split_and_mask() -> None:
    assert split_statements("SELECT 1; SELECT ';' ;;") == ["SELECT 1", "SELECT ';'"]
    assert split_statements("-- only comment") == []
    masked = mask_sql("SELECT 'a;b' -- x;\n")
    assert ";" not in masked and masked.startswith("SELECT '   '") and masked.endswith("\n")
    assert len(mask_sql('SELECT "a""b", [c;d]')) == len('SELECT "a""b", [c;d]')


def test_order_by_detection() -> None:
    assert has_top_level_order_by("SELECT a FROM t ORDER BY a")
    assert has_top_level_order_by("select a from t order\n by a")
    assert not has_top_level_order_by("SELECT a, ROW_NUMBER() OVER (ORDER BY a) FROM t")
    assert not has_top_level_order_by("SELECT 'ORDER BY' FROM t")
    assert not has_top_level_order_by("SELECT * FROM (SELECT a FROM t ORDER BY a LIMIT 3)")


# ---- corruption helpers ----------------------------------------------------------------------

GOLDS = [
    GOLD,
    "SELECT COUNT(*) FROM invoices WHERE amount_cents IS NULL",
    "SELECT a.name, COUNT(*) FROM accounts a JOIN users u ON u.account_id = a.account_id GROUP BY a.account_id ORDER BY 2 DESC, a.account_id LIMIT 5",
    "SELECT invoice_id, amount_cents FROM invoices WHERE amount_cents > 50000",
    "SELECT ticket_id, csat_score FROM support_tickets WHERE status = 'closed' AND priority = 'high'",
]


@pytest.mark.parametrize("gold", GOLDS)
def test_corrupt_sql_variants_execute_and_are_rejected(db: bytes, gold: str) -> None:
    variants = corrupt_sql(gold, db, random.Random(1))
    assert len(variants) >= 3
    order = has_top_level_order_by(gold)
    for v in variants:
        assert v.sql.strip() != gold.strip()
        assert not execution_match(db, v.sql, gold, order).ok
        assert "candidate error" not in execution_match(db, v.sql, gold, order).reason
    assert len({v.kind for v in variants}) >= 3


def test_corrupt_sql_deterministic_and_covers_null_kind(db: bytes) -> None:
    g = "SELECT ticket_id, csat_score FROM support_tickets WHERE status = 'closed'"
    a = corrupt_sql(g, db, random.Random(5), max_variants=50)
    b = corrupt_sql(g, db, random.Random(5), max_variants=50)
    assert a == b
    kinds = {v.kind for v in a}
    assert {"extra_row", "missing_row", "dropped_column", "null_to_empty"} <= kinds


def test_corrupt_sql_bad_gold_returns_empty(db: bytes) -> None:
    assert corrupt_sql("SELECT * FROM nope", db, random.Random(0)) == []
