"""The offline template audit is a CI gate: no template may have an ambiguous question, an
ordering tie, a LIMIT without ORDER BY, or gold that errors (the benchmark's gold is template SQL,
so a defective template is a defective benchmark)."""

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "audit_templates.py"


def _load():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("audit_templates", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["audit_templates"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_every_template_is_unambiguous_and_deterministic() -> None:
    rep = _load().audit(draws=60, seed=99)
    assert rep["templates"] == 89
    assert rep["flagged_templates"] == {}
    assert rep["questions_with_multiple_gold_sql"] == []


def test_a_question_whose_gold_has_a_hidden_filter_is_flagged(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    mod = _load()
    from distillery.taskpacks.sql import questions

    def hidden_filter(rng):  # type: ignore[no-untyped-def]
        c = rng.choice(("US", "DE"))  # the wording never says which country
        return (
            "How many accounts are there?",
            f"SELECT COUNT(*) FROM accounts WHERE country = '{c}'",
        )

    tpl = questions.Template("hidden_filter", "filter", ("accounts",), "easy", hidden_filter)
    monkeypatch.setattr(mod, "TEMPLATES", [tpl])
    rep = mod.audit(draws=20, seed=1)
    assert rep["n_questions_with_multiple_gold_sql"] == 1


def test_reversed_copy_exposes_an_order_tie(tmp_path: Path) -> None:
    import sqlite3

    mod = _load()
    src, dst = tmp_path / "a.sqlite", tmp_path / "b.sqlite"
    with sqlite3.connect(src) as db:
        db.execute("CREATE TABLE t (k INTEGER, v INTEGER)")
        db.executemany("INSERT INTO t VALUES (?, ?)", [(1, 1), (1, 2), (2, 3)])
    mod.reversed_copy(src, dst)
    with sqlite3.connect(dst) as db:
        assert db.execute("SELECT v FROM t").fetchall() == [(3,), (2,), (1,)]
