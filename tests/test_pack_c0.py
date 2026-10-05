"""C0: the pack interface and its SQL implementation (a pure refactor: same behaviour)."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from distillery import prompts
from distillery.server import autonomy
from distillery.server.views import with_answer_fields
from distillery.server.worker import Job
from distillery.taskpacks import base
from distillery.taskpacks.base import Pack, get_pack, pack_names
from distillery.taskpacks.sql import questions as q
from distillery.taskpacks.sql import schema as sql_schema
from distillery.taskpacks.sql.executor import LocalExecutor


def test_registry_lists_packs_and_rejects_unknown_names() -> None:
    assert pack_names() == ("sql", "toolcall")
    assert get_pack("sql").name == "sql"
    with pytest.raises(ValueError, match="unknown pack 'nope'"):
        get_pack("nope")


def test_sql_pack_satisfies_the_protocol_and_declares_its_constants() -> None:
    pack = get_pack("sql")
    assert isinstance(pack, Pack)
    assert (pack.language, pack.answer_label, pack.gold_field, pack.answer_key) == (
        "sql",
        "SQL",
        "gold_sql",
        "sql",
    )
    assert (
        pack.student_max_new_tokens == 160 and pack.default_skeleton_cap == q.DEFAULT_SKELETON_CAP
    )
    assert pack.families() == q.FAMILIES and pack.stress_families() == q.stress_families()


def test_sql_pack_env_tasks_and_verification_match_the_old_functions(tmp_path: Path) -> None:
    pack = get_pack("sql")
    env = pack.build_env(3, tmp_path / "db.sqlite")
    data = sql_schema.build_database(3)
    assert Path(env.ref).read_bytes() == data and env.size == len(data)
    assert env.context_text == sql_schema.schema_ddl(3)
    rep = pack.generate_tasks(env.ref, 12, 5, exclude_families=pack.stress_families())
    ref = q.generate_tasks(env.ref, 12, 5, exclude_families=q.stress_families())
    assert [t.task_id for t in rep.tasks] == [t.task_id for t in ref.tasks]
    t = rep.tasks[0]
    assert pack.task_from(t.model_dump(mode="json")) == t
    assert t.gold_answer == t.gold_sql and t.order_matters == t.requires_order  # type: ignore[attr-defined]
    ex = LocalExecutor()
    gold = ex.run_batch(env.ref, [t.gold_answer])[0]
    assert pack.gold_problem(gold) is None
    assert pack.compare(gold, pack.run_local(env.ref, t.gold_answer), t.order_matters).ok
    variants = pack.corrupt(t.gold_answer, env.ref, random.Random(1))
    assert variants and all(v.answer != t.gold_answer for v in variants)
    outs = ex.run_batch(env.ref, [v.answer for v in variants])
    assert not any(pack.compare(o, gold, t.order_matters).ok for o in outs)
    bad = ex.run_batch(env.ref, ["DROP TABLE accounts"])[0]
    assert pack.gold_problem(bad) == "gold_error"


def test_sql_pack_prompts_are_the_shared_builders() -> None:
    pack = get_pack("sql")
    assert pack.build_messages("Q?", "DDL", role="train") == prompts.build_messages(
        "Q?", "DDL", role="train"
    )
    assert pack.extract_answer("```sql\nSELECT 1;\n```") == "SELECT 1"
    assert "SQL" in pack.triage_prompt("SELECT 1")[0]["content"]
    msgs = pack.analysis_prompt(
        [{"family": "f", "question": "q", "gold_sql": "g", "student_sql": "s", "reason": "r"}],
        ["f"],
    )
    assert msgs[0]["role"] == "system" and "Allowed families: f" in msgs[1]["content"]


def test_registered_pack_is_resolved_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(base._REGISTRY, "alias", "distillery.taskpacks.sql.pack")
    assert get_pack("alias") is get_pack("sql")
    assert "alias" in pack_names()


def test_example_answer_aliases_keep_the_sql_fields() -> None:
    e = {"gold_sql": "g", "base_sql": "b", "student_sql": "s", "teacher_sql": "t", "base_ok": True}
    out = with_answer_fields(e)
    assert out["gold_answer"] == "g" and out["teacher_answer"] == "t"
    assert out["gold_sql"] == "g" and "base_ok_answer" not in out
    other = with_answer_fields({"gold_calls": "[]", "student_calls": "[1]"})
    assert other["gold_answer"] == "[]" and other["student_answer"] == "[1]"


def test_autonomy_records_the_pack_and_defaults_old_files_to_sql(tmp_path: Path) -> None:
    autonomy.write(tmp_path, Job("r1", "tiny", True, pack="sql"))
    data = autonomy.read(tmp_path)
    assert data is not None and data["pack"] == "sql"


def test_cli_rejects_an_unknown_pack() -> None:
    from distillery.cli import _parser

    assert _parser().parse_args(["run", "--pack", "sql"]).pack == "sql"
    with pytest.raises(SystemExit):
        _parser().parse_args(["run", "--pack", "nope"])
