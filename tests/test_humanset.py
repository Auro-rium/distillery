# ruff: noqa: S101
"""Drafting, confirming and finalizing the human held-out set (offline, scripted fakes only)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from distillery import humanset as hs
from distillery.pipeline_fakes import question_of
from distillery.prompts import build_messages
from distillery.taskpacks.sql import schema as s
from distillery.taskpacks.sql.executor import LocalExecutor
from distillery.taskpacks.sql.human import QuestionFile, human_task_id

COUNT = "SELECT COUNT(*) FROM accounts"
FILE_SHA = "ab12cd34" + "0" * 56
DB_SHA = "d" * 64


class ScriptedRunner:
    """Duck-typed LLMRunner: answers each prompt from a per-question list of 3 scripted replies.
    ``None`` = the call failed (the real runner returns None for an LLMError)."""

    def __init__(self, script: dict[str, list[str | None]]) -> None:
        self.script = script
        self.seen: dict[str, int] = {}
        self.calls: list[dict[str, Any]] = []

    def map(self, role: str, batch: Any, *, purpose: str, stage: str, schema: Any = None,
            temperature: float | None = 0.0) -> list[Any]:  # fmt: skip
        out: list[Any] = []
        for msgs in batch:
            question = question_of(msgs)
            assert question is not None
            i = self.seen.get(question, 0)
            self.seen[question] = i + 1
            self.calls.append(
                {"role": role, "purpose": purpose, "stage": stage, "temperature": temperature,
                 "messages": [dict(m) for m in msgs], "question": question}
            )  # fmt: skip
            reply = self.script[question][i]
            out.append(None if reply is None else SimpleNamespace(text=reply))
        return out


def _fence(sql: str) -> str:
    return f"```sql\n{sql}\n```"


def _qf(*questions: str) -> QuestionFile:
    from distillery.taskpacks.sql.human import HumanQuestion

    return QuestionFile(
        tuple(HumanQuestion(human_task_id(i, q), q) for i, q in enumerate(questions)),
        FILE_SHA,
        0,
    )


@pytest.fixture
def db(tmp_path: Path) -> tuple[str, str]:
    p = tmp_path / "db.sqlite"
    s.write_database(0, str(p))
    return str(p), s.schema_ddl(0)


def _draft(qf: QuestionFile, runner: ScriptedRunner, db: tuple[str, str]) -> dict[str, Any]:
    return hs.draft_questions(
        qf, runner=runner, executor=LocalExecutor(), db_ref=db[0], schema_ddl=db[1],
        db_sha256=DB_SHA, teacher_model="fake-teacher",
    )  # fmt: skip


def test_all_paths_and_discard_counts(db: tuple[str, str]) -> None:
    script: dict[str, list[str | None]] = {
        "agree?": [_fence(COUNT)] * 3,
        "agree differently?": [
            _fence(COUNT),
            _fence("SELECT COUNT(account_id) FROM accounts"),
            _fence("SELECT COUNT(1) FROM accounts"),
        ],
        "llm error?": [_fence(COUNT), None, _fence(COUNT)],
        "no sql?": [_fence(COUNT), "I cannot answer that.", _fence(COUNT)],
        "exec error?": [_fence(COUNT), _fence("SELECT * FROM no_such_table"), _fence(COUNT)],
        "empty?": [_fence("SELECT plan_id FROM plans WHERE 1=0")] * 3,
        "disagree?": [_fence(COUNT), _fence(COUNT), _fence("SELECT COUNT(*) FROM users")],
    }
    out = _draft(_qf(*script), ScriptedRunner(script), db)
    assert [k["question"] for k in out["kept"]] == ["agree?", "agree differently?"]
    reasons = {d["question"]: d["reason"] for d in out["discarded"]}
    assert reasons == {
        "llm error?": "llm_error",
        "no sql?": "no_sql",
        "exec error?": "exec_error",
        "empty?": "empty",
        "disagree?": "disagree",
    }
    assert out["discarded_by_reason"] == dict.fromkeys(hs.DISCARD_REASONS, 1)
    assert out["format"] == hs.DRAFTS_FORMAT and out["k"] == 3 and out["temperature"] == 0.8
    assert out["db_sha256"] == DB_SHA and out["question_file_sha256"] == FILE_SHA
    assert out["teacher_model"] == "fake-teacher" and out["n_questions"] == 7
    kept = out["kept"][0]
    assert kept["gold_sql"] == COUNT and len(kept["candidates"]) == 3
    assert (
        kept["preview"]["columns"]
        and kept["preview"]["rows"]
        and kept["task_id"].startswith("h-0-")
    )
    json.dumps(out)  # JSON-safe


def test_llm_prompts_are_k3_at_temperature_08_through_the_shared_builder(
    db: tuple[str, str],
) -> None:
    script: dict[str, list[str | None]] = {"a?": [_fence(COUNT)] * 3, "b?": [_fence(COUNT)] * 3}
    runner = ScriptedRunner(script)
    _draft(_qf("a?", "b?"), runner, db)
    assert len(runner.calls) == 6
    assert {c["temperature"] for c in runner.calls} == {0.8}
    assert {c["role"] for c in runner.calls} == {"teacher"}
    assert {c["purpose"] for c in runner.calls} == {hs.DRAFT_PURPOSE}
    for c in runner.calls:
        assert c["messages"] == build_messages(c["question"], db[1], role="train")


def test_requires_order_only_when_all_three_have_a_top_level_order_by(db: tuple[str, str]) -> None:
    ordered = "SELECT name FROM plans ORDER BY name"
    script: dict[str, list[str | None]] = {
        "all ordered?": [_fence(ordered)] * 3,
        "one unordered?": [_fence(ordered), _fence(ordered), _fence("SELECT name FROM plans")],
        "subquery order only?": [_fence("SELECT name FROM (SELECT name FROM plans ORDER BY name)")]
        * 3,
    }
    out = _draft(_qf(*script), ScriptedRunner(script), db)
    by_q = {k["question"]: k["requires_order"] for k in out["kept"]}
    assert by_q == {
        "all ordered?": True,
        "one unordered?": False,
        "subquery order only?": False,
    }


def test_ordered_results_must_agree_in_order(db: tuple[str, str]) -> None:
    script: dict[str, list[str | None]] = {
        "order?": [
            _fence("SELECT name FROM plans ORDER BY name ASC"),
            _fence("SELECT name FROM plans ORDER BY name ASC"),
            _fence("SELECT name FROM plans ORDER BY name DESC"),
        ]
    }
    out = _draft(_qf("order?"), ScriptedRunner(script), db)
    assert out["kept"] == [] and out["discarded"][0]["reason"] == "disagree"


def test_drafts_file_lands_under_humanset_sha8(tmp_path: Path, db: tuple[str, str]) -> None:
    out = _draft(_qf("a?"), ScriptedRunner({"a?": [_fence(COUNT)] * 3}), db)
    path = hs.write_drafts(tmp_path / "home", out)
    assert path == tmp_path / "home" / "humanset" / "ab12cd34" / "drafts.json"
    assert hs.read_drafts(tmp_path / "home", "ab12cd34") == out
    with pytest.raises(hs.HumanSetError):
        hs.read_drafts(tmp_path / "home", "../../etc")  # set ids are 8 hex chars, nothing else
    with pytest.raises(hs.HumanSetError):
        hs.read_drafts(tmp_path / "home", "00000000")  # unknown set


# ---- confirm + finalize -------------------------------------------------------------------


def _drafts(n: int = 4, discarded: int = 2) -> dict[str, Any]:
    kept = [
        {
            "task_id": f"h-{i}-0000000{i}",
            "question": f"Question {i}?",
            "gold_sql": COUNT,
            "requires_order": False,
            "candidates": [COUNT] * 3,
            "preview": {"columns": ["n"], "rows": [[3]], "row_count": 1},
        }
        for i in range(n)
    ]
    return {
        "format": hs.DRAFTS_FORMAT,
        "question_file_sha256": FILE_SHA,
        "db_sha256": DB_SHA,
        "teacher_model": "fake-teacher",
        "temperature": 0.8,
        "k": 3,
        "n_questions": n + discarded,
        "duplicates_dropped": 1,
        "kept": kept,
        "discarded": [
            {"task_id": f"h-{n + j}-x", "question": "x", "reason": "disagree", "candidates": []}
            for j in range(discarded)
        ],
        "discarded_by_reason": {**dict.fromkeys(hs.DISCARD_REASONS, 0), "disagree": discarded},
    }


def _script(answers: list[str]) -> Any:
    it = iter(answers)
    return lambda _prompt: next(it)


def test_confirm_shows_question_sql_and_table_and_records_decisions(tmp_path: Path) -> None:
    root = tmp_path / "home"
    hs.write_drafts(root, _drafts())
    shown: list[str] = []
    summary = hs.confirm_drafts(
        root, "ab12cd34", ask=_script(["y", "n", "s", "y"]), out=shown.append
    )
    text = "\n".join(shown)
    assert "Question 0?" in text and COUNT in text and "n" in text and "3" in text
    assert summary == {"confirmed": 2, "rejected": 1, "skipped": 1, "undecided": 0, "quit": False}
    lines = (root / "humanset" / "ab12cd34" / "decisions.jsonl").read_text().splitlines()
    assert [json.loads(x)["decision"] for x in lines] == ["confirm", "reject", "skip", "confirm"]
    assert [json.loads(x)["task_id"] for x in lines][0] == "h-0-00000000"


def test_confirm_is_resumable_and_skipped_items_come_back(tmp_path: Path) -> None:
    root = tmp_path / "home"
    hs.write_drafts(root, _drafts())
    first = hs.confirm_drafts(root, "ab12cd34", ask=_script(["y", "s", "q"]), out=lambda _m: None)
    assert first == {"confirmed": 1, "rejected": 0, "skipped": 1, "undecided": 2, "quit": True}
    shown: list[str] = []
    second = hs.confirm_drafts(root, "ab12cd34", ask=_script(["n", "y", "y"]), out=shown.append)
    text = "\n".join(shown)
    assert "Question 0?" not in text  # already confirmed: not asked again
    assert "Question 1?" in text  # the skipped one is offered again
    assert second == {"confirmed": 3, "rejected": 1, "skipped": 0, "undecided": 0, "quit": False}
    # last decision per id wins
    assert hs.read_decisions(root, "ab12cd34")["h-1-00000001"] == "reject"


def test_confirm_reprompts_on_invalid_key_and_quits_on_eof(tmp_path: Path) -> None:
    root = tmp_path / "home"
    hs.write_drafts(root, _drafts(2))

    def ask(_p: str) -> str:
        if not answers:
            raise EOFError
        return answers.pop(0)

    answers = ["maybe", "y"]
    summary = hs.confirm_drafts(root, "ab12cd34", ask=ask, out=lambda _m: None)
    assert summary["confirmed"] == 1 and summary["quit"] is True and summary["undecided"] == 1


def test_finalize_writes_only_confirmed_items_with_counts(tmp_path: Path) -> None:
    root = tmp_path / "home"
    hs.write_drafts(root, _drafts())
    hs.confirm_drafts(root, "ab12cd34", ask=_script(["y", "n", "s", "q"]), out=lambda _m: None)
    path = hs.finalize(root, "ab12cd34")
    assert path == root / "humanset" / "ab12cd34" / "human_confirmed.json"
    doc = json.loads(path.read_text())
    assert doc["format"] == hs.CONFIRMED_FORMAT and doc["db_sha256"] == DB_SHA
    assert doc["question_file_sha256"] == FILE_SHA and doc["teacher_model"] == "fake-teacher"
    assert [i["question"] for i in doc["items"]] == ["Question 0?"]
    item = doc["items"][0]
    assert item["family"] == "human" and item["source"] == "human"
    assert item["gold_sql"] == COUNT and item["requires_order"] is False
    assert "candidates" not in item and "preview" not in item
    assert doc["counts"] == {
        "questions": 6,
        "duplicates_dropped": 1,
        "kept": 4,
        "discarded": 2,
        "discarded_by_reason": {"llm_error": 0, "no_sql": 0, "exec_error": 0, "empty": 0,
                                "disagree": 2},
        "confirmed": 1,
        "rejected": 1,
        "skipped": 1,
        "undecided": 1,
    }  # fmt: skip
    assert hs.read_confirmed(path).items[0]["task_id"] == "h-0-00000000"


def test_finalize_refuses_when_nothing_was_confirmed(tmp_path: Path) -> None:
    root = tmp_path / "home"
    hs.write_drafts(root, _drafts(2))
    hs.confirm_drafts(root, "ab12cd34", ask=_script(["n", "n"]), out=lambda _m: None)
    with pytest.raises(hs.HumanSetError, match="no confirmed"):
        hs.finalize(root, "ab12cd34")


def test_decide_rejects_unknown_ids_and_decisions(tmp_path: Path) -> None:
    root = tmp_path / "home"
    hs.write_drafts(root, _drafts(2))
    hs.decide(root, "ab12cd34", "h-0-00000000", "confirm")
    assert hs.read_decisions(root, "ab12cd34") == {"h-0-00000000": "confirm"}
    with pytest.raises(hs.HumanSetError):
        hs.decide(root, "ab12cd34", "h-9-nope", "confirm")
    with pytest.raises(hs.HumanSetError):
        hs.decide(root, "ab12cd34", "h-0-00000000", "maybe")


def test_confirmed_file_never_contains_drafts_internals(tmp_path: Path) -> None:
    root = tmp_path / "home"
    hs.write_drafts(root, _drafts(1))
    hs.confirm_drafts(root, "ab12cd34", ask=_script(["y"]), out=lambda _m: None)
    text = hs.finalize(root, "ab12cd34").read_text()
    assert "preview" not in text and "candidates" not in text
