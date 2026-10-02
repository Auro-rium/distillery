# ruff: noqa: S101
"""Sealing and scoring of the human held-out set (Gate B)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from distillery.config import GateThresholds
from distillery.evaluator import (
    evaluate,
    human_set_fingerprint,
    seal_human_set,
)
from distillery.humanset import HumanSetError
from distillery.store import HeldoutIntegrityError, Store
from distillery.taskpacks.sql import questions as q
from distillery.taskpacks.sql import schema as s
from distillery.taskpacks.sql.executor import LocalExecutor

N = 40
NH = 12
DB_SHA = "d" * 64


class Lookup:
    def __init__(self, answers: dict[str, str], wrong_every: int = 0) -> None:
        self.answers = answers
        self.wrong_every = wrong_every

    def generate(self, messages_batch: list[list[dict[str, str]]]) -> list[str]:
        out: list[str] = []
        for i, msgs in enumerate(messages_batch):
            user = msgs[-1]["content"]
            sql = next(v for k, v in self.answers.items() if k in user)
            if self.wrong_every and i % self.wrong_every == 0:
                sql = "SELECT 1"
            out.append(f"```sql\n{sql}\n```")
        return out


def _item(t: Any, **extra: Any) -> dict[str, Any]:
    return {
        "task_id": t.task_id,
        "family": t.family,
        "question": t.question,
        "gold_sql": t.gold_sql,
        "requires_order": t.requires_order,
        "heldout_class": "seen",
        **extra,
    }


def _human(t: Any, i: int) -> dict[str, Any]:
    return {
        "task_id": f"h-{i}-{i:08x}",
        "family": "human",
        "source": "human",
        "question": f"[H] {t.question}",
        "gold_sql": t.gold_sql,
        "requires_order": t.requires_order,
    }


def _confirmed_file(path: Path, items: list[dict[str, Any]], db_sha: str = DB_SHA) -> Path:
    path.write_text(
        json.dumps(
            {
                "format": "distillery-human-confirmed-v1",
                "db_sha256": db_sha,
                "question_file_sha256": "f" * 64,
                "teacher_model": "fake-teacher",
                "counts": {
                    "questions": len(items) + 5,
                    "duplicates_dropped": 0,
                    "kept": len(items) + 2,
                    "discarded": 3,
                    "discarded_by_reason": {"disagree": 2, "empty": 1},
                    "confirmed": len(items),
                    "rejected": 2,
                    "skipped": 0,
                    "undecided": 0,
                },
                "items": items,
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def env(tmp_path: Path):  # type: ignore[no-untyped-def]
    db = tmp_path / "db.sqlite"
    s.write_database(0, str(db))
    tasks = q.generate_tasks(str(db), N + NH + 10, 0).tasks
    gate = [_item(t) for t in tasks[:N]]
    human_tasks = tasks[N : N + NH]
    human = [_human(t, i) for i, t in enumerate(human_tasks)]
    store = Store(tmp_path / "root")
    store.seal_heldout("r1", gate)
    train_q = [t.question for t in tasks[N + NH :]]
    answers = {str(it["question"]): str(it["gold_sql"]) for it in [*gate, *human]}
    return store, str(db), s.schema_ddl(0), gate, human, tasks, train_q, answers, tmp_path


def _gens(answers: dict[str, str]) -> dict[str, Lookup]:
    return {
        "base": Lookup(answers, wrong_every=2),
        "student": Lookup(answers, wrong_every=5),
        "teacher": Lookup(answers),
    }


def _eval(
    store: Store, db: str, ddl: str, answers: dict[str, str], gate: GateThresholds | None = None
):  # type: ignore[no-untyped-def]
    return evaluate(
        store, "r1", _gens(answers), LocalExecutor(), db_ref=db, schema_ddl=ddl,
        gate_cfg=gate or GateThresholds(),
    )  # fmt: skip


def _seal(store: Store, human: list[dict[str, Any]], tmp: Path, **kw: Any) -> dict[str, Any]:
    f = _confirmed_file(tmp / "human_confirmed.json", human)
    return seal_human_set(
        store, "r1", f, DB_SHA,
        train_questions=kw.get("train", []), dev_questions=kw.get("dev", []),
    )  # fmt: skip


def test_seal_returns_only_hash_and_counts(env) -> None:  # type: ignore[no-untyped-def]
    store, _db, _ddl, _gate, human, _t, train_q, _a, tmp = env
    out = _seal(store, human, tmp, train=train_q)
    assert out["n"] == NH and len(out["sha256"]) == 64
    assert out["dropped_exact_overlap"] == {"train": 0, "dev": 0, "gate": 0, "total": 0}
    assert out["counts"]["rejected"] == 2 and out["counts"]["discarded_by_reason"]["disagree"] == 2
    blob = json.dumps(out)
    assert not any(it["question"] in blob or it["gold_sql"] in blob for it in human)
    assert store.load_human("r1") == human  # sealed content is the file's items


def test_seal_drops_exact_matches_with_train_dev_and_gate_and_counts_them(env) -> None:  # type: ignore[no-untyped-def]
    store, _db, _ddl, gate, human, _t, _tq, _a, tmp = env
    human = [dict(h) for h in human]
    human[0]["question"] = (
        "  " + str(gate[3]["question"]).replace(" ", "  ") + " "
    )  # whitespace only
    human[1]["question"] = "train overlap question?"
    human[2]["question"] = "dev overlap question?"
    out = _seal(store, human, tmp, train=["train overlap question?"], dev=["dev overlap question?"])
    assert out["dropped_exact_overlap"] == {"train": 1, "dev": 1, "gate": 1, "total": 3}
    assert out["n"] == NH - 3
    kept = {h["task_id"] for h in store.load_human("r1") or []}
    assert human[0]["task_id"] not in kept and human[3]["task_id"] in kept


def test_seal_counts_skeleton_overlap_with_train(env) -> None:  # type: ignore[no-untyped-def]
    store, _db, _ddl, _g, human, tasks, _tq, _a, tmp = env
    human = [dict(h) for h in human]
    human[0]["question"] = tasks[N + NH].question.replace("1", "2")  # same skeleton, near-duplicate
    out = _seal(store, human, tmp, train=[tasks[N + NH].question])
    assert out["skeleton_in_train"] >= 0 and isinstance(out["skeleton_in_train"], int)


def test_seal_refuses_wrong_db_empty_result_and_malformed_files(env) -> None:  # type: ignore[no-untyped-def]
    store, _db, _ddl, _g, human, _t, _tq, _a, tmp = env
    f = _confirmed_file(tmp / "c.json", human, db_sha="0" * 64)
    with pytest.raises(HumanSetError, match="database"):
        seal_human_set(store, "r1", f, DB_SHA, train_questions=[], dev_questions=[])
    assert store.load_human("r1") is None  # nothing sealed on refusal
    f2 = _confirmed_file(tmp / "c2.json", [])
    with pytest.raises(HumanSetError, match="no confirmed"):
        seal_human_set(store, "r1", f2, DB_SHA, train_questions=[], dev_questions=[])
    bad = tmp / "bad.json"
    bad.write_text('{"format": "nope"}')
    with pytest.raises(HumanSetError):
        seal_human_set(store, "r1", bad, DB_SHA, train_questions=[], dev_questions=[])
    everything = [h["question"] for h in human]
    with pytest.raises(HumanSetError, match="every"):
        seal_human_set(
            store, "r1", _confirmed_file(tmp / "c3.json", human), DB_SHA,
            train_questions=everything, dev_questions=[],
        )  # fmt: skip


def test_seal_is_immutable_and_tamper_is_detected_at_evaluation(env) -> None:  # type: ignore[no-untyped-def]
    store, db, ddl, _g, human, _t, _tq, answers, tmp = env
    out = _seal(store, human, tmp)
    assert _seal(store, human, tmp)["sha256"] == out["sha256"]
    with pytest.raises(HeldoutIntegrityError):
        _seal(store, human[:-1], tmp)
    store._human_path("r1").write_bytes(b"[]")
    with pytest.raises(HeldoutIntegrityError):
        _eval(store, db, ddl, answers)


def test_fingerprint_changes_with_the_file(env) -> None:  # type: ignore[no-untyped-def]
    _s, _db, _ddl, _g, human, _t, _tq, _a, tmp = env
    f = _confirmed_file(tmp / "c.json", human)
    a = human_set_fingerprint(f)
    _confirmed_file(tmp / "c.json", human[:-1])
    assert human_set_fingerprint(f) != a


def test_no_human_seal_means_no_human_section(env) -> None:  # type: ignore[no-untyped-def]
    store, db, ddl, *_rest, answers, _tmp = env
    rep = _eval(store, db, ddl, answers)
    assert rep.human is None and rep.to_json()["human"] is None


def test_gate_b_uses_identical_thresholds_and_does_not_move_gate_a(env) -> None:  # type: ignore[no-untyped-def]
    store, db, ddl, _g, human, _t, _tq, answers, tmp = env
    thresholds = GateThresholds(ratio_lower_bound_min=0.7, mcnemar_alpha=0.1, seed=99)
    before = _eval(store, db, ddl, answers, thresholds)
    _seal(store, human, tmp)
    after = _eval(store, db, ddl, answers, thresholds)
    assert before.gate == after.gate  # Gate A unchanged by the human seal
    assert {m: s.correct for m, s in before.scores.items()} == {
        m: s.correct for m, s in after.scores.items()
    }
    assert after.human is not None
    gate_b = after.human["gate"]
    assert gate_b["thresholds"] == after.gate.model_dump(mode="json")["thresholds"]
    assert gate_b["n"] == after.human["n"] == NH
    assert after.human["accuracy"]["teacher"] == 1.0
    assert after.human["gate"]["decision"] in {"PROMOTE", "REJECT"}
    assert "selection" in after.human["note"].lower()


def test_gate_b_scores_only_human_items_and_stress_stays_out_of_both_gates(env) -> None:  # type: ignore[no-untyped-def]
    store, db, ddl, gate, human, tasks, _tq, answers, tmp = env
    stress = [
        {**_item(t), "task_id": f"s{i}", "family": "stressfam"}
        for i, t in enumerate(tasks[N + NH : N + NH + 5])
    ]
    stress_answers = {str(x["question"]): str(x["gold_sql"]) for x in stress}
    _seal(store, human, tmp)
    base = _eval(store, db, ddl, {**answers, **stress_answers})
    store.seal_stress("r1", stress)
    with_stress = _eval(store, db, ddl, {**answers, **stress_answers})
    assert base.human == with_stress.human  # stress cannot move Gate B
    assert base.gate == with_stress.gate  # nor Gate A
    assert with_stress.stress is not None and with_stress.stress["n"] == 5
    assert base.n == len(gate)
    js = with_stress.to_json()["human"]
    assert js["n"] == NH and js["sha256"] != js.get("none")
    assert {e["task_id"] for e in js["examples"]} <= {h["task_id"] for h in human}
