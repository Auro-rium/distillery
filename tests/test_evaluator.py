# ruff: noqa: S101
from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pytest

from distillery.config import GateThresholds
from distillery.evaluator import (
    ArtifactMismatchError,
    ExpectedArtifact,
    evaluate,
    verify_artifact,
)
from distillery.finetune import DownloadedFile, TrainedArtifact, sha256_file
from distillery.store import Store
from distillery.taskpacks.sql import questions as q
from distillery.taskpacks.sql import schema as s
from distillery.taskpacks.sql.executor import LocalExecutor

SRC = Path(__file__).resolve().parents[1] / "src" / "distillery"
N = 40


class Lookup:
    """Generator that answers each prompt via a question->sql map (optionally degraded)."""

    def __init__(self, answers: dict[str, str], wrong_every: int = 0) -> None:
        self.answers = answers
        self.wrong_every = wrong_every
        self.calls = 0

    def generate(self, messages_batch: list[list[dict[str, str]]]) -> list[str]:
        out: list[str] = []
        for i, msgs in enumerate(messages_batch):
            self.calls += 1
            user = msgs[-1]["content"]
            sql = next(v for k, v in self.answers.items() if k in user)
            if self.wrong_every and i % self.wrong_every == 0:
                sql = "SELECT 1"
            out.append(f"```sql\n{sql}\n```")
        return out


class Garbage:
    def generate(self, messages_batch: list[list[dict[str, str]]]) -> list[str]:
        return ["I cannot help with that." for _ in messages_batch]


@pytest.fixture
def env(tmp_path: Path):  # type: ignore[no-untyped-def]
    db = tmp_path / "db.sqlite"
    s.write_database(0, str(db))
    rep = q.generate_tasks(str(db), N, 0)
    tasks = rep.tasks[:N]
    items = [
        {
            "task_id": t.task_id,
            "family": t.family,
            "question": t.question,
            "gold_sql": t.gold_sql,
            "requires_order": t.requires_order,
            "heldout_class": "unseen_family" if i % 4 == 0 else "seen",
        }
        for i, t in enumerate(tasks)
    ]
    store = Store(tmp_path / "root")
    store.seal_heldout("r1", items)
    answers = {it["question"]: it["gold_sql"] for it in items}
    return store, str(db), s.schema_ddl(0), items, answers


def _gens(answers: dict[str, str]) -> dict[str, Lookup]:
    return {
        "base": Lookup(answers, wrong_every=2),  # ~50% right
        "student": Lookup(answers, wrong_every=10),  # ~90% right
        "teacher": Lookup(answers),  # 100% right
    }


def test_end_to_end_report(env) -> None:  # type: ignore[no-untyped-def]
    store, db, ddl, items, answers = env
    rep = evaluate(
        store,
        "r1",
        _gens(answers),
        LocalExecutor(),
        db_ref=db,
        schema_ddl=ddl,
        gate_cfg=GateThresholds(),
    )
    assert rep.n == N
    assert rep.scores["teacher"].accuracy == 1.0
    assert rep.scores["base"].accuracy < rep.scores["student"].accuracy < 1.0
    assert set(rep.accuracy_by_class) == {"seen", "unseen_family"}
    assert sum(rep.class_counts.values()) == N
    assert len(rep.heldout_sha256) == 64
    assert rep.to_json()["gate"]["decision"] in {"PROMOTE", "REJECT"}


def test_unparseable_output_counts_as_wrong_and_is_reported(env) -> None:  # type: ignore[no-untyped-def]
    store, db, ddl, _items, answers = env
    gens = {**_gens(answers), "student": Garbage()}
    rep = evaluate(
        store, "r1", gens, LocalExecutor(), db_ref=db, schema_ddl=ddl, gate_cfg=GateThresholds()
    )
    assert rep.scores["student"].accuracy == 0.0
    assert rep.scores["student"].unparseable == N
    assert rep.gate.decision == "REJECT"


def test_requires_exactly_three_roles(env) -> None:  # type: ignore[no-untyped-def]
    store, db, ddl, _items, answers = env
    gens = _gens(answers)
    del gens["base"]
    with pytest.raises(ValueError, match="exactly"):
        evaluate(
            store,
            "r1",
            gens,
            LocalExecutor(),
            db_ref=db,
            schema_ddl=ddl,
            gate_cfg=GateThresholds(),
        )


def _artifact(tmp_path: Path) -> tuple[TrainedArtifact, ExpectedArtifact]:
    f = tmp_path / "adapter_model.safetensors"
    f.write_bytes(b"weights-v1")
    art = TrainedArtifact(
        job_id="job1",
        checkpoint_id="ckpt1",
        base_model="Qwen/Qwen3-1.7B",
        fine_tuned_model_checkpoint=None,
        files=(DownloadedFile(file_id="f1", path=f, sha256=sha256_file(f)),),
    )
    return art, ExpectedArtifact("job1", "ckpt1", art.adapter_sha256)


def test_verify_artifact_ok(tmp_path: Path) -> None:
    art, exp = _artifact(tmp_path)
    verify_artifact(art, exp)


def test_verify_artifact_detects_tampered_file(tmp_path: Path) -> None:
    art, exp = _artifact(tmp_path)
    art.files[0].path.write_bytes(b"weights-v2")
    with pytest.raises(ArtifactMismatchError, match="digest"):
        verify_artifact(art, exp)


def test_verify_artifact_detects_wrong_job_or_checkpoint(tmp_path: Path) -> None:
    art, exp = _artifact(tmp_path)
    with pytest.raises(ArtifactMismatchError, match="job id"):
        verify_artifact(art, dataclasses.replace(exp, job_id="other"))
    with pytest.raises(ArtifactMismatchError, match="checkpoint id"):
        verify_artifact(art, dataclasses.replace(exp, checkpoint_id="other"))


def test_evaluate_refuses_mismatched_artifact_before_touching_heldout(
    env,
    tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    store, db, ddl, _items, answers = env
    art, exp = _artifact(tmp_path)
    art.files[0].path.write_bytes(b"tampered")
    gens = _gens(answers)
    with pytest.raises(ArtifactMismatchError):
        evaluate(
            store,
            "r1",
            gens,
            LocalExecutor(),
            db_ref=db,
            schema_ddl=ddl,
            gate_cfg=GateThresholds(),
            trained=art,
            expected=exp,
        )
    assert all(g.calls == 0 for g in gens.values())


def test_trained_and_expected_must_be_passed_together(env, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    store, db, ddl, _items, answers = env
    art, _exp = _artifact(tmp_path)
    with pytest.raises(ValueError, match="both"):
        evaluate(
            store,
            "r1",
            _gens(answers),
            LocalExecutor(),
            db_ref=db,
            schema_ddl=ddl,
            gate_cfg=GateThresholds(),
            trained=art,
        )


def test_examples_are_capped_deterministic_and_complete(env) -> None:  # type: ignore[no-untyped-def]
    store, db, ddl, items, answers = env

    def go():  # type: ignore[no-untyped-def]
        return evaluate(
            store, "r1", _gens(answers), LocalExecutor(), db_ref=db, schema_ddl=ddl,
            gate_cfg=GateThresholds(),
        ).to_json()["examples"]

    ex = go()
    assert ex == go() and 0 < len(ex) <= 60
    fields = {
        "task_id", "family", "heldout_class", "question", "gold_sql", "base_sql", "student_sql",
        "teacher_sql", "base_ok", "student_ok", "teacher_ok",
    }
    by_id = {it["task_id"]: it for it in items}
    kinds = {e["kind"] for e in ex}
    assert kinds <= {"fixed", "still_wrong", "regressed"} and "fixed" in kinds
    for e in ex:
        assert fields <= set(e)
        assert e["gold_sql"] == by_id[e["task_id"]]["gold_sql"]
        assert e["question"] == by_id[e["task_id"]]["question"]
        assert e["teacher_ok"] is True and e["student_sql"]
        if e["kind"] == "fixed":
            assert e["student_ok"] and not e["base_ok"]
        elif e["kind"] == "regressed":
            assert e["base_ok"] and not e["student_ok"]
        else:
            assert not e["student_ok"] and not e["base_ok"]
    assert not any(e["student_ok"] and e["base_ok"] for e in ex)


def test_examples_cap_is_20_per_kind(env) -> None:  # type: ignore[no-untyped-def]
    store, db, ddl, _items, answers = env
    gens = {**_gens(answers), "student": Garbage(), "base": Garbage()}
    ex = evaluate(
        store, "r1", gens, LocalExecutor(), db_ref=db, schema_ddl=ddl, gate_cfg=GateThresholds()
    ).to_json()["examples"]
    assert len(ex) == 20 and {e["kind"] for e in ex} == {"still_wrong"}
    assert ex[0]["student_sql"] == "I cannot help with that."  # raw output kept when unparseable


def test_only_evaluator_loads_heldout() -> None:
    """Integrity rule 5.1: no module except evaluator.py (and its definition) may load held-out."""
    offenders: list[str] = []
    for path in SRC.rglob("*.py"):
        if path.name in {"evaluator.py", "store.py"}:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            name = None
            if isinstance(node, ast.Attribute):
                name = node.attr
            elif isinstance(node, ast.Name):
                name = node.id
            if name == "load_heldout":
                offenders.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert offenders == []
