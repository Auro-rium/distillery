# ruff: noqa: S101
"""The human held-out set inside the pipeline: sealed in `split`, scored as Gate B, reported apart."""

from __future__ import annotations

import ast
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from distillery.humanset import CONFIRMED_FORMAT, HumanSetError
from distillery.orchestrator import Pipeline, Scale
from distillery.pipeline_fakes import build_dry_run
from distillery.sandbox_executor import AsyncBridge
from distillery.store import Store, sha256_hex
from distillery.taskpacks.sql import schema as sql_schema

NANO = Scale(name="nano", train=16, dev=8, heldout=12, stress=6)
SRC = Path(__file__).resolve().parents[1] / "src" / "distillery"
DB_SHA = sha256_hex(sql_schema.build_database(0))
QUESTIONS = [
    ("[H] accounts total?", "SELECT COUNT(*) FROM accounts"),
    ("[H] users total?", "SELECT COUNT(*) FROM users"),
    ("[H] plans total?", "SELECT COUNT(*) FROM plans"),
    ("[H] invoices total?", "SELECT COUNT(*) FROM invoices"),
    ("[H] payments total?", "SELECT COUNT(*) FROM payments"),
    ("[H] tickets total?", "SELECT COUNT(*) FROM support_tickets"),
]


def confirmed_file(
    path: Path, pairs: list[tuple[str, str]] = QUESTIONS, db_sha: str = DB_SHA
) -> Path:
    items = [
        {
            "task_id": f"h-{i}-{i:08x}",
            "family": "human",
            "source": "human",
            "heldout_class": "human",
            "question": q,
            "gold_sql": sql,
            "requires_order": False,
        }
        for i, (q, sql) in enumerate(pairs)
    ]
    path.write_text(
        json.dumps(
            {
                "format": CONFIRMED_FORMAT,
                "db_sha256": db_sha,
                "question_file_sha256": "f" * 64,
                "teacher_model": "fake-teacher",
                "counts": {
                    "questions": 10,
                    "duplicates_dropped": 1,
                    "kept": 8,
                    "discarded": 2,
                    "discarded_by_reason": {
                        "llm_error": 0,
                        "no_sql": 0,
                        "exec_error": 0,
                        "empty": 1,
                        "disagree": 1,
                    },
                    "confirmed": len(items),
                    "rejected": 2,
                    "skipped": 0,
                    "undecided": 0,
                },  # fmt: skip
                "items": items,
            }
        )
    )
    return path


def run_pipeline(tmp: Path, bridge: AsyncBridge, human: Path | None, run_id: str = "dry-h"):  # type: ignore[no-untyped-def]
    dr = build_dry_run(NANO, bridge, human_set_path=human)
    store = Store(tmp / "store")
    pipe = Pipeline(dr.pipeline_cfg, dr.config, dr.deps, store, run_id, say=lambda _s: None)
    return pipe, dr, store


@pytest.fixture(scope="module")
def human_run(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    tmp = tmp_path_factory.mktemp("human")
    f = confirmed_file(tmp / "human_confirmed.json")
    with AsyncBridge() as bridge:
        pipe, dr, store = run_pipeline(tmp, bridge, f)
        report = pipe.run()
        yield {"pipe": pipe, "dr": dr, "store": store, "report": report, "file": f, "tmp": tmp}
        store.close()


def test_report_has_decision_human_and_human_sections(human_run: dict[str, Any]) -> None:
    rep, store = human_run["report"], human_run["store"]
    human = rep["evaluation"]["human"]
    assert human["n"] == len(QUESTIONS) and human["gate"]["n"] == len(QUESTIONS)
    assert rep["decision_human"] == human["gate"]["decision"] in {"PROMOTE", "REJECT"}
    assert rep["decision"] == rep["evaluation"]["gate"]["decision"]  # Gate A untouched
    assert human["gate"]["thresholds"] == rep["evaluation"]["gate"]["thresholds"]
    d = rep["data"]["human"]
    assert d["n"] == len(QUESTIONS) and d["dropped_exact_overlap"]["total"] == 0
    assert d["counts"]["rejected"] == 2 and d["counts"]["discarded_by_reason"]["empty"] == 1
    assert d["sha256"] == human["sha256"] == store.seal_human("dry-h", store.load_human("dry-h"))
    assert "selection bias" in d["note"].lower()
    assert [i["question"] for i in store.load_human("dry-h")] == [q for q, _ in QUESTIONS]
    manifest = json.loads((human_run["pipe"].run_dir / "manifest.json").read_text())
    assert manifest["human_sha256"] == d["sha256"]
    # Gate A and the stress set are still scored on their own sets
    assert (
        rep["evaluation"]["n"] == NANO.heldout and rep["evaluation"]["stress"]["n"] == NANO.stress
    )


def test_report_does_not_leak_the_human_file_path_or_questions(human_run: dict[str, Any]) -> None:
    rep = human_run["report"]
    text = json.dumps({k: v for k, v in rep.items() if k != "evaluation"})
    assert str(human_run["tmp"]) not in text and "human_set_path" not in text
    assert not any(q in text for q, _ in QUESTIONS)  # only the evaluation block carries examples


def test_no_human_set_means_null_human_fields(tmp_path: Path) -> None:
    with AsyncBridge() as bridge:
        pipe, _dr, store = run_pipeline(tmp_path, bridge, None, "dry-nh")
        rep = pipe.run()
        store.close()
    assert rep["decision_human"] is None and rep["data"]["human"] is None
    assert rep["evaluation"]["human"] is None


def test_overlap_with_train_is_dropped_and_counted(
    human_run: dict[str, Any], tmp_path: Path
) -> None:
    split = human_run["pipe"]._results["split"]
    train_q = split["train"][0]["question"]
    pairs = [(train_q, "SELECT COUNT(*) FROM accounts"), *QUESTIONS]
    f = confirmed_file(tmp_path / "h2.json", pairs)
    with AsyncBridge() as bridge:
        pipe, _dr, store = run_pipeline(tmp_path, bridge, f, "dry-ov")
        rep = pipe.run()
        store.close()
    d = rep["data"]["human"]
    assert d["dropped_exact_overlap"] == {"train": 1, "dev": 0, "gate": 0, "total": 1}
    assert d["n"] == len(QUESTIONS)


def test_wrong_database_refuses_before_any_seal(tmp_path: Path) -> None:
    f = confirmed_file(tmp_path / "h.json", db_sha="0" * 64)
    with AsyncBridge() as bridge:
        pipe, _dr, store = run_pipeline(tmp_path, bridge, f, "dry-bad")
        with pytest.raises(HumanSetError, match="database"):
            pipe.run()
        assert store.load_human("dry-bad") is None
        store.close()


def test_split_cache_key_follows_the_file_content_not_just_its_path(tmp_path: Path) -> None:
    a = confirmed_file(tmp_path / "h.json")
    with AsyncBridge() as bridge:
        pipe, _dr, store = run_pipeline(tmp_path / "a", bridge, a, "dry-k1")
        pipe.run()
        first = pipe.hashes["split"]
        store.close()
        confirmed_file(a, QUESTIONS[:-1])
        pipe2, _dr2, store2 = run_pipeline(tmp_path / "b", bridge, a, "dry-k1")
        pipe2.run()
        assert pipe2.hashes["split"] != first
        store2.close()


def test_synthetic_fixture_matches_the_database_and_executes() -> None:
    """tests/fixtures/human_confirmed_synthetic.json feeds the committed sample report; if the
    synthetic database ever changes its sha, regenerate the fixture and the sample."""
    from distillery.humanset import read_confirmed
    from distillery.taskpacks.sql.executor import LocalExecutor

    path = Path(__file__).parent / "fixtures" / "human_confirmed_synthetic.json"
    conf = read_confirmed(path)
    assert conf.db_sha256 == DB_SHA
    assert all(i["question"].startswith("[synthetic]") for i in conf.items)
    outs = LocalExecutor({"db": sql_schema.build_database(0)}).run_batch(
        "db", [str(i["gold_sql"]) for i in conf.items]
    )
    assert all(o.ok and o.rows for o in outs)


def test_orchestrator_never_reads_the_confirmed_file() -> None:
    forbidden = {"read_confirmed", "load_human", "seal_human", "load_heldout", "load_stress"}
    tree = ast.parse((SRC / "orchestrator.py").read_text())
    names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {
        n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
    }
    assert not names & forbidden
    # nothing in the orchestrator opens or reads the human set path itself
    text = (SRC / "orchestrator.py").read_text()
    assert "human_set_path.read" not in text and "open(self.cfg.human_set_path" not in text
