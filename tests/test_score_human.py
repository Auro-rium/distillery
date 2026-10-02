# ruff: noqa: S101, S105
"""``distillery score-human``: Gate B for an already finished run, offline with the dry-run fakes."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from distillery import humanscore
from distillery.cli import EXIT_OK, EXIT_REFUSED, main
from distillery.humanscore import score_human_run
from distillery.humanset import CONFIRMED_FORMAT
from distillery.orchestrator import SCALES, Scale
from distillery.pipeline_fakes import build_dry_run
from distillery.sandbox_executor import AsyncBridge
from distillery.store import Store, sha256_hex
from distillery.taskpacks.sql import schema as sql_schema

SRC = Path(__file__).resolve().parents[1] / "src" / "distillery"
DB_SHA = sha256_hex(sql_schema.build_database(0))
RUN = "dry-sh"
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
    counts = {"questions": 10, "kept": 8, "discarded": 2, "confirmed": len(items), "rejected": 2}
    path.write_text(
        json.dumps(
            {
                "format": CONFIRMED_FORMAT,
                "db_sha256": db_sha,
                "question_file_sha256": "f" * 64,
                "teacher_model": "fake-teacher",
                "counts": counts,
                "items": items,
            }
        )
    )
    return path


def cli(tmp_path: Path, *args: str) -> tuple[int, list[str]]:
    lines: list[str] = []
    code = main(["--root", str(tmp_path / "home"), *args], out=lines.append)
    return code, lines


@pytest.fixture
def finished(tmp_path: Path) -> Path:
    """A finished dry run WITHOUT a human set (tiny scale). Returns the tmp dir."""
    code, lines = cli(tmp_path, "run", "--dry-run", "--scale", "tiny", "--run-id", RUN)
    assert code == EXIT_OK, lines
    return tmp_path


def run_dir(tmp_path: Path) -> Path:
    return tmp_path / "home" / "dry-runs" / "runs" / RUN


def report_of(tmp_path: Path) -> dict[str, Any]:
    return json.loads((run_dir(tmp_path) / "report.json").read_text())


def test_end_to_end_adds_gate_b_and_leaves_everything_else(finished: Path) -> None:
    before = report_of(finished)
    assert before["decision_human"] is None and before["evaluation"]["human"] is None
    f = confirmed_file(finished / "human_confirmed.json")
    code, lines = cli(finished, "score-human", RUN, "--human-set", str(f))
    assert code == EXIT_OK, lines
    after = report_of(finished)
    hr = json.loads((run_dir(finished) / "report_human.json").read_text())
    block = after["evaluation"]["human"]
    assert block == hr["gate_b"]
    assert after["decision_human"] == hr["decision_human"] == block["gate"]["decision"]
    assert block["n"] == len(QUESTIONS) and set(block["accuracy"]) == {"base", "student", "teacher"}
    assert block["file_sha256"] == sha256_hex(f.read_bytes())
    assert block["db_sha256"] == DB_SHA
    assert (
        block["adapter_sha256"] == before["rounds"][before["candidate_round"] - 1]["adapter_sha256"]
    )
    assert block["dropped_exact_overlap"]["total"] == 0
    assert block["counts"]["rejected"] == 2 and block["counts"]["discarded"] == 2
    assert block["scored_at"]
    note = "human set sealed AFTER training and after the Gate A decision"
    assert note in block["sealed_after_training_note"] and "not used for training" in hr["note"]
    # same thresholds as Gate A, Gate A and every other key untouched
    assert block["gate"]["thresholds"] == before["evaluation"]["gate"]["thresholds"]
    assert after["decision"] == before["decision"]
    changed = {k for k in after if after[k] != before[k]}
    assert changed == {"evaluation", "decision_human"}
    assert {
        k for k in after["evaluation"] if after["evaluation"][k] != before["evaluation"][k]
    } == {"human"}
    assert any("Gate B" in x for x in lines)
    # status/report commands still work on the amended report
    assert cli(finished, "report", RUN)[0] == EXIT_OK


def test_gate_b_matches_run_with_human_set_from_the_start(tmp_path: Path) -> None:
    """Scoring after the fact gives the same Gate B as sealing it in `split` (same fakes/seeds)."""
    f = confirmed_file(tmp_path / "h.json")
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    assert cli(a, "run", "--dry-run", "--scale", "tiny", "--run-id", RUN)[0] == EXIT_OK
    assert cli(a, "score-human", RUN, "--human-set", str(f))[0] == EXIT_OK
    assert (
        cli(b, "run", "--dry-run", "--scale", "tiny", "--run-id", RUN, "--human-set", str(f))[0]
        == EXIT_OK
    )
    late = report_of(a)["evaluation"]["human"]["gate"]
    early = report_of(b)["evaluation"]["human"]["gate"]
    assert late == early


def test_refuses_without_report(tmp_path: Path) -> None:
    f = confirmed_file(tmp_path / "h.json")
    code, lines = cli(tmp_path, "score-human", RUN, "--human-set", str(f))
    assert code == EXIT_REFUSED and "no report.json" in lines[-1]


def test_refuses_wrong_db_sha_and_seals_nothing(finished: Path) -> None:
    f = confirmed_file(finished / "bad.json", db_sha="0" * 64)
    before = (run_dir(finished) / "report.json").read_bytes()
    code, lines = cli(finished, "score-human", RUN, "--human-set", str(f))
    assert code == EXIT_REFUSED and "different database" in lines[-1]
    assert (run_dir(finished) / "report.json").read_bytes() == before
    assert not (run_dir(finished) / "report_human.json").exists()
    assert not (run_dir(finished) / "heldout" / "human.json").exists()


def test_refuses_when_gate_b_exists_and_never_overwrites(finished: Path) -> None:
    f = confirmed_file(finished / "h.json")
    assert cli(finished, "score-human", RUN, "--human-set", str(f))[0] == EXIT_OK
    snap = {p.name: p.read_bytes() for p in run_dir(finished).glob("report*.json")}
    other = confirmed_file(finished / "h2.json", QUESTIONS[:3])
    code, lines = cli(finished, "score-human", RUN, "--human-set", str(other))
    assert code == EXIT_REFUSED and "already has a Gate B" in lines[-1]
    assert {p.name: p.read_bytes() for p in run_dir(finished).glob("report*.json")} == snap


def test_refuses_when_report_human_file_exists(finished: Path) -> None:
    (run_dir(finished) / "report_human.json").write_text("{}")
    f = confirmed_file(finished / "h.json")
    code, lines = cli(finished, "score-human", RUN, "--human-set", str(f))
    assert code == EXIT_REFUSED and "report_human.json already exists" in lines[-1]


def test_refuses_without_final_eval_or_expected_artifact(finished: Path) -> None:
    import sqlite3

    f = confirmed_file(finished / "h.json")
    con = sqlite3.connect(finished / "home" / "dry-runs" / "index.sqlite")
    con.execute("UPDATE stages SET status='failed' WHERE run_id=? AND stage='final_eval'", (RUN,))
    con.commit()
    code, lines = cli(finished, "score-human", RUN, "--human-set", str(f))
    assert code == EXIT_REFUSED and "final_eval" in lines[-1]
    con.execute("UPDATE stages SET status='complete' WHERE run_id=? AND stage='final_eval'", (RUN,))
    con.execute("DELETE FROM experiments WHERE run_id=? AND name='expected_artifact'", (RUN,))
    con.commit()
    con.close()
    code, lines = cli(finished, "score-human", RUN, "--human-set", str(f))
    assert code == EXIT_REFUSED and "expected_artifact" in lines[-1]


def test_refuses_when_adapter_on_disk_changed(finished: Path) -> None:
    rep = report_of(finished)
    f = confirmed_file(finished / "h.json")
    adapters = [
        p for p in (run_dir(finished) / f"round{rep['candidate_round']}").rglob("*") if p.is_file()
    ]
    victim = next(p for p in adapters if p.suffix not in {".jsonl"})
    victim.write_bytes(victim.read_bytes() + b"tamper")
    code, lines = cli(finished, "score-human", RUN, "--human-set", str(f))
    assert code == EXIT_REFUSED and "adapter" in lines[-1]
    assert not (run_dir(finished) / "heldout" / "human.json").exists()


def test_exact_overlap_with_train_dev_gate_is_dropped_and_counted(finished: Path) -> None:
    store = Store(finished / "home" / "dry-runs")
    split = humanscore._outputs(store, RUN, "split")[0]
    train_q, dev_q = split["train"][0]["question"], split["dev"][0]["question"]
    gate_q = store.load_heldout(RUN)[0]["question"]  # tests may look; src may not
    store.close()
    pairs = [
        (train_q, "SELECT COUNT(*) FROM accounts"),
        ("  " + dev_q.replace(" ", "   ", 1) + " ", "SELECT COUNT(*) FROM users"),  # whitespace
        (gate_q, "SELECT COUNT(*) FROM plans"),
        *QUESTIONS,
    ]
    f = confirmed_file(finished / "h.json", pairs)
    code, lines = cli(finished, "score-human", RUN, "--human-set", str(f))
    assert code == EXIT_OK, lines
    block = report_of(finished)["evaluation"]["human"]
    assert block["dropped_exact_overlap"] == {"train": 1, "dev": 1, "gate": 1, "total": 3}
    assert block["n"] == len(QUESTIONS)
    assert any("dropped exact overlap" in x and '"total": 3' in x for x in lines)


def test_live_run_needs_token_and_explicit_approval(tmp_path: Path) -> None:
    f = confirmed_file(tmp_path / "h.json")
    code, lines = cli(tmp_path, "score-human", "sql-live", "--human-set", str(f))
    assert code == EXIT_REFUSED and "admin token" in lines[-1]
    # a valid token still is not enough without --i-approve-spend
    lines2: list[str] = []
    env = {"DISTILLERY_ADMIN_TOKEN": "tok", "DISTILLERY_ADMIN_TOKEN_SUPPLIED": "tok"}
    code = main(
        ["--root", str(tmp_path / "home"), "score-human", "sql-live", "--human-set", str(f)],
        env=env,
        out=lines2.append,
    )
    assert code == EXIT_REFUSED and "--i-approve-spend" in lines2[-1]


def _direct(finished: Path, f: Path, **kw: Any):  # type: ignore[no-untyped-def]
    rep = report_of(finished)
    store = Store(finished / "home" / "dry-runs")
    with AsyncBridge() as bridge:
        dr = build_dry_run(
            Scale.model_validate(rep["config"]["pipeline"]["scale"]), bridge, human_set_path=f, **kw
        )
        result = score_human_run(store, RUN, f, dr.config, dr.deps, say=lambda _s: None)
        store.close()
    return dr, result


def test_serving_servers_are_closed(finished: Path) -> None:
    dr, result = _direct(finished, confirmed_file(finished / "h.json"))
    assert result is not None
    assert dr.students.servers and dr.base.servers
    assert all(s.closed for s in (*dr.students.servers, *dr.base.servers))


def test_serving_servers_are_closed_when_scoring_fails(finished: Path) -> None:
    f = confirmed_file(finished / "h.json")
    rep = report_of(finished)
    store = Store(finished / "home" / "dry-runs")
    with AsyncBridge() as bridge:
        dr = build_dry_run(
            Scale.model_validate(rep["config"]["pipeline"]["scale"]), bridge, human_set_path=f
        )

        def boom(_batch: Any) -> list[str]:
            raise RuntimeError("generation exploded")

        # the base model blows up mid-scoring; every opened server must still be closed
        real_factory = dr.deps.base_factory
        assert real_factory is not None

        def bad_base() -> Any:
            srv = real_factory()
            srv.generate = boom  # type: ignore[method-assign]
            return srv

        deps = type(dr.deps)(**{**dr.deps.__dict__, "base_factory": bad_base})
        with pytest.raises(RuntimeError, match="exploded"):
            score_human_run(store, RUN, f, dr.config, deps, say=lambda _s: None)
        store.close()
    assert all(s.closed for s in (*dr.students.servers, *dr.base.servers))
    assert report_of(finished)["evaluation"]["human"] is None
    assert not (run_dir(finished) / "report_human.json").exists()


def test_existing_decision_human_is_never_overwritten(finished: Path) -> None:
    """Even if a refusal check were bypassed, the merge only fills absent keys."""
    f = confirmed_file(finished / "h.json")
    p = run_dir(finished) / "report.json"
    rep = report_of(finished)
    rep["decision_human"] = "REJECT"
    p.write_text(json.dumps(rep))
    code, lines = cli(finished, "score-human", RUN, "--human-set", str(f))
    assert code == EXIT_REFUSED and "already has a Gate B" in lines[-1]
    assert json.loads(p.read_text())["decision_human"] == "REJECT"


def test_only_evaluator_touches_sealed_loaders() -> None:
    forbidden = {"load_heldout", "load_stress", "load_human", "read_confirmed", "seal_human"}
    for name in ("humanscore.py", "cli.py"):
        tree = ast.parse((SRC / name).read_text())
        names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {
            n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
        }
        assert not names & forbidden, name


def test_scale_registry_unused_import_guard() -> None:
    assert "tiny" in SCALES  # the end-to-end tests rely on it
