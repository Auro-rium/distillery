# ruff: noqa: S101, S105
"""draft-heldout, confirm-heldout and run --human-set, offline with scripted fakes."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2
import openai
import pytest

from distillery import humanset as hs
from distillery.cli import EXIT_FAIL, EXIT_OK, EXIT_REFUSED, main
from distillery.orchestrator import Deps
from distillery.pipeline_fakes import FAKE_MODELS, question_of
from distillery.sandbox_executor import AsyncBridge
from distillery.store import Store, sha256_hex
from distillery.taskpacks.sql import schema as sql_schema
from distillery.taskpacks.sql.executor import LocalExecutor

TOKEN = "s3cret-admin-token"
COUNT = "SELECT COUNT(*) FROM accounts"
REQ = httpx2.Request("POST", "https://example.invalid/v1/chat/completions")


class ScriptedTransport:
    """Duck-typed async chat transport. ``script[question]`` = 3 replies; an Exception instance is
    raised instead (a terminal 400 for the llm_error path)."""

    def __init__(self, script: dict[str, list[Any]]) -> None:
        self.script = script
        self.seen: dict[str, int] = {}
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, *, model: str, messages: Any, temperature: Any = None, **_: Any) -> Any:
        question = question_of(messages)
        assert question is not None
        i = self.seen.get(question, 0)
        self.seen[question] = i + 1
        self.calls.append({"model": model, "temperature": temperature, "question": question})
        reply = self.script[question][i]
        if isinstance(reply, Exception):
            raise reply
        message = SimpleNamespace(content=f"```sql\n{reply}\n```", refusal=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
        )


def bad_request() -> Exception:
    return openai.APIStatusError("http 400", response=httpx2.Response(400, request=REQ), body=None)


@pytest.fixture
def live_env(tmp_path: Path) -> dict[str, str]:
    prices = tmp_path / "prices.json"
    price = {"input_per_mtok": 1.0, "output_per_mtok": 2.0, "source": "test", "date": "2026-09-29"}
    prices.write_text(json.dumps({m: price for m in FAKE_MODELS.values()}))
    env = {
        "DISTILLERY_ADMIN_TOKEN": TOKEN,
        "DISTILLERY_ADMIN_TOKEN_SUPPLIED": TOKEN,
        "DISTILLERY_PRICES_FILE": str(prices),
        "PATH": "/usr/bin",
    }
    env.update({f"DISTILLERY_MODEL_{r.upper()}": m for r, m in FAKE_MODELS.items()})
    return env


@pytest.fixture
def bridge() -> Iterator[AsyncBridge]:
    with AsyncBridge() as b:
        yield b


QFILE_TEXT = "# synthetic\nagree?\nllm error?\ndisagree?\nempty?\n"
SCRIPT: dict[str, list[Any]] = {
    "agree?": [COUNT] * 3,
    "llm error?": [COUNT, bad_request(), COUNT],
    "disagree?": [COUNT, COUNT, "SELECT COUNT(*) FROM users"],
    "empty?": ["SELECT plan_id FROM plans WHERE 1=0"] * 3,
}


def factory_for(transport: ScriptedTransport, calls: list[str] | None = None) -> Any:
    def factory(_c: Any, _p: Any, b: AsyncBridge) -> Deps:
        if calls is not None:
            calls.append("called")
        return Deps(
            transport=transport, finetune=None, executor=LocalExecutor(), bridge=b,  # type: ignore[arg-type]
            student_factory=None, llm_sleep=_no_sleep,
        )  # fmt: skip

    return factory


async def _no_sleep(_s: float) -> None:
    return None


def run_cli(tmp_path: Path, *args: str, **kw: Any) -> tuple[int, list[str]]:
    lines: list[str] = []
    code = main(["--root", str(tmp_path / "home"), *args], out=lines.append, **kw)
    return code, lines


def write_questions(tmp_path: Path) -> Path:
    p = tmp_path / "q.txt"
    p.write_text(QFILE_TEXT)
    return p


def set_id_of(path: Path) -> str:
    return sha256_hex(path.read_bytes())[:8]


def test_draft_requires_admin_token_and_approval_flag(
    tmp_path: Path, live_env: dict[str, str]
) -> None:
    q = write_questions(tmp_path)
    calls: list[str] = []
    fac = factory_for(ScriptedTransport(SCRIPT), calls)
    no_token = {**live_env, "DISTILLERY_ADMIN_TOKEN_SUPPLIED": "nope"}
    code, out = run_cli(tmp_path, "draft-heldout", "--questions", str(q), "--i-approve-spend",
                        env=no_token, deps_factory=fac)  # fmt: skip
    assert code == EXIT_REFUSED and "admin token" in out[-1] and calls == []
    code, out = run_cli(tmp_path, "draft-heldout", "--questions", str(q), env=live_env,
                        deps_factory=fac)  # fmt: skip
    assert code == EXIT_REFUSED and "--i-approve-spend" in out[-1] and calls == []
    assert not (tmp_path / "home" / "humanset").exists()


def test_draft_writes_drafts_with_discard_reasons_and_ledger_spend(
    tmp_path: Path, live_env: dict[str, str]
) -> None:
    q = write_questions(tmp_path)
    transport = ScriptedTransport(SCRIPT)
    code, out = run_cli(tmp_path, "draft-heldout", "--questions", str(q), "--i-approve-spend",
                        env=live_env, deps_factory=factory_for(transport))  # fmt: skip
    assert code == EXIT_OK, out
    sid = set_id_of(q)
    drafts = hs.read_drafts(tmp_path / "home", sid)
    assert [k["question"] for k in drafts["kept"]] == ["agree?"]
    assert drafts["discarded_by_reason"] == {
        "llm_error": 1, "no_sql": 0, "exec_error": 0, "empty": 1, "disagree": 1,
    }  # fmt: skip
    db = sql_schema.build_database(0)
    assert drafts["db_sha256"] == sha256_hex(db)
    assert drafts["teacher_model"] == FAKE_MODELS["teacher"]
    assert len(transport.calls) == 12 and {c["temperature"] for c in transport.calls} == {0.8}
    assert {c["model"] for c in transport.calls} == {FAKE_MODELS["teacher"]}
    assert any("1 kept" in line or "kept: 1" in line for line in out)
    store = Store(tmp_path / "home")
    assert store.total_spend(f"humanset-{sid}") > 0  # ledger-recorded, labelled by run id
    store.close()


def test_draft_is_capped_by_the_run_budget(tmp_path: Path, live_env: dict[str, str]) -> None:
    q = write_questions(tmp_path)
    transport = ScriptedTransport(SCRIPT)
    code, out = run_cli(tmp_path, "draft-heldout", "--questions", str(q), "--i-approve-spend",
                        "--budget-usd", "0.0000001", env=live_env,
                        deps_factory=factory_for(transport))  # fmt: skip
    assert code == EXIT_REFUSED and "budget" in out[-1].lower() and transport.calls == []


def test_draft_rejects_a_bad_question_file(tmp_path: Path, live_env: dict[str, str]) -> None:
    p = tmp_path / "q.txt"
    p.write_text("x" * 501 + "\n")
    calls: list[str] = []
    code, out = run_cli(tmp_path, "draft-heldout", "--questions", str(p), "--i-approve-spend",
                        env=live_env, deps_factory=factory_for(ScriptedTransport({}), calls))  # fmt: skip
    assert code == EXIT_REFUSED and "line 1" in out[-1] and calls == []


def _scripted_input(answers: list[str]) -> Callable[[str], str]:
    it = iter(answers)

    def ask(_prompt: str) -> str:
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None

    return ask


def _drafted(tmp_path: Path, live_env: dict[str, str]) -> str:
    q = write_questions(tmp_path)
    script = {**SCRIPT, "llm error?": [COUNT] * 3, "disagree?": [COUNT] * 3}
    code, _ = run_cli(tmp_path, "draft-heldout", "--questions", str(q), "--i-approve-spend",
                      env=live_env, deps_factory=factory_for(ScriptedTransport(script)))  # fmt: skip
    assert code == EXIT_OK
    return set_id_of(q)


def test_confirm_then_finalize_end_to_end(tmp_path: Path, live_env: dict[str, str]) -> None:
    sid = _drafted(tmp_path, live_env)  # kept: agree?, llm error?, disagree? ; discarded: empty?
    code, out = run_cli(tmp_path, "confirm-heldout", "--set", sid, env={},
                        ask=_scripted_input(["y", "q"]))  # fmt: skip
    assert (
        code == EXIT_OK and any("agree?" in line for line in out) and any(COUNT in x for x in out)
    )
    code, out = run_cli(tmp_path, "confirm-heldout", "--set", sid, env={},
                        ask=_scripted_input(["n", "y"]))  # fmt: skip
    assert code == EXIT_OK
    assert "Question: agree?" not in out  # already confirmed: resumed after item 1
    code, out = run_cli(tmp_path, "confirm-heldout", "--set", sid, "--finalize", env={})
    assert code == EXIT_OK
    doc = json.loads((tmp_path / "home" / "humanset" / sid / "human_confirmed.json").read_text())
    assert [i["question"] for i in doc["items"]] == ["agree?", "disagree?"]
    assert doc["counts"]["confirmed"] == 2 and doc["counts"]["rejected"] == 1
    assert doc["counts"]["discarded_by_reason"]["empty"] == 1
    assert any("confirmed: 2" in line or "2 confirmed" in line for line in out)
    # chained offline check: what finalize wrote is what a dry run accepts, seals and scores
    confirmed = tmp_path / "home" / "humanset" / sid / "human_confirmed.json"
    code, out = run_cli(tmp_path, "run", "--scale", "tiny", "--dry-run", "--human-set",
                        str(confirmed), env={})  # fmt: skip
    assert code == EXIT_OK, out
    run_dir = tmp_path / "home" / "dry-runs" / "runs" / "dry-sql-tiny"
    report = json.loads((run_dir / "report.json").read_text())
    assert report["evaluation"]["human"]["n"] == 2
    assert report["data"]["human"]["counts"]["rejected"] == 1
    assert report["data"]["human"]["counts"]["discarded_by_reason"]["empty"] == 1
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["human_sha256"] == report["data"]["human"]["sha256"]


def test_confirm_errors_are_clean(tmp_path: Path) -> None:
    code, out = run_cli(tmp_path, "confirm-heldout", "--set", "00000000", env={})
    assert code == EXIT_FAIL and "no drafts" in out[-1]
    code, out = run_cli(tmp_path, "confirm-heldout", "--set", "../../x", env={})
    assert code == EXIT_FAIL
    code, out = run_cli(tmp_path, "confirm-heldout", env={})  # no sets at all
    assert code == EXIT_FAIL and "no human sets" in out[-1].lower()


def test_run_with_human_set_flag_reports_gate_b(tmp_path: Path) -> None:
    items = [
        {"task_id": f"h-{i}-{i:08x}", "family": "human", "source": "human",
         "question": f"[H] total {t}?", "gold_sql": f"SELECT COUNT(*) FROM {t}",
         "requires_order": False}
        for i, t in enumerate(["accounts", "users", "plans", "invoices", "payments", "feature_usage"])
    ]  # fmt: skip
    f = tmp_path / "human_confirmed.json"
    f.write_text(json.dumps({
        "format": hs.CONFIRMED_FORMAT, "db_sha256": sha256_hex(sql_schema.build_database(0)),
        "question_file_sha256": "f" * 64, "teacher_model": "fake", "counts": {}, "items": items,
    }))  # fmt: skip
    code, out = run_cli(tmp_path, "run", "--scale", "tiny", "--dry-run", "--human-set", str(f),
                        env={})  # fmt: skip
    assert code == EXIT_OK, out
    text = "\n".join(out)
    assert "human held-out" in text and "Gate B" in text
    report = json.loads(
        (tmp_path / "home" / "dry-runs" / "runs" / "dry-sql-tiny" / "report.json").read_text()
    )
    assert report["decision_human"] in {"PROMOTE", "REJECT"}
    assert report["evaluation"]["human"]["n"] == 6


def test_run_with_a_missing_human_file_is_refused(tmp_path: Path) -> None:
    code, out = run_cli(tmp_path, "run", "--scale", "tiny", "--dry-run", "--human-set",
                        str(tmp_path / "absent.json"), env={})  # fmt: skip
    assert code == EXIT_REFUSED and "human" in out[-1].lower()
