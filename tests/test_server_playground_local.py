"""Base + student on sandbox CPU in the playground, with injected fakes (no network, no sandbox)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from server_helpers import (
    GOLD,
    QUESTION,
    fake_llm,
    make_config,
    make_settings,
    sample_report,
    seed_finished_run,
)

from distillery.sandbox_student import prepare_adapter
from distillery.server import create_app
from distillery.server.local_models import LocalModels
from distillery.student import ChatMessages, FakeStudent, StudentServer, StudentServingError

RUN = "sql-tiny-served"


def _seed_adapter(root: Path, *, base: str = "fake-s", tamper: bool = False) -> None:
    store = seed_finished_run(root, RUN, dry=False)
    d = store.run_dir(RUN) / "round1" / "checkpoints" / "ck"
    d.mkdir(parents=True)
    files = {
        "adapter_config.json": json.dumps({"base_model_name_or_path": base}).encode(),
        "adapter_model.safetensors": b"WEIGHTS",
    }
    entries = []
    for name, data in files.items():
        (d / name).write_bytes(data)
        entries.append({
            "file_id": name, "path": f"round1/checkpoints/ck/{name}",
            "sha256": hashlib.sha256(data).hexdigest(),
        })  # fmt: skip
    if tamper:
        (d / "adapter_model.safetensors").write_bytes(b"CHANGED")
    art = {"job_id": "j", "checkpoint_id": "c", "base_model": base,
           "fine_tuned_model_checkpoint": None, "files": entries, "adapter_sha256": "x"}  # fmt: skip
    store.get_or_run(
        RUN, f"finetune_r{sample_report()['candidate_round']}", {}, lambda: {"artifact": art}
    )
    store.close()


class CountingExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def run_batch(self, db_ref: str, sqls: Sequence[str]) -> list[Any]:
        from distillery.taskpacks.sql import schema
        from distillery.taskpacks.sql.runner import run_select

        self.calls += 1
        db = schema.build_database(0)
        return [run_select(db, q) for q in sqls]


def _sql(text: str) -> Callable[[ChatMessages], str]:
    return lambda _m: f"```sql\n{text}\n```"


def _client(
    tmp_path: Path, local: LocalModels | None, *, teacher: bool = True, **kw: Any
) -> TestClient:
    llm = fake_llm()[0] if teacher else None
    s = make_settings(tmp_path, playground_llm=llm, playground_local=local, **kw)
    return TestClient(create_app(s))


def _factory(sql: str = GOLD, seen: list[Sequence[Path]] | None = None) -> Any:
    def make(files: Sequence[Path]) -> StudentServer:
        prepare_adapter(files, "fake-s")
        if seen is not None:
            seen.append(files)
        return FakeStudent(_sql(sql))

    return make


def _ask(c: TestClient, **kw: Any) -> Any:
    return c.post("/api/playground", json={"question": QUESTION}, **kw)


def test_default_is_disabled_not_unavailable(tmp_path: Path) -> None:
    with _client(tmp_path, None) as c:
        r = _ask(c).json()["results"]
        for role in ("base", "student"):
            assert r[role]["state"] == "disabled" and r[role]["available"] is False
            assert r[role]["reason"] and r[role]["sql"] is None
        assert r["teacher"]["state"] == "ok"


def test_base_and_student_answer_and_are_verified_by_the_executor(tmp_path: Path) -> None:
    _seed_adapter(tmp_path / "data")
    ex = CountingExecutor()
    seen: list[Sequence[Path]] = []
    local = LocalModels(
        base=FakeStudent(_sql("SELECT 1")), base_reason=None, student_factory=_factory(GOLD, seen),
        student_run=RUN, executor=ex,
    )  # fmt: skip
    with _client(tmp_path, local) as c:
        body = _ask(c).json()
        r = body["results"]
        assert r["student"]["state"] == "ok" and r["student"]["sql"] == GOLD
        assert r["student"]["verified"] is True and r["student"]["rows_preview"]["rows"]
        assert r["base"]["state"] == "ok" and r["base"]["verified"] is False
        assert ex.calls >= 2  # model-written SQL ran through the injected (sandbox) executor
        assert [p.name for p in seen[0]] == ["adapter_config.json", "adapter_model.safetensors"]
        assert body["sandbox_cost_usd"] is None  # not measured, never shown as 0
        assert "not measured" in body["note"]
        pg = c.get("/api/config").json()["playground"]
        assert pg["models"]["student"]["state"] == "ok" and pg["models"]["base"]["state"] == "ok"


def test_student_states_unset_missing_run_mismatch_and_tampered(tmp_path: Path) -> None:
    def state(local: LocalModels, sub: str) -> dict[str, Any]:
        with _client(tmp_path / sub, local) as c:
            out: dict[str, Any] = _ask(c).json()["results"]["student"]
            return out

    unset = state(LocalModels(base=None, student_factory=_factory()), "a")
    assert (
        unset["state"] == "unavailable" and "DISTILLERY_PLAYGROUND_STUDENT_RUN" in unset["reason"]
    )
    gone = state(LocalModels(student_factory=_factory(), student_run="nope"), "b")
    assert gone["state"] == "unavailable" and "nope" in gone["reason"]
    _seed_adapter(tmp_path / "c" / "data", base="other/model")
    mism = state(LocalModels(student_factory=_factory(), student_run=RUN), "c")
    assert mism["state"] == "unavailable" and "base_model_name_or_path" in mism["reason"]
    _seed_adapter(tmp_path / "d" / "data", tamper=True)
    bad = state(LocalModels(student_factory=_factory(), student_run=RUN), "d")
    assert bad["state"] == "unavailable" and "sha256" in bad["reason"]


def test_disabled_flag_wins(tmp_path: Path) -> None:
    with _client(tmp_path, LocalModels(enabled=False, base=FakeStudent())) as c:
        r = _ask(c).json()["results"]
        assert r["base"]["state"] == "disabled" and r["student"]["state"] == "disabled"


def test_generation_error_is_an_error_state_never_a_fake_answer(tmp_path: Path) -> None:
    class Boom:
        def generate(self, _b: Sequence[ChatMessages]) -> list[str]:
            raise StudentServingError("sandbox exploded")

        def close(self) -> None: ...

    local = LocalModels(base=Boom(), base_reason=None)
    with _client(tmp_path, local) as c:
        b = _ask(c).json()["results"]["base"]
        assert b["state"] == "error" and "sandbox exploded" in b["error"] and b["sql"] is None
        empty = LocalModels(base=FakeStudent(lambda _m: "no sql here"), base_reason=None)
    with _client(tmp_path / "x", empty) as c:
        b = _ask(c).json()["results"]["base"]
        assert b["state"] == "error" and b["sql"] is None and "no SQL" in b["error"]


def test_per_ip_hour_cap_rate_limits_student_calls_only(tmp_path: Path) -> None:
    now = [0.0]
    student = FakeStudent(_sql(GOLD))
    local = LocalModels(base=student, base_reason=None)
    with _client(tmp_path, local, playground_student_per_ip_per_hour=1,
                 clock=lambda: now[0]) as c:  # fmt: skip
        assert _ask(c).json()["results"]["base"]["state"] == "ok"
        r = _ask(c)
        assert r.status_code == 200  # the request itself is allowed: the teacher still answers
        res = r.json()["results"]
        assert res["teacher"]["state"] == "ok"
        assert res["base"]["state"] == "rate_limited" and res["base"]["retry_after_s"] >= 1
        assert res["base"]["sql"] is None and student.calls == 1
        now[0] += 3601
        assert _ask(c).json()["results"]["base"]["state"] == "ok"


def test_daily_cap_rate_limits_student_calls_across_ips(tmp_path: Path) -> None:
    day = [datetime(2026, 1, 1, 12, tzinfo=UTC)]
    student = FakeStudent(_sql(GOLD))
    local = LocalModels(base=student, base_reason=None)
    with _client(tmp_path, local, playground_student_per_ip_per_hour=100,
                 playground_student_daily_cap=2, now=lambda: day[0]) as c:  # fmt: skip
        for _ in range(2):
            assert _ask(c).json()["results"]["base"]["state"] == "ok"
        limited = _ask(c).json()["results"]["base"]
        assert limited["state"] == "rate_limited" and "daily" in limited["reason"]
        assert limited["retry_after_s"] == 12 * 3600
        assert student.calls == 2
        day[0] = datetime(2026, 1, 2, 0, 0, 1, tzinfo=UTC)
        assert _ask(c).json()["results"]["base"]["state"] == "ok"


def test_unavailable_models_do_not_consume_the_caps(tmp_path: Path) -> None:
    with _client(tmp_path, LocalModels(base=None), playground_student_daily_cap=1) as c:
        for _ in range(3):
            assert _ask(c).json()["results"]["base"]["state"] == "unavailable"


def test_teacher_off_but_local_models_on_keeps_playground_enabled(tmp_path: Path) -> None:
    local = LocalModels(base=FakeStudent(_sql(GOLD)), base_reason=None)
    with _client(tmp_path, local, teacher=False, config=make_config()) as c:
        pg = c.get("/api/config").json()["playground"]
        assert pg["enabled"] is True and pg["models"]["teacher"]["state"] == "unavailable"
        r = _ask(c).json()["results"]
        assert r["teacher"]["state"] == "unavailable" and r["base"]["state"] == "ok"


def test_from_env_is_off_by_default_and_honest_without_credentials() -> None:
    from distillery.server.local_models import from_env

    off = from_env(make_config(), {}, demo_db=lambda: b"")
    assert off.enabled is False
    no_sandbox = from_env(
        make_config(), {"DISTILLERY_PLAYGROUND_SANDBOX_MODELS": "1"}, demo_db=lambda: b""
    )
    assert no_sandbox.enabled is True and no_sandbox.base is None
    assert "NEBIUS_AI_PROJECT" in (no_sandbox.base_reason or "")  # no project id in make_config


def test_from_env_with_credentials_builds_nothing_until_used() -> None:
    from pydantic import SecretStr

    from distillery.server.local_models import from_env

    cfg = make_config(nebius_project_id=SecretStr("proj-x"))
    env = {"DISTILLERY_PLAYGROUND_SANDBOX_MODELS": "1", "DISTILLERY_PLAYGROUND_STUDENT_RUN": RUN}
    lm = from_env(cfg, env, demo_db=lambda: (_ for _ in ()).throw(AssertionError("no db yet")))
    assert lm.base is not None and lm.base_reason is None and lm.student_run == RUN
    assert lm.student_factory is not None and lm.executor is not None
    lm.close()  # closing a never-used lazy server touches nothing
