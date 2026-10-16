"""scripts/w_e2e.py end to end against the in-process app with fakes (no network, no spend)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from test_gateway import LAMBDA, Env, student_for, write_router
from test_gateway_drift import drift

from distillery.server.local_models import LocalModels, ServedAdapter

REPO = Path(__file__).resolve().parents[1]


def _script() -> Any:
    spec = importlib.util.spec_from_file_location("w_e2e", REPO / "scripts" / "w_e2e.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["w_e2e"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    mod = _script()
    burst = "\n".join(f"How many credit grants were issued in month {m}?" for m in range(30))
    (tmp_path / "burst.txt").write_text(burst + "\n", encoding="utf-8")
    monkeypatch.setattr(mod, "BURST_SHA256", hashlib.sha256((burst + "\n").encode()).hexdigest())
    monkeypatch.setattr(mod, "LAMBDA_HAT", LAMBDA)
    monkeypatch.setattr(mod, "N1_ADAPTER", "aaa")
    monkeypatch.setattr(mod, "STUDENT_BASE", "fake-s")
    monkeypatch.setattr(mod, "_prereg", lambda c, sha: {"commit": c, "committed_at": "t"})
    monkeypatch.setattr(mod, "_code_pin", lambda c: {"head": c, "script_sha256": "s"})
    monkeypatch.setenv("DISTILLERY_GLOBAL_LEDGER", str(tmp_path / "ledger.sqlite"))
    router = write_router(tmp_path / "router")
    (router / "student.json").write_text(json.dumps({"adapter_sha256": "aaa"}), encoding="utf-8")
    student, _ = student_for()
    env = Env(
        tmp_path, router=router, student=student,
        # the burst and the out-of-scope question look unfamiliar, the rest look familiar
        nn_sim=lambda q: 0.1 if ("credit" in q or "retention" in q) else 0.9,
        gateway={"drift": drift(), "per_ip_per_hour": 1000},
        playground_local=LocalModels(served=ServedAdapter((), "aaa", "fake-s")),
    )  # fmt: skip
    monkeypatch.setattr(mod.httpx, "Client", lambda **kw: env.client)

    def go(*extra: str) -> tuple[int, dict[str, Any]]:
        out = tmp_path / f"out{len(extra)}.json"
        argv = ["--base-url", "http://testserver", "--out", str(out), "--prereg-commit", "abc"]
        code = mod.main([*argv, *extra])
        return code, json.loads(out.read_text(encoding="utf-8"))

    go.burst = tmp_path / "burst.txt"  # type: ignore[attr-defined]
    go.env = env  # type: ignore[attr-defined]
    return go


def test_the_local_run_passes_and_records_no_burst_text(run: Any) -> None:
    code, doc = run("--burst-questions", str(run.burst))
    assert code == 0 and doc["pass"] is True, doc["criteria"]
    c = doc["counts"]
    assert c["normal"] == 20 and c["normal_served_by_student"] == 19
    assert c["normal_deferred_by_router"] == 1 and c["burst_served_by_student"] == 0
    assert c["rollback_on_burst_request"] is not None and c["student_answers_after_rollback"] == 0
    assert doc["drift_before_burst"] == "CERTIFIED"
    assert [r["via"] for r in doc["requests"][:4]] == ["acme_ask"] * 3 + ["v1"]
    assert "credit" not in json.dumps(doc)  # sealed burst text never reaches the evidence


def test_the_hosted_check_skips_the_burst_and_the_ledger(run: Any) -> None:
    code, doc = run("--hosted")
    assert code == 0 and doc["mode"] == "hosted" and doc["counts"]["burst"] == 0
    assert "C5_drift" not in doc["criteria"] and doc["ledger"] is None


def test_without_a_pre_registration_nothing_passes(run: Any, tmp_path: Path) -> None:
    mod = sys.modules["w_e2e"]
    out = tmp_path / "noprereg.json"
    assert mod.main(["--base-url", "http://testserver", "--out", str(out), "--hosted"]) == 1
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["pass"] is False and doc["prereg"] is None and "prereg" in doc["stopped"]


def test_a_wrong_burst_file_or_existing_output_is_refused(run: Any, tmp_path: Path) -> None:
    (tmp_path / "other.txt").write_text("x\n", encoding="utf-8")
    mod = sys.modules["w_e2e"]
    assert mod.main(["--base-url", "http://t", "--out", str(tmp_path / "o.json"),
                     "--burst-questions", str(tmp_path / "other.txt")]) == 2  # fmt: skip
    (tmp_path / "exists.json").write_text("{}", encoding="utf-8")
    assert mod.main(["--base-url", "http://t", "--out", str(tmp_path / "exists.json")]) == 2


def test_a_failed_precondition_sends_nothing_and_writes_nothing(
    run: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = sys.modules["w_e2e"]
    monkeypatch.setattr(mod, "STUDENT_BASE", "Qwen/Qwen3-1.7B")  # the server reports fake-s
    out = tmp_path / "pre.json"
    argv = ["--base-url", "http://testserver", "--out", str(out), "--prereg-commit", "abc"]
    assert mod.main([*argv, "--burst-questions", str(run.burst)]) == 2
    assert not out.exists()
    assert run.env.upstream.calls == [] and run.env.upstream.embed_calls == []
