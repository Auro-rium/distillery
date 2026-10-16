"""The certified checkpoint shipped with the server (deploy/student) and the router's pin on it."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from server_helpers import QUESTION, fake_llm, make_config, make_settings
from test_gateway import Env, skill_body, student_for, write_router

from distillery.sandbox_student import prepare_adapter
from distillery.server import create_app
from distillery.server.gateway import WRONG_STUDENT_REASON, load_router
from distillery.server.local_models import (
    LocalModels,
    ServedAdapter,
    from_env,
    load_served_adapter,
)
from distillery.student import FakeStudent, StudentServer

REPO = Path(__file__).resolve().parents[1]
N1 = "7d6ce813882bf3a211c84bfab7f8109b3bcdd6ca4617349a03bff58076b85193"


def _digest(files: dict[str, bytes]) -> str:
    h = hashlib.sha256()
    for name in sorted(files):
        h.update(f"{name}:{hashlib.sha256(files[name]).hexdigest()}\n".encode())
    return h.hexdigest()


def _ship(d: Path, *, base: str = "fake-s", manifest_base: str | None = None) -> str:
    d.mkdir(parents=True)
    files = {
        "adapter_config.json": json.dumps({"base_model_name_or_path": base}).encode(),
        "adapter_model.safetensors": b"WEIGHTS",
        "tokenizer.json": b"{}",
    }
    for name, data in files.items():
        (d / name).write_bytes(data)
    sha = _digest(files)
    manifest = {"adapter_sha256": sha, "base_model": manifest_base or base}
    (d / "served.json").write_text(json.dumps(manifest), encoding="utf-8")
    return sha


def test_the_shipped_checkpoint_is_the_certified_n1_adapter() -> None:
    # the bytes in the image are the ones the router's certificate measured
    served = load_served_adapter(REPO / "deploy" / "student")
    assert served.adapter_sha256 == N1 and served.base_model == "Qwen/Qwen3-1.7B"
    router = load_router(REPO / "deploy" / "router")
    assert router.certified and router.student_adapter_sha256 == N1
    assert router.index_file is not None
    drift = json.loads((REPO / "deploy" / "router" / "drift.json").read_text(encoding="utf-8"))
    assert drift == {"window_w": 20, "tau": 0.9, "audit_rate": 0.1, "audit_window_m": 50,
                     "audit_k": 10, "min_requests": 20}  # fmt: skip


def test_the_image_and_blueprint_point_at_the_shipped_files() -> None:
    docker = (REPO / "Dockerfile").read_text(encoding="utf-8")
    blueprint = (REPO / "render.yaml").read_text(encoding="utf-8")
    for text in (docker, blueprint):
        assert "/app/deploy/router" in text and "/app/deploy/student" in text


def test_a_changed_extra_or_missing_file_is_rejected(tmp_path: Path) -> None:
    sha = _ship(tmp_path / "ok")
    assert load_served_adapter(tmp_path / "ok").adapter_sha256 == sha
    _ship(tmp_path / "changed")
    (tmp_path / "changed" / "adapter_model.safetensors").write_bytes(b"OTHER")
    _ship(tmp_path / "extra")
    (tmp_path / "extra" / "notes.txt").write_text("x", encoding="utf-8")
    _ship(tmp_path / "missing")
    (tmp_path / "missing" / "tokenizer.json").unlink()
    for sub in ("changed", "extra", "missing"):
        with pytest.raises(ValueError, match="do not hash"):
            load_served_adapter(tmp_path / sub)
    _ship(tmp_path / "base", base="fake-s", manifest_base="other/model")
    with pytest.raises(ValueError, match="not trained on"):
        load_served_adapter(tmp_path / "base")
    with pytest.raises(ValueError, match="cannot read served.json"):
        load_served_adapter(tmp_path / "nowhere")


def test_the_shipped_checkpoint_fixes_the_base_model(tmp_path: Path) -> None:
    sha = _ship(tmp_path / "student", base="Qwen/Qwen3-1.7B")
    cfg = make_config(
        nebius_api_key="k", nebius_project_id="p",
        model_ids={**make_config().model_ids, "student": "Qwen/Qwen3-0.6B"},
    )  # fmt: skip
    env = {
        "DISTILLERY_PLAYGROUND_SANDBOX_MODELS": "1",
        "DISTILLERY_PLAYGROUND_STUDENT_DIR": str(tmp_path / "student"),
    }
    local = from_env(cfg, env, demo_db=lambda: b"")
    assert local.served is not None and local.served.adapter_sha256 == sha
    # the stale 0.6B model id would reject the 1.7B adapter; the shipped checkpoint wins
    prepared = prepare_adapter(local.served.files, local.served.base_model)
    assert prepared.base_model_name_or_path == "Qwen/Qwen3-1.7B"
    (tmp_path / "student" / "tokenizer.json").write_bytes(b"changed")
    rejected = from_env(cfg, env, demo_db=lambda: b"")
    assert rejected.served is None and rejected.student_factory is None
    assert "shipped student rejected" in rejected.student_unset_reason


def test_the_playground_serves_the_shipped_checkpoint(tmp_path: Path) -> None:
    sha = _ship(tmp_path / "student")
    seen: list[Sequence[Path]] = []

    def factory(files: Sequence[Path]) -> StudentServer:
        prepare_adapter(files, "fake-s")
        seen.append(files)
        return FakeStudent(lambda _m: "```sql\nSELECT 1\n```")

    served = load_served_adapter(tmp_path / "student")
    local = LocalModels(student_factory=factory, served=served)
    s = make_settings(tmp_path, playground_llm=fake_llm()[0], playground_local=local)
    with TestClient(create_app(s)) as c:
        r = c.post("/api/playground", json={"question": QUESTION}).json()["results"]["student"]
        assert r["state"] == "ok" and r["sql"] == "SELECT 1"
        assert c.get("/api/config").json()["models"]["student"] == "fake-s"
    assert sorted(p.name for p in seen[0]) == sorted(p.name for p in served.files)
    assert served.adapter_sha256 == sha


def _served(sha: str) -> LocalModels:
    return LocalModels(served=ServedAdapter((), sha, "fake-s"))


def _pinned_router(d: Path, sha: str) -> Path:
    write_router(d)
    (d / "student.json").write_text(json.dumps({"adapter_sha256": sha}), encoding="utf-8")
    return d


def test_a_pinned_router_serves_only_the_certified_checkpoint(tmp_path: Path) -> None:
    student, nn = student_for()
    env = Env(
        tmp_path / "a", router=_pinned_router(tmp_path / "ra", "aaa"), student=student,
        nn_sim=nn, playground_local=_served("aaa"),
    )  # fmt: skip
    assert env.post(skill_body()).json()["distillery"]["served_by"] == "student"
    status = env.client.get("/api/gateway/status").json()
    assert status["student_adapter_sha256"] == status["certified_adapter_sha256"] == "aaa"


@pytest.mark.parametrize("local", [_served("bbb"), None])
def test_another_or_unknown_checkpoint_goes_to_the_teacher_before_any_spend(
    tmp_path: Path, local: LocalModels | None
) -> None:
    student, _ = student_for()
    looked_up: list[str] = []
    kw: dict[str, Any] = {"playground_local": local} if local is not None else {}
    env = Env(
        tmp_path, router=_pinned_router(tmp_path / "r", "aaa"), student=student,
        nn_sim=lambda q: looked_up.append(q) or 0.9, **kw,
    )  # fmt: skip
    d = env.post(skill_body()).json()["distillery"]
    assert d["served_by"] == "teacher" and d["reason"] == WRONG_STUDENT_REASON
    assert student.calls == 0 and looked_up == [] and d["embed_cost_usd"] == 0
