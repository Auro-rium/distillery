# ruff: noqa: S101
"""EndpointStudent offline with a fake control plane and chat client. Nothing here touches the
network; real-API behaviour (status field, routing, custom weights) is UNVERIFIED."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from distillery.budget import BudgetExceeded, Ledger
from distillery.endpoint_student import (
    EndpointBackend,
    EndpointSpec,
    EndpointStudent,
    HttpControlPlane,
    make_base_factory,
    make_student_factory,
)
from distillery.finetune import TrainedArtifact
from distillery.student import StudentServer, StudentServingError

SPEC = EndpointSpec(
    flavor_name="base", gpu_type="gpu-h100-sxm", gpu_count=1, region="eu-north1",
    hourly_cost_usd=3.0, model_name="ft-model", base_model_name="Qwen/Qwen3-1.7B",
    poll_interval_s=1.0, ready_timeout_s=10.0, concurrency=3,
)  # fmt: skip


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s


class FakeControl:
    def __init__(self, statuses: list[str], delete_fails: int = 0) -> None:
        self.statuses = list(statuses)
        self.created: list[Mapping[str, Any]] = []
        self.deleted: list[str] = []
        self.delete_fails = delete_fails

    def templates(self) -> Any:
        return []

    def create(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        self.created.append(payload)
        return {"endpoint_id": "ep-1", "routing_key": "rk-1"}

    def status(self, endpoint_id: str) -> str:
        return self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]

    def delete(self, endpoint_id: str) -> None:
        if self.delete_fails > 0:
            self.delete_fails -= 1
            raise StudentServingError("HTTP 500")
        self.deleted.append(endpoint_id)


class FakeChat:
    def __init__(self, not_routable: int = 0, fail: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.not_routable = not_routable
        self.fail = fail
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw: Any) -> Any:
        self.calls.append(kw)
        if self.not_routable > 0:
            self.not_routable -= 1
            raise RuntimeError("Error code: 404")
        if self.fail:
            raise RuntimeError("boom")
        text = kw["messages"][-1]["content"]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def build(
    control: FakeControl, chat: FakeChat, clock: Clock, ledger: Ledger | None = None
) -> EndpointStudent:
    return EndpointStudent(
        control, lambda url: chat, SPEC, model_name="ft-model", ledger=ledger,
        sleep=clock.sleep, clock=clock,
    )  # fmt: skip


def q(i: int) -> list[dict[str, str]]:
    return [{"role": "user", "content": f"q{i}"}]


def test_create_wait_ready_generate_delete_and_cost() -> None:
    clock, control, chat = Clock(), FakeControl(["provisioning", "not ready", "ready"]), FakeChat(1)
    ledger = Ledger("r", 10.0, 100.0, 1.0)
    s: StudentServer = build(control, chat, clock, ledger)
    payload = control.created[0]
    assert payload["model_name"] == "ft-model" and payload["scaling"] == {
        "min_replicas": 1,
        "max_replicas": 1,
    }
    assert payload["gpu_type"] == "gpu-h100-sxm" and payload["region"] == "eu-north1"
    assert s.generate([q(0), q(1), q(2), q(3)]) == ["q0", "q1", "q2", "q3"]
    assert chat.calls[-1]["model"] == "rk-1" and chat.calls[-1]["temperature"] == 0
    clock.t += 1800.0  # half an hour of billed time after ready
    s.close()
    assert control.deleted == ["ep-1"]
    assert ledger.spent() == pytest.approx(1.5, rel=1e-2)  # 0.5h x explicit $3/h
    s.close()  # idempotent
    assert control.deleted == ["ep-1"]


def test_endpoint_deleted_when_never_ready() -> None:
    clock, control = Clock(), FakeControl(["provisioning"])
    with pytest.raises(StudentServingError, match="not ready"):
        build(control, FakeChat(), clock)
    assert control.deleted == ["ep-1"]


def test_endpoint_deleted_on_failed_status_and_no_billing_before_ready() -> None:
    clock, control = Clock(), FakeControl(["failed"])
    ledger = Ledger("r", 10.0, 100.0, 1.0)
    with pytest.raises(StudentServingError, match="failed"):
        build(control, FakeChat(), clock, ledger)
    assert control.deleted == ["ep-1"] and ledger.spent() == 0.0


def test_endpoint_deleted_when_generation_fails_and_caller_closes() -> None:
    clock, control = Clock(), FakeControl(["ready"])
    chat = FakeChat()
    s = build(control, chat, clock)
    chat.fail = True
    try:
        with pytest.raises(StudentServingError, match="chat completion failed"):
            s.generate([q(0)])
    finally:
        s.close()
    assert control.deleted == ["ep-1"]


def test_delete_retries_then_succeeds() -> None:
    clock, control = Clock(), FakeControl(["ready"], delete_fails=2)
    build(control, FakeChat(), clock).close()
    assert control.deleted == ["ep-1"]


def test_delete_failure_is_loud() -> None:
    clock, control = Clock(), FakeControl(["ready"], delete_fails=99)
    s = build(control, FakeChat(), clock)
    with pytest.raises(StudentServingError, match="may still be billing"):
        s.close()


def test_budget_preflight_refuses_before_creating() -> None:
    clock, control = Clock(), FakeControl(["ready"])
    with pytest.raises(BudgetExceeded):
        build(control, FakeChat(), clock, Ledger("r", 2.0, 100.0, 1.0))  # 1h x $3 > $2
    assert control.created == []


def test_closed_student_refuses() -> None:
    clock = Clock()
    s = build(FakeControl(["ready"]), FakeChat(), clock)
    s.close()
    with pytest.raises(RuntimeError, match="closed"):
        s.generate([q(0)])


def test_hourly_cost_is_required() -> None:
    with pytest.raises(ValueError):
        EndpointSpec(flavor_name="base", gpu_type="g", gpu_count=1, region="r")  # type: ignore[call-arg]


def _trained(ckpt: str | None) -> TrainedArtifact:
    return TrainedArtifact(
        job_id="j", checkpoint_id="c", base_model="Qwen/Qwen3-1.7B",
        fine_tuned_model_checkpoint=ckpt, files=(),
    )  # fmt: skip


def test_factories_choose_model_names() -> None:
    clock, chat = Clock(), FakeChat()
    kw: dict[str, Any] = {"sleep": clock.sleep, "clock": clock}
    spec = SPEC.model_copy(update={"model_name": None})
    control = FakeControl(["ready"])
    backend = EndpointBackend(control, lambda url: chat)
    # the fine-tuned checkpoint name is used when model_name is unset
    make_student_factory(backend, spec, **kw)(_trained("ckpt-name")).close()
    assert control.created[0]["model_name"] == "ckpt-name"
    with pytest.raises(StudentServingError, match="no endpoint model name"):
        make_student_factory(backend, spec, **kw)(_trained(None))
    make_base_factory(backend, spec, **kw)().close()
    assert control.created[1]["model_name"] == "Qwen/Qwen3-1.7B"
    no_base = spec.model_copy(update={"base_model_name": None})
    with pytest.raises(StudentServingError, match="base_model_name"):
        make_base_factory(backend, no_base, **kw)()


def test_http_control_plane_routes_and_no_key_in_repr() -> None:
    seen: list[tuple[str, str, dict[str, str], bytes | None]] = []
    responses = {
        ("POST", "/v0/dedicated_endpoints"): (200, b'{"endpoint_id": "e", "routing_key": "k"}'),
        ("GET", "/v0/dedicated_endpoints"): (
            200,
            b'{"endpoints": [{"endpoint_id": "e", "status": "READY"}]}',
        ),
        ("DELETE", "/v0/dedicated_endpoints/e"): (404, b"gone"),
    }  # fmt: skip

    def send(method: str, url: str, headers: Mapping[str, str], body: bytes | None):  # type: ignore[no-untyped-def]
        seen.append((method, url, dict(headers), body))
        return responses[(method, url.removeprefix("https://api.tokenfactory.nebius.com"))]

    cp = HttpControlPlane(lambda: "sekret", send=send)
    assert "sekret" not in repr(cp)
    assert cp.create({"name": "n"})["routing_key"] == "k"
    assert json.loads(seen[0][3] or b"") == {"name": "n"}
    assert seen[0][2]["Authorization"] == "Bearer sekret"
    assert cp.status("e") == "READY" and cp.status("zzz") == "not_found"
    cp.delete("e")  # 404 == already deleted
    responses[("POST", "/v0/dedicated_endpoints")] = (500, b"oops")
    with pytest.raises(StudentServingError, match="HTTP 500"):
        cp.create({"name": "n"})


def test_data_plane_url_default_and_override(tmp_path: Path) -> None:
    assert SPEC.data_plane_url() == "https://api.tokenfactory.eu-north1.nebius.com/v1"
    o = SPEC.model_copy(update={"data_plane_base_url": "https://x/v1"})
    assert o.data_plane_url() == "https://x/v1"


def test_endpoint_student_reports_served_model() -> None:
    clock, control = Clock(), FakeControl(["ready"])
    s = build(control, FakeChat(), clock)
    assert s.served_model == "ft-model"  # type: ignore[attr-defined]
    s.close()
