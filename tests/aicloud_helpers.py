# ruff: noqa: S101, S105, S106
"""Fakes for the H1 tests: a scripted `nebius ai endpoint` CLI and a clock that only advances
when someone sleeps. Nothing here touches the network or a real process."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence

from distillery.aicloud_endpoint import Clock, EndpointPlan

TOKEN = "TOK-SECRET-123"


def make_plan(**kw: object) -> EndpointPlan:
    base: dict[str, object] = {
        "vllm_tag": "v0.0.test",
        "hourly_usd": 2.0,
        "adapter_s3_source": "s3://bkt/n1",
        "auth_token": TOKEN,
        "name": "t-ep",
    }
    base.update(kw)
    return EndpointPlan(**base)  # type: ignore[arg-type]


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def monotonic(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s

    def now_iso(self) -> str:
        return f"T+{self.t - 1000.0:g}"

    def clock(self) -> Clock:
        return Clock(monotonic=self.monotonic, sleep=self.sleep, now_iso=self.now_iso)


def proc(rc: int = 0, out: str = "", err: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], rc, out, err)


class FakeNebius:
    """Scripted CLI. ``states`` is what successive ``get`` calls report (the last one repeats)."""

    def __init__(
        self,
        *,
        states: Sequence[str] = ("PENDING", "RUNNING"),
        create_rc: int = 0,
        create_stderr: str = "",
        create_creates: bool = True,
        create_stdout: str | None = None,
        delete_rc: int = 0,
        list_rc: int = 0,
        gone_after_list: int = 0,  # `list` still shows the endpoint this many times after delete
        create_advance_s: float = 0.0,
        clock: FakeClock | None = None,
    ) -> None:
        self.states = list(states)
        self.create_rc, self.create_stderr = create_rc, create_stderr
        self.create_creates, self.create_stdout = create_creates, create_stdout
        self.delete_rc, self.list_rc = delete_rc, list_rc
        self.gone_after_list = gone_after_list
        self.create_advance_s, self.clock = create_advance_s, clock
        self.calls: list[list[str]] = []
        self.endpoints: dict[str, str] = {}
        self._gets = 0
        self._deleted_pending = 0

    def verbs(self) -> list[str]:
        return [c[3] for c in self.calls]

    def run(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        a = list(argv)
        self.calls.append(a)
        assert a[:3] == ["nebius", "ai", "endpoint"], a
        verb = a[3]
        if verb == "create":
            name = a[a.index("--name") + 1]
            if self.create_advance_s and self.clock:
                self.clock.t += self.create_advance_s
            if self.create_creates:
                self.endpoints["ep-1"] = name
            if self.create_rc != 0:
                return proc(self.create_rc, "", self.create_stderr)
            out = self.create_stdout if self.create_stdout is not None else '{"resource_id":"ep-1"}'
            return proc(0, out)
        if verb == "get":
            i = min(self._gets, len(self.states) - 1)
            self._gets += 1
            body = {"status": {"state": self.states[i], "public_endpoints": ["ep-1.example.test"]}}
            return proc(0, json.dumps(body))
        if verb == "list":
            if self.list_rc != 0:
                return proc(self.list_rc, "", "list boom")
            if self._deleted_pending > 0:
                self._deleted_pending -= 1
                shown = {**self.endpoints, "ep-1": self.endpoints.get("ep-1", "t-ep")}
            else:
                shown = self.endpoints
            rows = [{"metadata": {"id": i, "name": n}} for i, n in shown.items()]
            return proc(0, json.dumps({"items": rows}))
        if verb == "delete":
            if self.delete_rc != 0:
                return proc(self.delete_rc, "", f"denied {TOKEN}")
            self.endpoints.pop(a[a.index("--id") + 1], None)
            self._deleted_pending = self.gone_after_list
            return proc(0)
        raise AssertionError(f"unexpected verb {verb}")
