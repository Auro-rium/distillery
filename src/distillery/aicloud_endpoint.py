"""H1: a time-boxed vLLM container on a Nebius AI Cloud endpoint, always deleted.

What this module does: builds the ``nebius ai endpoint ...`` commands, creates ONE endpoint that
serves Qwen3-1.7B + the N1 LoRA with vLLM, waits until it is ready, hands the caller a handle
(base URL + token) and, in a ``finally``, deletes it and CONFIRMS via ``list`` that it is gone.
The spend (hourly price x wall seconds from the create call to the confirmed delete) is recorded
in the ledger on every path after the create call, success or failure.

Nothing here talks to the network itself: every CLI call goes through an injected
``CommandRunner`` (the real one wraps ``subprocess``; tests use a fake).

CLI surface (``nebius ai endpoint create --help``, nebius 0.12.282, read 2026-10-13):
``create --image IMG --name N --platform gpu-l40s-d [--preset P] --container-port 8000
[--args "..."] [--volume s3://BUCKET:/path[:ro[:PROFILE]]] [--auth token --token T]
[--on-demand | --preemptible] [--disk-size 100Gi] [--async] [--format json]``.

UNVERIFIED (the orchestrator checks each with ``--help`` before the live run; every argv lives in
exactly one small function so a fix is a one-line change):

* ``nebius ai endpoint get --id ID --format json`` exists (``get_argv``).
* ``nebius ai endpoint list --format json`` exists (``list_argv``).
* ``nebius ai endpoint delete --id ID`` exists (``delete_argv``).
* Where the id sits in the ``create --async`` Operation JSON (``_ID_PATHS``; the help documents
  ``jsonpath={.resource_id}``, we try that first).
* The name of the state field in ``get`` output and its values (``_STATE_PATHS``, ``READY_STATES``,
  ``FAILED_STATES``), and the shape of ``status.public_endpoints`` (strings or objects carrying a
  URL; ``_public_url``).
* The shape of the ``list`` output (a list, or an object holding one under ``items``/``endpoints``;
  ids and names are read from the row or its ``metadata``; ``_rows``/``_row_id``/``_row_name``).
* That ``--volume s3://BUCKET[/prefix]:/adapters/n1:ro`` accepts a key prefix after the bucket.
* That ``--auth token`` makes the managed HTTPS front door require ``Authorization: Bearer T``.
* That the image entrypoint (``vllm/vllm-openai``) takes ``--args`` as its arguments.
"""

from __future__ import annotations

import json
import logging
import re
import secrets
import sqlite3
import subprocess  # noqa: S404 - the real runner is the one place the CLI is spawned
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from distillery.budget import Ledger
from distillery.global_ledger import GlobalLedgerError

log = logging.getLogger(__name__)

KIND_AICLOUD_ENDPOINT = "aicloud_endpoint"  # ledger kind; input_tokens holds wall milliseconds
BASE_MODEL = "Qwen/Qwen3-1.7B"
SERVED_ADAPTER = "n1"
ADAPTER_MOUNT = "/adapters/n1"
CONTAINER_PORT = 8000
DEFAULT_PLATFORM = "gpu-l40s-d"
DEFAULT_MAX_LORA_RANK = 64
REDACTED = "***"

READY_STATES = frozenset({"RUNNING", "READY"})  # UNVERIFIED
FAILED_STATES = frozenset({"ERROR", "FAILED", "ERRORED", "DELETED", "DELETING"})  # UNVERIFIED
_STATE_PATHS = (("status", "state"), ("status", "phase"), ("status", "status"), ("state",))
_ID_PATHS = (("resource_id",), ("metadata", "resource_id"), ("resource", "metadata", "id"))
_ROW_ID_PATHS = (("metadata", "id"), ("id",), ("resource_id",))
_ROW_NAME_PATHS = (("metadata", "name"), ("name",))


class AICloudEndpointError(RuntimeError):
    """Base class: the endpoint lifecycle failed."""


class EndpointStartError(AICloudEndpointError):
    """Create failed, the endpoint went to an error state, or it was not ready in time."""


class EndpointDeleteError(AICloudEndpointError):
    """The endpoint could not be deleted/confirmed gone: it MAY STILL BE BILLING."""

    def __init__(self, message: str, endpoint_id: str | None = None) -> None:
        super().__init__(message)
        self.endpoint_id = endpoint_id


# ---------------------------------------------------------------- plan


class EndpointPlan(BaseModel):
    """Everything the live run needs, validated up front. Frozen; the token is never printed."""

    model_config = ConfigDict(frozen=True)

    vllm_tag: str = Field(min_length=1)  # EXPLICIT: no default, "latest" refused
    hourly_usd: float = Field(gt=0)  # EXPLICIT: no price is known to this codebase
    adapter_s3_source: str  # s3://BUCKET[/prefix] holding adapter_config.json + weights
    adapter_s3_profile: str | None = None
    max_lora_rank: int = Field(default=DEFAULT_MAX_LORA_RANK, ge=1)  # N1 adapter_config.json "r"
    platform: str = DEFAULT_PLATFORM
    preset: str | None = None
    # HARD: create -> confirmed delete. The last ``delete_reserve_minutes`` are kept for stopping
    # the measurement, deleting and confirming, so the endpoint is gone by the end of the box.
    time_box_minutes: float = Field(default=45.0, gt=0)
    delete_reserve_minutes: float = Field(default=5.0, ge=0)
    ready_timeout_minutes: float = Field(default=25.0, gt=0)  # also capped by the time box
    preemptible: bool = False
    disk_size: str = "100Gi"
    name: str = Field(default_factory=lambda: f"distillery-h1-{secrets.token_hex(3)}")
    base_model: str = BASE_MODEL
    served_model_name: str = SERVED_ADAPTER
    auth_token: str = Field(default_factory=lambda: secrets.token_urlsafe(32), repr=False)

    @field_validator("vllm_tag")
    @classmethod
    def _tag(cls, v: str) -> str:
        if v.strip().lower() in ("latest", "") or re.search(r"\s", v):
            raise ValueError("vllm_tag must be an explicit version tag, never 'latest'")
        return v

    @field_validator("adapter_s3_source")
    @classmethod
    def _s3(cls, v: str) -> str:
        if not v.startswith("s3://") or len(v) <= len("s3://") or ":" in v[5:]:
            raise ValueError("adapter_s3_source must look like s3://BUCKET[/prefix]")
        return v

    @model_validator(mode="after")
    def _reserve_fits(self) -> EndpointPlan:
        if self.delete_reserve_minutes >= self.time_box_minutes:
            raise ValueError("delete_reserve_minutes must be shorter than time_box_minutes")
        return self

    @property
    def image(self) -> str:
        return f"vllm/vllm-openai:{self.vllm_tag}"

    @property
    def estimated_cost_usd(self) -> float:
        """The preflight figure: the whole time box at the hourly price."""
        return self.hourly_usd * self.time_box_minutes / 60.0

    def volume(self) -> str:
        parts = [f"{self.adapter_s3_source}:{ADAPTER_MOUNT}:ro"]
        if self.adapter_s3_profile:
            parts.append(self.adapter_s3_profile)
        return ":".join(parts)

    def public_dict(self) -> dict[str, Any]:
        """The plan as it may be printed or written: the token is replaced."""
        d = self.model_dump(exclude={"auth_token"})
        d["auth_token"] = REDACTED
        d["image"] = self.image
        d["estimated_cost_usd"] = self.estimated_cost_usd
        return d


def vllm_args(plan: EndpointPlan) -> str:
    """The single ``--args`` string handed to the vLLM OpenAI server."""
    return (
        f"--model {plan.base_model} --enable-lora "
        f"--lora-modules {plan.served_model_name}={ADAPTER_MOUNT} "
        f"--max-lora-rank {plan.max_lora_rank} --max-logprobs 5 --dtype bfloat16 "
        f"--max-model-len 4096 --port {CONTAINER_PORT}"
    )


# ---------------------------------------------------------------- argv (one function each)

_BASE = ("nebius", "ai", "endpoint")


def create_argv(plan: EndpointPlan) -> list[str]:
    argv = [
        *_BASE, "create",
        "--image", plan.image,
        "--name", plan.name,
        "--platform", plan.platform,
    ]  # fmt: skip
    if plan.preset:
        argv += ["--preset", plan.preset]
    argv += [
        "--container-port", str(CONTAINER_PORT),
        "--args", vllm_args(plan),
        "--volume", plan.volume(),
        "--auth", "token", "--token", plan.auth_token,
        "--preemptible" if plan.preemptible else "--on-demand",
        "--disk-size", plan.disk_size,
        "--async", "--format", "json",
    ]  # fmt: skip
    return argv


def get_argv(endpoint_id: str) -> list[str]:  # UNVERIFIED subcommand
    return [*_BASE, "get", "--id", endpoint_id, "--format", "json"]


def delete_argv(endpoint_id: str) -> list[str]:  # UNVERIFIED subcommand
    return [*_BASE, "delete", "--id", endpoint_id]


def list_argv() -> list[str]:  # UNVERIFIED subcommand
    return [*_BASE, "list", "--format", "json"]


def redact_argv(argv: Sequence[str], token: str | None = None) -> list[str]:
    """``argv`` with the value after ``--token`` (and any literal ``token``) replaced."""
    out: list[str] = []
    hide = False
    for a in argv:
        if hide:
            out.append(REDACTED)
            hide = False
            continue
        out.append(REDACTED if token and a == token else a)
        hide = a == "--token"
    return out


def render_argv(argv: Sequence[str], token: str | None = None) -> str:
    """Shell-pasteable rendering of a REDACTED argv (what plan-only prints)."""
    import shlex

    return " ".join(shlex.quote(a) for a in redact_argv(argv, token))


# ---------------------------------------------------------------- command runner


class CommandRunner(Protocol):
    def run(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        """Run ``argv`` and return the finished process; never raises for a failing command."""
        ...


class SubprocessRunner:
    """Real runner: ``subprocess.run`` with a timeout. Errors become a non-zero return code, and
    the argv (which holds the token) is never logged."""

    def __init__(self, timeout_s: float = 180.0) -> None:
        self.timeout_s = timeout_s

    def run(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(  # noqa: S603 - fixed argv, no shell
                list(argv), capture_output=True, text=True, timeout=self.timeout_s, check=False
            )
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(list(argv), 124, "", f"timeout {self.timeout_s}s")
        except OSError as exc:
            return subprocess.CompletedProcess(list(argv), 127, "", f"cannot run: {exc}")


# ---------------------------------------------------------------- clock


@dataclass(frozen=True)
class Clock:
    """Injected time: ``monotonic`` for durations, ``sleep``, ``now_iso`` for the artifact."""

    monotonic: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    now_iso: Callable[[], str] = lambda: datetime.now(UTC).isoformat(timespec="seconds")  # noqa: E731


# ---------------------------------------------------------------- parsing (UNVERIFIED shapes)


def _dig(obj: Any, path: Sequence[str]) -> Any:
    for key in path:
        if not isinstance(obj, Mapping) or key not in obj:
            return None
        obj = obj[key]
    return obj


def _first(obj: Any, paths: Sequence[Sequence[str]]) -> Any:
    for p in paths:
        v = _dig(obj, p)
        if v not in (None, ""):
            return v
    return None


def _loads(text: str) -> Any:
    try:
        return json.loads(text)
    except ValueError:
        return None


def parse_endpoint_id(stdout: str) -> str | None:
    v = _first(_loads(stdout), _ID_PATHS)
    return str(v) if v is not None else None


def parse_state(stdout: str) -> str | None:
    v = _first(_loads(stdout), _STATE_PATHS)
    return str(v).upper() if v is not None else None


def _public_url(stdout: str) -> str | None:
    pub = _dig(_loads(stdout), ("status", "public_endpoints"))
    if isinstance(pub, (str, Mapping)):
        pub = [pub]
    for item in pub if isinstance(pub, list) else []:
        raw = item
        if isinstance(item, Mapping):
            raw = next((item[k] for k in ("url", "endpoint", "address", "host") if k in item), "")
        text = str(raw).strip().rstrip("/")
        if not text:
            continue
        return text if "://" in text else f"https://{text}"
    return None


def _rows(stdout: str) -> list[Mapping[str, Any]]:
    obj = _loads(stdout)
    if isinstance(obj, Mapping):
        obj = next((obj[k] for k in ("items", "endpoints", "data") if k in obj), [])
    return [r for r in obj if isinstance(r, Mapping)] if isinstance(obj, list) else []


def _row_id(row: Mapping[str, Any]) -> str | None:
    v = _first(row, _ROW_ID_PATHS)
    return str(v) if v is not None else None


def _row_name(row: Mapping[str, Any]) -> str | None:
    v = _first(row, _ROW_NAME_PATHS)
    return str(v) if v is not None else None


# ---------------------------------------------------------------- handle


@dataclass
class EndpointHandle:
    """What the caller measures against; also the lifecycle record written to the artifact.

    ``expired()`` is the watchdog: the measurement must stop once it is True (the context manager
    deletes the endpoint on exit regardless)."""

    endpoint_id: str
    name: str
    base_url: str  # ".../v1"
    token: str = field(repr=False)
    deadline: float  # monotonic time at which measuring must stop (box end minus delete reserve)
    clock: Clock = field(repr=False)
    created_at: str = ""
    ready_at: str = ""
    deleted_at: str = ""
    deleted_confirmed: bool = False
    timed_out: bool = False
    duration_s: float = 0.0
    spend_usd: float = 0.0
    error: str | None = None

    def time_left_s(self) -> float:
        return max(0.0, self.deadline - self.clock.monotonic())

    def expired(self) -> bool:
        return self.clock.monotonic() >= self.deadline

    def lifecycle(self) -> dict[str, Any]:
        """The token-free lifecycle record for the evidence file."""
        return {
            "endpoint_id": self.endpoint_id,
            "name": self.name,
            "base_url": self.base_url,
            "created_at": self.created_at,
            "ready_at": self.ready_at,
            "deleted_at": self.deleted_at,
            "deleted_confirmed": self.deleted_confirmed,
            "timed_out": self.timed_out,
            "duration_s": self.duration_s,
            "spend_usd": self.spend_usd,
            "error": self.error,
        }


ModelProbe = Callable[[EndpointHandle], bool]


# ---------------------------------------------------------------- lifecycle


class _Lifecycle:
    def __init__(self, plan: EndpointPlan, runner: CommandRunner, clock: Clock) -> None:
        self.plan, self.runner, self.clock = plan, runner, clock

    def scrub(self, text: str) -> str:
        return text.replace(self.plan.auth_token, REDACTED).strip()[:500]

    def run(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return self.runner.run(argv)

    def list_rows(self) -> list[Mapping[str, Any]] | None:
        res = self.run(list_argv())
        return _rows(res.stdout or "") if res.returncode == 0 else None

    def find_by_name(self) -> str | None:
        for row in self.list_rows() or []:
            if _row_name(row) == self.plan.name:
                return _row_id(row)
        return None

    def delete_and_confirm(
        self, endpoint_id: str | None, *, attempts: int, confirm_timeout_s: float
    ) -> bool:
        """Delete (retrying), then poll ``list`` until the endpoint is gone. True = confirmed."""
        if endpoint_id is None:  # the create answer was unusable: look for our unique name
            endpoint_id = self.find_by_name()
            if endpoint_id is None:
                rows = self.list_rows()
                return rows is not None  # list worked and our name is absent: nothing exists
        last = ""
        for i in range(attempts):
            res = self.run(delete_argv(endpoint_id))
            if res.returncode == 0:
                break
            last = self.scrub(res.stderr or res.stdout or "")
            log.warning("delete of endpoint %s failed (attempt %d): %s", endpoint_id, i + 1, last)
            if i + 1 < attempts:
                self.clock.sleep(2.0**i)
        deadline = self.clock.monotonic() + confirm_timeout_s
        while True:
            rows = self.list_rows()
            if rows is not None and all(_row_id(r) != endpoint_id for r in rows):
                return True
            if self.clock.monotonic() >= deadline:
                return False
            self.clock.sleep(5.0)


@contextmanager
def aicloud_endpoint(
    plan: EndpointPlan,
    runner: CommandRunner,
    ledger: Ledger,
    clock: Clock | None = None,
    *,
    probe: ModelProbe | None = None,
    on_created: Callable[[str], None] | None = None,
    poll_interval_s: float = 10.0,
    delete_attempts: int = 3,
    confirm_timeout_s: float | None = None,
) -> Iterator[EndpointHandle]:
    """Create the endpoint, yield a ready handle, and ALWAYS delete + confirm in ``finally``.

    ``on_created(id)`` (optional) is called right after the create answered with an id.
    ``probe(handle)`` (optional) is called once the CLI state is ready, until it returns True: the
    caller's check that vLLM actually answers (e.g. ``GET /v1/models`` lists the adapter).
    Raises ``BudgetExceeded`` before anything exists, ``EndpointStartError`` when the endpoint
    never becomes ready, ``EndpointDeleteError`` (with the id) when deletion cannot be confirmed.
    """
    clock = clock or Clock()
    life = _Lifecycle(plan, runner, clock)
    reserve_s = plan.delete_reserve_minutes * 60.0
    if confirm_timeout_s is None:
        confirm_timeout_s = 0.6 * reserve_s if reserve_s else 60.0
    ledger.preflight(plan.estimated_cost_usd)  # before anything exists
    gl = ledger.global_ledger
    reservation: str | None = None
    if gl is not None:  # hold the headroom for the life of the endpoint (other processes see it)
        reservation = gl.reserve(
            "aicloud-endpoint",
            plan.estimated_cost_usd,
            ledger.project_cap,
            ttl_s=plan.time_box_minutes * 60.0 + 3600.0,
            note=plan.name,
        )

    t0 = clock.monotonic()
    deadline = t0 + (plan.time_box_minutes * 60.0 - reserve_s)
    ready_deadline = min(deadline, t0 + plan.ready_timeout_minutes * 60.0)
    created_at = clock.now_iso()
    endpoint_id: str | None = None
    handle: EndpointHandle | None = None
    primary: BaseException | None = None
    confirmed = False
    try:
        res = life.run(create_argv(plan))
        if res.returncode != 0:
            detail = life.scrub(res.stderr or res.stdout or "")
            raise EndpointStartError(f"create failed (exit {res.returncode}): {detail}")
        endpoint_id = parse_endpoint_id(res.stdout or "")
        if endpoint_id is None:
            raise EndpointStartError(
                f"create succeeded but no endpoint id found in its output (name {plan.name})"
            )
        log.info("created AI Cloud endpoint %s (name %s)", endpoint_id, plan.name)
        if on_created is not None:
            on_created(endpoint_id)
        base_url = _wait_ready(life, endpoint_id, ready_deadline, poll_interval_s)
        handle = EndpointHandle(
            endpoint_id=endpoint_id,
            name=plan.name,
            base_url=f"{base_url}/v1",
            token=plan.auth_token,
            deadline=deadline,
            clock=clock,
            created_at=created_at,
        )
        _wait_probe(life, handle, probe, ready_deadline, poll_interval_s)
        handle.ready_at = clock.now_iso()
        yield handle
    except BaseException as exc:
        primary = exc
        raise
    finally:
        body_ended = clock.monotonic()
        try:
            confirmed = life.delete_and_confirm(
                endpoint_id, attempts=delete_attempts, confirm_timeout_s=confirm_timeout_s
            )
        finally:
            duration = max(0.0, clock.monotonic() - t0)
            spend = plan.hourly_usd * duration / 3600.0
            try:
                ledger.record(
                    KIND_AICLOUD_ENDPOINT, None, spend, input_tokens=round(duration * 1000)
                )
            finally:
                _release(gl, reservation)
                if handle is not None:
                    handle.deleted_at = clock.now_iso()
                    handle.deleted_confirmed = confirmed
                    handle.duration_s = duration
                    handle.spend_usd = spend
                    handle.timed_out = body_ended >= deadline
                    handle.error = f"{type(primary).__name__}: {primary}" if primary else None
        if not confirmed:
            shown = endpoint_id or f"name={plan.name} (id unknown)"
            raise EndpointDeleteError(
                f"FAILED to delete/confirm endpoint {shown}: it may still be billing "
                f"(${plan.hourly_usd:.2f}/h). Delete it manually in the console / "
                f"`nebius ai endpoint delete --id ...`."
                + (f" Original error: {primary}" if primary else ""),
                endpoint_id,
            ) from primary


def _release(gl: Any, reservation: str | None) -> None:
    """Drop the project-cap hold once the real spend is recorded (never masks the main error)."""
    if gl is None or reservation is None:
        return
    try:
        gl.release(reservation)
    except (GlobalLedgerError, sqlite3.Error) as exc:
        log.warning("could not release reservation %s: %s", reservation, exc)


def _wait_ready(life: _Lifecycle, endpoint_id: str, ready_deadline: float, poll_s: float) -> str:
    clock = life.clock
    last = "unknown"
    while True:
        res = life.run(get_argv(endpoint_id))
        if res.returncode == 0:
            state = parse_state(res.stdout or "")
            last = state or "no state field"
            if state in FAILED_STATES:
                raise EndpointStartError(f"endpoint {endpoint_id} entered state {state}")
            url = _public_url(res.stdout or "")
            if state in READY_STATES and url:
                return url
        else:
            last = f"get exit {res.returncode}: {life.scrub(res.stderr or '')}"
        if clock.monotonic() >= ready_deadline:
            raise EndpointStartError(
                f"endpoint {endpoint_id} not ready within the time allowed (last: {last})"
            )
        clock.sleep(poll_s)


def _wait_probe(
    life: _Lifecycle,
    handle: EndpointHandle,
    probe: ModelProbe | None,
    ready_deadline: float,
    poll_s: float,
) -> None:
    if probe is None:
        return
    while not probe(handle):
        if life.clock.monotonic() >= ready_deadline:
            raise EndpointStartError(
                f"endpoint {handle.endpoint_id} is running but vLLM never answered the probe"
            )
        life.clock.sleep(poll_s)
