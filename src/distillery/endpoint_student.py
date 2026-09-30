"""Student serving path 2: a dedicated Token Factory endpoint (billed while ``ready``).

Doc sources (spikes/out): ai-models-inference_dedicated-endpoints_{deploy-api,operating,
billing-policy,custom-weights}.md. UNVERIFIED against the real API, and there is a blocker:
serving CUSTOM (fine-tuned) weights is "beta, available on request, contact Support"; the docs
do not say how a LoRA is referenced, so ``model_name`` is an explicit input (never guessed).

What is UNVERIFIED: the create/list response shape beyond ``endpoint_id`` and ``routing_key``,
the name of the status field and the value meaning ready (we read ``status`` == "ready",
case-insensitive), the data-plane URL for regions other than the documented example, and every
price. The hourly price is a REQUIRED explicit parameter; nothing here knows any price.

Lifecycle: the endpoint is created in ``__init__`` under ``budget.paid_resource`` semantics and
deleted by ``close()`` on every path, including a failed wait-for-ready. If deletion fails the
error is raised loudly (the endpoint may still be billing).
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from distillery.budget import Ledger, paid_resource
from distillery.finetune import TrainedArtifact
from distillery.student import ChatMessages, StudentServer, StudentServingError

log = logging.getLogger(__name__)

CONTROL_PLANE_URL = "https://api.tokenfactory.nebius.com"
_FAILED_STATUSES = {"failed", "error", "deleted"}


class EndpointSpec(BaseModel):
    """Everything needed to create an endpoint. Valid ``flavor_name``/``gpu_type``/``region``
    combinations come from ``ControlPlane.templates()`` (the docs' source of truth)."""

    model_config = ConfigDict(frozen=True)

    flavor_name: str
    gpu_type: str
    gpu_count: int = Field(ge=1)
    region: str
    hourly_cost_usd: float = Field(ge=0)  # EXPLICIT: the price is unknown to this codebase
    model_name: str | None = None  # None -> the trained artifact's fine_tuned_model_checkpoint
    base_model_name: str | None = None  # template model for the un-tuned base endpoint
    estimated_hours: float = Field(default=1.0, gt=0)  # for the budget preflight only
    name_prefix: str = "distillery"
    data_plane_base_url: str | None = None  # default follows the doc example pattern (UNVERIFIED)
    ready_timeout_s: float = 1800.0
    poll_interval_s: float = 15.0
    concurrency: int = Field(default=8, ge=1)
    max_tokens: int = Field(default=256, ge=1)

    def data_plane_url(self) -> str:
        return self.data_plane_base_url or f"https://api.tokenfactory.{self.region}.nebius.com/v1"


class ControlPlane(Protocol):
    def templates(self) -> Any: ...
    def create(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def status(self, endpoint_id: str) -> str: ...
    def delete(self, endpoint_id: str) -> None: ...


Send = Callable[[str, str, Mapping[str, str], bytes | None], tuple[int, bytes]]


def _urllib_send(
    method: str, url: str, headers: Mapping[str, str], body: bytes | None
) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=dict(headers), method=method)  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


class HttpControlPlane:
    """``/v0/dedicated_endpoints`` client. The API key is fetched per request, never stored."""

    def __init__(
        self,
        api_key_getter: Callable[[], str],
        base_url: str = CONTROL_PLANE_URL,
        *,
        send: Send = _urllib_send,
    ) -> None:
        self._key = api_key_getter
        self._base = base_url.rstrip("/")
        self._send = send

    def __repr__(self) -> str:
        return f"HttpControlPlane(base_url={self._base!r})"

    def _call(self, method: str, path: str, payload: Mapping[str, Any] | None = None) -> Any:
        headers = {"Authorization": f"Bearer {self._key()}", "Content-Type": "application/json"}
        body = None if payload is None else json.dumps(payload).encode()
        code, raw = self._send(method, f"{self._base}{path}", headers, body)
        if code >= 400:
            raise StudentServingError(f"{method} {path} -> HTTP {code}: {raw[:300]!r}")
        return json.loads(raw) if raw else None

    def templates(self) -> Any:
        return self._call("GET", "/v0/dedicated_endpoints/templates")

    def create(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        out = self._call("POST", "/v0/dedicated_endpoints", payload)
        if not isinstance(out, dict) or "endpoint_id" not in out or "routing_key" not in out:
            raise StudentServingError("create response lacks endpoint_id/routing_key")
        return out

    def status(self, endpoint_id: str) -> str:
        # Only the list route is documented for reads, so filter it (shape UNVERIFIED).
        out = self._call("GET", "/v0/dedicated_endpoints")
        rows = out if isinstance(out, list) else next(
            (out[k] for k in ("endpoints", "data", "items") if isinstance(out, dict) and k in out),
            [],
        )  # fmt: skip
        for row in rows:
            if isinstance(row, dict) and row.get("endpoint_id") == endpoint_id:
                return str(row.get("status", "unknown"))
        return "not_found"

    def delete(self, endpoint_id: str) -> None:
        try:
            self._call("DELETE", f"/v0/dedicated_endpoints/{endpoint_id}")
        except StudentServingError as exc:
            if "HTTP 404" not in str(exc):  # already gone counts as deleted
                raise


@dataclass
class EndpointBackend:
    """What the orchestrator needs to build endpoint students (Deps.endpoint)."""

    control: ControlPlane
    chat_client: Callable[[str], Any]  # data-plane base URL -> duck-typed OpenAI client


class EndpointStudent:
    """``StudentServer`` on a dedicated endpoint alive exactly from ``__init__`` to ``close``."""

    def __init__(
        self,
        control: ControlPlane,
        chat_client: Callable[[str], Any],
        spec: EndpointSpec,
        *,
        model_name: str,
        ledger: Ledger | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        delete_attempts: int = 3,
    ) -> None:
        self._control, self._spec, self._model_name = control, spec, model_name
        self._ledger, self._sleep, self._clock = ledger, sleep, clock
        self._delete_attempts = delete_attempts
        self._closed = False
        self._ready_at: float | None = None
        self._delete_error: str | None = None
        self.billed_usd = 0.0
        if ledger is not None:  # refuse before anything exists
            ledger.preflight(spec.hourly_cost_usd * spec.estimated_hours)
        payload = {
            "name": f"{spec.name_prefix}-{uuid.uuid4().hex[:8]}",
            "description": "Distillery evaluation endpoint (auto-created, auto-deleted)",
            "model_name": model_name,
            "flavor_name": spec.flavor_name,
            "gpu_type": spec.gpu_type,
            "gpu_count": spec.gpu_count,
            "region": spec.region,
            "scaling": {"min_replicas": 1, "max_replicas": 1},
        }
        self._cm: AbstractContextManager[Mapping[str, Any]] = paid_resource(
            lambda: control.create(payload), self._delete
        )
        created = self._cm.__enter__()
        self.endpoint_id = str(created["endpoint_id"])
        self._routing_key = str(created["routing_key"])
        try:
            self._client = chat_client(spec.data_plane_url())
            self._wait_ready()
        except BaseException:
            self._teardown()  # never leave a created endpoint behind
            raise

    @property
    def served_model(self) -> str:
        """The model name this endpoint was created with (recorded in the report)."""
        return self._model_name

    def __repr__(self) -> str:
        return f"EndpointStudent(endpoint_id={self.endpoint_id!r}, model={self._model_name!r})"

    # ---- lifecycle -------------------------------------------------------
    def _delete(self, _created: Mapping[str, Any]) -> None:
        last: Exception | None = None
        for attempt in range(self._delete_attempts):
            try:
                self._control.delete(self.endpoint_id)
                return
            except StudentServingError as exc:
                last = exc
                if attempt + 1 < self._delete_attempts:
                    self._sleep(2.0**attempt)
        self._delete_error = str(last)
        raise StudentServingError(str(last))

    def _wait_ready(self) -> None:
        deadline = self._clock() + self._spec.ready_timeout_s
        while True:
            status = self._control.status(self.endpoint_id).lower()
            if status == "ready":
                break
            if status in _FAILED_STATUSES:
                raise StudentServingError(f"endpoint {self.endpoint_id} status {status!r}")
            if self._clock() >= deadline:
                raise StudentServingError(
                    f"endpoint {self.endpoint_id} not ready after {self._spec.ready_timeout_s}s "
                    f"(last status {status!r})"
                )
            self._sleep(self._spec.poll_interval_s)
        self._ready_at = self._clock()  # billing starts at ready (billing-policy.md)
        while True:  # inference returns 404 until routable
            try:
                self._chat([{"role": "user", "content": "ping"}], max_tokens=1)
                return
            except StudentServingError as exc:
                if self._clock() >= deadline or "404" not in str(exc):
                    raise
                self._sleep(self._spec.poll_interval_s)

    def _teardown(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._cm.__exit__(None, None, None)  # paid_resource: delete, errors only logged
        finally:
            if self._ready_at is not None:
                hours = max(0.0, self._clock() - self._ready_at) / 3600.0
                self.billed_usd = hours * self._spec.hourly_cost_usd
                if self._ledger is not None:
                    self._ledger.record("endpoint", self._model_name, self.billed_usd)
        if self._delete_error is not None:
            raise StudentServingError(
                f"FAILED to delete endpoint {self.endpoint_id} (it may still be billing): "
                f"{self._delete_error}"
            )

    def close(self) -> None:
        self._teardown()

    # ---- generation ------------------------------------------------------
    def _chat(self, messages: ChatMessages, *, max_tokens: int) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self._routing_key,
                messages=[dict(m) for m in messages],
                temperature=0,
                max_tokens=max_tokens,
            )
            return str(resp.choices[0].message.content or "")
        except Exception as exc:  # noqa: BLE001 - foreign SDK errors are re-raised typed
            raise StudentServingError(f"chat completion failed: {exc}") from exc

    def generate(self, messages_batch: Sequence[ChatMessages]) -> list[str]:
        if self._closed:
            raise RuntimeError("EndpointStudent is closed")
        with ThreadPoolExecutor(max_workers=self._spec.concurrency) as pool:
            return list(
                pool.map(lambda m: self._chat(m, max_tokens=self._spec.max_tokens), messages_batch)
            )


def make_student_factory(
    backend: EndpointBackend,
    spec: EndpointSpec,
    ledger: Ledger | None = None,
    **clock_kw: Any,  # sleep=/clock= overrides, for tests
) -> Callable[[TrainedArtifact], StudentServer]:
    def factory(trained: TrainedArtifact) -> StudentServer:
        name = spec.model_name or trained.fine_tuned_model_checkpoint
        if not name:
            raise StudentServingError(
                "no endpoint model name: set EndpointSpec.model_name (how custom weights are "
                "referenced is undocumented; ask Support)"
            )
        return EndpointStudent(
            backend.control, backend.chat_client, spec, model_name=name, ledger=ledger, **clock_kw
        )

    return factory


def make_base_factory(
    backend: EndpointBackend,
    spec: EndpointSpec,
    ledger: Ledger | None = None,
    **clock_kw: Any,  # sleep=/clock= overrides, for tests
) -> Callable[[], StudentServer]:
    def factory() -> StudentServer:
        if not spec.base_model_name:
            raise StudentServingError("EndpointSpec.base_model_name is required to serve the base")
        return EndpointStudent(
            backend.control, backend.chat_client, spec, model_name=spec.base_model_name,
            ledger=ledger, **clock_kw,
        )  # fmt: skip

    return factory
