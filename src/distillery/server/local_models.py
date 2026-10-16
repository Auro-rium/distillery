"""Base and student served for the playground on Nebius sandbox CPU (opt-in).

``LocalModels`` is what the playground consumes; tests inject fakes into it. ``from_env`` builds the
real thing: nothing touches the network until the first question that needs it (the sandbox, the
image and the connection are created lazily), so starting the server stays free and offline.

# UNVERIFIED live in this module: the wiring below is exercised only with fakes. The pieces it
# uses (SandboxCpuStudent, SandboxExecutor) were measured live in the S4/S2 spikes.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from distillery.config import Config
from distillery.finetune import sha256_file
from distillery.student import (
    ChatMessages,
    StudentOutput,
    StudentServer,
    StudentServingError,
    generate_detailed,
)
from distillery.taskpacks.sql.executor import Executor

ENV_ENABLE = "DISTILLERY_PLAYGROUND_SANDBOX_MODELS"
ENV_STUDENT_RUN = "DISTILLERY_PLAYGROUND_STUDENT_RUN"
ENV_STUDENT_DIR = "DISTILLERY_PLAYGROUND_STUDENT_DIR"
SERVED_MANIFEST = "served.json"
DISABLED_REASON = (
    f"sandbox-served base/student are switched off on this server (set {ENV_ENABLE}=1 to enable; "
    "sandbox CPU price is not known, so this is opt-in and rate limited)"
)


@dataclass(frozen=True)
class ServedAdapter:
    """A trained checkpoint shipped with the server (``deploy/student/``): every file, the digest
    over all of them (``TrainedArtifact.adapter_sha256``'s formula) and the base it was trained on.
    """

    files: tuple[Path, ...]
    adapter_sha256: str
    base_model: str


def load_served_adapter(directory: Path) -> ServedAdapter:
    """Read ``served.json`` and check the directory against it. Raises ``ValueError`` with the
    reason when a file is missing, extra or changed, or the base model disagrees: a student whose
    bytes are not the certified ones must never be served under that certificate."""
    d = Path(directory)
    try:
        doc = json.loads((d / SERVED_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read {SERVED_MANIFEST} in {d}: {exc}") from None
    want, base = doc.get("adapter_sha256"), doc.get("base_model")
    if not (isinstance(want, str) and want and isinstance(base, str) and base):
        raise ValueError(f"{SERVED_MANIFEST} needs adapter_sha256 and base_model")
    files = tuple(sorted(p for p in d.iterdir() if p.is_file() and p.name != SERVED_MANIFEST))
    h = hashlib.sha256()
    for f in files:  # sorted by name, as TrainedArtifact.adapter_sha256 does
        digest = sha256_file(f)
        h.update(f"{f.name}:{digest}\n".encode())
    if h.hexdigest() != want:
        raise ValueError(f"the files in {d} do not hash to the adapter_sha256 in {SERVED_MANIFEST}")
    try:
        cfg = json.loads((d / "adapter_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read adapter_config.json in {d}: {exc}") from None
    if not isinstance(cfg, dict) or cfg.get("base_model_name_or_path") != base:
        raise ValueError(f"adapter_config.json was not trained on {base!r}")
    return ServedAdapter(files, want, base)


@dataclass
class LocalModels:
    """What the playground needs to serve the base and the student on sandbox CPU."""

    enabled: bool = True
    disabled_reason: str = DISABLED_REASON
    base: StudentServer | None = None
    base_reason: str | None = "base model is not configured on this server"
    # adapter files (config + weights, already checked against the run) -> a student server.
    # Must raise StudentServingError (or ValueError) when the adapter does not fit the base model.
    student_factory: Callable[[Sequence[Path]], StudentServer] | None = None
    student_run: str | None = None
    student_unset_reason: str = f"no student run configured (set {ENV_STUDENT_RUN}=<run id>)"
    # a checkpoint shipped with the server; when set it is served instead of ``student_run``
    served: ServedAdapter | None = None
    # runs the model-written SQL (sandbox); None = the in-process read-only runner
    executor: Executor | None = None

    def close(self) -> None:
        if self.base is not None:
            self.base.close()


class _Lazy:
    """A ``StudentServer`` that is built on first use (so no network at startup)."""

    def __init__(self, build: Callable[[], StudentServer]) -> None:
        self._build = build
        self._inner: StudentServer | None = None
        self._lock = threading.Lock()
        self._closed = False

    def generate(self, messages_batch: Sequence[ChatMessages]) -> list[str]:
        with self._lock:
            if self._closed:
                raise StudentServingError("closed")
            if self._inner is None:
                self._inner = self._build()
            inner = self._inner
        return inner.generate(messages_batch)

    def generate_detailed(self, messages_batch: Sequence[ChatMessages]) -> list[StudentOutput]:
        with self._lock:
            if self._closed:
                raise StudentServingError("closed")
            if self._inner is None:
                self._inner = self._build()
            inner = self._inner
        return generate_detailed(inner, messages_batch)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            inner, self._inner = self._inner, None
        if inner is not None:
            inner.close()


class _LazyExecutor:
    def __init__(self, build: Callable[[], Executor]) -> None:
        self._build = build
        self._inner: Executor | None = None
        self._lock = threading.Lock()

    def run_batch(self, db_ref: str, sqls: Sequence[str]) -> Any:
        with self._lock:
            inner = self._inner
            if inner is None:
                inner = self._inner = self._build()
        return inner.run_batch(db_ref, sqls)


def from_env(
    config: Config, env: Mapping[str, str], *, demo_db: Callable[[], bytes]
) -> LocalModels:
    if (env.get(ENV_ENABLE) or "").strip() != "1":
        return LocalModels(enabled=False)
    student_run = (env.get(ENV_STUDENT_RUN) or "").strip() or None
    key, project = config.nebius_api_key, config.nebius_project_id
    base_model = config.model_ids.get("student")
    served: ServedAdapter | None = None
    if student_dir := (env.get(ENV_STUDENT_DIR) or "").strip():
        try:
            served = load_served_adapter(Path(student_dir))
        except ValueError as exc:
            why = f"shipped student rejected: {exc}"[:300]
            return LocalModels(base_reason=why, student_unset_reason=why)
        base_model = served.base_model  # the adapter fixes its base, not DISTILLERY_MODEL_STUDENT
    if key is None or project is None or not base_model:
        why = (
            "the sandbox is not configured on this server (needs NEBIUS_API_KEY, "
            "NEBIUS_AI_PROJECT and a student model id)"
        )
        return LocalModels(base_reason=why, student_run=student_run, student_unset_reason=why)
    from distillery.sandbox import ContreeSandbox
    from distillery.sandbox_executor import AsyncBridge, SandboxExecutor
    from distillery.sandbox_student import SandboxCpuStudent, ServingImages, prepare_adapter

    secret, proj = key.get_secret_value(), project.get_secret_value()
    lock = threading.Lock()
    shared: dict[str, Any] = {}

    def parts() -> tuple[Any, Any, str, ServingImages]:
        with lock:
            if not shared:
                from distillery.cli import SANDBOX_BASE_IMAGE

                bridge = AsyncBridge()
                sandbox = ContreeSandbox(
                    lambda: secret, env.get("DISTILLERY_SANDBOX_URL") or None, project_id=proj
                )
                image = bridge.run(
                    sandbox.ensure_image(env.get("DISTILLERY_SANDBOX_IMAGE") or SANDBOX_BASE_IMAGE)
                )
                shared.update(
                    bridge=bridge, sandbox=sandbox, image=image,
                    images=ServingImages(sandbox, image, bridge, base_model=base_model),
                )  # fmt: skip
            return shared["sandbox"], shared["bridge"], shared["image"], shared["images"]

    def make(files: Sequence[Path]) -> StudentServer:
        sandbox, bridge, image, images = parts()
        # a person is waiting: one short retry of a lost job and no timeout re-runs; a failure
        # becomes an error output, which the gateway hands to the teacher
        return SandboxCpuStudent(
            sandbox, image, bridge, base_model=base_model, adapter_files=files, images=images,
            batch_size=1, lost_job_waits=(2.0,), timeout_retries=0,
        )  # fmt: skip

    def student_factory(files: Sequence[Path]) -> StudentServer:
        prepare_adapter(files, base_model)  # validate now, before anything is built or billed
        return _Lazy(lambda: make(files))

    def executor() -> Executor:
        sandbox, bridge, image, _ = parts()
        ex = SandboxExecutor(sandbox, image, bridge=bridge)
        ex.register("playground-demo", demo_db())
        return ex

    return LocalModels(
        base=_Lazy(lambda: make(())),
        base_reason=None,
        student_factory=student_factory,
        student_run=student_run,
        served=served,
        executor=_LazyExecutor(executor),
    )
