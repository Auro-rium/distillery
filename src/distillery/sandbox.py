"""Sandbox abstraction (Nebius Sandboxes / ConTree SDK) with an in-process fake.

Doc sources (spikes/out): sandboxes_sdk_python_sdk_getting-started.md,
running-commands.md, branching.md, images.md, reference_image.md, sandboxes_overview.md.

Key doc facts used here:
* ``image.run(shell=|command=, args, env, cwd, stdin, files, timeout, disposable, ...)``;
  result has ``stdout``, ``stderr``, ``exit_code``, ``uuid`` (the new image version).
* ``disposable=False`` keeps the resulting image (checkpoint/branch primitive); running
  the same command on the same image yields the same uuid (caching).
* Beta cap: 50 simultaneously running operations.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

MAX_CONCURRENCY = 40  # our ceiling; the documented beta cap is 50 in-flight operations
BETA_INFLIGHT_CAP = 50
DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/sandboxes"
_TIMEOUT_GRACE_S = 5.0
# The SDK truncates stdout/stderr at 65535 bytes by default, which would cut a JSON result set
# mid-way (the S4 spike already needed 400000). Ask for more explicitly on every run.
TRUNCATE_OUTPUT_AT = 1_000_000
# Transient transport failures (SDK ``ApiTimeoutError`` / ``ContreeTransportError``):
# * POLL_RETRIES: consecutive failed polls of one operation status that are tolerated before the
#   error is let through (polling is a GET by operation uuid, so repeating it is idempotent);
# * RUN_RETRIES: extra whole-run attempts for DISPOSABLE runs only (no image is kept, so a repeat
#   has no side effect beyond compute). Branch (non-disposable) runs are never resubmitted.
POLL_RETRIES = 6
RUN_RETRIES = 2
RETRY_BACKOFF_S = 1.0
_BACKOFF_CAP_S = 8.0

FileSource = bytes | str | Path


class SandboxError(Exception):
    """Base class for sandbox failures we surface."""


class SandboxTimeoutError(SandboxError):
    """A run exceeded its timeout."""


@dataclass(frozen=True)
class RunResult:
    stdout: str
    stderr: str
    exit_code: int
    # UUID of the image produced by the run. # UNVERIFIED: whether disposable runs return
    # a uuid at all; the fake returns None for them and consumers must not rely on it.
    image_uuid: str | None
    timed_out: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and self.error is None


@dataclass(frozen=True)
class Job:
    """One unit of work for :meth:`Sandbox.run_batch` (always disposable)."""

    shell: str | None = None
    command: str | None = None
    args: tuple[str, ...] = ()
    files: Mapping[str, FileSource] | None = None
    stdin: str | bytes | None = None
    timeout: float | None = None

    def __post_init__(self) -> None:
        if (self.shell is None) == (self.command is None):
            raise ValueError("exactly one of shell / command is required")


class Sandbox(Protocol):
    async def run(
        self,
        image_ref: str,
        *,
        shell: str | None = None,
        command: str | None = None,
        args: Sequence[str] = (),
        files: Mapping[str, FileSource] | None = None,
        stdin: str | bytes | None = None,
        timeout: float | None = None,
        disposable: bool = True,
    ) -> RunResult: ...

    async def branch(
        self,
        image_ref: str,
        setup_shell: str,
        *,
        files: Mapping[str, FileSource] | None = None,
        timeout: float | None = None,
    ) -> str:
        """Non-disposable run; returns the new image uuid (the checkpoint)."""
        ...

    async def run_batch(
        self, image_ref: str, jobs: Sequence[Job], concurrency: int = MAX_CONCURRENCY
    ) -> list[RunResult]: ...

    async def ensure_image(self, ref: str) -> str:
        """Import ``ref`` (e.g. ``docker://python:3.12-slim``) if needed; returns the image ref
        to pass to ``run``/``branch``."""
        ...

    def lineage(self) -> list[tuple[str, str | None, str]]:
        """``(uuid, parent_uuid, label)`` for every image this sandbox produced/used."""
        ...


def _check_concurrency(concurrency: int) -> None:
    if not 1 <= concurrency <= MAX_CONCURRENCY:
        raise ValueError(f"concurrency must be in 1..{MAX_CONCURRENCY}, got {concurrency}")


def _label(shell: str | None, command: str | None, args: Sequence[str]) -> str:
    if shell is not None:
        return shell
    return " ".join([command or "", *args]).strip()


class _SandboxBase:
    """Shared batching + lineage bookkeeping."""

    def __init__(self) -> None:
        self._lineage: dict[str, tuple[str | None, str]] = {}

    async def run(  # pragma: no cover - overridden
        self,
        image_ref: str,
        *,
        shell: str | None = None,
        command: str | None = None,
        args: Sequence[str] = (),
        files: Mapping[str, FileSource] | None = None,
        stdin: str | bytes | None = None,
        timeout: float | None = None,
        disposable: bool = True,
    ) -> RunResult:
        raise NotImplementedError

    def _note_root(self, ref: str) -> None:
        self._lineage.setdefault(ref, (None, f"root:{ref}"))

    def _note_child(self, parent: str, child: str, label: str) -> None:
        self._note_root(parent)
        self._lineage.setdefault(child, (parent, label))

    def lineage(self) -> list[tuple[str, str | None, str]]:
        return [(uuid, parent, label) for uuid, (parent, label) in self._lineage.items()]

    async def branch(
        self,
        image_ref: str,
        setup_shell: str,
        *,
        files: Mapping[str, FileSource] | None = None,
        timeout: float | None = None,
    ) -> str:
        result = await self.run(
            image_ref, shell=setup_shell, files=files, timeout=timeout, disposable=False
        )
        if result.error is not None or result.image_uuid is None:
            raise SandboxError(f"branch produced no image (error={result.error!r})")
        if result.exit_code != 0:
            raise SandboxError(f"branch setup exited {result.exit_code}: {result.stderr[:300]!r}")
        return result.image_uuid

    async def run_batch(
        self, image_ref: str, jobs: Sequence[Job], concurrency: int = MAX_CONCURRENCY
    ) -> list[RunResult]:
        """Run disposable jobs with at most ``concurrency`` in flight; order preserved.

        A timeout or sandbox error in one job yields a failed ``RunResult`` for that job
        only; it never aborts its siblings.
        """
        _check_concurrency(concurrency)
        sem = asyncio.Semaphore(concurrency)

        async def one(job: Job) -> RunResult:
            async with sem:
                try:
                    return await self.run(
                        image_ref,
                        shell=job.shell,
                        command=job.command,
                        args=job.args,
                        files=job.files,
                        stdin=job.stdin,
                        timeout=job.timeout,
                        disposable=True,
                    )
                except SandboxTimeoutError as exc:
                    return RunResult("", "", -1, None, timed_out=True, error=str(exc))
                except SandboxError as exc:
                    return RunResult("", "", -1, None, error=str(exc))

        return list(await asyncio.gather(*(one(j) for j in jobs)))


# --------------------------------------------------------------------------- fake


@dataclass
class FakeExecution:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    duration_s: float = 0.0


@dataclass
class _FakeImage:
    files: dict[str, bytes] = field(default_factory=dict)


Handler = Callable[[Job, dict[str, bytes]], FakeExecution]


def default_handler(job: Job, fs: dict[str, bytes]) -> FakeExecution:
    """Tiny deterministic interpreter: ``echo``, ``cat``, ``> / >>`` redirects, ``exit N``.

    Commands are split on ``&&`` and ``;``. Unknown commands succeed with no output.
    """
    script = job.shell if job.shell is not None else " ".join([job.command or "", *job.args])
    out: list[str] = []
    code = 0
    for raw in script.replace("&&", ";").split(";"):
        part = raw.strip()
        if not part:
            continue
        if part.startswith("exit"):
            tail = part[4:].strip()
            code = int(tail) if tail.isdigit() else 0
            break
        if part.startswith("echo "):
            body = part[5:]
            target, append = None, False
            if ">>" in body:
                body, target = (s.strip() for s in body.split(">>", 1))
                append = True
            elif ">" in body:
                body, target = (s.strip() for s in body.split(">", 1))
            text = body.strip().strip("'\"") + "\n"
            if target is None:
                out.append(text)
            else:
                old = fs.get(target, b"") if append else b""
                fs[target] = old + text.encode()
        elif part.startswith("cat "):
            path = part[4:].strip()
            if path in fs:
                out.append(fs[path].decode())
            else:
                return FakeExecution("".join(out), f"cat: {path}: No such file\n", 1)
    return FakeExecution("".join(out), "", code)


class FakeSandbox(_SandboxBase):
    """In-process deterministic sandbox with lineage tracking.

    Mirrors documented semantics: non-disposable runs create a child image whose uuid is a
    pure function of (parent uuid, command, files, stdin), so identical commands on the same
    image give identical uuids. ``peak_inflight`` records max concurrent runs for tests.
    """

    def __init__(
        self,
        handler: Handler = default_handler,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        super().__init__()
        self._handler = handler
        self._sleep = sleep
        self._images: dict[str, _FakeImage] = {}
        self._inflight = 0
        self.peak_inflight = 0
        self.run_count = 0

    def _resolve(self, ref: str) -> str:
        if ref not in self._images:
            self._images[ref] = _FakeImage()
            self._note_root(ref)
        return ref

    async def ensure_image(self, ref: str) -> str:
        return self._resolve(ref)

    @staticmethod
    def _child_uuid(parent: str, job: Job, disposable_flag: bool) -> str:
        h = hashlib.sha256()
        h.update(parent.encode())
        h.update(repr((job.shell, job.command, job.args, job.stdin)).encode())
        for name in sorted(job.files or {}):
            src = (job.files or {})[name]
            h.update(name.encode())
            h.update(src if isinstance(src, bytes) else str(src).encode())
        h.update(b"keep" if not disposable_flag else b"drop")
        return h.hexdigest()[:32]

    async def run(
        self,
        image_ref: str,
        *,
        shell: str | None = None,
        command: str | None = None,
        args: Sequence[str] = (),
        files: Mapping[str, FileSource] | None = None,
        stdin: str | bytes | None = None,
        timeout: float | None = None,
        disposable: bool = True,
    ) -> RunResult:
        job = Job(
            shell=shell,
            command=command,
            args=tuple(args),
            files=files,
            stdin=stdin,
            timeout=timeout,
        )
        parent = self._resolve(image_ref)
        fs = dict(self._images[parent].files)
        for dest, src in (files or {}).items():
            data = (
                src
                if isinstance(src, bytes)
                else (
                    src.encode()
                    if isinstance(src, str) and not Path(src).exists()
                    else Path(src).read_bytes()
                )
            )
            fs["/" + dest.lstrip("/")] = data
        self.run_count += 1
        self._inflight += 1
        self.peak_inflight = max(self.peak_inflight, self._inflight)
        try:
            execution = self._handler(job, fs)
            try:
                await asyncio.wait_for(self._sleep(execution.duration_s), timeout=timeout)
            except TimeoutError:
                raise SandboxTimeoutError(f"run exceeded timeout={timeout}s") from None
        finally:
            self._inflight -= 1
        new_uuid: str | None = None
        if not disposable:
            new_uuid = self._child_uuid(parent, job, disposable)
            if new_uuid not in self._images:
                self._images[new_uuid] = _FakeImage(fs)
            self._note_child(parent, new_uuid, _label(shell, command, args))
        return RunResult(execution.stdout, execution.stderr, execution.exit_code, new_uuid)


# --------------------------------------------------------------------------- real


class ContreeSandbox(_SandboxBase):
    """Real sandbox via the Nebius ConTree SDK (``contree-sdk`` 0.3.6). Imports are lazy.

    Auth follows the published SDK, not the saved docs: ``Contree(ContreeConfig(auth=IAMAuth(
    token=..., project_id=..., base_url=...)))``. The live API needs a Project header, whose id
    comes from ``project_id`` or the env var ``NEBIUS_AI_PROJECT`` (read at connect time).

    # UNVERIFIED live: only exercised against a stub SDK (project id was not available).
    # UNVERIFIED: exception types raised on timeout/failure; we enforce our own
    # ``asyncio.wait_for`` (timeout + grace) as a backstop.
    """

    def __init__(
        self,
        api_key_getter: Callable[[], str],
        base_url: str | None = None,
        *,
        project_id: str | None = None,
        max_inflight: int = BETA_INFLIGHT_CAP,
        sdk: Any | None = None,
        truncate_output_at: int = TRUNCATE_OUTPUT_AT,
        poll_retries: int = POLL_RETRIES,
        run_retries: int = RUN_RETRIES,
        retry_backoff_s: float = RETRY_BACKOFF_S,
    ) -> None:
        super().__init__()
        self._poll_retries = poll_retries
        self._run_retries = run_retries
        self._retry_backoff_s = retry_backoff_s
        self._hardened = False
        self._truncate_output_at = truncate_output_at
        self._sdk = sdk
        self._api_key_getter = api_key_getter  # key is fetched only at connect time
        self._base_url = base_url or DEFAULT_BASE_URL
        self._project_id = project_id
        self._global_sem = asyncio.Semaphore(max_inflight)

    def __repr__(self) -> str:  # never includes the key or the project id
        return f"ContreeSandbox(base_url={self._base_url!r})"

    def _get_sdk(self) -> Any:
        if self._sdk is None:
            project = (
                self._project_id
                or os.environ.get("NEBIUS_PROJECT_ID")  # name used by the official docs/SDK
                or os.environ.get("NEBIUS_AI_PROJECT")
            )
            if not project:
                raise SandboxError(
                    "no Sandboxes project id: set NEBIUS_PROJECT_ID (or NEBIUS_AI_PROJECT); "
                    "the API rejects requests without a Project header"
                )
            sdk_mod = importlib.import_module("contree_sdk")
            auth_mod = importlib.import_module("contree_sdk.auth")
            cfg_mod = importlib.import_module("contree_sdk.config")
            auth = auth_mod.IAMAuth(
                token=self._api_key_getter(), project_id=project, base_url=self._base_url
            )
            self._sdk = sdk_mod.Contree(cfg_mod.ContreeConfig(auth=auth))
        if not self._hardened:
            self._harden_polling(self._sdk)
            self._hardened = True
        return self._sdk

    def _backoff(self, attempt: int) -> float:
        return min(self._retry_backoff_s * 2**attempt, _BACKOFF_CAP_S)

    def _harden_polling(self, sdk: Any) -> None:
        """Make the SDK's operation poll tolerate transient transport errors.

        contree-sdk 0.3.6 ``image.run`` starts an operation and then calls
        ``_wait_operation(uuid)``, which polls ``_api.get_operation_status(uuid)``. If one poll
        raises ``ApiTimeoutError`` the SDK's ``_operation_canceller`` CANCELS the (possibly
        finished) operation and the error escapes without the uuid. Wrapping the poll call itself
        retries the same uuid in place: a GET, so idempotent, and the operation is never
        resubmitted or cancelled for a blip. Bounded by ``poll_retries`` consecutive failures;
        beyond that the original error propagates (and becomes a ``SandboxError``). Skipped
        quietly if the SDK internals are not as expected (stub SDKs, other versions).
        """
        api = getattr(sdk, "_api", None)
        orig = getattr(api, "get_operation_status", None)
        if api is None or not callable(orig):
            return
        transient = _transient_error_types()
        retries, backoff = self._poll_retries, self._backoff

        async def tolerant(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(retries + 1):
                try:
                    return await orig(*args, **kwargs)
                except transient:
                    if attempt >= retries:
                        raise
                    await asyncio.sleep(backoff(attempt))
            raise AssertionError("unreachable")  # pragma: no cover

        try:
            api.get_operation_status = tolerant
        except (AttributeError, TypeError):  # frozen / slotted client: leave as is
            return

    async def ensure_image(self, ref: str) -> str:
        """Import ``ref`` (e.g. ``python:3.12-slim``) if it is not present; returns its uuid."""
        image = await self._get_sdk().images.oci(ref)
        return str(image.uuid)

    async def run(
        self,
        image_ref: str,
        *,
        shell: str | None = None,
        command: str | None = None,
        args: Sequence[str] = (),
        files: Mapping[str, FileSource] | None = None,
        stdin: str | bytes | None = None,
        timeout: float | None = None,
        disposable: bool = True,
    ) -> RunResult:
        if (shell is None) == (command is None):
            raise ValueError("exactly one of shell / command is required")
        sdk = self._get_sdk()
        self._note_root(image_ref)
        kwargs: dict[str, Any] = {
            "disposable": disposable,
            "truncate_output_at": self._truncate_output_at,
        }
        if shell is not None:
            kwargs["shell"] = shell
        else:
            kwargs["command"] = command
            kwargs["args"] = list(args)
        if stdin is not None:
            kwargs["stdin"] = stdin
        if files:
            # absolute destinations, as the docs' "file.sh -> /file.sh" rule describes
            kwargs["files"] = {"/" + str(dest).lstrip("/"): src for dest, src in files.items()}
        if timeout is not None:
            kwargs["timeout"] = timeout
        contree_error, timed_out_error = _sdk_error_types()
        transient = _transient_error_types()
        # only a disposable run may be repeated whole: it keeps no image, so a repeat is harmless
        attempts = 1 + (self._run_retries if disposable else 0)
        async with self._global_sem:
            backstop = None if timeout is None else timeout + _TIMEOUT_GRACE_S
            for attempt in range(attempts):
                try:
                    image = await sdk.images.use(image_ref)  # no API call per getting-started.md
                    res = await asyncio.wait_for(image.run(**kwargs), timeout=backstop)
                    break
                except TimeoutError:
                    raise SandboxTimeoutError(f"run exceeded timeout={timeout}s") from None
                except timed_out_error as exc:
                    raise SandboxTimeoutError(_short(exc)) from exc
                except transient as exc:
                    if attempt + 1 < attempts:
                        await asyncio.sleep(self._backoff(attempt))
                        continue
                    note = "" if disposable else " (branch run: not resubmitted, not idempotent)"
                    raise SandboxError(_short(exc) + note) from exc
                except contree_error as exc:
                    raise SandboxError(_short(exc)) from exc
        uuid = None if res.uuid is None else str(res.uuid)
        if uuid is not None and not disposable:
            self._note_child(image_ref, uuid, _label(shell, command, args))
        return RunResult(_text(res.stdout), _text(res.stderr), int(res.exit_code), uuid)


def _sdk_error_types() -> tuple[type[BaseException], type[BaseException]]:
    """``(ContreeError, OperationTimedOutError)``; without the SDK nothing is translated.

    ``except ()`` catches nothing, and a class that never gets raised is a safe stand-in.
    """
    try:
        mod = importlib.import_module("contree_sdk.sdk.exceptions")
        return mod.ContreeError, mod.OperationTimedOutError
    except (ImportError, AttributeError):  # stub SDK in tests
        return _Never, _Never


def _transient_error_types() -> tuple[type[BaseException], ...]:
    """SDK transport errors worth retrying: ``ApiTimeoutError`` and ``ContreeTransportError``."""
    try:
        mod = importlib.import_module("contree_sdk.sdk.exceptions")
        return (mod.ApiTimeoutError, mod.ContreeTransportError)
    except (ImportError, AttributeError):  # stub SDK in tests
        return (_Never,)


class _Never(BaseException):  # sentinel that is never raised
    pass


def _short(exc: BaseException, limit: int = 300) -> str:
    return f"{type(exc).__name__}: {str(exc)[:limit]}"


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
