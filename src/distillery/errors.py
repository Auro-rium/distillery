"""Run-level error classes: is a failure worth an unattended retry, or is it final?

The supervisor (``server/worker.py``) restarts a run only when the CLI exits with
``EXIT_RETRYABLE`` (75, ``EX_TEMPFAIL``). Everything here decides which exceptions earn that code.

Rules (plan A2):
* retryable: provider/network trouble that a later attempt can get past without changing anything
  (HTTP 429/5xx, timeouts, connection drops, LLM retries exhausted, sandbox transport errors,
  lost or timed-out sandbox jobs, a fine-tune job that was unreachable while polled);
* terminal: anything a retry would hit again (budget caps, refusals, a failed verifier self-test,
  no headroom, a fine-tune job that FAILED, 4xx, artifact or held-out integrity errors).

Wrappers hide the original type (``SandboxError(...) from exc``, ``TerminalLLMError from
APIStatusError``), so the cause chain is consulted after the explicit classes.
"""

from __future__ import annotations

from typing import Literal

Kind = Literal["retryable", "terminal"]

EXIT_OK, EXIT_FAIL, EXIT_REFUSED = 0, 1, 2
EXIT_RETRYABLE = 75  # EX_TEMPFAIL (sysexits.h): the supervisor restarts the same argv


class RetryableRunError(RuntimeError):
    """The current attempt cannot finish, but a resumed attempt can (stage cache + job adoption).
    Raised instead of producing a verdict from infrastructure failures."""


class InfraError(RetryableRunError):
    """Sandbox infrastructure kept failing for a batch after bounded in-process retries."""


class RunSuspended(BaseException):  # noqa: N818 - a control-flow signal, like KeyboardInterrupt
    """SIGTERM (server shutdown or redeploy): stop now WITHOUT cancelling paid jobs, so the next
    attempt adopts them. Derives from BaseException so ``except Exception`` cannot swallow it."""


def _explicit(exc: BaseException) -> Kind | None:
    """Classification by the exception's own type, or None when the type alone does not say."""
    import openai

    from distillery.budget import BudgetExceeded, UnknownPriceError
    from distillery.evaluator import ArtifactMismatchError
    from distillery.finetune import JobFailedError, PollTimeoutError
    from distillery.humanset import HumanSetError
    from distillery.llm import RetriesExhaustedError, TerminalLLMError, is_retryable
    from distillery.orchestrator import (
        ConfigRefusal,
        ShortfallError,
        TaskTooEasyError,
        VerifierSelfTestError,
    )
    from distillery.sandbox import SandboxTimeoutError, _transient_error_types
    from distillery.store import HeldoutIntegrityError
    from distillery.student import StudentServingError

    terminal: tuple[type[BaseException], ...] = (
        BudgetExceeded, UnknownPriceError, ConfigRefusal, VerifierSelfTestError, TaskTooEasyError,
        ShortfallError, JobFailedError, TerminalLLMError, ArtifactMismatchError,
        HeldoutIntegrityError, HumanSetError, KeyboardInterrupt,
    )  # fmt: skip
    if isinstance(exc, terminal):
        return "terminal"
    if isinstance(exc, RetryableRunError | RetriesExhaustedError | SandboxTimeoutError):
        return "retryable"
    if isinstance(exc, _transient_error_types()):
        return "retryable"
    if isinstance(exc, PollTimeoutError):
        return "retryable" if exc.unreachable else "terminal"
    if isinstance(exc, StudentServingError):
        return "retryable" if exc.transient else "terminal"
    if isinstance(exc, openai.APIStatusError | openai.APIConnectionError):
        return "retryable" if is_retryable(exc) else "terminal"
    return None


def chain(exc: BaseException) -> list[BaseException]:
    """``exc`` then its causes/contexts, outermost first (cycle-safe)."""
    out: list[BaseException] = []
    cur: BaseException | None = exc
    while cur is not None and all(cur is not seen for seen in out):
        out.append(cur)
        cur = cur.__cause__ or cur.__context__
    return out


def is_known(exc: BaseException) -> bool:
    """True when some exception in the chain has an explicit classification (not a bug)."""
    return any(_explicit(e) is not None for e in chain(exc))


def classify(exc: BaseException) -> Kind:
    """``"retryable"`` or ``"terminal"`` for a run-level failure. Unknown types are terminal:
    a restart loop on a bug would burn the restart budget and hide the traceback."""
    if isinstance(exc, RunSuspended):
        return "terminal"  # the supervisor resumes a suspension itself; never count it a retry
    for e in chain(exc):  # the outermost explicit verdict wins: wrappers state intent
        if (kind := _explicit(e)) is not None:
            return kind
    return "terminal"


__all__ = [
    "EXIT_FAIL",
    "EXIT_OK",
    "EXIT_REFUSED",
    "EXIT_RETRYABLE",
    "InfraError",
    "Kind",
    "RetryableRunError",
    "RunSuspended",
    "chain",
    "classify",
    "is_known",
]
