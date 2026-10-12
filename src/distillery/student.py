"""Student model serving: interface only.

The live path serves the fine-tuned Qwen3-1.7B LoRA adapter (and the un-tuned base) with
``peft`` on CPU inside Nebius Sandboxes: ``sandbox_student.SandboxCpuStudent``. Only the
interface and the fake live here.

``StudentServer.generate`` returns plain text. A server that also measures how confident the model
was (the mean token log-probability of the answer, the certified router's ``mean_logprob``
signal) additionally offers ``generate_detailed`` -> ``StudentOutput``; use the module-level
``generate_detailed`` helper, which falls back to text-only outputs for servers without it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

ChatMessages = Sequence[dict[str, Any]]


class StudentServingError(RuntimeError):
    """A serving backend failed (setup, generation, endpoint lifecycle). Never swallowed.

    ``transient=True``: the sandbox lost the work (timeout, exit -1, transport error) and a later
    attempt can succeed. ``False`` (default): the model/script itself failed (bad adapter, OOM,
    too many per-sample errors); a retry would fail the same way."""

    def __init__(self, message: str, *, transient: bool = False) -> None:
        super().__init__(message)
        self.transient = transient


def text_sha256(text: str) -> str:
    """sha256 (hex) of the UTF-8 text; the ``output_sha256`` of a generation."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StudentOutput:
    """One generation with the confidence signals of the answer.

    ``mean_logprob`` / ``min_logprob`` are over the ANSWER tokens only (the ``<think>`` block and
    the end-of-sequence/pad tokens are excluded, see ``sandbox_student.answer_logprob_stats``);
    ``None`` when the server does not measure them, or the answer has no tokens.
    ``error`` is set for a sample that failed inside the sandbox (``text`` is then ``""``).
    """

    text: str
    mean_logprob: float | None = None
    min_logprob: float | None = None
    n_answer_tokens: int | None = None
    output_sha256: str = ""
    seconds: float | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.output_sha256:
            object.__setattr__(self, "output_sha256", text_sha256(self.text))


class StudentServer(Protocol):
    def generate(self, messages_batch: Sequence[ChatMessages]) -> list[str]: ...

    def close(self) -> None: ...


@runtime_checkable
class DetailedStudentServer(Protocol):
    """A ``StudentServer`` that also reports per-request confidence signals."""

    def generate(self, messages_batch: Sequence[ChatMessages]) -> list[str]: ...

    def generate_detailed(self, messages_batch: Sequence[ChatMessages]) -> list[StudentOutput]: ...

    def close(self) -> None: ...


def generate_detailed(server: Any, messages_batch: Sequence[ChatMessages]) -> list[StudentOutput]:
    """``server.generate_detailed`` when it has one, else its text outputs with no signals."""
    detailed = getattr(server, "generate_detailed", None)
    if callable(detailed):
        outs = list(detailed(messages_batch))
        if len(outs) != len(messages_batch):
            raise StudentServingError(
                f"generate_detailed returned {len(outs)} outputs for {len(messages_batch)} prompts"
            )
        return outs
    return [StudentOutput(text=str(t)) for t in server.generate(messages_batch)]


def fake_logprobs(text: str) -> tuple[float, float, int]:
    """Deterministic plausible ``(mean_logprob, min_logprob, n_answer_tokens)`` derived from a
    hash of the text (for fakes and dry runs)."""
    h = hashlib.sha256(b"fake-logprob:" + text.encode("utf-8")).digest()
    mean = -0.02 - 1.5 * (int.from_bytes(h[:4], "big") / 0xFFFFFFFF)  # in [-1.52, -0.02]
    low = mean - 0.1 - 3.0 * (int.from_bytes(h[4:8], "big") / 0xFFFFFFFF)  # <= mean
    return round(mean, 6), round(low, 6), max(1, len(text.split()))


class FakeStudent:
    """Deterministic in-memory student for tests. ``generate_detailed`` reports hash-derived
    log-probabilities, or ``logprob(messages, text)`` -> mean log-probability when given."""

    def __init__(
        self,
        responder: Callable[[ChatMessages], str] | None = None,
        logprob: Callable[[ChatMessages, str], float] | None = None,
    ) -> None:
        self._responder = responder or (lambda m: f"fake:{len(m)}")
        self._logprob = logprob
        self.closed = False
        self.calls = 0

    def generate(self, messages_batch: Sequence[ChatMessages]) -> list[str]:
        return [o.text for o in self.generate_detailed(messages_batch)]

    def generate_detailed(self, messages_batch: Sequence[ChatMessages]) -> list[StudentOutput]:
        if self.closed:
            raise RuntimeError("FakeStudent is closed")
        self.calls += 1
        outs: list[StudentOutput] = []
        for m in messages_batch:
            text = self._responder(m)
            mean, low, n = fake_logprobs(text)
            if self._logprob is not None:
                mean = round(float(self._logprob(m, text)), 6)
                low = min(low, mean)
            outs.append(
                StudentOutput(
                    text=text, mean_logprob=mean, min_logprob=low, n_answer_tokens=n, seconds=0.0
                )
            )
        return outs

    def close(self) -> None:
        self.closed = True
