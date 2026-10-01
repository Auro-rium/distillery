"""Student model serving: interface only.

The live path serves the fine-tuned Qwen3-1.7B LoRA adapter (and the un-tuned base) with
``peft`` on CPU inside Nebius Sandboxes: ``sandbox_student.SandboxCpuStudent``. Only the
interface and the fake live here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

ChatMessages = Sequence[dict[str, Any]]


class StudentServingError(RuntimeError):
    """A serving backend failed (setup, generation, endpoint lifecycle). Never swallowed."""


class StudentServer(Protocol):
    def generate(self, messages_batch: Sequence[ChatMessages]) -> list[str]: ...

    def close(self) -> None: ...


class FakeStudent:
    """Deterministic in-memory student for tests."""

    def __init__(self, responder: Callable[[ChatMessages], str] | None = None) -> None:
        self._responder = responder or (lambda m: f"fake:{len(m)}")
        self.closed = False
        self.calls = 0

    def generate(self, messages_batch: Sequence[ChatMessages]) -> list[str]:
        if self.closed:
            raise RuntimeError("FakeStudent is closed")
        self.calls += 1
        return [self._responder(m) for m in messages_batch]

    def close(self) -> None:
        self.closed = True
