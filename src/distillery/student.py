"""Student model serving: interface only. The serving path is NOT chosen yet.

Phase 0 spike S4 must pick one of three candidate paths for serving the fine-tuned
Qwen3-1.7B LoRA adapter, based on what the docs/live platform actually allow:

1. Sandbox-CPU with ``peft``: load base model + downloaded LoRA adapter inside a ConTree
   sandbox and generate on CPU. Cheap and always available, but slow; tests batching.
2. Dedicated endpoint: deploy via ``POST /v0/dedicated_endpoints`` (billed while >=1 replica
   runs). Docs say custom fine-tuned weights are beta / on request only
   (ai-models-inference_dedicated-endpoints_custom-weights.md), so this may need support.
3. Serverless job: some managed batch/serverless inference for the adapter, if the platform
   offers one (not found in the saved docs).

Implementations (all UNVERIFIED against the real APIs): ``sandbox_student.SandboxCpuStudent``
(path 1) and ``endpoint_student.EndpointStudent`` (path 2). Only the interface and the fake
live here.
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
