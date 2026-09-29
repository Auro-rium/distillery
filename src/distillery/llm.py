"""Thin role-routed chat client over an OpenAI-compatible API (Nebius Token Factory).

Everything that touches the network is injected (the transport client, sleep, clock,
jitter source, model routing, pricing, usage sink) so the whole module is testable
offline. The real ``openai.AsyncOpenAI`` is only built by :func:`make_openai_client`.

Doc sources (spikes/out): ai-models-inference_json.md (structured output),
ai-models-inference_rate-limits.md (429 / Retry-After), quickstart.md (base_url).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeVar, overload

import openai
from pydantic import BaseModel, ValidationError

Role = Literal["planner", "teacher", "triage", "student"]
ChatMessage = dict[str, Any]
ModelT = TypeVar("ModelT", bound=BaseModel)

MAX_SCHEMA_RETRIES = 2
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


class LLMError(Exception):
    """Base class for errors raised by this module."""


class TerminalLLMError(LLMError):
    """A non-retryable failure (4xx other than 429, refusal, bad config)."""


class RetriesExhaustedError(LLMError):
    """Retryable failures persisted through every allowed attempt."""


class SchemaValidationError(LLMError):
    """The model never produced output matching the requested schema."""


# --------------------------------------------------------------------------- protocols


ChatTransport = Any
"""Duck-typed ``openai.AsyncOpenAI``: only ``client.chat.completions.create(**kw)`` is used."""


@dataclass(frozen=True)
class CallRecord:
    """One logged model call. Never contains the API key or prompt text (hash only)."""

    role: str
    model: str
    purpose: str
    stage: str
    prompt_sha256: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cost_usd: float
    attempt: int
    ok: bool
    error: str | None = None


class UsageSink(Protocol):
    def record_call(self, record: CallRecord) -> None: ...


PricingFn = Callable[[str, int, int], float]
"""``(model_id, input_tokens, output_tokens) -> cost in USD``. Supplied by config/budget."""

SleepFn = Callable[[float], Awaitable[None]]


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 5
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0


@dataclass(frozen=True)
class ChatResult[ModelT2]:
    """Result of a chat call. Only ``text`` / ``parsed`` may be used downstream.

    ``reasoning`` is kept separately purely for logging/debugging.
    """

    text: str
    reasoning: str | None
    parsed: ModelT2 | None
    model: str
    input_tokens: int
    output_tokens: int
    attempts: int = 1
    raw_finish_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- helpers


def prompt_hash(messages: Sequence[Mapping[str, Any]]) -> str:
    blob = json.dumps(list(messages), sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def is_retryable(exc: BaseException) -> bool:
    """Retryable: HTTP 429 / 5xx, timeouts, connection errors. Terminal: other 4xx."""
    if isinstance(exc, openai.APITimeoutError | openai.APIConnectionError):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    return False


def retry_after_seconds(exc: BaseException) -> float | None:
    """Honour the ``Retry-After`` header (seconds) per the rate-limits doc."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        return max(0.0, float(str(raw)))
    except ValueError:
        return None


def backoff_delay(attempt: int, policy: RetryPolicy, jitter: float) -> float:
    """Exponential backoff capped at ``max_delay_s`` with full-ish jitter (0.5x-1.0x)."""
    raw = min(policy.max_delay_s, policy.base_delay_s * (2.0 ** (attempt - 1)))
    return raw * (0.5 + 0.5 * jitter)


def split_reasoning(message: Any) -> tuple[str, str | None]:
    """Separate final content from reasoning.

    # UNVERIFIED: the docs do not say how Nebius returns reasoning. We check, in order,
    # a ``reasoning_content`` / ``reasoning`` field on the message, then inline
    # ``<think>...</think>`` blocks in ``content`` (Qwen3 convention).
    """
    content = getattr(message, "content", None) or ""
    reasoning_parts: list[str] = []
    for attr in ("reasoning_content", "reasoning"):
        val = getattr(message, attr, None)
        if val is None:
            extra = getattr(message, "model_extra", None) or {}
            val = extra.get(attr)
        if isinstance(val, str) and val:
            reasoning_parts.append(val)
            break
    for m in _THINK_RE.finditer(content):
        reasoning_parts.append(m.group(1).strip())
    final = _THINK_RE.sub("", content)
    # An unterminated <think> (truncated output) must never leak into final content.
    if "<think>" in final:
        head, _, tail = final.partition("<think>")
        reasoning_parts.append(tail.strip())
        final = head
    reasoning = "\n".join(p for p in reasoning_parts if p) or None
    return final.strip(), reasoning


def response_format_for(schema: type[BaseModel]) -> dict[str, Any]:
    """Structured-output ``response_format``.

    # UNVERIFIED: ai-models-inference_json.md shows two shapes (the Python sample passes
    # the raw schema as ``json_schema``; the 'valid JSON schema' example and the JS
    # ``zodResponseFormat`` use OpenAI's ``{name, schema, strict}``). We use the latter.
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "schema": schema.model_json_schema(),
            "strict": True,
        },
    }


def make_openai_client(base_url: str, api_key: str, *, timeout_s: float = 120.0) -> ChatTransport:
    """Build the real client. ``max_retries=0``: retry policy lives in :class:`LLMClient`."""
    return openai.AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=0)


# --------------------------------------------------------------------------- client


class LLMClient:
    def __init__(
        self,
        transport: ChatTransport,
        models: Mapping[str, str],
        *,
        sink: UsageSink | None = None,
        pricing: PricingFn | None = None,
        retry: RetryPolicy | None = None,
        sleep: SleepFn = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self._transport = transport
        self._models = dict(models)  # role -> model id, supplied by config; never hardcoded
        self._sink = sink
        self._pricing = pricing
        self._retry = retry or RetryPolicy()
        self._sleep = sleep
        self._clock = clock
        self._jitter = jitter

    def model_for(self, role: str) -> str:
        try:
            return self._models[role]
        except KeyError:
            raise TerminalLLMError(f"no model configured for role {role!r}") from None

    @overload
    async def chat(
        self,
        role: Role,
        messages: Sequence[ChatMessage],
        *,
        purpose: str,
        json_schema: type[ModelT],
        max_retries: int = ...,
        temperature: float | None = ...,
        stage: str = ...,
    ) -> ChatResult[ModelT]: ...

    @overload
    async def chat(
        self,
        role: Role,
        messages: Sequence[ChatMessage],
        *,
        purpose: str,
        json_schema: None = ...,
        max_retries: int = ...,
        temperature: float | None = ...,
        stage: str = ...,
    ) -> ChatResult[None]: ...

    async def chat(
        self,
        role: Role,
        messages: Sequence[ChatMessage],
        *,
        purpose: str,
        json_schema: type[ModelT] | None = None,
        max_retries: int = 2,
        temperature: float | None = None,
        stage: str = "",
    ) -> ChatResult[ModelT] | ChatResult[None]:
        """Call the model for ``role``.

        ``max_retries`` bounds *schema-validation* retries (capped at 2); transport retries
        follow :class:`RetryPolicy`. Raises TerminalLLMError, RetriesExhaustedError or
        SchemaValidationError.
        """
        schema_retries = max(0, min(max_retries, MAX_SCHEMA_RETRIES))
        model = self.model_for(role)
        convo: list[ChatMessage] = list(messages)
        if json_schema is not None:
            # Doc tip: give the schema in the prompt as well as in response_format.
            convo = [
                *convo,
                {
                    "role": "system",
                    "content": "Respond only with JSON matching this JSON Schema: "
                    + json.dumps(json_schema.model_json_schema()),
                },
            ]
        last_error = ""
        for attempt in range(1, schema_retries + 2):
            message, in_tok, out_tok, finish = await self._call(
                role, model, convo, purpose, stage, json_schema, temperature, attempt
            )
            text, reasoning = split_reasoning(message)
            refusal = getattr(message, "refusal", None)
            if refusal:
                raise TerminalLLMError(f"model refused: {refusal}")
            if json_schema is None:
                return ChatResult(text, reasoning, None, model, in_tok, out_tok, attempt, finish)
            try:
                parsed = json_schema.model_validate_json(text)
            except ValidationError as exc:
                last_error = str(exc)[:500]
                convo = [
                    *convo,
                    {"role": "assistant", "content": text},
                    {
                        "role": "user",
                        "content": "That did not validate against the schema: "
                        f"{last_error}\nReply again with corrected JSON only.",
                    },
                ]
                continue
            return ChatResult(text, reasoning, parsed, model, in_tok, out_tok, attempt, finish)
        raise SchemaValidationError(
            f"no schema-valid output after {schema_retries + 1} attempts: {last_error}"
        )

    async def _call(
        self,
        role: str,
        model: str,
        convo: Sequence[ChatMessage],
        purpose: str,
        stage: str,
        json_schema: type[BaseModel] | None,
        temperature: float | None,
        schema_attempt: int,
    ) -> tuple[Any, int, int, str | None]:
        kwargs: dict[str, Any] = {"model": model, "messages": list(convo)}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if json_schema is not None:
            kwargs["response_format"] = response_format_for(json_schema)
        digest = prompt_hash(convo)
        last_exc: BaseException | None = None
        for attempt in range(1, self._retry.max_attempts + 1):
            start = self._clock()
            try:
                completion = await self._transport.chat.completions.create(**kwargs)
            except (openai.APIStatusError, openai.APIConnectionError) as exc:
                latency = self._clock() - start
                retryable = is_retryable(exc)
                self._log(
                    role,
                    model,
                    purpose,
                    stage,
                    digest,
                    0,
                    0,
                    latency,
                    schema_attempt,
                    False,
                    f"{type(exc).__name__}: {getattr(exc, 'status_code', '')}",
                )
                if not retryable:
                    raise TerminalLLMError(f"{type(exc).__name__}: {exc}") from exc
                last_exc = exc
                if attempt == self._retry.max_attempts:
                    break
                delay = retry_after_seconds(exc)
                if delay is None:
                    delay = backoff_delay(attempt, self._retry, self._jitter())
                await self._sleep(min(delay, self._retry.max_delay_s))
                continue
            latency = self._clock() - start
            usage = getattr(completion, "usage", None)
            in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
            out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
            self._log(
                role,
                model,
                purpose,
                stage,
                digest,
                in_tok,
                out_tok,
                latency,
                schema_attempt,
                True,
                None,
            )
            choice = completion.choices[0]
            return choice.message, in_tok, out_tok, getattr(choice, "finish_reason", None)
        raise RetriesExhaustedError(
            f"gave up after {self._retry.max_attempts} attempts: {type(last_exc).__name__}"
        ) from last_exc

    def _log(
        self,
        role: str,
        model: str,
        purpose: str,
        stage: str,
        digest: str,
        in_tok: int,
        out_tok: int,
        latency: float,
        attempt: int,
        ok: bool,
        error: str | None,
    ) -> None:
        if self._sink is None:
            return
        cost = self._pricing(model, in_tok, out_tok) if self._pricing else 0.0
        self._sink.record_call(
            CallRecord(
                role,
                model,
                purpose,
                stage,
                digest,
                in_tok,
                out_tok,
                latency,
                cost,
                attempt,
                ok,
                error,
            )
        )
