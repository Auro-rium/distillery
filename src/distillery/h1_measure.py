"""H1 measurement: throughput/latency of the endpoint, parity with the sandbox student, and the
cost / break-even arithmetic. Pure functions plus an injectable HTTP client; no I/O of its own
beyond the real ``HttpChatClient`` (never constructed by tests).

* ``measure_throughput`` drives OpenAI-style chat completions (model ``n1``, temperature 0) with
  the CANONICAL prompt construction (``Pack.build_messages(..., role="eval_student")``: exactly
  what the sandbox student is given in ``evaluator.score_model``).
* ``parity`` compares the endpoint's outputs with the sandbox student's on the same items: the
  identical-SQL rate, each side's execution-match accuracy against gold (the pack verifier the
  evaluator uses) and their agreement.
* ``cost_per_1k`` / ``blended_curve`` / ``breakeven_volume`` are PROJECTIONS (labelled
  ``"projected": true``): the hourly price is an input, not a measurement of this module.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from distillery.costing import percentile, teacher_stats
from distillery.evaluator import identical_rate
from distillery.taskpacks.base import Pack, get_pack
from distillery.taskpacks.sql.executor import Executor

ChatMessages = list[dict[str, str]]
SERVED_MODEL = "n1"
DEFAULT_LEVELS = (1, 8, 32, 64)
DEFAULT_MAX_TOKENS = 160


class ChatError(RuntimeError):
    """One chat completion failed (HTTP error, timeout, malformed answer)."""


@dataclass(frozen=True)
class ChatResponse:
    text: str
    completion_tokens: int
    latency_s: float  # measured by the client around the HTTP call


class ChatClient(Protocol):
    def chat(self, messages: ChatMessages, *, max_tokens: int) -> ChatResponse:
        """One chat completion at temperature 0 on the served model; raises ``ChatError``."""
        ...

    def lists_model(self) -> bool:
        """True once the server is up and lists the served model (the readiness probe)."""
        ...


# ---------------------------------------------------------------- real HTTP client

HttpSend = Callable[[str, str, Mapping[str, str], bytes | None, float], tuple[int, bytes]]


def _urllib_send(
    method: str, url: str, headers: Mapping[str, str], body: bytes | None, timeout: float
) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers=dict(headers), method=method)  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ChatError(f"{method} {url.split('?')[0]} failed: {exc}") from exc


class HttpChatClient:
    """OpenAI chat-completions over HTTPS with a bearer token. ``base_url`` ends in ``/v1``.
    The token never appears in an error message or ``repr``."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        model: str = SERVED_MODEL,
        timeout_s: float = 120.0,
        send: HttpSend = _urllib_send,
        monotonic: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._base, self._token, self._model = base_url.rstrip("/"), token, model
        self._timeout, self._send, self._mono = timeout_s, send, monotonic

    def __repr__(self) -> str:
        return f"HttpChatClient(base_url={self._base!r}, model={self._model!r})"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"}

    def chat(self, messages: ChatMessages, *, max_tokens: int) -> ChatResponse:
        body = json.dumps(
            {
                "model": self._model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": max_tokens,
            }
        ).encode()
        t0 = self._mono()
        code, raw = self._send(
            "POST", f"{self._base}/chat/completions", self._headers(), body, self._timeout
        )
        latency = self._mono() - t0
        if code != 200:
            raise ChatError(f"HTTP {code}: {raw[:200]!r}".replace(self._token, "***"))
        try:
            data = json.loads(raw)
            text = str(data["choices"][0]["message"]["content"] or "")
            tokens = int(data.get("usage", {}).get("completion_tokens", 0))
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ChatError(f"malformed completion: {raw[:200]!r}") from exc
        return ChatResponse(text, tokens, latency)

    def lists_model(self) -> bool:
        """True once ``GET /models`` answers 200 and lists the served model (vLLM is up)."""
        try:
            code, raw = self._send("GET", f"{self._base}/models", self._headers(), None, 30.0)
            if code != 200:
                return False
            return any(m.get("id") == self._model for m in json.loads(raw).get("data", []))
        except (ChatError, ValueError, AttributeError, TypeError):
            return False


# ---------------------------------------------------------------- prompts / items


def load_items(path: Path) -> list[dict[str, Any]]:
    """Items from a JSON list or a JSONL file. Each needs ``question`` + ``schema_ddl`` (or a
    ready ``messages`` list); parity additionally needs ``gold_sql`` and ``requires_order``."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise ValueError(f"{path}: expected a list of objects")
    return rows


def build_prompts(
    items: Sequence[Mapping[str, Any]],
    pack: Pack | None = None,
    *,
    default_schema_ddl: str | None = None,
) -> list[ChatMessages]:
    """The messages the sandbox student gets (``evaluator.score_model``): same function, same
    role. An item carrying its own ``messages`` is used as given; one without a schema takes
    ``default_schema_ddl`` (stored split items usually share one database)."""
    pk = pack or get_pack("sql")
    out: list[ChatMessages] = []
    for i, it in enumerate(items):
        if "messages" in it:
            out.append([dict(m) for m in it["messages"]])
            continue
        question = it.get("question")
        ddl = it.get("schema_ddl", it.get("context_text", default_schema_ddl))
        if not isinstance(question, str) or not isinstance(ddl, str):
            raise ValueError(f"item {i}: needs 'question' and 'schema_ddl' (or 'messages')")
        out.append(pk.build_messages(question, ddl, role="eval_student"))
    return out


def load_outputs(path: Path) -> list[str]:
    """Sandbox-student outputs: a JSON list of strings or of ``{"output"|"text": str}``."""
    rows = load_items_or_strings(path)
    return [r if isinstance(r, str) else str(r.get("output", r.get("text", ""))) for r in rows]


def load_items_or_strings(path: Path) -> list[Any]:
    text = Path(path).read_text(encoding="utf-8").strip()
    return json.loads(text) if text.startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()
    ]  # fmt: skip


# ---------------------------------------------------------------- throughput


@dataclass(frozen=True)
class LevelStats:
    concurrency: int
    requests: int  # attempted (ok + errors)
    ok: int
    errors: int
    skipped: int  # not sent because the time box expired
    wall_s: float
    req_per_s: float
    latency_p50_s: float | None
    latency_p95_s: float | None
    output_tokens: int
    output_tokens_per_s: float
    first_error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MeasureResult:
    levels: list[LevelStats]
    samples: list[dict[str, Any]]  # one per attempted request, with the raw output text
    stopped_early: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "levels": [lv.to_json() for lv in self.levels],
            "stopped_early": self.stopped_early,
            "n_samples": len(self.samples),
        }

    def outputs_at(self, concurrency: int, n: int) -> list[str] | None:
        """Outputs (by prompt index) of one level, or ``None`` if any request is missing."""
        by_idx = {s["index"]: s for s in self.samples if s["level"] == concurrency and s["ok"]}
        return [by_idx[i]["text"] for i in range(n)] if len(by_idx) == n else None


def measure_throughput(
    client: ChatClient,
    prompts: Sequence[ChatMessages],
    concurrency_levels: Sequence[int] = DEFAULT_LEVELS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    *,
    monotonic: Callable[[], float] = time.monotonic,
    should_stop: Callable[[], bool] = lambda: False,
) -> MeasureResult:
    """Send ``prompts`` once per concurrency level. ``should_stop`` (the time-box watchdog) is
    checked before every request; once true the remaining requests are skipped, the level ends
    and no further level starts."""
    if not prompts:
        raise ValueError("no prompts")
    levels: list[LevelStats] = []
    samples: list[dict[str, Any]] = []
    stopped = False
    for level in concurrency_levels:
        if level < 1:
            raise ValueError("concurrency levels must be >= 1")
        if should_stop():
            stopped = True
            break

        def one(i: int, level: int = level) -> dict[str, Any]:
            if should_stop():
                return {"level": level, "index": i, "skipped": True}
            try:
                r = client.chat(prompts[i], max_tokens=max_tokens)
                return {
                    "level": level, "index": i, "ok": True, "skipped": False,
                    "latency_s": r.latency_s, "completion_tokens": r.completion_tokens,
                    "text": r.text, "error": None,
                }  # fmt: skip
            except ChatError as exc:
                return {
                    "level": level, "index": i, "ok": False, "skipped": False,
                    "latency_s": None, "completion_tokens": 0, "text": "", "error": str(exc),
                }  # fmt: skip

        t0 = monotonic()
        with ThreadPoolExecutor(max_workers=level) as pool:
            rows = list(pool.map(one, range(len(prompts))))
        wall = monotonic() - t0
        sent = [r for r in rows if not r["skipped"]]
        ok = [r for r in sent if r["ok"]]
        lat = [float(r["latency_s"]) for r in ok]
        tokens = sum(int(r["completion_tokens"]) for r in ok)
        first_err = next((r["error"] for r in sent if not r["ok"]), None)
        levels.append(
            LevelStats(
                concurrency=level,
                requests=len(sent),
                ok=len(ok),
                errors=len(sent) - len(ok),
                skipped=len(rows) - len(sent),
                wall_s=wall,
                req_per_s=len(ok) / wall if wall > 0 else 0.0,
                latency_p50_s=percentile(lat, 50) if lat else None,
                latency_p95_s=percentile(lat, 95) if lat else None,
                output_tokens=tokens,
                output_tokens_per_s=tokens / wall if wall > 0 else 0.0,
                first_error=first_err,
            )
        )
        samples.extend({k: v for k, v in r.items() if k != "skipped"} for r in sent)
        if len(sent) < len(rows):
            stopped = True
            break
    return MeasureResult(levels, samples, stopped)


# ---------------------------------------------------------------- parity


def normalise_sql(sql: str | None) -> str | None:
    """Whitespace-collapsed SQL (``None`` stays ``None``)."""
    return None if sql is None else " ".join(sql.split())


def accuracy(
    outputs: Sequence[str],
    items: Sequence[Mapping[str, Any]],
    *,
    executor: Executor,
    db_ref: str,
    pack: Pack | None = None,
) -> dict[str, Any]:
    """Execution-match of each output against its item's gold, with the pack verifier that
    ``evaluator.score_model`` uses. ``correct`` is per item."""
    pk = pack or get_pack("sql")
    if len(outputs) != len(items):
        raise ValueError(f"{len(outputs)} outputs for {len(items)} items")
    sqls = [pk.extract_answer(o) for o in outputs]
    runnable = [i for i, s in enumerate(sqls) if s is not None]
    outcomes = dict(
        zip(runnable, executor.run_batch(db_ref, [sqls[i] or "" for i in runnable]), strict=True)
    )
    golds = executor.run_batch(db_ref, [str(it[pk.gold_field]) for it in items])
    correct = [
        sqls[i] is not None and pk.compare(outcomes[i], golds[i], bool(it["requires_order"])).ok
        for i, it in enumerate(items)
    ]
    n = len(items)
    return {
        "n": n,
        "correct": correct,
        "accuracy": sum(correct) / n if n else 0.0,
        "unparseable": sum(s is None for s in sqls),
    }


def parity(
    endpoint_outputs: Sequence[str],
    sandbox_outputs: Sequence[str],
    items: Sequence[Mapping[str, Any]],
    *,
    executor: Executor,
    db_ref: str,
    pack: Pack | None = None,
) -> dict[str, Any]:
    """Endpoint vs sandbox student on the same items.

    ``identical_sql_rate``: extracted SQL equal after whitespace normalisation (both must have
    parsed). ``agreement``: both right or both wrong against gold."""
    pk = pack or get_pack("sql")
    if not (len(endpoint_outputs) == len(sandbox_outputs) == len(items)):
        raise ValueError("endpoint outputs, sandbox outputs and items differ in length")
    n = len(items)
    ep = [normalise_sql(pk.extract_answer(o)) for o in endpoint_outputs]
    sb = [normalise_sql(pk.extract_answer(o)) for o in sandbox_outputs]
    identical = sum(a is not None and a == b for a, b in zip(ep, sb, strict=True))
    acc_e = accuracy(endpoint_outputs, items, executor=executor, db_ref=db_ref, pack=pk)
    acc_s = accuracy(sandbox_outputs, items, executor=executor, db_ref=db_ref, pack=pk)
    ce, cs = acc_e["correct"], acc_s["correct"]
    return {
        "n": n,
        "identical_sql_rate": identical / n if n else 0.0,
        "identical_output_rate": identical_rate(list(endpoint_outputs), list(sandbox_outputs)),
        "endpoint_accuracy": acc_e["accuracy"],
        "sandbox_accuracy": acc_s["accuracy"],
        "accuracy_delta": acc_e["accuracy"] - acc_s["accuracy"],
        "agreement": sum(a == b for a, b in zip(ce, cs, strict=True)) / n if n else 0.0,
        "both_correct": sum(a and b for a, b in zip(ce, cs, strict=True)),
        "only_endpoint_correct": sum(a and not b for a, b in zip(ce, cs, strict=True)),
        "only_sandbox_correct": sum(b and not a for a, b in zip(ce, cs, strict=True)),
        "endpoint_unparseable": acc_e["unparseable"],
        "sandbox_unparseable": acc_s["unparseable"],
    }


# ---------------------------------------------------------------- cost (all projections)


def cost_per_1k(hourly_usd: float, req_per_hour: float, utilization: float = 1.0) -> dict[str, Any]:
    """Student USD per 1,000 requests when the endpoint is busy ``utilization`` of the time at
    ``req_per_hour`` (a measured saturation rate). A projection, not a measurement."""
    if hourly_usd < 0 or req_per_hour <= 0 or not 0 < utilization <= 1:
        raise ValueError("need hourly >= 0, req_per_hour > 0 and 0 < utilization <= 1")
    effective = req_per_hour * utilization
    return {
        "hourly_usd": hourly_usd,
        "req_per_hour": req_per_hour,
        "utilization": utilization,
        "usd_per_1k": hourly_usd / effective * 1000,
        "projected": True,
        "basis": "hourly price / (measured req/hour x utilization) x 1000",
    }


def teacher_cost_per_1k(
    root: Path, model: str, run_id: str | None = None, limit: int | None = None
) -> dict[str, Any]:
    """Teacher USD per 1k from a store's ledger rows (``costing.teacher_stats``, read-only)."""
    t = teacher_stats(Path(root), model, run_id, limit)
    return {
        "model": model,
        "n_calls": t["n_calls"],
        "usd_per_1k": t["usd_per_1k_questions"],
        "latency_p50_s": t["latency_p50_s"],
        "latency_p95_s": t["latency_p95_s"],
        "basis": t["basis"],
        "projected": False,
    }


def blended_per_1k(
    coverage: float, student_per_1k: float, teacher_per_1k: float, audit_rate: float
) -> float:
    """Marginal USD per 1k when a ``coverage`` share goes to the student and ``audit_rate`` of
    those is ALSO sent to the teacher (the audit sample); the rest goes to the teacher."""
    if not (0 <= coverage <= 1 and 0 <= audit_rate <= 1):
        raise ValueError("coverage and audit_rate must be in [0, 1]")
    return (
        coverage * student_per_1k
        + (1 - coverage) * teacher_per_1k
        + coverage * audit_rate * teacher_per_1k
    )


def breakeven_volume(
    coverage: float,
    student_per_1k: float,
    teacher_per_1k: float,
    audit_rate: float,
    fixed_daily_usd: float = 0.0,
) -> float | None:
    """Requests/day above which blending beats all-teacher: fixed/(saving per request), or
    ``None`` when the blend never saves (``coverage*(teacher*(1-audit)-student) <= 0``)."""
    saving_per_1k = teacher_per_1k - blended_per_1k(
        coverage, student_per_1k, teacher_per_1k, audit_rate
    )
    if saving_per_1k <= 0:
        return None
    return fixed_daily_usd * 1000 / saving_per_1k


def blended_curve(
    coverage: float,
    student_per_1k: float,
    teacher_per_1k: float,
    audit_rate: float,
    volumes: Sequence[float],
    *,
    student_fixed_daily_usd: float = 0.0,
    student_capacity_per_day: float | None = None,
) -> dict[str, Any]:
    """Daily cost of all-teacher vs the blend at each volume (requests/day), plus the break-even.

    Two ways to pay for the student: marginally (``student_per_1k``, an endpoint kept busy) or a
    fixed always-on daily cost (``student_fixed_daily_usd``, with ``student_per_1k`` = 0)."""
    blend = blended_per_1k(coverage, student_per_1k, teacher_per_1k, audit_rate)
    points = []
    for v in volumes:
        teacher_only = v / 1000 * teacher_per_1k
        blended = student_fixed_daily_usd + v / 1000 * blend
        cap_ok = (
            None if student_capacity_per_day is None else coverage * v <= student_capacity_per_day
        )
        points.append(
            {
                "volume_per_day": v,
                "teacher_only_usd": teacher_only,
                "blended_usd": blended,
                "savings_usd": teacher_only - blended,
                "student_capacity_ok": cap_ok,
            }
        )
    be = breakeven_volume(
        coverage, student_per_1k, teacher_per_1k, audit_rate, student_fixed_daily_usd
    )
    return {
        "projected": True,
        "coverage": coverage,
        "audit_rate": audit_rate,
        "student_per_1k": student_per_1k,
        "teacher_per_1k": teacher_per_1k,
        "student_fixed_daily_usd": student_fixed_daily_usd,
        "blended_per_1k": blend,
        "breakeven_volume_per_day": be,
        "breakeven_never": be is None,
        "points": points,
    }


def best_req_per_hour(result: MeasureResult) -> float | None:
    """The highest measured throughput over all levels with no errors, in requests/hour."""
    rates = [lv.req_per_s for lv in result.levels if lv.ok and not lv.errors and lv.req_per_s > 0]
    best = max(rates, default=None)
    return None if best is None or not math.isfinite(best) else best * 3600.0
