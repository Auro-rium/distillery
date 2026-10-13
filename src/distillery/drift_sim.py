# ruff: noqa: S311
"""Free replay simulator for the drift guard (``distillery.drift``): no network, no model calls.

Everything is driven by RECORDED signal rows (``router_data`` rows: ``served_by`` or ``score``,
``correct``, ...). A row is routed the way the router routed it: its recorded ``served_by``, else
``score >= lambda`` (student) / below (teacher). An audit of a student-served request is decided by
the row's ``correct`` (a wrong student answer is a disagreement with the teacher).

* ``stream`` builds a deterministic request sequence: in-distribution traffic, then a burst.
* ``calibrate_tau`` picks the smallest deferral threshold whose false-rollback rate on
  in-distribution traffic is at most a target per 1,000 requests.
* ``run_burst`` replays a stream through a ``DriftMonitor`` and reports detection latency, wrong
  answers served before the rollback, and false alarms.

Everything is seeded with ``random.Random`` instances built from the explicit seed (never the
module-level RNG), so the same inputs give the same JSON.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from distillery.drift import (
    AUDIT_AGREE,
    AUDIT_DISAGREE,
    DriftMonitor,
    DriftParams,
    RouteEvent,
)

DEFAULT_BLOCK_LEN = 10
DEFAULT_N_STREAMS = 200
SIM_CERT_SHA = "0" * 64


class DriftSimError(ValueError):
    """The simulation inputs cannot be used (nothing was computed)."""


# ---------------------------------------------------------------- rows


def load_rows(path: Path) -> list[dict[str, Any]]:
    """Signal rows from a JSON list, ``{"rows": [...]}`` / ``{"items": [...]}`` (the router's
    ``signals_*.json`` files are the former), or JSON lines."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise DriftSimError(f"cannot read {path}: {exc}") from None
    try:
        if str(path).endswith(".jsonl"):
            doc: Any = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
        else:
            doc = json.loads(text)
    except ValueError as exc:
        raise DriftSimError(f"{path} is not valid JSON: {exc}") from None
    if isinstance(doc, dict):
        doc = doc.get("rows", doc.get("items"))
    if not isinstance(doc, list) or not all(isinstance(r, dict) for r in doc):
        raise DriftSimError(f"{path} must hold a list of row objects")
    if not doc:
        raise DriftSimError(f"{path} has no rows")
    return list(doc)


def route_of(row: Mapping[str, Any], lam: float | None = None) -> Literal["student", "teacher"]:
    """The router's recorded decision for one row (see the module docstring)."""
    sb = row.get("served_by")
    if sb in ("student", "teacher"):
        return sb  # type: ignore[no-any-return]
    score = row.get("score")
    if isinstance(score, int | float) and not isinstance(score, bool) and lam is not None:
        return "student" if float(score) >= lam else "teacher"
    raise DriftSimError(
        "row has neither served_by nor (score and a lambda): "
        f"{sorted(row)[:8]} (pass lam=, or score the rows first)"
    )


@dataclass(frozen=True)
class SimRequest:
    index: int
    request_id: str
    source: Literal["in_dist", "burst"]
    served_by: Literal["student", "teacher"]  # the router's recorded decision
    correct: bool | None  # was the student's answer right (None when the row has no label)
    row_id: str


def _make(
    rows: Sequence[Mapping[str, Any]], lam: float | None, rng: random.Random, n: int,
    source: Literal["in_dist", "burst"], start: int, tag: str,
) -> list[SimRequest]:  # fmt: skip
    if not rows:
        raise DriftSimError(f"no {source} rows to draw from")
    out: list[SimRequest] = []
    for k in range(n):
        i = rng.randrange(len(rows))
        row = rows[i]
        c = row.get("correct")
        out.append(
            SimRequest(
                index=start + k,
                request_id=f"sim-{tag}-{start + k}",
                source=source,
                served_by=route_of(row, lam),
                correct=None if c is None else bool(c),
                row_id=str(row.get("task_id", f"{source}[{i}]")),
            )
        )
    return out


def stream(
    in_dist_rows: Sequence[Mapping[str, Any]],
    burst_rows: Sequence[Mapping[str, Any]],
    *,
    n_before: int,
    burst_at: int | None = None,
    n_burst: int = 500,
    seed: int,
    lam: float | None = None,
) -> list[SimRequest]:
    """``n_before`` in-distribution requests (indices ``0..n_before-1``), then ``n_burst`` burst
    requests starting at index ``burst_at`` (= ``n_before``; if given it must equal ``n_before``:
    the burst starts exactly where the in-distribution prefix ends). Rows are drawn with
    replacement by seeded RNGs, one per source, so the in-distribution prefix is the same whatever
    the burst. ``request_id`` is ``sim-<seed>-<index>`` (the audit hash input)."""
    if n_before < 0 or n_burst < 0:
        raise DriftSimError("n_before and n_burst must be >= 0")
    if burst_at is not None and burst_at != n_before:
        raise DriftSimError(f"burst_at ({burst_at}) must equal n_before ({n_before})")
    tag = str(seed)
    pre = _make(in_dist_rows, lam, random.Random(f"{seed}:in"), n_before, "in_dist", 0, tag)
    burst = _make(burst_rows, lam, random.Random(f"{seed}:burst"), n_burst, "burst", n_before, tag)
    return pre + burst


# ---------------------------------------------------------------- tau calibration


def _tau_for(j: int, w: int) -> float:
    """The largest short decimal that is <= j / w (so ``Fraction(str(tau))`` is a few digits below
    j/w, far less than 1 / w**2): a window holding exactly j deferrals of w trips, the next lower
    share does not. A plain ``j / w`` float can round UP (5/7 -> 0.7142857142857143) and miss it."""
    dp = 2 * len(str(w)) + 1
    tau = ((j * 10**dp) // w) / 10**dp
    return float(max(tau, 10.0**-dp))


def _defer_flags(rows: Sequence[Mapping[str, Any]], lam: float | None) -> list[int]:
    return [1 if route_of(r, lam) == "teacher" else 0 for r in rows]


def _bootstrap(flags: Sequence[int], length: int, block: int, rng: random.Random) -> list[int]:
    n = len(flags)
    out: list[int] = []
    while len(out) < length:
        s = rng.randrange(n)
        out.extend(flags[(s + k) % n] for k in range(block))
    return out[:length]


def _first_trips(flags: Sequence[int], w: int, min_requests: int) -> list[int | None]:
    """For one stream: for each integer j in 1..w, the first request index at which the window
    holds at least a j/w share of deferrals once ``min_requests`` were seen (None = never).
    share >= j/w  <=>  count * w >= j * n  (exact integer arithmetic)."""
    first: list[int | None] = [None] * (w + 1)
    best = 0
    count = 0
    for t, f in enumerate(flags):
        count += f
        if t >= w:
            count -= flags[t - w]
        if t + 1 < min_requests:
            continue
        jmax = min(w, (count * w) // min(t + 1, w))
        while best < jmax:
            best += 1
            first[best] = t
    return first


def calibrate_tau(
    fit_rows: Sequence[Mapping[str, Any]],
    window_w: int,
    target_false_alarm_per_1000: float,
    n_streams: int = DEFAULT_N_STREAMS,
    seed: int = 0,
    *,
    lam: float | None = None,
    stream_len: int | None = None,
    block_len: int = DEFAULT_BLOCK_LEN,
    min_requests: int | None = None,
) -> dict[str, Any]:
    """Pick the smallest tau (a multiple of 1/window_w, as a short decimal) whose false-rollback
    rate on in-distribution traffic is at most ``target_false_alarm_per_1000``.

    Method. The deferral indicators of ``fit_rows`` (routed as recorded) are resampled by a
    circular BLOCK BOOTSTRAP (``block_len`` consecutive rows per block, which keeps clustering) into
    ``n_streams`` streams of ``stream_len`` requests (default ``max(2000, 20 * window_w)``), stream
    i seeded ``Random(f"{seed}:cal:{i}")``. Detector (a) is replayed on each stream. Because a
    rollback ends a stream (everything is teacher afterwards), the rate is a hazard:
    ``rollbacks / requests observed until the rollback (or the stream's end) * 1000``. This is
    non-increasing in tau, so the first tau (j / window_w, j = 1..w) meeting the target is the
    smallest. It is then re-measured on ``n_streams`` FRESH streams (``:val:``) so the reported
    rate is not the one the tau was selected on. Raises ``DriftSimError`` if even tau = 1 cannot
    meet the target. Only detector (a) is calibrated here (see ``audit_tail`` for detector (b))."""
    if window_w < 1 or n_streams < 1 or block_len < 1:
        raise DriftSimError("window_w, n_streams and block_len must be >= 1")
    if target_false_alarm_per_1000 < 0 or not math.isfinite(target_false_alarm_per_1000):
        raise DriftSimError("target false alarm rate must be a finite number >= 0")
    flags = _defer_flags(fit_rows, lam)
    if not flags:
        raise DriftSimError("no fit rows")
    mreq = window_w if min_requests is None else min_requests
    length = stream_len or max(2000, 20 * window_w)
    if length < mreq:
        raise DriftSimError("stream_len must be >= min_requests")

    def measure(kind: str) -> tuple[list[int], list[int]]:
        trips = [0] * (window_w + 1)
        exposure = [0] * (window_w + 1)
        for i in range(n_streams):
            s = _bootstrap(flags, length, block_len, random.Random(f"{seed}:{kind}:{i}"))
            first = _first_trips(s, window_w, mreq)
            for j in range(1, window_w + 1):
                t = first[j]
                if t is None:
                    exposure[j] += length
                else:
                    trips[j] += 1
                    exposure[j] += t + 1
        return trips, exposure

    trips, exposure = measure("cal")
    chosen = next(
        (
            j
            for j in range(1, window_w + 1)
            if trips[j] / exposure[j] * 1000.0 <= target_false_alarm_per_1000
        ),
        None,
    )
    if chosen is None:
        raise DriftSimError(
            f"no tau <= 1 reaches {target_false_alarm_per_1000} false alarms per 1000 requests "
            f"with window {window_w} (at tau = 1: {trips[window_w]} of {n_streams} streams trip); "
            "use a larger window or a larger target"
        )
    v_trips, v_exposure = measure("val")
    return {
        "tau": _tau_for(chosen, window_w),
        "tau_fraction": f"{chosen}/{window_w}",
        "window_w": window_w,
        "min_requests": mreq,
        "target_false_alarm_per_1000": target_false_alarm_per_1000,
        "false_alarm_per_1000": trips[chosen] / exposure[chosen] * 1000.0,
        "validation_false_alarm_per_1000": v_trips[chosen] / v_exposure[chosen] * 1000.0,
        "baseline_deferral_share": sum(flags) / len(flags),
        "n_fit_rows": len(flags),
        "n_streams": n_streams,
        "stream_len": length,
        "block_len": block_len,
        "seed": seed,
        "rollbacks": trips[chosen],
        "validation_rollbacks": v_trips[chosen],
        "method": "block bootstrap of fit-row deferral indicators; hazard = rollbacks / "
        "requests observed until rollback * 1000; smallest tau = j/window_w meeting the target; "
        "re-measured on fresh streams",
    }


def audit_tail(m: int, k: int, p: float) -> float:
    """P(Binomial(m, p) >= k): the chance that an in-distribution window of m audited student
    answers already holds k disagreements when the student's true error rate is p. Detector (b)
    false-alarms at roughly this rate per window, so choose k with it (p = the certified selective
    risk, or the measured error of served rows)."""
    if m < 1 or not 0 <= k <= m or not 0.0 <= p <= 1.0:
        raise DriftSimError("need m >= 1, 0 <= k <= m, 0 <= p <= 1")
    return float(math.fsum(math.comb(m, i) * p**i * (1.0 - p) ** (m - i) for i in range(k, m + 1)))


# ---------------------------------------------------------------- the burst replay


def simulate(
    requests: Sequence[SimRequest], params: DriftParams, *, cert_sha256: str = SIM_CERT_SHA
) -> dict[str, Any]:
    """Replay ``requests`` through a fresh ``DriftMonitor``. After a rollback every request is
    served by the teacher whatever the row recorded (the gate is ``allows_student()``)."""
    burst_start = next((r.index for r in requests if r.source == "burst"), None)
    mon = DriftMonitor(params, cert_sha256)
    events: list[dict[str, Any]] = []
    rollback_index: int | None = None
    detector: str | None = None
    wrong_before = wrong_before_burst = student_after = blocked = audits = disagreements = 0
    served_student = 0
    for r in requests:
        allowed = mon.allows_student()
        served = r.served_by if allowed else "teacher"
        if r.served_by == "student" and not allowed:
            blocked += 1
        audit = None
        if served == "student":
            if r.correct is None:
                raise DriftSimError(
                    f"request {r.index} ({r.row_id}) is student-served but has no 'correct'"
                )
            served_student += 1
            if not r.correct:
                wrong_before += 1
                if burst_start is None or r.index < burst_start:
                    wrong_before_burst += 1
            if rollback_index is not None:
                student_after += 1
            if mon.wants_audit(r.request_id):
                audits += 1
                audit = AUDIT_AGREE if r.correct else AUDIT_DISAGREE
                disagreements += audit == AUDIT_DISAGREE
        for e in mon.observe(RouteEvent(served, r.request_id, audit)):  # type: ignore[arg-type]
            events.append({"index": r.index, "source": r.source, **e.to_row()})
            if e.kind == "rollback":
                rollback_index, detector = r.index, e.detector
    false_alarm = rollback_index is not None and (
        burst_start is None or rollback_index < burst_start
    )
    latency = (
        rollback_index - burst_start + 1
        if rollback_index is not None and burst_start is not None and not false_alarm
        else None
    )
    return {
        "params": params.to_json(),
        "n_requests": len(requests),
        "n_in_dist": sum(1 for r in requests if r.source == "in_dist"),
        "n_burst": sum(1 for r in requests if r.source == "burst"),
        "burst_start_index": burst_start,
        "rolled_back": rollback_index is not None,
        "rollback_index": rollback_index,
        "detector": detector,
        "detection_latency_requests": latency,
        "latency_note": "requests from the first burst request to the rollback, counting both "
        "(1 = caught on the very first burst request); null = no rollback during the burst"
        if latency is not None
        else "null: no rollback during the burst"
        + (" (a false alarm fired before it)" if false_alarm else ""),
        "false_alarms_before_burst": 1 if false_alarm else 0,
        "student_answers_served": served_student,
        "wrong_student_answers_served_before_rollback": wrong_before,
        "wrong_student_answers_served_before_burst": wrong_before_burst,
        "wrong_student_answers_served_during_burst": wrong_before - wrong_before_burst,
        "student_answers_served_after_rollback": student_after,
        "student_answers_blocked_by_rollback": blocked,
        "audits": audits,
        "audit_disagreements": disagreements,
        "events": events,
    }


def run_burst(
    in_dist_rows: Sequence[Mapping[str, Any]],
    burst_rows: Sequence[Mapping[str, Any]],
    params: DriftParams,
    *,
    n_before: int,
    n_burst: int = 500,
    seed: int,
    lam: float | None = None,
    burst_at: int | None = None,
) -> dict[str, Any]:
    """``stream`` + ``simulate``: the burst report (JSON-serialisable)."""
    reqs = stream(
        in_dist_rows, burst_rows, n_before=n_before, burst_at=burst_at, n_burst=n_burst, seed=seed,
        lam=lam,
    )  # fmt: skip
    out = simulate(reqs, params)
    out["seed"] = seed
    return out
