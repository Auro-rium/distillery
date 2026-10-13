"""Drift guard for the certified router: when traffic stops looking like the calibration mix, roll
back to teacher-only automatically.

The Learn-then-Test certificate holds only for traffic drawn like the calibration split, so the
gateway watches two independent signals and, if either trips, stops serving the student until a
NEW certificate (a new gate) is installed with ``recertify``.

State machine (``DriftState``)::

    CERTIFIED --(detector trips)--> ROLLBACK --(settles at once)--> TEACHER
    TEACHER --recertify(new cert sha != any rolled-back sha, prereg commit)--> CERTIFIED

``ROLLBACK`` is the transition instant: ``observe`` emits the events for CERTIFIED -> ROLLBACK ->
TEACHER in one call, so the steady state after a rollback is always ``TEACHER``. Both states refuse
the student (``allows_student()`` is False). A rollback also emits ``redistill_queued``: a request
to re-distil on fresh traffic. It is only a queued event here; nothing in this module (or in the
gateway) runs a distillation.

Detectors (evaluated after every observed request; either trips):

* (a) ``deferral_rate``: among the last ``window_w`` requests, the share served by the teacher is
  ``>= tau``, evaluated only once ``min_requests`` requests have been seen since the monitor was
  armed (start, or the last recertify). The window is partial before it fills (share over the
  requests present). The comparison is exact (``Fraction``), so "exactly tau" trips.
* (b) ``audit_disagreement``: among the last ``audit_window_m`` AUDITED student-served requests,
  at least ``audit_k`` had a teacher answer that disagreed. Not gated by ``min_requests`` (a burst
  of wrong answers must not wait for volume). Audits are selected deterministically by
  ``should_audit(request_id, rate)`` (a hash), so a replay audits the same requests.

Pure and deterministic: no clock, no I/O, no randomness. Thread-safe (one lock).
"""

from __future__ import annotations

import hashlib
import re
import threading
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction
from typing import Any, Literal

AUDIT_AGREE = "agree"
AUDIT_DISAGREE = "disagree"
AuditResult = Literal["agree", "disagree"]

DETECTOR_DEFERRAL = "deferral_rate"
DETECTOR_AUDIT = "audit_disagreement"
DETECTOR_REVOKED = "revoked_certificate"

ROLLBACK_REASON = "rollback: drift detected → teacher"

KIND_ROLLBACK = "rollback"
KIND_TEACHER_ONLY = "teacher_only"
KIND_REDISTILL = "redistill_queued"
KIND_RECERTIFY = "recertify"

_SHA_RE = re.compile(r"[0-9a-f]{64}")


class DriftError(ValueError):
    """An illegal drift operation (bad params, illegal transition, bad recertify)."""


class DriftState(StrEnum):
    CERTIFIED = "CERTIFIED"
    ROLLBACK = "ROLLBACK"
    TEACHER = "TEACHER"


@dataclass(frozen=True)
class DriftParams:
    window_w: int  # detector (a): look at the last W requests
    tau: float  # detector (a): roll back when the deferral share is >= tau (0 < tau <= 1)
    audit_rate: float  # fraction of student-served requests re-checked by the teacher
    audit_window_m: int  # detector (b): the last M audited student-served requests
    audit_k: int  # detector (b): roll back at >= k disagreements among them
    min_requests: int  # detector (a) is armed only after this many requests

    def __post_init__(self) -> None:
        for name in ("window_w", "audit_window_m", "audit_k", "min_requests"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise DriftError(f"{name} must be an integer >= 1, got {v!r}")
        if not (isinstance(self.tau, int | float) and 0.0 < self.tau <= 1.0):
            raise DriftError(f"tau must be in (0, 1], got {self.tau!r}")
        if not (isinstance(self.audit_rate, int | float) and 0.0 <= self.audit_rate <= 1.0):
            raise DriftError(f"audit_rate must be in [0, 1], got {self.audit_rate!r}")
        if self.audit_k > self.audit_window_m:
            raise DriftError("audit_k cannot exceed audit_window_m (it could never trip)")

    def to_json(self) -> dict[str, Any]:
        return {
            "window_w": self.window_w,
            "tau": self.tau,
            "audit_rate": self.audit_rate,
            "audit_window_m": self.audit_window_m,
            "audit_k": self.audit_k,
            "min_requests": self.min_requests,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> DriftParams:
        """Parse ``drift.json``-style config. ``window_w`` and ``tau`` are required (tau must be
        calibrated, see ``drift_sim.calibrate_tau``); the audit is off (rate 0) unless asked for."""
        unknown = set(data) - {
            "window_w", "tau", "audit_rate", "audit_window_m", "audit_k", "min_requests",
        }  # fmt: skip
        if unknown:
            raise DriftError(f"unknown drift parameter(s): {sorted(unknown)}")
        for req in ("window_w", "tau"):
            if req not in data:
                raise DriftError(f"drift config needs {req!r}")
        try:
            window_w = int(data["window_w"])
            return cls(
                window_w=window_w,
                tau=float(data["tau"]),
                audit_rate=float(data.get("audit_rate", 0.0)),
                audit_window_m=int(data.get("audit_window_m", 50)),
                audit_k=int(data.get("audit_k", 5)),
                min_requests=int(data.get("min_requests", window_w)),
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, DriftError):
                raise
            raise DriftError(f"invalid drift config: {exc}") from None


@dataclass(frozen=True)
class RouteEvent:
    """One request as the router handled it. ``audit`` is only for an audited student answer."""

    served_by: Literal["student", "teacher"]
    request_id: str
    audit: AuditResult | None = None

    def __post_init__(self) -> None:
        if self.served_by not in ("student", "teacher"):
            raise DriftError(f"served_by must be 'student' or 'teacher', got {self.served_by!r}")
        if self.audit not in (None, AUDIT_AGREE, AUDIT_DISAGREE):
            raise DriftError(f"audit must be 'agree', 'disagree' or None, got {self.audit!r}")
        if self.audit is not None and self.served_by != "student":
            raise DriftError("only a student-served request can carry an audit result")


@dataclass(frozen=True)
class DriftEvent:
    """A state-machine event; the fields mirror the ``drift_events`` table columns."""

    kind: str
    state_from: DriftState
    state_to: DriftState
    detector: str | None
    details: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        return {
            "state_from": self.state_from.value,
            "state_to": self.state_to.value,
            "kind": self.kind,
            "detector": self.detector,
            "details": self.details,
        }


def should_audit(request_id: str, rate: float) -> bool:
    """Deterministic audit selection: the request id hashes to a uniform number in [0, 1); audit
    iff it is below ``rate``. Same ids, same rate -> same audits (replays are reproducible)."""
    if rate <= 0.0:
        return False
    if rate >= 1.0:
        return True
    digest = hashlib.sha256(("drift-audit:" + request_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2.0**64 < rate


def _tau_frac(tau: float) -> Fraction:
    return Fraction(str(tau))  # 0.3 means exactly 3/10, not the nearest float


class DriftMonitor:
    """Holds the state machine and the two windows. See the module docstring."""

    def __init__(self, params: DriftParams, cert_sha256: str | None = None) -> None:
        self.params = params
        self._tau = _tau_frac(params.tau)
        self._lock = threading.RLock()
        self._state = DriftState.CERTIFIED
        self.cert_sha256 = cert_sha256
        self._rolled_back: list[str] = []  # cert shas this deployment rolled back from
        self._deferrals: deque[bool] = deque(maxlen=params.window_w)  # True = teacher-served
        self._audits: deque[bool] = deque(maxlen=params.audit_window_m)  # True = disagreement
        self._seen = 0
        self.last_detector: str | None = None

    # ---- read side --------------------------------------------------------------------------
    @property
    def state(self) -> DriftState:
        with self._lock:
            return self._state

    def allows_student(self) -> bool:
        with self._lock:
            return self._state is DriftState.CERTIFIED

    @property
    def rolled_back_shas(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._rolled_back)

    def wants_audit(self, request_id: str) -> bool:
        return should_audit(request_id, self.params.audit_rate)

    def window_stats(self) -> dict[str, Any]:
        with self._lock:
            n = len(self._deferrals)
            defers = sum(self._deferrals)
            return {
                "requests_seen": self._seen,
                "window_requests": n,
                "window_deferred": defers,
                "deferral_share": (defers / n) if n else None,
                "audits_in_window": len(self._audits),
                "audit_disagreements": sum(self._audits),
                "armed": self._seen >= self.params.min_requests,
            }

    def reason(self) -> str:
        """Why the student is off, for the call log / response header."""
        with self._lock:
            if self.last_detector:
                return f"{ROLLBACK_REASON} ({self.last_detector})"
            return ROLLBACK_REASON

    # ---- observing --------------------------------------------------------------------------
    def observe(self, ev: RouteEvent) -> list[DriftEvent]:
        """Feed one request. Returns the events it caused (empty unless a detector tripped).
        Requests observed while in ROLLBACK/TEACHER are ignored: nothing is served by the student,
        so there is no deferral share or audit to measure."""
        with self._lock:
            if self._state is not DriftState.CERTIFIED:
                return []
            self._seen += 1
            self._deferrals.append(ev.served_by == "teacher")
            if ev.audit is not None:
                self._audits.append(ev.audit == AUDIT_DISAGREE)
            fired: list[str] = []
            n, defers = len(self._deferrals), sum(self._deferrals)
            if self._seen >= self.params.min_requests and Fraction(defers, n) >= self._tau:
                fired.append(DETECTOR_DEFERRAL)
            disagreements = sum(self._audits)
            if disagreements >= self.params.audit_k:
                fired.append(DETECTOR_AUDIT)
            if not fired:
                return []
            return self._rollback(
                fired,
                {
                    "request_id": ev.request_id,
                    "requests_seen": self._seen,
                    "deferral": {
                        "deferred": defers,
                        "window_requests": n,
                        "share": defers / n,
                        "tau": self.params.tau,
                        "window_w": self.params.window_w,
                        "min_requests": self.params.min_requests,
                    },
                    "audit": {
                        "disagreements": disagreements,
                        "audited": len(self._audits),
                        "k": self.params.audit_k,
                        "window_m": self.params.audit_window_m,
                    },
                },
            )

    def _rollback(self, fired: list[str], numbers: dict[str, Any]) -> list[DriftEvent]:
        detector = fired[0]
        self.last_detector = "+".join(fired)
        base = {"cert_sha256": self.cert_sha256, "fired": fired, **numbers}
        if self.cert_sha256 and self.cert_sha256 not in self._rolled_back:
            self._rolled_back.append(self.cert_sha256)
        self._clear_windows()
        c, r, t = DriftState.CERTIFIED, DriftState.ROLLBACK, DriftState.TEACHER
        self._state = t  # the ROLLBACK instant is the event below; the steady state is TEACHER
        return [
            DriftEvent(KIND_ROLLBACK, c, r, detector, base),
            DriftEvent(KIND_TEACHER_ONLY, r, t, detector, {"cert_sha256": self.cert_sha256}),
            DriftEvent(
                KIND_REDISTILL,
                t,
                t,
                detector,
                {
                    "cert_sha256": self.cert_sha256,
                    "queued": True,
                    "note": "queued only: nothing re-distils automatically",
                },
            ),
        ]

    def _clear_windows(self) -> None:
        self._deferrals.clear()
        self._audits.clear()
        self._seen = 0

    # ---- certificates -----------------------------------------------------------------------
    def bind_cert(self, cert_sha256: str | None) -> list[DriftEvent]:
        """Tell the monitor which certificate is loaded (at startup / after a reload). If that
        certificate was already rolled back and the state is CERTIFIED (a stale cert.json put back
        on disk), roll back again instead of silently trusting it."""
        with self._lock:
            self.cert_sha256 = cert_sha256
            if (
                cert_sha256 is not None
                and self._state is DriftState.CERTIFIED
                and cert_sha256 in self._rolled_back
            ):
                return self._rollback(
                    [DETECTOR_REVOKED],
                    {"request_id": None, "note": "certificate already rolled back"},
                )
            return []

    def recertify(self, new_cert_sha256: str, prereg_commit: str) -> list[DriftEvent]:
        """TEACHER -> CERTIFIED, the only way back. Requires a lowercase hex sha256 that differs
        from every certificate this deployment rolled back from (a NEW gate, not the old one) and a
        non-empty pre-registration commit. Clears both windows."""
        with self._lock:
            if self._state is not DriftState.TEACHER:
                raise DriftError(f"recertify needs state TEACHER, not {self._state.value}")
            if not isinstance(new_cert_sha256, str) or not _SHA_RE.fullmatch(new_cert_sha256):
                raise DriftError("new_cert_sha256 must be 64 lowercase hex characters")
            if not isinstance(prereg_commit, str) or not prereg_commit.strip():
                raise DriftError("prereg_commit is required")
            if new_cert_sha256 in self._rolled_back:
                raise DriftError(
                    "the new certificate must differ from the one(s) rolled back from: "
                    "re-certifying with the same gate is refused"
                )
            old = self.cert_sha256
            self.cert_sha256 = new_cert_sha256
            self._state = DriftState.CERTIFIED
            self.last_detector = None
            self._clear_windows()
            return [
                DriftEvent(
                    KIND_RECERTIFY,
                    DriftState.TEACHER,
                    DriftState.CERTIFIED,
                    None,
                    {
                        "previous_cert_sha256": old,
                        "new_cert_sha256": new_cert_sha256,
                        "cert_sha256": new_cert_sha256,
                        "prereg_commit": prereg_commit.strip(),
                    },
                )
            ]

    # ---- persistence ------------------------------------------------------------------------
    def restore(self, events: Iterable[Mapping[str, Any]]) -> None:
        """Rebuild the state from persisted events (oldest first; each with ``kind``, ``state_to``,
        ``detector`` and a ``details`` dict). The windows start empty. A trailing ROLLBACK (crash
        between the events) settles to TEACHER. Never moves to CERTIFIED except via a recorded
        ``recertify``: a restart cannot silently re-trust the student."""
        with self._lock:
            state = DriftState.CERTIFIED
            cert = self.cert_sha256
            rolled: list[str] = []
            detector: str | None = None
            for e in events:
                state = DriftState(str(e["state_to"]))
                details = e.get("details") or {}
                if e["kind"] == KIND_ROLLBACK:
                    fired = details.get("fired")
                    detector = "+".join(fired) if fired else e.get("detector")
                    sha = details.get("cert_sha256")
                    if sha and sha not in rolled:
                        rolled.append(str(sha))
                elif e["kind"] == KIND_RECERTIFY:
                    cert = details.get("new_cert_sha256") or cert
                    detector = None
            if state is DriftState.ROLLBACK:
                state = DriftState.TEACHER
            self._state = state
            self._rolled_back = rolled
            self.last_detector = detector if state is not DriftState.CERTIFIED else None
            self.cert_sha256 = cert
            self._clear_windows()
