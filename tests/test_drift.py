# ruff: noqa: S101
"""The drift guard's state machine and detectors (pure, no I/O)."""

from __future__ import annotations

import pytest

from distillery.drift import (
    DETECTOR_AUDIT,
    DETECTOR_DEFERRAL,
    DETECTOR_REVOKED,
    DriftError,
    DriftEvent,
    DriftMonitor,
    DriftParams,
    DriftState,
    RouteEvent,
    should_audit,
)

SHA_A, SHA_B, SHA_C = "a" * 64, "b" * 64, "c" * 64


def params(**kw: object) -> DriftParams:
    base: dict[str, object] = {
        "window_w": 10, "tau": 0.5, "audit_rate": 0.0, "audit_window_m": 5, "audit_k": 3,
        "min_requests": 10,
    }  # fmt: skip
    base.update(kw)
    return DriftParams(**base)  # type: ignore[arg-type]


def ev(served_by: str, i: int = 0, audit: str | None = None) -> RouteEvent:
    return RouteEvent(served_by, f"r{i}", audit)  # type: ignore[arg-type]


def feed(m: DriftMonitor, kinds: str, audits: str | None = None) -> list[DriftEvent]:
    """'s' = student, 't' = teacher; audits: same length, 'a' agree, 'd' disagree, '-' none."""
    out: list[DriftEvent] = []
    for i, c in enumerate(kinds):
        a = {"a": "agree", "d": "disagree", "-": None}[audits[i]] if audits else None
        out += m.observe(ev("student" if c == "s" else "teacher", i, a))
    return out


def rows_of(events: list[DriftEvent]) -> list[dict[str, object]]:
    return [e.to_row() | {"kind": e.kind} for e in events]


def test_param_validation() -> None:
    for bad in (
        {"window_w": 0}, {"tau": 0.0}, {"tau": 1.1}, {"audit_rate": -0.1}, {"audit_rate": 1.5},
        {"audit_window_m": 0}, {"audit_k": 0}, {"audit_k": 6}, {"min_requests": 0},
        {"window_w": True},
    ):  # fmt: skip
        with pytest.raises(DriftError):
            params(**bad)
    params(tau=1.0, audit_k=5)  # boundaries are legal


def test_from_mapping_defaults_and_errors() -> None:
    p = DriftParams.from_mapping({"window_w": 40, "tau": 0.6})
    assert (p.audit_rate, p.audit_window_m, p.audit_k, p.min_requests) == (0.0, 50, 5, 40)
    with pytest.raises(DriftError):
        DriftParams.from_mapping({"window_w": 40})  # tau is required (it must be calibrated)
    with pytest.raises(DriftError):
        DriftParams.from_mapping({"window_w": 40, "tau": 0.5, "typo": 1})
    with pytest.raises(DriftError):
        DriftParams.from_mapping({"window_w": "x", "tau": 0.5})


def test_route_event_validation() -> None:
    with pytest.raises(DriftError):
        RouteEvent("teacher", "r", "agree")  # an audit belongs to a student answer
    with pytest.raises(DriftError):
        RouteEvent("nobody", "r")  # type: ignore[arg-type]
    with pytest.raises(DriftError):
        RouteEvent("student", "r", "maybe")  # type: ignore[arg-type]


def test_starts_certified_and_stays_while_below_tau() -> None:
    m = DriftMonitor(params(), SHA_A)
    assert m.state is DriftState.CERTIFIED and m.allows_student()
    assert feed(m, "sssst" * 4) == []  # 20% deferral, far below tau 0.5
    assert m.state is DriftState.CERTIFIED


def test_deferral_trips_at_exactly_tau() -> None:
    m = DriftMonitor(params(), SHA_A)
    out = feed(m, "sssssttttt")  # first full window: 5/10 == tau
    assert [e.kind for e in out] == ["rollback", "teacher_only", "redistill_queued"]
    assert out[0].detector == DETECTOR_DEFERRAL
    assert out[0].details["deferral"]["deferred"] == 5


def test_just_below_tau_does_not_trip() -> None:
    m = DriftMonitor(params(), SHA_A)
    assert feed(m, "ssssstttts") == []  # 4/10
    assert m.state is DriftState.CERTIFIED


def test_tau_comparison_is_exact_not_float() -> None:
    # 0.3 * 10 is 3.0000000000000004 in floats; 3/10 must still hit tau = 0.3 exactly
    m = DriftMonitor(params(tau=0.3), SHA_A)
    out = feed(m, "sssssssttt")
    assert out and out[0].kind == "rollback"


def test_min_requests_gates_the_deferral_detector() -> None:
    m = DriftMonitor(params(min_requests=20), SHA_A)
    assert feed(m, "t" * 19) == []  # share 1.0 but only 19 seen
    assert m.state is DriftState.CERTIFIED
    out = feed(m, "t")  # the 20th request arms it
    assert out and out[0].detector == DETECTOR_DEFERRAL


def test_min_requests_below_window_uses_partial_window() -> None:
    m = DriftMonitor(params(window_w=10, min_requests=2, tau=0.9), SHA_A)
    assert feed(m, "t") == []
    out = feed(m, "t")  # 2 seen, 2/2 = 1.0 >= 0.9
    assert out and out[0].details["deferral"]["window_requests"] == 2


def test_window_slides_old_deferrals_age_out() -> None:
    m = DriftMonitor(params(), SHA_A)
    assert feed(m, "tttt" + "s" * 30) == []  # 4 early deferrals leave the window
    assert m.window_stats()["deferral_share"] == 0.0


def test_audit_trips_at_exactly_k_in_window_m() -> None:
    m = DriftMonitor(params(audit_window_m=5, audit_k=3), SHA_A)
    assert feed(m, "sssss", "ddaa-") == []  # 2 disagreements
    out = feed(m, "s", "d")  # 3 disagreements among the last 5 audited (d, d, a, a, d)
    assert out and out[0].detector == DETECTOR_AUDIT
    assert out[0].details["audit"]["disagreements"] == 3
    assert m.state is DriftState.TEACHER


def test_audit_just_below_k_and_old_disagreements_age_out() -> None:
    m = DriftMonitor(params(audit_window_m=3, audit_k=2), SHA_A)
    # one disagreement per 3 audits at most (d a a a a d a a d a)
    assert feed(m, "s" * 10, "daaaadaada") == []
    assert m.state is DriftState.CERTIFIED
    assert m.window_stats()["audit_disagreements"] == 1


def test_audit_not_gated_by_min_requests() -> None:
    m = DriftMonitor(params(min_requests=1000, audit_k=2, audit_window_m=2), SHA_A)
    out = feed(m, "ss", "dd")
    assert out and out[0].detector == DETECTOR_AUDIT


def test_unaudited_students_do_not_dilute_the_audit_window() -> None:
    m = DriftMonitor(params(audit_window_m=3, audit_k=3), SHA_A)
    out = feed(m, "s" * 9, "d--d--d--")
    assert out and out[0].detector == DETECTOR_AUDIT


def test_both_detectors_firing_together_reports_both() -> None:
    p = params(window_w=1, min_requests=1, tau=1.0, audit_k=1, audit_window_m=1)
    m = DriftMonitor(p, SHA_A)
    # a student request cannot defer, so only (b) fires on it
    first = m.observe(ev("student", 0, "disagree"))
    assert first[0].detector == DETECTOR_AUDIT and first[0].details["fired"] == [DETECTOR_AUDIT]
    # both at once: deferral already 1/1 on a teacher request while a disagreement sits in the window
    m2 = DriftMonitor(
        params(window_w=2, min_requests=3, tau=0.5, audit_k=1, audit_window_m=1), SHA_A
    )
    assert m2.observe(ev("teacher", 0)) == [] and m2.observe(ev("teacher", 1)) == []
    out = m2.observe(ev("student", 2, "disagree"))  # deferral 1/2 >= 0.5 and 1 disagreement
    assert out[0].details["fired"] == [DETECTOR_DEFERRAL, DETECTOR_AUDIT]
    assert (
        out[0].detector == DETECTOR_DEFERRAL
        and m2.last_detector == "deferral_rate+audit_disagreement"
    )


def test_rollback_event_chain_and_numbers() -> None:
    m = DriftMonitor(params(), SHA_A)
    out = feed(m, "t" * 10)
    c, r, t = DriftState.CERTIFIED, DriftState.ROLLBACK, DriftState.TEACHER
    assert [(e.kind, e.state_from, e.state_to) for e in out] == [
        ("rollback", c, r),
        ("teacher_only", r, t),
        ("redistill_queued", t, t),
    ]
    d = out[0].details
    assert d["cert_sha256"] == SHA_A and d["fired"] == [DETECTOR_DEFERRAL]
    assert d["deferral"] == {
        "deferred": 10, "window_requests": 10, "share": 1.0, "tau": 0.5, "window_w": 10,
        "min_requests": 10,
    }  # fmt: skip
    assert d["request_id"] == "r9" and d["requests_seen"] == 10
    assert out[2].details["queued"] is True
    assert m.state is DriftState.TEACHER


def test_no_student_after_rollback_and_observe_is_inert() -> None:
    m = DriftMonitor(params(), SHA_A)
    feed(m, "t" * 10)
    assert not m.allows_student()
    assert feed(m, "s" * 50) == []  # ignored in TEACHER: no second rollback, no state change
    assert m.state is DriftState.TEACHER and not m.allows_student()
    assert "rollback: drift detected → teacher" in m.reason() and DETECTOR_DEFERRAL in m.reason()


def test_rollback_to_row_shape() -> None:
    row = feed(DriftMonitor(params(), SHA_A), "t" * 10)[0].to_row()
    assert set(row) == {"state_from", "state_to", "kind", "detector", "details"}
    assert row["state_from"] == "CERTIFIED" and row["state_to"] == "ROLLBACK"


def test_recertify_needs_teacher_state() -> None:
    m = DriftMonitor(params(), SHA_A)
    with pytest.raises(DriftError, match="TEACHER"):
        m.recertify(SHA_B, "abc")


def test_recertify_rules() -> None:
    m = DriftMonitor(params(), SHA_A)
    feed(m, "t" * 10)
    with pytest.raises(DriftError, match="differ"):
        m.recertify(SHA_A, "abc")  # same gate
    with pytest.raises(DriftError, match="64"):
        m.recertify("nothex", "abc")
    with pytest.raises(DriftError, match="64"):
        m.recertify(SHA_B.upper(), "abc")
    with pytest.raises(DriftError, match="prereg"):
        m.recertify(SHA_B, "  ")
    assert m.state is DriftState.TEACHER  # every refusal leaves the state alone
    out = m.recertify(SHA_B, " deadbeef ")
    assert [(e.kind, e.state_from, e.state_to) for e in out] == [
        ("recertify", DriftState.TEACHER, DriftState.CERTIFIED)
    ]
    assert out[0].details["prereg_commit"] == "deadbeef"
    assert out[0].details["previous_cert_sha256"] == SHA_A
    assert m.state is DriftState.CERTIFIED and m.allows_student() and m.cert_sha256 == SHA_B


def test_recertify_clears_windows_and_rearms() -> None:
    m = DriftMonitor(params(), SHA_A)
    feed(m, "t" * 10)
    m.recertify(SHA_B, "abc")
    assert m.window_stats()["requests_seen"] == 0
    assert feed(m, "t" * 9) == []  # min_requests applies afresh: stale window did not carry over
    assert feed(m, "t")[0].kind == "rollback"
    assert m.rolled_back_shas == (SHA_A, SHA_B)


def test_cannot_recertify_back_to_any_rolled_back_cert() -> None:
    m = DriftMonitor(params(), SHA_A)
    feed(m, "t" * 10)
    m.recertify(SHA_B, "abc")
    feed(m, "t" * 10)  # B rolls back too
    with pytest.raises(DriftError, match="differ"):
        m.recertify(SHA_A, "abc")
    with pytest.raises(DriftError, match="differ"):
        m.recertify(SHA_B, "abc")
    m.recertify(SHA_C, "abc")
    assert m.state is DriftState.CERTIFIED


def test_restore_teacher_state_survives_a_restart() -> None:
    m = DriftMonitor(params(), SHA_A)
    fresh = DriftMonitor(params(), SHA_A)
    fresh.restore(rows_of(feed(m, "t" * 10)))
    assert fresh.state is DriftState.TEACHER and not fresh.allows_student()
    assert fresh.rolled_back_shas == (SHA_A,)
    assert DETECTOR_DEFERRAL in fresh.reason()
    # a different cert.json on disk does not bring the student back by itself
    assert fresh.bind_cert(SHA_B) == []
    assert fresh.state is DriftState.TEACHER and fresh.cert_sha256 == SHA_B


def test_restore_trailing_rollback_settles_to_teacher() -> None:
    m = DriftMonitor(params(), SHA_A)
    only_first = rows_of(feed(m, "t" * 10)[:1])  # crash after the first event
    fresh = DriftMonitor(params(), SHA_A)
    fresh.restore(only_first)
    assert fresh.state is DriftState.TEACHER


def test_restore_after_recertify_is_certified_with_the_new_cert() -> None:
    m = DriftMonitor(params(), SHA_A)
    rows = rows_of(feed(m, "t" * 10)) + rows_of(m.recertify(SHA_B, "abc"))
    fresh = DriftMonitor(params(), SHA_A)
    fresh.restore(rows)
    assert fresh.state is DriftState.CERTIFIED and fresh.cert_sha256 == SHA_B
    assert fresh.rolled_back_shas == (SHA_A,)
    assert fresh.allows_student()


def test_restore_empty_is_certified() -> None:
    m = DriftMonitor(params(), SHA_A)
    m.restore([])
    assert m.state is DriftState.CERTIFIED


def test_stale_rolled_back_cert_on_disk_is_revoked_on_bind() -> None:
    m = DriftMonitor(params(), SHA_A)
    rows = rows_of(feed(m, "t" * 10)) + rows_of(m.recertify(SHA_B, "abc"))
    fresh = DriftMonitor(params(), SHA_A)
    fresh.restore(rows)
    out = fresh.bind_cert(SHA_A)  # somebody put the old cert.json back
    assert out and out[0].detector == DETECTOR_REVOKED
    assert fresh.state is DriftState.TEACHER
    assert fresh.bind_cert(SHA_C) == [] and fresh.state is DriftState.TEACHER


def test_should_audit_is_deterministic_and_proportional() -> None:
    ids = [f"req_{i}" for i in range(4000)]
    a = [should_audit(i, 0.1) for i in ids]
    assert a == [should_audit(i, 0.1) for i in ids]
    assert 0.07 < sum(a) / len(a) < 0.13
    assert not any(should_audit(i, 0.0) for i in ids)
    assert all(should_audit(i, 1.0) for i in ids)
    # the selection at a lower rate is a subset of a higher rate's selection
    assert all(should_audit(i, 0.2) for i in ids if should_audit(i, 0.1))


def test_wants_audit_uses_the_params_rate() -> None:
    assert DriftMonitor(params(audit_rate=1.0)).wants_audit("x")
    assert not DriftMonitor(params(audit_rate=0.0)).wants_audit("x")
