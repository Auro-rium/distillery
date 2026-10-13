# ruff: noqa: S101, S105, S106, E501
"""Drift guard in the gateway: rollback flips routing to the teacher, restarts keep it there, audits
spend through the daily cap, endpoint shapes, recertify. Everything is fake: no network."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from server_helpers import ADMIN, GOLD, make_config
from test_gateway import (
    LAMBDA,
    NANO,
    TEACHER,
    Env,
    _chat,
    skill_body,
    write_router,
)

from distillery.drift import DriftError, DriftParams
from distillery.server.gateway import load_router
from distillery.server.gateway_config import drift_from_env
from distillery.student import FakeStudent

WRONG = "SELECT COUNT(*) FROM accounts WHERE 1 = 0"  # runs, returns one row, wrong answer
ARROW_REASON = "rollback: drift detected → teacher"
SHA = "f" * 64


def drift(**kw: Any) -> DriftParams:
    base: dict[str, Any] = {
        "window_w": 4, "tau": 0.75, "audit_rate": 0.0, "audit_window_m": 4, "audit_k": 2,
        "min_requests": 4,
    }  # fmt: skip
    base.update(kw)
    return DriftParams(**base)


class Knobs:
    """A student whose answer and confidence the test changes between requests."""

    def __init__(self, sql: str = GOLD, logprob: float = -0.1) -> None:
        self.sql, self.logprob = sql, logprob
        self.hook: Any = None
        self.student = FakeStudent(self._sql, lambda m, t: self.logprob)

    def _sql(self, messages: Any) -> str:
        if self.hook is not None:
            self.hook()
        return self.sql


def env_for(
    tmp_path: Path, knobs: Knobs | None, params: DriftParams | None = None, **kw: Any
) -> Env:
    gw: dict[str, Any] = dict(kw.pop("gateway", {}))
    if params is not None:
        gw["drift"] = params
    router = kw.pop("router", None) or (
        tmp_path / "router"
        if (tmp_path / "router" / "cert.json").exists()
        else write_router(tmp_path / "router")
    )
    return Env(
        tmp_path,
        router=router,
        student=None if knobs is None else knobs.student,
        nn_sim=lambda q: 0.9,
        gateway=gw,
        **kw,
    )


def served(env: Env, n: int = 1) -> list[dict[str, Any]]:
    return [env.post(skill_body()).json()["distillery"] for _ in range(n)]


def status(env: Env) -> dict[str, Any]:
    return env.client.get("/api/gateway/drift").json()  # type: ignore[no-any-return]


def events(env: Env) -> list[dict[str, Any]]:
    return env.app.state.reader.real.drift_events()  # type: ignore[no-any-return]


# ---- rollback flips routing -----------------------------------------------------------------


def test_deferral_burst_rolls_back_and_the_student_is_never_called_again(tmp_path: Path) -> None:
    k = Knobs(logprob=-3.0)  # low confidence: the router itself defers every request
    env = env_for(tmp_path, k, drift())
    assert {d["served_by"] for d in served(env, 4)} == {"teacher"}
    assert k.student.calls == 4 and status(env)["state"] == "TEACHER"
    k.logprob = -0.1  # a confident student again: still not trusted
    calls_before = k.student.calls
    r = env.post(skill_body())
    d = r.json()["distillery"]
    assert d["served_by"] == "teacher"
    assert d["reason"] == f"{ARROW_REASON} (deferral_rate)"
    assert (
        "rollback" in r.headers["x-distillery-reason"] and "->" in r.headers["x-distillery-reason"]
    )
    assert k.student.calls == calls_before  # not even invoked: no sandbox call, no embedding
    last = env.rows()[-1]
    assert (
        last["served_by"] == "teacher"
        and last["student_called"] == 0
        and last["reason"] == d["reason"]
    )
    assert [e["kind"] for e in events(env)] == ["rollback", "teacher_only", "redistill_queued"]
    assert events(env)[0]["details"]["deferral"]["share"] == 1.0
    assert events(env)[0]["details"]["cert_sha256"] == load_router(tmp_path / "router").cert_sha256


def test_student_served_requests_do_not_trip_it(tmp_path: Path) -> None:
    env = env_for(tmp_path, Knobs(), drift())
    assert {d["served_by"] for d in served(env, 12)} == {"student"}
    s = status(env)
    assert (
        s["state"] == "CERTIFIED"
        and s["allows_student"] is True
        and s["window"]["deferral_share"] == 0.0
    )
    assert events(env) == []


def test_infrastructure_deferrals_do_not_count_as_drift(tmp_path: Path) -> None:
    env = env_for(tmp_path, None, drift())  # no student sandbox: every request defers for infra
    assert {d["served_by"] for d in served(env, 10)} == {"teacher"}
    s = status(env)
    assert s["state"] == "CERTIFIED" and s["window"]["requests_seen"] == 0
    assert events(env) == []


def test_uncertified_router_never_trips_the_guard(tmp_path: Path) -> None:
    router = write_router(tmp_path / "router", lam=None)
    env = env_for(tmp_path, Knobs(), drift(), router=router)
    served(env, 10)
    assert status(env)["state"] == "CERTIFIED" and events(env) == []


def test_precheck_short_circuits_count_as_router_decisions(tmp_path: Path) -> None:
    env = Env(
        tmp_path, router=write_router(tmp_path / "router"), student=Knobs().student,
        nn_sim=lambda q: 0.0, gateway={"drift": drift()},
    )  # fmt: skip
    ds = served(env, 4)  # unfamiliar questions: the router answers "teacher" before the student
    assert all(d["short_circuit"] for d in ds)
    assert status(env)["state"] == "TEACHER"


def test_rollback_mid_flight_is_honoured_before_the_student_answer_is_served(
    tmp_path: Path,
) -> None:
    k = Knobs()
    env = env_for(tmp_path, k, drift())
    gw = env.app.state.gateway

    def trip() -> None:  # another request rolls the gateway back while this one is in the student
        from distillery.drift import RouteEvent

        for i in range(4):
            gw.drift.observe(RouteEvent("teacher", f"x{i}"))

    k.hook = trip
    d = env.post(skill_body()).json()["distillery"]
    assert d["served_by"] == "teacher" and d["reason"].startswith(ARROW_REASON)


def test_only_skill_requests_feed_the_monitor(tmp_path: Path) -> None:
    env = env_for(tmp_path, Knobs(), drift())
    for _ in range(6):
        assert env.post(_chat(NANO)).status_code == 200
    assert status(env)["window"]["requests_seen"] == 0


# ---- restart ----------------------------------------------------------------------------------


def test_restart_restores_teacher_state(tmp_path: Path) -> None:
    env = env_for(tmp_path, Knobs(logprob=-3.0), drift())
    served(env, 4)
    assert status(env)["state"] == "TEACHER"
    k2 = Knobs()
    env2 = env_for(tmp_path, k2, drift())  # same data root = the same store
    s = status(env2)
    assert s["state"] == "TEACHER" and s["allows_student"] is False
    assert len(s["rolled_back_cert_sha256"]) == 1
    d = env2.post(skill_body()).json()["distillery"]
    assert d["served_by"] == "teacher" and d["reason"].startswith(ARROW_REASON)
    assert k2.student.calls == 0


def test_restart_without_drift_config_still_stays_rolled_back(tmp_path: Path) -> None:
    env = env_for(tmp_path, Knobs(logprob=-3.0), drift())
    served(env, 4)
    env2 = env_for(tmp_path, Knobs(), None)  # the config was removed
    s = status(env2)
    assert s["enabled"] is True and s["state"] == "TEACHER" and s["params"]["tau"] == 0.75
    assert env2.post(skill_body()).json()["distillery"]["served_by"] == "teacher"


def test_restart_with_a_new_cert_on_disk_does_not_recertify_by_itself(tmp_path: Path) -> None:
    env = env_for(tmp_path, Knobs(logprob=-3.0), drift())
    served(env, 4)
    write_router(tmp_path / "router", lam=0.7)  # a new cert.json appears
    env2 = env_for(tmp_path, Knobs(), drift())
    assert status(env2)["state"] == "TEACHER"
    assert env2.post(skill_body()).json()["distillery"]["served_by"] == "teacher"


def test_restart_when_never_rolled_back_is_certified_and_unconfigured_is_disabled(
    tmp_path: Path,
) -> None:
    env = env_for(tmp_path, Knobs(), drift())
    served(env, 2)
    assert status(env_for(tmp_path, Knobs(), drift()))["state"] == "CERTIFIED"
    off = status(env_for(tmp_path, Knobs(), None))
    assert off["enabled"] is False and off["state"] is None and "not configured" in off["note"]


# ---- audits ----------------------------------------------------------------------------------


def test_audit_spend_goes_through_the_cap_and_is_recorded(tmp_path: Path) -> None:
    env = env_for(tmp_path, Knobs(), drift(audit_rate=1.0))
    ds = served(env, 3)
    assert {d["served_by"] for d in ds} == {"student"}
    assert len(env.upstream.calls) == 3 and {c[0] for c in env.upstream.calls} == {TEACHER}
    a = status(env)["audits"]
    assert (
        a["audits"] == 3
        and a["skipped_cap"] == 0
        and a["spent_usd"] == pytest.approx(3 * (1200 * 1.0 + 30 * 2.0) / 1e6)
    )
    spent = env.client.get("/api/config").json()["playground"]["spent_today_usd"]
    assert spent == pytest.approx(
        a["spent_usd"]
    )  # the same daily ledger the cap is checked against
    rows = [r for r in env.rows() if r["skill"] == 1]
    assert all(
        r["served_by"] == "student" and r["cost_usd"] > 0 for r in rows
    )  # audit $ on the row
    s = status(env)
    assert s["window"]["audits_in_window"] == 3 and s["window"]["audit_disagreements"] == 0
    assert s["state"] == "CERTIFIED"


def test_audit_is_skipped_and_logged_when_the_cap_refuses(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    env = env_for(
        tmp_path,
        Knobs(),
        drift(audit_rate=1.0),
        config=make_config(playground_daily_cap_usd=0.0005),
    )
    with caplog.at_level("WARNING", logger="distillery.gateway"):
        ds = served(env, 3)
    assert {d["served_by"] for d in ds} == {"student"}  # the user's answer is unaffected
    assert env.upstream.calls == []  # no teacher call was made past the cap
    a = status(env)["audits"]
    assert a["skipped_cap"] == 3 and a["audits"] == 0 and a["spent_usd"] == 0
    assert "drift audit skipped" in caplog.text and "daily_budget_exhausted" in caplog.text
    assert status(env)["window"]["audits_in_window"] == 0  # a skipped audit is no verdict


def test_audit_without_an_upstream_is_skipped(tmp_path: Path) -> None:
    env = env_for(tmp_path, Knobs(), drift(audit_rate=1.0))
    env.app.state.gateway.upstream = None
    assert served(env, 1)[0]["served_by"] == "student"
    assert status(env)["audits"]["skipped_error"] == 1


def test_audit_disagreements_roll_back_and_the_audit_catches_a_confidently_wrong_student(
    tmp_path: Path,
) -> None:
    k = Knobs(sql=WRONG)  # confident, runs fine, wrong answer: the router cannot see this
    env = env_for(tmp_path, k, drift(audit_rate=1.0, audit_k=2, audit_window_m=4))
    ds = served(env, 2)
    assert {d["served_by"] for d in ds} == {"student"}  # both wrong answers did go out
    s = status(env)
    assert s["state"] == "TEACHER" and s["detector"] == "audit_disagreement"
    ev = events(env)[0]
    assert ev["detector"] == "audit_disagreement" and ev["details"]["audit"]["disagreements"] == 2
    nxt = env.post(skill_body()).json()["distillery"]
    assert nxt["served_by"] == "teacher" and nxt["reason"] == f"{ARROW_REASON} (audit_disagreement)"
    assert k.student.calls == 2


def test_agreeing_audits_compare_executed_results_not_sql_text(tmp_path: Path) -> None:
    # different SQL text, same result set as the teacher's `SELECT COUNT(*) FROM accounts`
    env = env_for(
        tmp_path, Knobs(sql="SELECT COUNT(account_id) FROM accounts"), drift(audit_rate=1.0)
    )
    served(env, 3)
    s = status(env)
    assert s["window"]["audit_disagreements"] == 0 and s["window"]["audits_in_window"] == 3


def test_audit_where_the_teacher_sql_does_not_run_is_inconclusive(tmp_path: Path) -> None:
    from server_helpers import MODELS
    from test_gateway import FakeUpstream

    up = FakeUpstream({MODELS["teacher"]: "```sql\nSELECT nope FROM missing\n```"})
    env = env_for(tmp_path, Knobs(), drift(audit_rate=1.0), upstream=up)
    served(env, 2)
    s = status(env)
    assert s["audits"]["inconclusive"] == 2 and s["window"]["audits_in_window"] == 0


def test_audit_selection_is_deterministic_by_request_id(tmp_path: Path) -> None:
    env = env_for(tmp_path, Knobs(), drift(audit_rate=0.5, window_w=100, min_requests=100))
    ds = served(env, 10)
    mon = env.app.state.gateway.drift
    expect = sum(mon.wants_audit(d["request_id"]) for d in ds)
    assert status(env)["audits"]["audits"] == expect == len(env.upstream.calls)


# ---- endpoints --------------------------------------------------------------------------------


def test_drift_endpoint_shape_when_enabled(tmp_path: Path) -> None:
    env = env_for(tmp_path, Knobs(logprob=-3.0), drift(audit_rate=0.2))
    served(env, 4)
    s = status(env)
    assert set(s) == {
        "enabled", "state", "allows_student", "reason", "detector", "params", "cert_sha256",
        "rolled_back_cert_sha256", "window", "audits", "events", "note",
    }  # fmt: skip
    assert s["enabled"] is True and s["state"] == "TEACHER" and s["allows_student"] is False
    assert s["reason"] == f"{ARROW_REASON} (deferral_rate)" and s["detector"] == "deferral_rate"
    assert s["params"] == {
        "window_w": 4, "tau": 0.75, "audit_rate": 0.2, "audit_window_m": 4, "audit_k": 2,
        "min_requests": 4,
    }  # fmt: skip
    assert len(s["cert_sha256"]) == 64 and s["rolled_back_cert_sha256"] == [s["cert_sha256"]]
    assert set(s["window"]) == {
        "requests_seen", "window_requests", "window_deferred", "deferral_share",
        "audits_in_window", "audit_disagreements", "armed",
    }  # fmt: skip
    assert set(s["audits"]) == {
        "audits",
        "inconclusive",
        "skipped_cap",
        "skipped_error",
        "spent_usd",
    }
    assert [e["kind"] for e in s["events"]] == [
        "redistill_queued",
        "teacher_only",
        "rollback",
    ]  # newest first
    e = s["events"][-1]
    assert set(e) == {"id", "created_at", "state_from", "state_to", "kind", "detector", "details"}
    assert (e["state_from"], e["state_to"]) == ("CERTIFIED", "ROLLBACK")
    assert env.client.get("/api/gateway/status").json()["drift"] == {
        "enabled": True,
        "state": "TEACHER",
    }


def test_drift_endpoint_shape_when_disabled(tmp_path: Path) -> None:
    env = env_for(tmp_path, Knobs(), None)
    s = status(env)
    assert (
        s["enabled"] is False and s["state"] is None and s["window"] is None and s["events"] == []
    )
    assert env.client.get("/api/gateway/status").json()["drift"] == {
        "enabled": False,
        "state": None,
    }
    assert served(env, 6)[0]["served_by"] == "student"


def test_drift_routes_beat_the_spa_catch_all(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>SPA</html>")
    env = env_for(tmp_path, Knobs(), drift(), frontend_dist=dist)
    r = env.client.get("/api/gateway/drift")
    assert r.headers["content-type"].startswith("application/json") and r.json()["enabled"] is True
    r = env.client.post("/api/gateway/drift/recertify", json={})
    assert r.status_code == 401 and r.json()["error"] == "admin_token_required"


# ---- recertify ---------------------------------------------------------------------------------


def recert(env: Env, body: Any, token: str | None = ADMIN) -> Any:
    return env.client.post(
        "/api/gateway/drift/recertify",
        json=body,
        headers={} if token is None else {"x-admin-token": token},
    )


def cert_sha(d: Path) -> str:
    return hashlib.sha256((d / "cert.json").read_bytes()).hexdigest()


def test_recertify_is_admin_only(tmp_path: Path) -> None:
    env = env_for(tmp_path, Knobs(logprob=-3.0), drift())
    served(env, 4)
    body = {"cert_sha256": SHA, "prereg_commit": "abc"}
    r = recert(env, body, None)
    assert r.status_code == 401 and r.json()["error"] == "admin_token_required"
    r = recert(env, body, "wrong-token")
    assert r.status_code == 403 and r.json()["error"] == "forbidden"
    assert status(env)["state"] == "TEACHER"


def test_recertify_when_disabled_or_not_rolled_back(tmp_path: Path) -> None:
    off = env_for(tmp_path / "a", Knobs(), None)
    assert recert(off, {"cert_sha256": SHA, "prereg_commit": "c"}).status_code == 404
    on = env_for(tmp_path / "b", Knobs(), drift())
    r = recert(on, {"cert_sha256": SHA, "prereg_commit": "c"})
    assert r.status_code == 409 and r.json()["error"] == "not_rolled_back"


def test_recertify_rules_and_full_recovery(tmp_path: Path) -> None:
    k = Knobs(logprob=-3.0)
    env = env_for(tmp_path, k, drift())
    served(env, 4)
    old = cert_sha(tmp_path / "router")
    r = env.client.post(
        "/api/gateway/drift/recertify", content=b"{nope", headers={"x-admin-token": ADMIN}
    )
    assert r.status_code == 400 and r.json()["error"] == "invalid_json"
    for bad in ([], {"cert_sha256": SHA}, {"cert_sha256": 5, "prereg_commit": "c"},
                {"cert_sha256": SHA, "prereg_commit": " "}, {"cert_sha256": "XYZ", "prereg_commit": "c"},
                {"cert_sha256": SHA, "prereg_commit": "c", "extra": 1}):  # fmt: skip
        r = recert(env, bad)
        assert r.status_code == 422 and r.json()["error"] == "invalid_request", bad
    r = recert(env, {"cert_sha256": old, "prereg_commit": "c"})  # the gate we rolled back from
    assert r.status_code == 409 and r.json()["error"] == "same_certificate"
    r = recert(env, {"cert_sha256": SHA, "prereg_commit": "c"})  # not what cert.json holds
    assert (
        r.status_code == 409 and r.json()["error"] == "cert_mismatch" and old in r.json()["message"]
    )
    assert status(env)["state"] == "TEACHER"

    write_router(tmp_path / "router", lam=0.7)  # a NEW certificate (different bytes)
    new = cert_sha(tmp_path / "router")
    assert new != old
    k.logprob = -0.1
    r = recert(env, {"cert_sha256": new, "prereg_commit": " 1234abc "})
    assert r.status_code == 200, r.text
    b = r.json()
    assert (
        b["state"] == "CERTIFIED"
        and b["cert_sha256"] == new
        and b["prereg_commit"] == "1234abc"
        and b["lambda"] == 0.7
    )
    assert (b["event"]["kind"], b["event"]["state_from"], b["event"]["state_to"]) == (
        "recertify",
        "TEACHER",
        "CERTIFIED",
    )
    d = env.post(skill_body()).json()[
        "distillery"
    ]  # the student serves again, under the new lambda
    assert d["served_by"] == "student" and d["lambda"] == 0.7
    s = status(env)
    assert (
        s["state"] == "CERTIFIED"
        and s["cert_sha256"] == new
        and s["rolled_back_cert_sha256"] == [old]
    )
    assert s["window"]["requests_seen"] == 1  # windows restarted at the recertify
    assert env.client.get("/api/gateway/status").json()["lambda"] == 0.7
    assert [e["kind"] for e in events(env)][-1] == "recertify"
    assert env.app.state.reader.real._query("SELECT actor, action FROM audit")[-1] == (
        "admin",
        "drift_recertify",
    )  # noqa: SLF001
    # a restart after the recertify stays certified, with the new cert
    env2 = env_for(tmp_path, Knobs(), drift())
    assert status(env2)["state"] == "CERTIFIED" and status(env2)["cert_sha256"] == new
    # and the old gate can never come back, not even after a second rollback
    r2 = recert(env2, {"cert_sha256": old, "prereg_commit": "c"})
    assert r2.status_code == 409


def test_recertify_with_an_uncertified_new_cert_is_refused(tmp_path: Path) -> None:
    env = env_for(tmp_path, Knobs(logprob=-3.0), drift())
    served(env, 4)
    write_router(tmp_path / "router", lam=None)  # a cert that found no threshold
    r = recert(env, {"cert_sha256": cert_sha(tmp_path / "router"), "prereg_commit": "c"})
    assert r.status_code == 409 and r.json()["error"] == "not_certified"
    assert status(env)["state"] == "TEACHER"


def test_a_stale_rolled_back_cert_put_back_on_disk_is_rolled_back_again_at_startup(
    tmp_path: Path,
) -> None:
    env = env_for(tmp_path, Knobs(logprob=-3.0), drift())
    served(env, 4)
    old = cert_sha(tmp_path / "router")
    write_router(tmp_path / "router", lam=0.7)
    assert (
        recert(
            env, {"cert_sha256": cert_sha(tmp_path / "router"), "prereg_commit": "c"}
        ).status_code
        == 200
    )
    write_router(tmp_path / "router", lam=LAMBDA)  # somebody restores the old cert.json
    assert cert_sha(tmp_path / "router") == old
    env2 = env_for(tmp_path, Knobs(), drift())
    s = status(env2)
    assert s["state"] == "TEACHER" and s["detector"] == "revoked_certificate"


# ---- configuration ------------------------------------------------------------------------------


def test_cert_sha256_is_the_hash_of_the_cert_json_bytes(tmp_path: Path) -> None:
    d = write_router(tmp_path / "r")
    assert load_router(d).cert_sha256 == hashlib.sha256((d / "cert.json").read_bytes()).hexdigest()
    assert load_router(None).cert_sha256 is None


def test_drift_config_from_env() -> None:
    assert drift_from_env({}) is None
    p = drift_from_env({"DISTILLERY_DRIFT_WINDOW": "50", "DISTILLERY_DRIFT_TAU": "0.7"})
    assert p == DriftParams(
        window_w=50, tau=0.7, audit_rate=0.0, audit_window_m=50, audit_k=5, min_requests=50
    )
    full = drift_from_env({
        "DISTILLERY_DRIFT_WINDOW": "20", "DISTILLERY_DRIFT_TAU": "0.6", "DISTILLERY_DRIFT_AUDIT_RATE": "0.1",
        "DISTILLERY_DRIFT_AUDIT_WINDOW": "30", "DISTILLERY_DRIFT_AUDIT_K": "4", "DISTILLERY_DRIFT_MIN_REQUESTS": "10",
    })  # fmt: skip
    assert full == DriftParams(20, 0.6, 0.1, 30, 4, 10)
    with pytest.raises(DriftError):
        drift_from_env({"DISTILLERY_DRIFT_TAU": "0.7"})  # window missing: half-configured
    with pytest.raises(DriftError):
        drift_from_env({"DISTILLERY_DRIFT_WINDOW": "x", "DISTILLERY_DRIFT_TAU": "0.7"})


def test_drift_json_in_the_router_dir_enables_the_guard(tmp_path: Path) -> None:
    router = write_router(tmp_path / "router")
    (router / "drift.json").write_text(json.dumps({"window_w": 4, "tau": 0.75, "min_requests": 4}))
    env = env_for(tmp_path, Knobs(logprob=-3.0), None, router=router)
    assert status(env)["params"]["tau"] == 0.75
    served(env, 4)
    assert status(env)["state"] == "TEACHER"


def test_explicit_settings_beat_drift_json(tmp_path: Path) -> None:
    router = write_router(tmp_path / "router")
    (router / "drift.json").write_text(json.dumps({"window_w": 4, "tau": 0.75}))
    env = env_for(tmp_path, Knobs(), drift(tau=0.5), router=router)
    assert status(env)["params"]["tau"] == 0.5


def test_malformed_drift_json_stops_the_server_instead_of_running_unguarded(tmp_path: Path) -> None:
    router = write_router(tmp_path / "router")
    (router / "drift.json").write_text(json.dumps({"window_w": 4}))  # tau missing
    with pytest.raises(DriftError):
        env_for(tmp_path, Knobs(), None, router=router)
