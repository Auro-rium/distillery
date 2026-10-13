# ruff: noqa: S101, S105, S106
from __future__ import annotations

import sys

import pytest
from aicloud_helpers import TOKEN, FakeClock, FakeNebius, make_plan
from pydantic import ValidationError

from distillery import aicloud_endpoint as ae
from distillery.budget import BudgetExceeded, Ledger
from distillery.store import Store


def ledger(run: float = 10.0, proj: float = 19.5, store: Store | None = None) -> Ledger:
    return Ledger("h1", run, proj, 1.0, store=store)


# ---------------------------------------------------------------- argv / plan


def test_create_argv_has_adapter_volume_port_and_vllm_args() -> None:
    plan = make_plan(adapter_s3_profile="prof", max_lora_rank=32)
    argv = ae.create_argv(plan)
    assert argv[:4] == ["nebius", "ai", "endpoint", "create"]
    assert argv[argv.index("--image") + 1] == "vllm/vllm-openai:v0.0.test"
    assert argv[argv.index("--platform") + 1] == "gpu-l40s-d"
    assert argv[argv.index("--container-port") + 1] == "8000"
    assert argv[argv.index("--volume") + 1] == "s3://bkt/n1:/adapters/n1:ro:prof"
    assert argv[argv.index("--auth") + 1] == "token" and argv[argv.index("--token") + 1] == TOKEN
    assert "--on-demand" in argv and "--preemptible" not in argv
    assert argv[-3:] == ["--async", "--format", "json"]
    args = argv[argv.index("--args") + 1]
    assert args == (
        "--model Qwen/Qwen3-1.7B --enable-lora --lora-modules n1=/adapters/n1 "
        "--max-lora-rank 32 --max-logprobs 5 --dtype bfloat16 --max-model-len 4096 --port 8000"
    )
    assert "--preset" not in argv
    assert "--preset" in ae.create_argv(make_plan(preset="1gpu-8vcpu-32gb"))
    assert "--preemptible" in ae.create_argv(make_plan(preemptible=True))
    assert ae.vllm_args(make_plan()).count("--max-lora-rank 64") == 1  # the default rank


def test_other_argv() -> None:
    assert ae.get_argv("e1") == [
        "nebius",
        "ai",
        "endpoint",
        "get",
        "--id",
        "e1",
        "--format",
        "json",
    ]
    assert ae.delete_argv("e1") == ["nebius", "ai", "endpoint", "delete", "--id", "e1"]
    assert ae.list_argv() == ["nebius", "ai", "endpoint", "list", "--format", "json"]


def test_token_is_redacted_everywhere_printable() -> None:
    plan = make_plan()
    shown = ae.render_argv(ae.create_argv(plan), TOKEN)
    assert TOKEN not in shown and "--token '***'" in shown
    assert TOKEN not in str(ae.redact_argv(ae.create_argv(plan)))
    assert TOKEN not in repr(plan) and TOKEN not in str(plan.public_dict())
    assert plan.public_dict()["auth_token"] == "***"


def test_default_token_is_generated_and_distinct() -> None:
    a = ae.EndpointPlan(vllm_tag="v1", hourly_usd=1.0, adapter_s3_source="s3://b")
    b = ae.EndpointPlan(vllm_tag="v1", hourly_usd=1.0, adapter_s3_source="s3://b")
    assert len(a.auth_token) >= 32 and a.auth_token != b.auth_token and a.name != b.name


def test_plan_validation() -> None:
    with pytest.raises(ValidationError):
        make_plan(vllm_tag="latest")
    with pytest.raises(ValidationError):
        make_plan(adapter_s3_source="/local/path")
    with pytest.raises(ValidationError):
        make_plan(hourly_usd=0)
    with pytest.raises(ValidationError):  # hourly price has no default
        ae.EndpointPlan(vllm_tag="v1", adapter_s3_source="s3://b")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):  # tag has no default
        ae.EndpointPlan(hourly_usd=1.0, adapter_s3_source="s3://b")  # type: ignore[call-arg]
    assert make_plan().time_box_minutes == 45
    assert make_plan(hourly_usd=3.0, time_box_minutes=30).estimated_cost_usd == pytest.approx(1.5)


# ---------------------------------------------------------------- lifecycle


def test_happy_path_deletes_confirms_and_records_spend() -> None:
    fc = FakeClock()
    cli = FakeNebius()
    lg = ledger()
    with ae.aicloud_endpoint(
        make_plan(time_box_minutes=90), cli, lg, fc.clock(), poll_interval_s=10.0
    ) as h:
        assert h.endpoint_id == "ep-1" and h.base_url == "https://ep-1.example.test/v1"
        assert h.token == TOKEN and TOKEN not in repr(h)
        assert cli.endpoints == {"ep-1": "t-ep"}  # alive while measuring
        fc.sleep(3600.0 - 10.0 - 1.0)  # an hour of use in total (10 s spent polling)
        assert not h.expired() and h.time_left_s() > 0
    assert cli.endpoints == {} and cli.verbs() == ["create", "get", "get", "delete", "list"]
    assert h.deleted_confirmed and not h.timed_out and h.error is None
    assert h.created_at == "T+0" and h.ready_at != "" and h.deleted_at != ""
    assert h.duration_s == pytest.approx(3600.0 - 1.0)
    assert h.spend_usd == pytest.approx(2.0 * h.duration_s / 3600.0)
    assert lg.spent() == pytest.approx(h.spend_usd)  # 'aicloud_endpoint' kind, in the ledger


def test_spend_row_kind_and_not_an_llm_call(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import sqlite3

    fc = FakeClock()
    store = Store(tmp_path)
    with ae.aicloud_endpoint(make_plan(), FakeNebius(), ledger(store=store), fc.clock()):
        fc.sleep(1800)
    store.close()
    con = sqlite3.connect(tmp_path / "index.sqlite")
    rows = con.execute("SELECT kind, usd, input_tokens FROM spend").fetchall()
    assert [r[0] for r in rows] == ["aicloud_endpoint"] and rows[0][1] > 0.9
    assert con.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0] == 0
    con.close()


def test_exception_in_body_still_deletes_and_records_spend() -> None:
    fc = FakeClock()
    cli = FakeNebius()
    lg = ledger()
    with pytest.raises(ValueError, match="boom"):
        with ae.aicloud_endpoint(make_plan(), cli, lg, fc.clock()):
            fc.sleep(60)
            raise ValueError("boom")
    assert cli.endpoints == {} and "delete" in cli.verbs() and cli.verbs()[-1] == "list"
    assert lg.spent() > 0


def test_keyboard_interrupt_still_deletes() -> None:
    cli = FakeNebius()
    with pytest.raises(KeyboardInterrupt):
        with ae.aicloud_endpoint(make_plan(), cli, ledger(), FakeClock().clock()):
            raise KeyboardInterrupt
    assert cli.endpoints == {}


def test_delete_failure_raises_loudly_with_id_and_still_records_spend(caplog) -> None:  # type: ignore[no-untyped-def]
    caplog.set_level("WARNING")
    fc = FakeClock()
    cli = FakeNebius(delete_rc=1)
    lg = ledger()
    with pytest.raises(ae.EndpointDeleteError, match="ep-1") as ei:
        with ae.aicloud_endpoint(
            make_plan(), cli, lg, fc.clock(), delete_attempts=2, confirm_timeout_s=20.0
        ) as h:
            fc.sleep(120)
    assert "may still be billing" in str(ei.value) and ei.value.endpoint_id == "ep-1"
    assert TOKEN not in str(ei.value)
    assert cli.verbs().count("delete") == 2  # retried
    assert cli.endpoints == {"ep-1": "t-ep"}  # really still there
    assert lg.spent() > 0 and h.deleted_confirmed is False
    assert "delete of endpoint ep-1 failed" in caplog.text
    assert TOKEN not in caplog.text and TOKEN not in repr(h)  # logs never carry the token


def test_delete_failure_on_top_of_a_body_error_mentions_both() -> None:
    cli = FakeNebius(delete_rc=1)
    with pytest.raises(ae.EndpointDeleteError, match="ep-1.*original error: kaput|ep-1.*kaput"):
        with ae.aicloud_endpoint(
            make_plan(), cli, ledger(), FakeClock().clock(), delete_attempts=1, confirm_timeout_s=5
        ):
            raise RuntimeError("kaput")


def test_delete_confirmation_waits_for_list_to_drop_the_id() -> None:
    cli = FakeNebius(gone_after_list=2)
    with ae.aicloud_endpoint(make_plan(), cli, ledger(), FakeClock().clock()) as h:
        pass
    assert cli.verbs().count("list") == 3 and h.deleted_confirmed


def test_unconfirmable_list_means_not_confirmed() -> None:
    cli = FakeNebius(list_rc=1)
    with pytest.raises(ae.EndpointDeleteError):
        with ae.aicloud_endpoint(
            make_plan(), cli, ledger(), FakeClock().clock(), confirm_timeout_s=5
        ):
            pass


def test_time_box_watchdog_expires_then_deletes() -> None:
    fc = FakeClock()
    cli = FakeNebius()
    lg = ledger()
    plan = make_plan(time_box_minutes=10, delete_reserve_minutes=2)
    steps = 0
    with ae.aicloud_endpoint(plan, cli, lg, fc.clock()) as h:
        while not h.expired():  # what measure_throughput's should_stop does
            fc.sleep(30)
            steps += 1
        assert steps > 0
    assert h.timed_out and h.deleted_confirmed and cli.endpoints == {}
    # measuring stopped at box - reserve = 8 min; the endpoint was gone before the box ended
    assert 480.0 <= h.duration_s <= 600.0
    assert h.spend_usd == pytest.approx(2.0 * h.duration_s / 3600)
    assert h.spend_usd <= plan.estimated_cost_usd  # never above the figure that was approved
    assert lg.spent() == pytest.approx(h.spend_usd)


def test_worst_case_path_stays_within_the_estimate() -> None:
    """Slow list confirmation after a delete issued at the measurement deadline: the whole
    create -> confirmed-delete span still fits the time box, so spend <= preflight estimate."""
    fc = FakeClock()
    cli = FakeNebius(gone_after_list=20)  # list keeps showing it for 20 polls (5 s apart)
    plan = make_plan(time_box_minutes=10, delete_reserve_minutes=5)
    with ae.aicloud_endpoint(plan, cli, ledger(), fc.clock()) as h:
        while not h.expired():
            fc.sleep(1)
    assert h.deleted_confirmed and h.duration_s <= 600.0
    assert h.spend_usd <= plan.estimated_cost_usd + 1e-9


def test_reserve_must_fit_inside_the_box() -> None:
    with pytest.raises(ValidationError):
        make_plan(time_box_minutes=5, delete_reserve_minutes=5)
    assert make_plan().delete_reserve_minutes == 5.0


def test_global_ledger_hold_is_taken_for_the_endpoint_and_released(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from distillery.global_ledger import GlobalLedger

    gl = GlobalLedger(tmp_path / "g.sqlite")
    lg = Ledger("h1", 10.0, 19.5, 1.0, global_ledger=gl)
    fc = FakeClock()
    with ae.aicloud_endpoint(make_plan(time_box_minutes=60), FakeNebius(), lg, fc.clock()) as h:
        held = gl.summary()["reserved_usd"]
        assert held == pytest.approx(2.0)  # hourly 2.0 x 60 min: other processes see it
        fc.sleep(600)
    summary = gl.summary()
    assert summary["reserved_usd"] == 0 and summary["spend_usd"] == pytest.approx(h.spend_usd)


def test_global_ledger_hold_is_released_even_when_delete_fails(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from distillery.global_ledger import GlobalLedger

    gl = GlobalLedger(tmp_path / "g.sqlite")
    lg = Ledger("h1", 10.0, 19.5, 1.0, global_ledger=gl)
    with pytest.raises(ae.EndpointDeleteError):
        with ae.aicloud_endpoint(
            make_plan(), FakeNebius(delete_rc=1), lg, FakeClock().clock(),
            delete_attempts=1, confirm_timeout_s=5,
        ):  # fmt: skip
            pass
    assert gl.summary()["reserved_usd"] == 0 and gl.summary()["spend_usd"] > 0


def test_global_ledger_hold_respects_the_project_cap(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from distillery.global_ledger import GlobalLedger

    gl = GlobalLedger(tmp_path / "g.sqlite")
    gl.reserve("someone-else", 18.5, 19.5)
    lg = Ledger("h1", 10.0, 19.5, 1.0, global_ledger=gl)
    cli = FakeNebius()
    with pytest.raises(BudgetExceeded):
        with ae.aicloud_endpoint(make_plan(), cli, lg, FakeClock().clock()):
            pass
    assert cli.calls == []


def test_preflight_refuses_over_cap_before_anything_is_created() -> None:
    cli = FakeNebius()
    plan = make_plan(hourly_usd=4.0, time_box_minutes=60)  # estimate $4.00
    with pytest.raises(BudgetExceeded, match="run cap"):
        with ae.aicloud_endpoint(plan, cli, ledger(run=3.0), FakeClock().clock()):
            raise AssertionError("must not be reached")
    with pytest.raises(BudgetExceeded, match="project cap"):
        with ae.aicloud_endpoint(plan, cli, ledger(proj=3.0), FakeClock().clock()):
            raise AssertionError("must not be reached")
    assert cli.calls == []


def test_endpoint_error_state_fails_start_deletes_and_records_spend() -> None:
    cli = FakeNebius(states=("PENDING", "ERROR"))
    lg = ledger()
    with pytest.raises(ae.EndpointStartError, match="ERROR"):
        with ae.aicloud_endpoint(make_plan(), cli, lg, FakeClock().clock()):
            raise AssertionError("not ready")
    assert cli.endpoints == {} and lg.spent() > 0


def test_never_ready_times_out_then_deletes() -> None:
    cli = FakeNebius(states=("PENDING",))
    fc = FakeClock()
    with pytest.raises(ae.EndpointStartError, match="not ready"):
        with ae.aicloud_endpoint(make_plan(ready_timeout_minutes=2), cli, ledger(), fc.clock()):
            raise AssertionError("not ready")
    assert cli.endpoints == {} and fc.t - 1000.0 < 600


def test_ready_state_without_public_url_keeps_polling() -> None:
    cli = FakeNebius(states=("RUNNING",))
    orig = cli.run

    def run(argv):  # type: ignore[no-untyped-def]
        res = orig(argv)
        if argv[3] == "get":
            res.stdout = '{"status": {"state": "RUNNING"}}'
        return res

    cli.run = run  # type: ignore[method-assign]
    with pytest.raises(ae.EndpointStartError, match="not ready"):
        with ae.aicloud_endpoint(
            make_plan(ready_timeout_minutes=1), cli, ledger(), FakeClock().clock()
        ):
            pass


def test_probe_is_awaited_after_state_ready() -> None:
    fc = FakeClock()
    seen: list[str] = []

    def probe(h: ae.EndpointHandle) -> bool:
        seen.append(h.base_url)
        return len(seen) >= 3

    with ae.aicloud_endpoint(make_plan(), FakeNebius(), ledger(), fc.clock(), probe=probe):
        assert len(seen) == 3


def test_probe_never_answering_fails_start_and_deletes() -> None:
    cli = FakeNebius()
    with pytest.raises(ae.EndpointStartError, match="probe"):
        with ae.aicloud_endpoint(
            make_plan(ready_timeout_minutes=1), cli, ledger(), FakeClock().clock(),
            probe=lambda h: False,
        ):  # fmt: skip
            pass
    assert cli.endpoints == {}


def test_create_failure_cleans_up_by_name_if_it_half_succeeded() -> None:
    fc = FakeClock()
    cli = FakeNebius(
        create_rc=1, create_stderr=f"quota {TOKEN}", create_creates=True,
        create_advance_s=60.0, clock=fc,
    )  # fmt: skip
    lg = ledger()
    with pytest.raises(ae.EndpointStartError) as ei:
        with ae.aicloud_endpoint(make_plan(), cli, lg, fc.clock()):
            raise AssertionError("never created")
    assert TOKEN not in str(ei.value) and "***" in str(ei.value)  # stderr scrubbed
    assert cli.endpoints == {} and "delete" in cli.verbs()  # found by its unique name
    assert lg.spent() > 0  # an 'aicloud_endpoint' row exists: a half-created endpoint may bill


def test_create_failure_with_nothing_created_is_a_clean_confirmed_failure() -> None:
    cli = FakeNebius(create_rc=1, create_creates=False)
    with pytest.raises(ae.EndpointStartError, match="create failed"):
        with ae.aicloud_endpoint(make_plan(), cli, ledger(), FakeClock().clock()):
            pass
    assert "delete" not in cli.verbs()


def test_unparseable_create_output_falls_back_to_name_lookup() -> None:
    cli = FakeNebius(create_stdout="not json")
    with pytest.raises(ae.EndpointStartError, match="no endpoint id"):
        with ae.aicloud_endpoint(make_plan(), cli, ledger(), FakeClock().clock()):
            pass
    assert cli.endpoints == {}  # still deleted, found by name


def test_unknown_id_and_unlistable_raises_naming_the_endpoint() -> None:
    cli = FakeNebius(create_stdout="not json", list_rc=1)
    with pytest.raises(ae.EndpointDeleteError, match="t-ep"):
        with ae.aicloud_endpoint(make_plan(), cli, ledger(), FakeClock().clock()):
            pass


def test_on_created_reports_the_id() -> None:
    got: list[str] = []
    with ae.aicloud_endpoint(
        make_plan(), FakeNebius(), ledger(), FakeClock().clock(), on_created=got.append
    ):
        pass
    assert got == ["ep-1"]


def test_public_endpoint_shapes() -> None:
    assert ae._public_url('{"status":{"public_endpoints":["a.b"]}}') == "https://a.b"
    assert (
        ae._public_url('{"status":{"public_endpoints":[{"url":"https://x.y/"}]}}') == "https://x.y"
    )
    assert ae._public_url('{"status":{}}') is None
    assert ae.parse_state('{"status":{"state":"running"}}') == "RUNNING"
    assert ae.parse_endpoint_id('{"metadata":{"resource_id":"z"}}') == "z"


# ---------------------------------------------------------------- real runner (local, no CLI)


def test_subprocess_runner_maps_failures_to_return_codes() -> None:
    r = ae.SubprocessRunner(timeout_s=10)
    ok = r.run([sys.executable, "-c", "print('hi')"])
    assert ok.returncode == 0 and ok.stdout.strip() == "hi"
    assert r.run([sys.executable, "-c", "raise SystemExit(3)"]).returncode == 3
    assert r.run(["/nonexistent/definitely-not-a-binary"]).returncode == 127
    slow = ae.SubprocessRunner(timeout_s=0.2).run(
        [sys.executable, "-c", "import time; time.sleep(5)"]
    )
    assert slow.returncode == 124
