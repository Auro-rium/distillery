"""Tool-call pack: env determinism, gold replay, corruptions, families/stress, extraction."""

from __future__ import annotations

import copy
import json
import random
from collections import Counter

import pytest

from distillery.paraphrase import pseudo_task
from distillery.taskpacks.toolcall import env as E
from distillery.taskpacks.toolcall import prompts as P
from distillery.taskpacks.toolcall import questions as Q
from distillery.taskpacks.toolcall import verifier as V
from distillery.taskpacks.toolcall.executor import LocalExecutor
from distillery.taskpacks.toolcall.pack import ToolcallPack

SEEDS = (1, 2, 3, 4)


@pytest.fixture(scope="module")
def pools() -> dict[int, tuple[E.State, list[Q.ToolTask]]]:
    out = {}
    for s in SEEDS:
        st = E.build_state(s)
        out[s] = (st, Q.generate_tasks(st, 400, 7, state_key=s).tasks)
    return out


# ---- env ---------------------------------------------------------------------------------------


def test_env_deterministic_and_seed_sensitive() -> None:
    assert E.canonical_state(E.build_state(5)) == E.canonical_state(E.build_state(5))
    assert E.canonical_state(E.build_state(5)) != E.canonical_state(E.build_state(6))
    assert E.render_context(E.build_state(5)) == E.render_context(E.build_state(5))


def test_env_uses_sql_enums_and_has_traps() -> None:
    from distillery.taskpacks.sql import schema as S

    st = E.build_state(1)
    assert {a["status"] for a in st["accounts"].values()} <= set(S.ACCOUNT_STATUSES)
    assert {u["role"] for u in st["users"].values()} <= set(S.USER_ROLES)
    assert {a["plan"] for a in st["accounts"].values()} <= set(S.TIERS)
    names = Counter(a["name"] for a in st["accounts"].values())
    assert max(names.values()) == 2  # duplicated account name
    for aid, a in st["accounts"].items():
        lim = E.SEAT_LIMITS[a["plan"]]
        assert lim is None or a["seats"] <= lim
        assert len(E.live_users(st, aid)) <= a["seats"]


def _ok(state: E.State, *calls: E.Call) -> E.ReplayResult:
    return E.replay(state, list(calls))


def test_tool_preconditions_reject() -> None:
    st = E.build_state(1)
    st["accounts"][1].update(status="active", plan="free", seats=3)
    st["accounts"][2].update(status="suspended", plan="pro", seats=5)
    st["invoices"][1].update(status="open", amount_cents=100)
    st["tickets"][1].update(status="open", assignee=None)
    c = E.call_of
    bad = [
        c("nope"),
        {"tool": "add_seat"},
        c("add_seat", account_id=1, count=1),  # free plan
        c("add_seat", account_id=2, count=1),  # suspended
        c("add_seat", account_id=3, count=0),  # below minimum
        c("add_seat", account_id=3, count=1, extra=1),  # extra arg
        c("add_seat", account_id="3", count=1),  # wrong type
        c("add_seat", account_id=999, count=1),  # missing entity
        c("refund_invoice", invoice_id=1),  # not paid
        c("close_ticket", ticket_id=1, resolution="resolved"),  # unassigned
        c("assign_ticket", ticket_id=1, agent="nobody"),  # enum
        c("suspend_account", account_id=2, reason="abuse"),  # already suspended
        c("update_plan", account_id=1, plan="free"),  # same plan
        c("create_user", account_id=1, email="a@b.c", full_name="A B", role="boss"),
        c("lookup_account", account_id=999),
    ]
    for b in bad:
        assert not _ok(st, b).ok, b
    assert _ok(
        st,
        c("assign_ticket", ticket_id=1, agent="dana"),
        c("close_ticket", ticket_id=1, resolution="resolved"),
    ).ok
    assert not _ok(
        st,
        c("close_ticket", ticket_id=1, resolution="resolved"),
        c("assign_ticket", ticket_id=1, agent="dana"),
    ).ok


def test_replay_does_not_mutate_input_and_invalid_call_fails() -> None:
    st = E.build_state(2)
    before = E.canonical_state(st)
    r = _ok(
        st, E.call_of("lookup_account", account_id=1), E.call_of("lookup_account", account_id=999)
    )
    assert not r.ok and r.failed_index == 1
    assert E.canonical_state(st) == before


def test_user_email_unique_and_seat_required() -> None:
    st = E.build_state(1)
    aid = next(a for a in st["accounts"] if st["accounts"][a]["status"] in ("active", "trial"))
    st["accounts"][aid]["seats"] = len(E.live_users(st, aid))
    new = E.call_of(
        "create_user", account_id=aid, email="new@x.example", full_name="N N", role="member"
    )
    assert not _ok(st, new).ok  # no free seat
    st["accounts"][aid]["seats"] += 1
    assert _ok(st, new).ok
    dup = E.call_of(
        "create_user",
        account_id=aid,
        email=next(iter(st["users"].values()))["email"],
        full_name="N N",
        role="member",
    )
    assert not _ok(st, dup).ok


# ---- tasks / gold ------------------------------------------------------------------------------


def test_every_gold_replays_and_matches_gold_state(
    pools: dict[int, tuple[E.State, list[Q.ToolTask]]],
) -> None:
    ex = LocalExecutor()
    for seed, (st, tasks) in pools.items():
        assert len(tasks) >= 300
        ex.register(E.env_ref(seed), st)
        for t in tasks:
            res = E.replay(st, t.gold_calls)
            assert res.ok, (t.task_id, res.error)
            (out,) = ex.run_batch(E.env_ref(seed), [t.gold_answer])
            assert out.ok and out.rows[0][0] == E.canonical_state(res.state)
            assert t.gold_answer == E.canonical_calls(t.gold_calls)
            if t.gold_calls and t.family not in Q.GUARD_FAMILIES:
                assert E.canonical_state(res.state) != E.canonical_state(st)  # gold does something


def test_all_families_and_guard_empty_golds(
    pools: dict[int, tuple[E.State, list[Q.ToolTask]]],
) -> None:
    fams = Counter(t.family for _, ts in pools.values() for t in ts)
    assert set(fams) == set(Q.FAMILIES) and len(Q.FAMILIES) == 8
    empties = [t for _, ts in pools.values() for t in ts if t.gold_answer == "[]"]
    assert empties and {t.family for t in empties} <= Q.GUARD_FAMILIES


def test_generation_deterministic_and_distinct() -> None:
    st = E.build_state(3)
    a = Q.generate_tasks(st, 200, 11, state_key=3).tasks
    b = Q.generate_tasks(st, 200, 11, state_key=3).tasks
    assert [t.model_dump() for t in a] == [t.model_dump() for t in b]
    assert len({t.gold_answer for t in a}) == len(a) and len({t.question for t in a}) == len(a)
    assert len({t.task_id for t in a}) == len(a)
    assert [t.task_id for t in a] != [
        t.task_id for t in Q.generate_tasks(st, 200, 12, state_key=3).tasks
    ]


def test_family_filters_and_stress_reservation() -> None:
    st = E.build_state(3)
    assert Q.stress_families() == Q.stress_families()
    assert Q.stress_families() == tuple(sorted(random.Random(777).sample(sorted(Q.FAMILIES), 2)))
    assert Q.stress_families() == (
        "dependent_chain",
        "single_call",
    )  # pinned: fixed rule, never retuned
    stress = Q.stress_families()
    train = Q.generate_tasks(st, 150, 1, exclude_families=stress, state_key=3).tasks
    held = Q.generate_tasks(st, 60, 2, families=stress, state_key=3).tasks
    assert {t.family for t in train}.isdisjoint(stress)
    assert {t.family for t in held} == set(stress)


def test_goals_have_no_leaked_gold_and_skeleton_masks_literals() -> None:
    assert Q.skeleton("Refund invoice 17.") == Q.skeleton("Refund invoice 4.")
    assert Q.skeleton("Assign ticket 3 to eli.") == Q.skeleton("Assign ticket 9 to ivo.")
    assert Q.skeleton("Suspend the account named 'Fathom' in JP for abuse.") == Q.skeleton(
        "Suspend the account named 'Ironleaf' in DE for non-payment."
    )
    assert Q.skeleton("Refund invoice 1.") != Q.skeleton("Close ticket 1 as resolved.")


def test_gold_fits_student_budget(pools: dict[int, tuple[E.State, list[Q.ToolTask]]]) -> None:
    assert P.STUDENT_MAX_NEW_TOKENS == 384 and ToolcallPack.student_max_new_tokens == 384
    for _, ts in pools.values():
        for t in ts:
            assert len(P.format_answer(t.gold_answer)) <= 600  # ~3 chars/token worst case => <= 200


def test_pseudo_task_compatible() -> None:
    t = Q.generate_tasks(E.build_state(1), 5, 1).tasks[0]
    p = pseudo_task(t, 1, "reworded")  # type: ignore[arg-type]
    assert p.gold_answer == t.gold_answer and p.question == "reworded"


# ---- corruptions -------------------------------------------------------------------------------


def test_every_corruption_rejected_independent_oracle(
    pools: dict[int, tuple[E.State, list[Q.ToolTask]]],
) -> None:
    kinds: Counter[str] = Counter()
    for _, (st, tasks) in pools.items():
        for n, t in enumerate(tasks[:80]):
            gold_state = E.replay(st, t.gold_calls).state
            for v in V.corrupt_calls(t.gold_calls, st, random.Random(n), max_variants=6):
                kinds[v.kind] += 1
                calls = json.loads(v.answer)
                r = E.replay(st, calls)
                # oracle: either an invalid call, or a different final state
                assert (not r.ok) or E.canonical_state(r.state) != E.canonical_state(gold_state)
                assert not V.verify(st, v.answer, t.gold_answer).ok
    assert set(kinds) == {
        "drop_call",
        "wrong_id",
        "swapped_args",
        "extra_call",
        "broken_order",
        "wrong_enum",
        "wrong_number",
    }


def test_broken_order_only_for_dependent_calls() -> None:
    st = E.build_state(1)
    t = next(
        t
        for t in Q.generate_tasks(st, 400, 1, state_key=1).tasks
        if t.template == "ms_assign_close"
    )
    kinds = {v.kind for v in V.corrupt_calls(t.gold_calls, st, random.Random(0), max_variants=20)}
    assert "broken_order" in kinds


def test_verifier_accepts_equivalent_orders_and_lookups() -> None:
    st = E.build_state(1)
    paid = sorted(
        i
        for i, v in st["invoices"].items()
        if v["status"] == "paid" and v["amount_cents"] is not None
    )
    assert len(paid) >= 2
    c = [E.call_of("refund_invoice", invoice_id=i) for i in paid[:2]]
    assert V.verify(st, E.canonical_calls(c[::-1]), E.canonical_calls(c)).ok  # independent calls
    extra = [E.call_of("lookup_invoice", invoice_id=paid[0]), *c]
    assert V.verify(st, E.canonical_calls(extra), E.canonical_calls(c)).ok  # read-only no-op


def test_verifier_rejects_bad_text_and_empty_vs_nonempty() -> None:
    st = E.build_state(1)
    assert not V.verify(st, "not json", "[]").ok
    assert not V.verify(st, '{"a": 1}', "[]").ok
    assert V.verify(st, "[]", "[]").ok
    assert not V.verify(
        st, E.canonical_calls([E.call_of("lookup_account", account_id=999)]), "[]"
    ).ok


def test_selftest_zero_tolerance(pools: dict[int, tuple[E.State, list[Q.ToolTask]]]) -> None:
    st, tasks = pools[1]
    ex = LocalExecutor({E.env_ref(1): st})
    rep = V.selftest(tasks, E.env_ref(1), ex, st, seed=3, sample=60, k=4)
    assert rep["corruptions_tested"] > 100 and len(rep["by_kind"]) >= 6

    class Lax(LocalExecutor):  # an executor that wrongly reports every corruption as the gold state
        def run_batch(self, env_ref, answers):  # type: ignore[no-untyped-def]
            outs = super().run_batch(env_ref, answers)
            return [outs[0]] * len(outs)

    with pytest.raises(V.SelfTestError, match="ACCEPTED"):
        V.selftest(tasks, E.env_ref(1), Lax({E.env_ref(1): st}), st, seed=3, sample=20, k=3)
    with pytest.raises(V.SelfTestError, match="vacuous"):
        V.selftest([t for t in tasks if False] or [], E.env_ref(1), ex, st, seed=3)


# ---- extraction --------------------------------------------------------------------------------

GOLD = [
    E.call_of("assign_ticket", ticket_id=3, agent="eli"),
    E.call_of("close_ticket", ticket_id=3, resolution="resolved"),
]
GOLD_TXT = E.canonical_calls(GOLD)


@pytest.mark.parametrize(
    "raw",
    [
        "```json\n" + json.dumps(GOLD) + "\n```",
        "Sure!\n```json\n" + json.dumps(GOLD, indent=2) + "\n```\nDone.",
        "```\n" + json.dumps(GOLD) + "\n```",
        "<think>maybe [1, 2] first</think>\n```json\n" + json.dumps(GOLD) + "\n```",
        "reasoning </think>```json\n" + json.dumps(GOLD),  # stray close tag + dangling fence
        "```json\n[]\n```\nActually:\n```json\n" + json.dumps(GOLD) + "\n```",  # last fence wins
        "The calls are " + json.dumps(GOLD) + " as requested.",
        json.dumps(GOLD),
        "```JSON\n"
        + json.dumps([{"args": {"agent": "eli", "ticket_id": 3}, "tool": "assign_ticket"}, GOLD[1]])
        + "\n```",
    ],
)
def test_extract_answer_variants(raw: str) -> None:
    assert P.extract_answer(raw) == GOLD_TXT


def test_extract_answer_empty_and_none() -> None:
    assert P.extract_answer("```json\n[]\n```") == "[]"
    assert P.extract_answer("[]") == "[]"
    assert P.extract_answer("Nothing to do [] really") is None
    assert P.extract_answer("") is None
    assert P.extract_answer("I cannot do that.") is None
    assert P.extract_answer("```json\n[1, 2, 3]\n```") is None
    assert P.extract_answer("<think>unclosed ```json\n[]\n```") is None
    one = P.extract_answer('```json\n{"tool": "lookup_account", "args": {"account_id": 1}}\n```')
    assert one is not None and json.loads(one) == [E.call_of("lookup_account", account_id=1)]


def test_malformed_calls_extracted_but_rejected() -> None:
    txt = P.extract_answer(
        '```json\n[{"tool": "add_seat", "args": {"account_id": 1}, "x": 2}]\n```'
    )
    assert txt is not None
    assert not V.verify(E.build_state(1), txt, "[]").ok


def test_gold_roundtrips_through_extraction(
    pools: dict[int, tuple[E.State, list[Q.ToolTask]]],
) -> None:
    for _, ts in pools.values():
        for t in ts:
            assert P.extract_answer(P.format_answer(t.gold_answer)) == t.gold_answer


# ---- prompts / pack ----------------------------------------------------------------------------


def test_prefix_identical_across_roles_and_row_shape() -> None:
    pack = ToolcallPack()
    env = pack.build_env(2)
    task = pack.generate_tasks(env.ref, 5, 1).tasks[0]
    msgs = [
        pack.build_messages(task.question, env.context_text, role=r)
        for r in ("train", "eval_student", "eval_teacher", "eval_base")
    ]
    assert all(m == msgs[0] for m in msgs)
    row = pack.to_training_row(task, task.gold_answer, context_text=env.context_text)
    assert row["messages"][:2] == msgs[0] and row["messages"][-1]["role"] == "assistant"
    assert pack.extract_answer(row["messages"][-1]["content"]) == task.gold_answer
    with pytest.raises(ValueError):
        pack.build_messages("g", "c", role="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        pack.to_training_row(task, " ", context_text="c")


def test_pack_end_to_end_and_executor_shape() -> None:
    pack = ToolcallPack()
    env = pack.build_env(4)
    tasks = pack.generate_tasks(env.ref, 80, 3).tasks
    outs = pack.executor.run_batch(env.ref, [t.gold_answer for t in tasks] + ["junk"])
    assert len(outs) == len(tasks) + 1 and all(o.ok for o in outs[:-1]) and not outs[-1].ok
    assert pack.compare(outs[0], outs[0]).ok and not pack.compare(outs[-1], outs[0]).ok
    assert pack.selftest(tasks, env.ref, seed=1)["corruptions_tested"] > 0
    assert pack.corrupt(tasks[0], env.ref, random.Random(1))
    assert pack.families == Q.FAMILIES and pack.answer_label == "tool calls"
    msgs = pack.analysis_prompt(
        [
            {
                "family": "bulk",
                "question": "q",
                "gold_answer": "[]",
                "student_answer": "x",
                "reason": "r",
            }
        ],
        ["bulk"],
    )
    assert msgs[1]["content"].startswith("Allowed families: bulk")
    with pytest.raises(ValueError):
        E.parse_env_ref("sql:1")


def test_state_not_shared_between_executor_calls() -> None:
    ex = LocalExecutor()
    ref = E.env_ref(1)
    st = E.build_state(1)
    before = E.canonical_state(st)
    paid = next(i for i, v in st["invoices"].items() if v["status"] == "paid" and v["amount_cents"])
    a = E.canonical_calls([E.call_of("refund_invoice", invoice_id=paid)])
    o1, o2 = ex.run_batch(ref, [a, a])
    assert o1.ok and o2.ok and o1.rows == o2.rows
    assert E.canonical_state(ex._state(ref)) == before
    assert copy.deepcopy(st) == st
