# ruff: noqa: S311, E501
"""Final-state verification of predicted tool calls, plus deliberate corruptions and a self-test.

A candidate is correct iff replaying its calls on the seeded state succeeds (every call valid) and
its canonical final state equals the gold's. Call order and redundant read-only ``lookup_*`` calls
do not matter by themselves; they matter only through their effect on the state (so swapping two
dependent calls, e.g. ``close_ticket`` before ``assign_ticket``, is rejected).

``corrupt_calls`` produces deliberately wrong variants of the gold: drop a call, wrong id, swapped
args, extra call, broken order of dependent calls, wrong enum, wrong number. Each variant is kept
only if the verifier rejects it. ``selftest`` re-checks that with zero tolerance.
"""

from __future__ import annotations

import copy
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from distillery.taskpacks.sql.verifier import VerifyResult
from distillery.taskpacks.toolcall import env as E
from distillery.taskpacks.toolcall.executor import ExecOutcome, Executor

__all__ = [
    "Corruption",
    "SelfTestError",
    "VerifyResult",
    "compare_outcomes",
    "corrupt_calls",
    "selftest",
    "verify",
]


class SelfTestError(Exception):
    """The verifier accepted a corruption, rejected a gold, or the self-test was vacuous. (The
    orchestrator's ``SelfTestError`` lives in orchestrator.py; wiring maps this to it.)"""


_ID_TABLE = {"account_id": "accounts", "invoice_id": "invoices", "ticket_id": "tickets"}


def _diff(a: E.State, b: E.State) -> str:
    out: list[str] = []
    for table in sorted(set(a) | set(b)):
        ta, tb = a.get(table), b.get(table)
        if ta == tb:
            continue
        if isinstance(ta, dict) and isinstance(tb, dict):
            keys = sorted(k for k in set(ta) | set(tb) if ta.get(k) != tb.get(k))
            out.append(f"{table}: {keys[:6]}")
        else:
            out.append(table)
    return "; ".join(out)


def compare_outcomes(
    candidate: ExecOutcome, gold: ExecOutcome, requires_order: bool = False
) -> VerifyResult:
    """Compare two executed outcomes (same signature as the SQL pack's; ``requires_order`` unused)."""
    if not gold.ok:
        return VerifyResult(False, f"gold error ({gold.error_kind}): {gold.error}")
    if not candidate.ok:
        return VerifyResult(False, f"candidate error ({candidate.error_kind}): {candidate.error}")
    if candidate.rows == gold.rows:
        return VerifyResult(True, "match")
    got = json.loads(str(candidate.rows[0][0]))
    want = json.loads(str(gold.rows[0][0]))
    return VerifyResult(False, f"final state differs ({_diff(got, want)})")


def verify(state: E.State, candidate_answer: str, gold_answer: str) -> VerifyResult:
    """Replay both answers on ``state`` and compare canonical final states."""
    from distillery.taskpacks.toolcall.executor import run_answer

    return compare_outcomes(run_answer(state, candidate_answer), run_answer(state, gold_answer))


# ---- corruptions ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Corruption:
    kind: str
    sql: str  # the corrupted answer text (field name matches the SQL pack's Corruption)

    @property
    def answer(self) -> str:
        return self.sql


def _enum_args(call: dict[str, Any]) -> list[tuple[str, list[str]]]:
    props = E.TOOLS[call["tool"]].schema["properties"]
    return [(k, list(p["enum"])) for k, p in props.items() if "enum" in p]


def _candidates(
    gold: list[E.Call], state: E.State, rng: random.Random
) -> list[tuple[str, list[E.Call]]]:
    out: list[tuple[str, list[E.Call]]] = []

    def variant(i: int, key: str, value: Any) -> list[E.Call]:
        c = copy.deepcopy(gold)
        c[i]["args"][key] = value
        return c

    for i in range(len(gold)):
        out.append(("drop_call", gold[:i] + gold[i + 1 :]))
    for i, c in enumerate(gold):
        for k, v in c["args"].items():
            if k in _ID_TABLE:
                others = [x for x in sorted(state[_ID_TABLE[k]]) if x != v]
                for x in rng.sample(others, min(2, len(others))):
                    out.append(("wrong_id", variant(i, k, x)))
            elif k == "count":
                out.append(("wrong_number", variant(i, k, v + 1)))
                if v > 1:
                    out.append(("wrong_number", variant(i, k, v - 1)))
        for k, values in _enum_args(c):
            for x in rng.sample([x for x in values if x != c["args"][k]], 2):
                out.append(("wrong_enum", variant(i, k, x)))
    # swapped args: between two calls of one tool (their first id), else inside one call
    for i in range(len(gold)):
        for j in range(i + 1, len(gold)):
            if gold[i]["tool"] == gold[j]["tool"]:
                for k in gold[i]["args"]:
                    if k in _ID_TABLE and gold[i]["args"][k] != gold[j]["args"][k]:
                        sw = copy.deepcopy(gold)
                        sw[i]["args"][k], sw[j]["args"][k] = gold[j]["args"][k], gold[i]["args"][k]
                        out.append(("swapped_args", sw))
                        break
    for i, c in enumerate(gold):
        keys = list(c["args"])
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                if c["args"][keys[a]] != c["args"][keys[b]]:
                    cc = copy.deepcopy(gold)
                    cc[i]["args"][keys[a]], cc[i]["args"][keys[b]] = (
                        c["args"][keys[b]],
                        c["args"][keys[a]],
                    )
                    out.append(("swapped_args", cc))
    # broken order: every adjacent swap and the full reversal (kept only if the state differs)
    for i in range(len(gold) - 1):
        order = list(gold)
        order[i], order[i + 1] = order[i + 1], order[i]
        out.append(("broken_order", order))
    if len(gold) > 2:
        out.append(("broken_order", gold[::-1]))
    # extra call: a duplicate of the last call, or a different valid mutation after the gold
    if gold:
        out.append(("extra_call", [*gold, copy.deepcopy(gold[-1])]))
    after = E.replay(state, gold).state
    extras: list[E.Call] = [
        E.call_of("add_seat", account_id=a, count=1) for a in sorted(after["accounts"])
    ]
    extras += [E.call_of("refund_invoice", invoice_id=i) for i in sorted(after["invoices"])]
    extras += [
        E.call_of("suspend_account", account_id=a, reason="abuse")
        for a in sorted(after["accounts"])
    ]
    extras += [
        E.call_of("assign_ticket", ticket_id=t, agent="ivo") for t in sorted(after["tickets"])
    ]
    extras += [
        E.call_of("close_ticket", ticket_id=t, resolution="resolved")
        for t in sorted(after["tickets"])
    ]
    rng.shuffle(extras)
    for e in extras[:30]:
        out.append(("extra_call", [*gold, e]))
    return out


def corrupt_calls(
    gold: Sequence[E.Call], state: E.State, rng: random.Random, *, max_variants: int = 8
) -> list[Corruption]:
    """Wrong-but-plausible variants of the gold call list that the verifier rejects.

    Each variant is replayed (it may be invalid or valid-but-different) and kept only if
    ``verify`` rejects it; kinds are interleaved so the sample is varied. Deterministic for a
    given ``rng`` state.
    """
    gold = list(gold)
    gold_answer = E.canonical_calls(gold)
    if not E.replay(state, gold).ok:
        return []
    by_kind: dict[str, list[str]] = {}
    seen = {gold_answer}
    for kind, calls in _candidates(gold, state, rng):
        ans = E.canonical_calls(calls)
        if ans in seen:
            continue
        seen.add(ans)
        if not verify(state, ans, gold_answer).ok:
            by_kind.setdefault(kind, []).append(ans)
    kinds = sorted(by_kind)
    rng.shuffle(kinds)
    out: list[Corruption] = []
    depth = 0
    while len(out) < max_variants and any(depth < len(v) for v in by_kind.values()):
        for k in kinds:
            if depth < len(by_kind[k]) and len(out) < max_variants:
                out.append(Corruption(k, by_kind[k][depth]))
        depth += 1
    return out


# ---- zero-tolerance self-test ---------------------------------------------------------------


def selftest(
    tasks: Sequence[Any],
    env_ref: str,
    executor: Executor,
    state: E.State,
    *,
    seed: int,
    sample: int = 50,
    k: int = 4,
) -> dict[str, Any]:
    """Run gold and corruptions of a seeded task sample through ``executor``; any corruption that
    the verifier accepts, any gold it rejects, or a vacuous run (no corruption tested) raises
    ``SelfTestError``. Returns counters per corruption kind."""
    from collections import Counter

    rng = random.Random(seed)
    chosen = rng.sample(list(tasks), min(sample, len(tasks)))
    failures: list[str] = []
    by_kind: Counter[str] = Counter()
    tested = 0
    for n, t in enumerate(chosen):
        variants = corrupt_calls(
            t.gold_calls, state, random.Random(seed * 1000 + n), max_variants=k
        )
        outs = executor.run_batch(env_ref, [t.gold_answer, *[v.answer for v in variants]])
        g = outs[0]
        if not g.ok:
            failures.append(f"gold rejected for {t.task_id}: {g.error}")
            continue
        for v, o in zip(variants, outs[1:], strict=True):
            tested += 1
            by_kind[v.kind] += 1
            if compare_outcomes(o, g).ok:
                failures.append(f"{t.task_id} [{v.kind}]: corruption ACCEPTED: {v.answer!r}")
    if tested == 0:
        failures.append("no corruptions could be generated: self-test would be vacuous")
    if failures:
        shown = "\n  ".join(failures[:10])
        raise SelfTestError(f"verifier self-test failed ({len(failures)} problems):\n  {shown}")
    return {"by_kind": dict(by_kind), "tasks_sampled": len(chosen), "corruptions_tested": tested}
