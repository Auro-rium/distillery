# ruff: noqa: S311, E501
"""Templated tool-call task generator: goal text + gold call sequence over the seeded SaaS state.

Every template is ``build(rng, state) -> (goal, calls) | None`` (None = no suitable entity in this
state). A task is kept only if its gold calls replay without error on the seeded state, and
(unless the family is ``error_guard``, whose correct answer may be "do nothing") change it.
Gold final state = ``replay(state, gold)``. Families, skeletons, caps and the two reserved
stress families follow the SQL pack's rules.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict

from distillery.taskpacks.sql import schema as _schema
from distillery.taskpacks.toolcall import env as E
from distillery.taskpacks.toolcall.env import Call, State

Difficulty = Literal["easy", "medium", "hard"]
Build = Callable[[random.Random, State], tuple[str, list[Call]] | None]


class ToolTask(BaseModel):
    """One goal with a replay-verified gold call sequence (``gold_answer`` = its canonical JSON)."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    family: str
    question: str  # the goal text (named ``question`` like SqlTask so shared helpers work)
    gold_answer: str
    requires_order: bool = False  # kept for SqlTask shape; the verifier compares final states
    difficulty: Difficulty = "medium"
    tables: tuple[str, ...] = ()
    template: str = ""

    @property
    def order_matters(self) -> bool:
        """Pack-interface name for ``requires_order`` (always False here: final states compare)."""
        return self.requires_order

    @property
    def gold_calls(self) -> list[Call]:
        import json

        return list(json.loads(self.gold_answer))


@dataclass(frozen=True)
class Template:
    name: str
    family: str
    tables: tuple[str, ...]
    difficulty: Difficulty
    build: Build


_TEMPLATES: list[Template] = []


def _register(
    family: str, tables: tuple[str, ...], difficulty: Difficulty
) -> Callable[[Build], Build]:
    def deco(fn: Build) -> Build:
        _TEMPLATES.append(Template(fn.__name__, family, tables, difficulty, fn))
        return fn

    return deco


# ---- helpers -----------------------------------------------------------------------------------

REASON_TEXT = {"nonpayment": "non-payment", "abuse": "abuse", "requested": "a customer request"}
RES_TEXT = {"resolved": "resolved", "wont_fix": "won't fix", "duplicate": "a duplicate"}


call = E.call_of


def _open(state: State, aid: int) -> bool:
    return bool(state["accounts"][aid]["status"] in ("active", "trial"))


def _ref(state: State, aid: int, rng: random.Random, by_name: bool) -> str:
    a = state["accounts"][aid]
    if by_name:
        return f"the account named '{a['name']}' in {a['country']}"
    return f"account {aid}"


def _tickets(state: State, aid: int | None = None, **where: object) -> list[int]:
    return sorted(
        t for t, r in state["tickets"].items()
        if (aid is None or r["account_id"] == aid) and all(r[k] == v for k, v in where.items())
    )  # fmt: skip


def _paid(state: State, aid: int | None = None) -> list[int]:
    return sorted(
        i for i, v in state["invoices"].items()
        if v["status"] == "paid" and v["amount_cents"] is not None and (aid is None or v["account_id"] == aid)
    )  # fmt: skip


def _free_seats(state: State, aid: int) -> int:
    return int(state["accounts"][aid]["seats"]) - len(E.live_users(state, aid))


def _tier_room(state: State, aid: int) -> int | None:
    """Seats still purchasable on the current tier (None = unlimited; 0 on free)."""
    a = state["accounts"][aid]
    if a["plan"] == "free":
        return 0
    lim = E.SEAT_LIMITS[a["plan"]]
    return None if lim is None else lim - int(a["seats"])


def _new_user(rng: random.Random, state: State, aid: int) -> tuple[str, str, str]:
    existing = {u["email"] for u in state["users"].values()}
    for _ in range(50):
        first, last = rng.choice(E._FIRST), rng.choice(E._LAST)
        name = f"{first} {last}"
        email = f"{first.lower()}.{last.lower()}@{state['accounts'][aid]['name'].lower().split()[0].strip('&')}.example"
        if email not in existing:
            return name, email, rng.choice(_schema.USER_ROLES[1:])
    raise RuntimeError("no free email")  # pragma: no cover


def _pick(rng: random.Random, xs: list[int]) -> int | None:
    return rng.choice(xs) if xs else None


def _accs(state: State, pred: Callable[[int, dict[str, object]], bool]) -> list[int]:
    return sorted(i for i, a in state["accounts"].items() if pred(i, a))


# ---- single_call -------------------------------------------------------------------------------


@_register("single_call", ("invoices",), "easy")
def sc_refund(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    i = _pick(rng, _paid(s))
    if i is None:
        return None
    return rng.choice((f"Refund invoice {i}.", f"Please issue a refund for invoice {i}.")), [
        call("refund_invoice", invoice_id=i)
    ]


@_register("single_call", ("tickets",), "easy")
def sc_close(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    t = _pick(rng, _tickets(s, status="pending"))
    if t is None:
        return None
    r = rng.choice(E.RESOLUTIONS)
    return rng.choice(
        (
            f"Close ticket {t} as {RES_TEXT[r]}.",
            f"Mark ticket {t} closed; resolution: {RES_TEXT[r]}.",
        )
    ), [call("close_ticket", ticket_id=t, resolution=r)]


@_register("single_call", ("tickets",), "easy")
def sc_assign(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    cands = [t for t in _tickets(s) if s["tickets"][t]["status"] != "closed"]
    t = _pick(rng, cands)
    if t is None:
        return None
    ag = rng.choice([a for a in E.AGENTS if a != s["tickets"][t]["assignee"]])
    return rng.choice((f"Assign ticket {t} to {ag}.", f"Hand ticket {t} over to agent {ag}.")), [
        call("assign_ticket", ticket_id=t, agent=ag)
    ]


@_register("single_call", ("accounts",), "easy")
def sc_suspend(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    a = _pick(rng, _accs(s, lambda i, _a: _open(s, i)))
    if a is None:
        return None
    r = rng.choice(E.SUSPEND_REASONS)
    return rng.choice(
        (
            f"Suspend account {a} for {REASON_TEXT[r]}.",
            f"Account {a} must be suspended due to {REASON_TEXT[r]}.",
        )
    ), [call("suspend_account", account_id=a, reason=r)]


@_register("single_call", ("accounts",), "easy")
def sc_add_seat(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    cands = [a for a in _accs(s, lambda i, _a: _open(s, i)) if _tier_room(s, a) != 0]
    a = _pick(rng, cands)
    if a is None:
        return None
    room = _tier_room(s, a)
    n = rng.randint(1, min(5, room if room is not None else 5))
    return rng.choice((f"Add {n} seats to account {a}.", f"Buy {n} more seats for account {a}.")), [
        call("add_seat", account_id=a, count=n)
    ]


@_register("single_call", ("accounts",), "easy")
def sc_update_plan(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    a = _pick(rng, _accs(s, lambda i, _a: _open(s, i)))
    if a is None:
        return None
    acc = s["accounts"][a]
    tiers = [
        t
        for t in _schema.TIERS
        if t != acc["plan"]
        and (E.SEAT_LIMITS[t] is None or acc["seats"] <= (E.SEAT_LIMITS[t] or 0))
    ]
    if not tiers:
        return None
    t = rng.choice(tiers)
    return rng.choice(
        (f"Move account {a} to the {t} plan.", f"Change the plan of account {a} to {t}.")
    ), [call("update_plan", account_id=a, plan=t)]


@_register("single_call", ("users", "accounts"), "easy")
def sc_create_user(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    a = _pick(rng, [x for x in _accs(s, lambda i, _a: _open(s, i)) if _free_seats(s, x) > 0])
    if a is None:
        return None
    name, email, role = _new_user(rng, s, a)
    return (
        rng.choice(
            (
                f"Create a {role} user {name} with email {email} on account {a}.",
                f"Add {name} <{email}> to account {a} as {role}.",
            )
        ),
        [call("create_user", account_id=a, email=email, full_name=name, role=role)],
    )


# ---- lookup_then_act (entities described, not numbered) ----------------------------------------


@_register("lookup_then_act", ("invoices", "accounts"), "medium")
def la_refund_largest(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    cands = []
    for a in s["accounts"]:
        paid = sorted(_paid(s, a), key=lambda i: -s["invoices"][i]["amount_cents"])
        if paid and (
            len(paid) == 1
            or s["invoices"][paid[0]]["amount_cents"] > s["invoices"][paid[1]]["amount_cents"]
        ):
            cands.append((a, paid[0]))
    if not cands:
        return None
    a, i = rng.choice(cands)
    return f"Refund the largest paid invoice of {_ref(s, a, rng, True)}.", [
        call("refund_invoice", invoice_id=i)
    ]


@_register("lookup_then_act", ("tickets", "accounts"), "medium")
def la_close_ticket(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    cands = [
        (a, t)
        for a in s["accounts"]
        for p in _schema.PRIORITIES
        if len(ts := _tickets(s, a, status="pending", priority=p)) == 1
        for t in ts
    ]
    if not cands:
        return None
    a, t = rng.choice(cands)
    r = rng.choice(E.RESOLUTIONS)
    p = s["tickets"][t]["priority"]
    return f"Close the pending {p}-priority ticket of {_ref(s, a, rng, True)} as {RES_TEXT[r]}.", [
        call("close_ticket", ticket_id=t, resolution=r)
    ]


@_register("lookup_then_act", ("tickets", "accounts"), "medium")
def la_assign_open(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    cands = [
        (a, t)
        for a in s["accounts"]
        for p in _schema.PRIORITIES
        if len(ts := _tickets(s, a, status="open", priority=p)) == 1
        for t in ts
    ]
    if not cands:
        return None
    a, t = rng.choice(cands)
    ag = rng.choice(E.AGENTS)
    p = s["tickets"][t]["priority"]
    return f"Assign the open {p}-priority ticket of {_ref(s, a, rng, True)} to {ag}.", [
        call("assign_ticket", ticket_id=t, agent=ag)
    ]


@_register("lookup_then_act", ("accounts",), "medium")
def la_suspend_by_name(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    a = _pick(rng, _accs(s, lambda i, _a: _open(s, i)))
    if a is None:
        return None
    r = rng.choice(E.SUSPEND_REASONS)
    return f"Suspend {_ref(s, a, rng, True)} for {REASON_TEXT[r]}.", [
        call("suspend_account", account_id=a, reason=r)
    ]


@_register("lookup_then_act", ("accounts", "users"), "medium")
def la_user_by_owner(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    cands = [
        (a, u["email"])
        for a in _accs(s, lambda i, _a: _open(s, i) and _free_seats(s, i) > 0)
        for u in s["users"].values()
        if u["account_id"] == a and u["role"] == "owner"
    ]
    if not cands:
        return None
    a, owner = rng.choice(cands)
    name, email, role = _new_user(rng, s, a)
    return (
        f"Add {name} ({email}) as {role} to the account whose owner is {owner}.",
        [call("create_user", account_id=a, email=email, full_name=name, role=role)],
    )


@_register("lookup_then_act", ("accounts",), "medium")
def la_plan_by_name(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    a = _pick(rng, _accs(s, lambda i, _a: _open(s, i)))
    if a is None:
        return None
    acc = s["accounts"][a]
    tiers = [
        t
        for t in _schema.TIERS
        if t != acc["plan"]
        and (E.SEAT_LIMITS[t] is None or acc["seats"] <= (E.SEAT_LIMITS[t] or 0))
    ]
    if not tiers:
        return None
    t = rng.choice(tiers)
    return f"Put {_ref(s, a, rng, True)} on the {t} plan.", [
        call("update_plan", account_id=a, plan=t)
    ]


# ---- multi_step ---------------------------------------------------------------------------------


@_register("multi_step", ("tickets",), "medium")
def ms_assign_close(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    t = _pick(rng, _tickets(s, status="open"))
    if t is None:
        return None
    ag, r = rng.choice(E.AGENTS), rng.choice(E.RESOLUTIONS)
    return (
        rng.choice(
            (
                f"Assign ticket {t} to {ag} and then close it as {RES_TEXT[r]}.",
                f"Ticket {t} should go to {ag} and be closed ({RES_TEXT[r]}).",
            )
        ),
        [
            call("assign_ticket", ticket_id=t, agent=ag),
            call("close_ticket", ticket_id=t, resolution=r),
        ],
    )


@_register("multi_step", ("tickets",), "medium")
def ms_reassign_close(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    t = _pick(rng, _tickets(s, status="pending"))
    if t is None:
        return None
    ag, r = (
        rng.choice([a for a in E.AGENTS if a != s["tickets"][t]["assignee"]]),
        rng.choice(E.RESOLUTIONS),
    )
    return f"Reassign ticket {t} to {ag}, then close it as {RES_TEXT[r]}.", [
        call("assign_ticket", ticket_id=t, agent=ag),
        call("close_ticket", ticket_id=t, resolution=r),
    ]


@_register("multi_step", ("accounts", "users"), "hard")
def ms_seat_then_user(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    cands = [
        a
        for a in _accs(s, lambda i, _a: _open(s, i) and _free_seats(s, i) == 0)
        if _tier_room(s, a) != 0
    ]
    a = _pick(rng, cands)
    if a is None:
        return None
    name, email, role = _new_user(rng, s, a)
    return (
        rng.choice(
            (
                f"Add {name} ({email}) as {role} to account {a}, buying a seat if needed.",
                f"Account {a} needs a new {role}: {name}, {email}. Make room if the account is full.",
            )
        ),
        [
            call("add_seat", account_id=a, count=1),
            call("create_user", account_id=a, email=email, full_name=name, role=role),
        ],
    )


@_register("multi_step", ("accounts",), "hard")
def ms_upgrade_then_seats(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    cands = [
        a for a in _accs(s, lambda i, _a: _open(s, i)) if s["accounts"][a]["plan"] == "starter"
    ]
    a = _pick(rng, cands)
    if a is None:
        return None
    n = rng.randint(max(1, 11 - s["accounts"][a]["seats"]), 10)
    return f"Upgrade account {a} to pro and add {n} seats.", [
        call("update_plan", account_id=a, plan="pro"),
        call("add_seat", account_id=a, count=n),
    ]


@_register("multi_step", ("invoices", "accounts"), "medium")
def ms_refund_suspend(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    cands = [i for i in _paid(s) if _open(s, s["invoices"][i]["account_id"])]
    i = _pick(rng, cands)
    if i is None:
        return None
    a, r = s["invoices"][i]["account_id"], rng.choice(E.SUSPEND_REASONS)
    return f"Refund invoice {i} and suspend its account for {REASON_TEXT[r]}.", [
        call("refund_invoice", invoice_id=i),
        call("suspend_account", account_id=a, reason=r),
    ]


# ---- conditional --------------------------------------------------------------------------------


@_register("conditional", ("accounts",), "medium")
def cd_plan(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    a = _pick(rng, _accs(s, lambda i, _a: _open(s, i)))
    if a is None:
        return None
    n = rng.randint(1, 3)
    acc = s["accounts"][a]
    gold = (
        [call("update_plan", account_id=a, plan="starter")]
        if acc["plan"] == "free"
        else [call("add_seat", account_id=a, count=n)]
    )
    return (
        f"If account {a} is on the free plan, move it to starter; otherwise add {n} seats to it.",
        gold,
    )


@_register("conditional", ("tickets",), "medium")
def cd_ticket(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    t = _pick(rng, _tickets(s))
    if t is None:
        return None
    ag = rng.choice(E.AGENTS)
    st = s["tickets"][t]["status"]
    gold = (
        [call("assign_ticket", ticket_id=t, agent=ag)]
        if st == "open"
        else [call("close_ticket", ticket_id=t, resolution="resolved")]
        if st == "pending"
        else []
    )
    return (
        f"For ticket {t}: if it is open, assign it to {ag}; if it is pending, close it as resolved; if it is closed, do nothing.",
        gold,
    )


@_register("conditional", ("invoices",), "medium")
def cd_invoice(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    i = _pick(rng, sorted(s["invoices"]))
    if i is None:
        return None
    inv = s["invoices"][i]
    gold = (
        [call("refund_invoice", invoice_id=i)]
        if inv["status"] == "paid" and inv["amount_cents"] is not None
        else []
    )
    return f"If invoice {i} is paid, refund it; otherwise leave it as it is.", gold


@_register("conditional", ("accounts",), "hard")
def cd_status(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    a = _pick(rng, sorted(s["accounts"]))
    if a is None:
        return None
    acc = s["accounts"][a]
    r = rng.choice(E.SUSPEND_REASONS)
    if acc["status"] == "active":
        gold = [call("suspend_account", account_id=a, reason=r)]
    elif acc["status"] == "trial" and acc["plan"] != "starter":
        gold = [call("update_plan", account_id=a, plan="starter")]
    else:
        gold = []
    return (
        f"If account {a} is active, suspend it for {REASON_TEXT[r]}; if it is a trial, move it to starter; in any other case do nothing.",
        gold,
    )


# ---- bulk ---------------------------------------------------------------------------------------


@_register("bulk", ("invoices",), "medium")
def bk_refund(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    cands = [a for a in s["accounts"] if len(_paid(s, a)) >= 2]
    a = _pick(rng, cands)
    if a is None:
        return None
    return f"Refund every paid invoice of account {a}.", [
        call("refund_invoice", invoice_id=i) for i in _paid(s, a)
    ]


@_register("bulk", ("tickets",), "medium")
def bk_close(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    cands = [a for a in s["accounts"] if len(_tickets(s, a, status="pending")) >= 2]
    a = _pick(rng, cands)
    if a is None:
        return None
    r = rng.choice(E.RESOLUTIONS)
    return f"Close all pending tickets of account {a} as {RES_TEXT[r]}.", [
        call("close_ticket", ticket_id=t, resolution=r) for t in _tickets(s, a, status="pending")
    ]


@_register("bulk", ("tickets",), "medium")
def bk_assign(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    cands = [a for a in s["accounts"] if len(_tickets(s, a, status="open")) >= 2]
    a = _pick(rng, cands)
    if a is None:
        return None
    ag = rng.choice(E.AGENTS)
    return f"Assign every open ticket of account {a} to {ag}.", [
        call("assign_ticket", ticket_id=t, agent=ag) for t in _tickets(s, a, status="open")
    ]


@_register("bulk", ("tickets",), "hard")
def bk_priority(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    p = rng.choice(_schema.PRIORITIES)
    ts = _tickets(s, status="open", priority=p)
    if not 2 <= len(ts) <= 5:
        return None
    ag = rng.choice(E.AGENTS)
    return f"Assign all open {p}-priority tickets (across every account) to {ag}.", [
        call("assign_ticket", ticket_id=t, agent=ag) for t in ts
    ]


@_register("bulk", ("users",), "hard")
def bk_users(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    cands = [a for a in _accs(s, lambda i, _a: _open(s, i)) if _free_seats(s, a) >= 2]
    a = _pick(rng, cands)
    if a is None:
        return None
    n = rng.randint(2, min(3, _free_seats(s, a)))
    gold: list[Call] = []
    names: list[str] = []
    tmp = {"users": dict(s["users"]), "accounts": s["accounts"]}
    for _ in range(n):
        name, email, role = _new_user(rng, tmp, a)
        tmp["users"][-len(gold) - 1] = {"email": email}
        gold.append(call("create_user", account_id=a, email=email, full_name=name, role=role))
        names.append(f"{name} <{email}> ({role})")
    return f"Create these users on account {a}: " + "; ".join(names) + ".", gold


# ---- error_guard (the right answer may be "do nothing") ----------------------------------------


@_register("error_guard", ("invoices",), "medium")
def eg_refund(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    i = _pick(rng, sorted(s["invoices"]))
    if i is None:
        return None
    inv = s["invoices"][i]
    ok = inv["status"] == "paid" and inv["amount_cents"] is not None
    return f"Refund invoice {i} only if it is paid and has an amount; otherwise make no changes.", [
        call("refund_invoice", invoice_id=i)
    ] if ok else []


@_register("error_guard", ("tickets",), "medium")
def eg_close(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    t = _pick(rng, _tickets(s))
    if t is None:
        return None
    tk = s["tickets"][t]
    ok = tk["status"] != "closed" and tk["assignee"] is not None
    return (
        f"Close ticket {t} as resolved, but only if it is not closed and an agent is already assigned; otherwise do nothing.",
        [call("close_ticket", ticket_id=t, resolution="resolved")] if ok else [],
    )


@_register("error_guard", ("accounts",), "medium")
def eg_suspend(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    a = _pick(rng, sorted(s["accounts"]))
    if a is None:
        return None
    return f"Suspend account {a} for abuse unless it is already suspended or churned.", [
        call("suspend_account", account_id=a, reason="abuse")
    ] if _open(s, a) else []


@_register("error_guard", ("accounts",), "hard")
def eg_seats(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    a = _pick(rng, sorted(s["accounts"]))
    if a is None:
        return None
    n = rng.randint(1, 8)
    room = _tier_room(s, a)
    ok = _open(s, a) and room != 0 and (room is None or n <= room)
    return (
        f"Add {n} seats to account {a} if that is allowed (account active or trial, on a paid plan, within the plan's seat limit); otherwise do nothing.",
        [call("add_seat", account_id=a, count=n)] if ok else [],
    )


@_register("error_guard", ("users", "accounts"), "hard")
def eg_user(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    a = _pick(rng, sorted(s["accounts"]))
    if a is None:
        return None
    name, email, role = _new_user(rng, s, a)
    ok = _open(s, a) and _free_seats(s, a) > 0
    return (
        f"Add {name} ({email}) as {role} to account {a} only if it is active or trial and has a free seat; do not buy seats.",
        [call("create_user", account_id=a, email=email, full_name=name, role=role)] if ok else [],
    )


# ---- dependent_chain (3+ ordered steps) --------------------------------------------------------


@_register("dependent_chain", ("accounts", "users"), "hard")
def dc_onboard(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    a = _pick(
        rng,
        [
            x
            for x in _accs(s, lambda i, _a: _open(s, i))
            if _free_seats(s, x) < 2 and _tier_room(s, x) != 0
        ],
    )
    if a is None:
        return None
    need = 2 - _free_seats(s, a)
    tmp = {"users": dict(s["users"]), "accounts": s["accounts"]}
    users = []
    for k in range(2):
        name, email, role = _new_user(rng, tmp, a)
        tmp["users"][-k - 1] = {"email": email}
        users.append(call("create_user", account_id=a, email=email, full_name=name, role=role))
    txt = "; ".join(
        f"{u['args']['full_name']} <{u['args']['email']}> ({u['args']['role']})" for u in users
    )
    return (
        f"Onboard two users on account {a}, buying seats first if it does not have room: {txt}.",
        [call("add_seat", account_id=a, count=need), *users],
    )


@_register("dependent_chain", ("accounts", "users"), "hard")
def dc_free_full(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    a = _pick(
        rng,
        [
            x
            for x in _accs(s, lambda i, _a: _open(s, i) and s["accounts"][i]["plan"] == "free")
            if _free_seats(s, x) == 0
        ],
    )
    if a is None:
        return None
    name, email, role = _new_user(rng, s, a)
    return (
        f"Account {a} is on the free plan and all its seats are taken. Move it to starter, buy one seat, then add {name} ({email}) as {role}.",
        [
            call("update_plan", account_id=a, plan="starter"),
            call("add_seat", account_id=a, count=1),
            call("create_user", account_id=a, email=email, full_name=name, role=role),
        ],
    )


@_register("dependent_chain", ("tickets",), "hard")
def dc_ticket_pair(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    cands = [a for a in s["accounts"] if len(_tickets(s, a, status="open")) >= 2]
    a = _pick(rng, cands)
    if a is None:
        return None
    t1, t2 = _tickets(s, a, status="open")[:2]
    ag1, ag2 = rng.sample(E.AGENTS, 2)
    r = rng.choice(E.RESOLUTIONS)
    return (
        f"For account {a}: assign ticket {t1} to {ag1} and close it as {RES_TEXT[r]}, then assign ticket {t2} to {ag2}.",
        [
            call("assign_ticket", ticket_id=t1, agent=ag1),
            call("close_ticket", ticket_id=t1, resolution=r),
            call("assign_ticket", ticket_id=t2, agent=ag2),
        ],
    )


# ---- cross_entity -------------------------------------------------------------------------------


@_register("cross_entity", ("accounts",), "hard")
def ce_dup_names(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    by_name: dict[str, list[int]] = {}
    for i, a in s["accounts"].items():
        by_name.setdefault(a["name"], []).append(i)
    pairs = [v for v in by_name.values() if len(v) == 2]
    if not pairs:
        return None
    x, y = rng.sample(pairs[0], 2)
    ax, ay = s["accounts"][x], s["accounts"][y]
    r, n = rng.choice(E.SUSPEND_REASONS), rng.randint(1, 3)
    room = _tier_room(s, y)
    if not (_open(s, x) and _open(s, y) and room != 0 and (room is None or n <= room)):
        return None
    return (
        f"Two accounts are named '{ax['name']}'. Suspend the one in {ax['country']} for {REASON_TEXT[r]} and add {n} seats to the one in {ay['country']}.",
        [call("suspend_account", account_id=x, reason=r), call("add_seat", account_id=y, count=n)],
    )


@_register("cross_entity", ("invoices", "accounts"), "hard")
def ce_industry_overdue(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    ind = rng.choice(_schema.INDUSTRIES)
    accs = [
        a
        for a, acc in sorted(s["accounts"].items())
        if acc["industry"] == ind
        and _open(s, a)
        and any(v["account_id"] == a and v["status"] == "overdue" for v in s["invoices"].values())
    ]
    if not 1 <= len(accs) <= 5:
        return None
    return (
        f"Suspend every active or trial {ind} account that has an overdue invoice, for non-payment.",
        [call("suspend_account", account_id=a, reason="nonpayment") for a in accs],
    )


@_register("cross_entity", ("accounts",), "hard")
def ce_tier_seats(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    tier = rng.choice(("starter", "pro"))
    accs = [
        a
        for a in _accs(s, lambda i, _a: _open(s, i))
        if s["accounts"][a]["plan"] == tier and _tier_room(s, a) != 0
    ]
    if not 1 <= len(accs) <= 5:
        return None
    return f"Add one seat to every active or trial account on the {tier} plan.", [
        call("add_seat", account_id=a, count=1) for a in accs
    ]


@_register("cross_entity", ("invoices", "accounts"), "hard")
def ce_two_accounts(rng: random.Random, s: State) -> tuple[str, list[Call]] | None:
    cands = [a for a in s["accounts"] if _paid(s, a)]
    if len(cands) < 2:
        return None
    a, b = sorted(rng.sample(cands, 2))
    ia = max(_paid(s, a), key=lambda i: (s["invoices"][i]["amount_cents"], i))
    ib = max(_paid(s, b), key=lambda i: (s["invoices"][i]["amount_cents"], i))
    ties = [
        sum(
            s["invoices"][i]["amount_cents"] == s["invoices"][x]["amount_cents"]
            for i in _paid(s, x)
        )
        for x in (a, b)
    ]
    if max(ties) > 1:
        return None
    return (
        f"Refund the largest paid invoice of account {a} and the largest paid invoice of account {b}.",
        [call("refund_invoice", invoice_id=ia), call("refund_invoice", invoice_id=ib)],
    )


TEMPLATES: tuple[Template, ...] = tuple(_TEMPLATES)
FAMILIES: tuple[str, ...] = tuple(sorted({t.family for t in TEMPLATES}))
GUARD_FAMILIES = frozenset({"error_guard", "conditional"})  # gold may legitimately be empty


def task_id(family: str, gold_answer: str) -> str:
    return "tc-" + hashlib.sha256(f"{family}|{gold_answer}".encode()).hexdigest()[:12]


# ---- reserved stress families (same rule as the SQL pack) -------------------------------------

# The STRESS set is drawn only from these families; they never appear in train, dev or the gate
# set. Chosen by a fixed written rule, not by past results:
#     random.Random(STRESS_SEED).sample(sorted(FAMILIES), k=STRESS_FAMILY_COUNT)
STRESS_SEED = 777
STRESS_FAMILY_COUNT = 2


def stress_families(families: tuple[str, ...] | None = None) -> tuple[str, ...]:
    pool = sorted(FAMILIES if families is None else families)
    return tuple(sorted(random.Random(STRESS_SEED).sample(pool, k=STRESS_FAMILY_COUNT)))


# ---- skeletons ------------------------------------------------------------------------------------

_VOCAB: frozenset[str] = frozenset(
    v.lower()
    for group in (
        _schema.INDUSTRIES, _schema.COUNTRIES, _schema.ACCOUNT_STATUSES, _schema.USER_ROLES,
        _schema.TIERS, _schema.INVOICE_STATUSES, _schema.PRIORITIES, E.AGENTS,
        ("non-payment", "abuse", "won't fix", "resolved", "a duplicate"),
    )
    for v in group
)  # fmt: skip
_VOCAB_RE = re.compile(
    r"(?<!\w)(" + "|".join(sorted(map(re.escape, _VOCAB), key=len, reverse=True)) + r")(?!\w)"
)
_QUOTED_RE = re.compile(r"'[A-Za-z][^']*'(?!\w)")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+")
_NAME_RE = re.compile(
    r"(?<=add )[A-Z][a-z]+ [A-Z][a-z]+|[A-Z][a-z]+ [A-Z][a-z]+(?= (?:\(|<|with))|[A-Z][a-z]+ [A-Z][a-z]+(?=,)|(?<=: )[A-Z][a-z]+ [A-Z][a-z]+|(?<=; )[A-Z][a-z]+ [A-Z][a-z]+"
)
_NUMBER_RE = re.compile(r"\d[\d,.:-]*")


def skeleton(question: str) -> str:
    """The goal with literals masked: quoted names and emails -> <s>, person names -> <s>, numbers
    -> <n>, enumerated vocabulary (countries, statuses, roles, tiers, agents, reasons) -> <v>."""
    s = _NAME_RE.sub("<s>", question)
    s = _EMAIL_RE.sub("<s>", s)
    s = _QUOTED_RE.sub("<s>", s.replace("’", "'"))
    s = _NUMBER_RE.sub("<n>", s.lower())
    s = _VOCAB_RE.sub("<v>", s)
    return " ".join(s.split())


# ---- capacity / generation ---------------------------------------------------------------------

CAPACITY_PROBE_DRAWS = 200
_CAPACITY_CACHE: dict[tuple[str, int, int], int] = {}


def template_capacity(
    tpl: Template, state: State, state_key: int = 0, draws: int = CAPACITY_PROBE_DRAWS
) -> int:
    """Distinct gold answers the template produced in ``draws`` deterministic draws on ``state``."""
    key = (tpl.name, state_key, id(tpl.build))
    if key not in _CAPACITY_CACHE:
        rng = random.Random(f"capacity|{tpl.name}")
        out = set()
        for _ in range(draws):
            r = tpl.build(rng, state)
            if r is not None:
                out.add(E.canonical_calls(r[1]))
        _CAPACITY_CACHE[key] = len(out)
    return _CAPACITY_CACHE[key]


DEFAULT_TEMPLATE_SHARE = 0.10
DEFAULT_FAMILY_SHARE = 0.30
_EXHAUSTED_AFTER = 8
_SKELETON_EXHAUSTED_AFTER = 30


@dataclass
class GenerationReport:
    tasks: list[ToolTask]
    dropped_invalid: int = 0  # gold replay failed
    dropped_noop: int = (
        0  # non-empty gold that leaves the state unchanged (or empty gold outside guard families)
    )
    dropped_none: int = 0  # template found no suitable entity
    dropped_duplicate: int = 0
    dropped_duplicate_question: int = 0
    dropped_cap: int = 0
    dropped_skeleton_cap: int = 0
    attempts: int = 0
    dropped_by_template: Counter[str] = field(default_factory=Counter)
    capped_templates: set[str] = field(default_factory=set)
    capped_families: set[str] = field(default_factory=set)

    @property
    def dropped_error(self) -> int:
        """Pack-interface name: gold replay failed."""
        return self.dropped_invalid

    @property
    def dropped_empty(self) -> int:
        """Pack-interface name: gold that changes nothing."""
        return self.dropped_noop

    @property
    def dropped_total(self) -> int:
        return (self.dropped_invalid + self.dropped_noop + self.dropped_none + self.dropped_duplicate
                + self.dropped_duplicate_question + self.dropped_cap + self.dropped_skeleton_cap)  # fmt: skip


def generate_tasks(
    state: State,
    n: int,
    seed: int,
    *,
    families: tuple[str, ...] | None = None,
    exclude_families: tuple[str, ...] = (),
    max_attempt_factor: int = 60,
    template_share: float = DEFAULT_TEMPLATE_SHARE,
    family_share: float = DEFAULT_FAMILY_SHARE,
    skeleton_cap: int | None = None,
    state_key: int = 0,
) -> GenerationReport:
    """Up to ``n`` distinct tasks whose gold replays cleanly on ``state``. Templates are cycled
    round-robin over the selected families with per-template, per-family and per-skeleton caps
    (see the SQL pack's ``generate_tasks``). Deterministic for (state, n, seed, filters)."""
    rng = random.Random(seed)
    excluded = set(exclude_families)
    pool = [
        t
        for t in TEMPLATES
        if (families is None or t.family in families) and t.family not in excluded
    ]
    if not pool:
        raise ValueError(f"no templates for families {families!r} minus {exclude_families!r}")
    tpl_cap = max(1, math.ceil(template_share * n), math.ceil(n / len(pool)))
    fam_cap = max(1, math.ceil(family_share * n), math.ceil(n / len({t.family for t in pool})))
    report = GenerationReport(tasks=[])
    capacity = {t.name: template_capacity(t, state, state_key) for t in pool}
    pool = [t for t in pool if capacity[t.name] > 0]
    seen: set[str] = set()
    seen_q: set[str] = set()
    streak: Counter[str] = Counter()
    sk_streak: Counter[str] = Counter()
    per_tpl: Counter[str] = Counter()
    per_fam: Counter[str] = Counter()
    per_sk: Counter[str] = Counter()
    base = E.canonical_state(state)

    def drop(tpl: Template, attr: str) -> None:
        setattr(report, attr, getattr(report, attr) + 1)
        report.dropped_by_template[tpl.name] += 1

    def retire(tpl: Template) -> None:
        nonlocal pool
        pool = [t for t in pool if t.name != tpl.name]

    i = 0
    while pool and len(report.tasks) < n and report.attempts < n * max_attempt_factor:
        tpl = pool[i % len(pool)]
        i += 1
        report.attempts += 1
        built = tpl.build(rng, state)
        if built is None:
            drop(tpl, "dropped_none")
            streak[tpl.name] += 1
            if streak[tpl.name] >= _EXHAUSTED_AFTER:
                retire(tpl)
            continue
        goal, calls = built
        answer = E.canonical_calls(calls)
        sk = skeleton(goal)
        if answer in seen or goal in seen_q:
            drop(tpl, "dropped_duplicate" if answer in seen else "dropped_duplicate_question")
            streak[tpl.name] += 1
            if streak[tpl.name] >= _EXHAUSTED_AFTER:
                retire(tpl)
            continue
        streak[tpl.name] = 0
        if skeleton_cap is not None and per_sk[sk] >= skeleton_cap:
            drop(tpl, "dropped_skeleton_cap")
            sk_streak[tpl.name] += 1
            if sk_streak[tpl.name] >= _SKELETON_EXHAUSTED_AFTER:
                retire(tpl)
            continue
        sk_streak[tpl.name] = 0
        res = E.replay(state, calls)
        if not res.ok:
            drop(tpl, "dropped_invalid")
            continue
        changed = E.canonical_state(res.state) != base
        if (not changed) if calls else (tpl.family not in GUARD_FAMILIES):
            drop(tpl, "dropped_noop")
            continue
        if per_tpl[tpl.name] >= min(tpl_cap, capacity[tpl.name]) or per_fam[tpl.family] >= fam_cap:
            drop(tpl, "dropped_cap")
            if per_tpl[tpl.name] >= min(tpl_cap, capacity[tpl.name]):
                report.capped_templates.add(tpl.name)
                retire(tpl)
            if per_fam[tpl.family] >= fam_cap:
                report.capped_families.add(tpl.family)
                pool = [t for t in pool if t.family != tpl.family]
            continue
        seen.add(answer)
        seen_q.add(goal)
        per_tpl[tpl.name] += 1
        per_fam[tpl.family] += 1
        per_sk[sk] += 1
        if per_tpl[tpl.name] >= min(tpl_cap, capacity[tpl.name]):
            report.capped_templates.add(tpl.name)
            retire(tpl)
        if per_fam[tpl.family] >= fam_cap:
            report.capped_families.add(tpl.family)
            pool = [t for t in pool if t.family != tpl.family]
        report.tasks.append(
            ToolTask(
                task_id=task_id(tpl.family, answer),
                family=tpl.family,
                question=goal,
                gold_answer=answer,
                difficulty=tpl.difficulty,
                tables=tuple(sorted(tpl.tables)),
                template=tpl.name,
            )  # fmt: skip
        )
    return report


IN_DISTRIBUTION = "in_distribution"
STRESS = "stress"
