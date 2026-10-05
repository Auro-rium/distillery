# ruff: noqa: S311, E501
"""Deterministic seeded SaaS-admin state and the tools that mutate it (pure Python, no model code).

The state reuses the enums of the SQL pack's "Lumenstack" schema (industries, countries, account
statuses, roles, tiers, invoice/ticket vocab). It is a plain JSON-able dict: ``accounts``,
``users``, ``invoices``, ``tickets`` (integer ids) plus ``next_user_id``. ``build_state(seed)`` is
a pure function of the seed (no wall clock).

A *call* is ``{"tool": name, "args": {...}}``. ``apply_call`` validates the args against the
tool's JSON schema (required keys, types, enums, bounds, no extra keys), then checks the
tool's preconditions on the current state, and only then mutates it. Any violation raises
``ToolError``; ``replay`` turns that into a failed run (an invalid call fails the task).
``lookup_*`` tools are read-only (they validate and require the entity to exist but never change
the state). ``canonical_state`` is the stable string that final states are compared by.
"""

from __future__ import annotations

import copy
import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from distillery.taskpacks.sql.schema import (
    ACCOUNT_STATUSES,
    COUNTRIES,
    INDUSTRIES,
    INVOICE_STATUSES,
    PRIORITIES,
    TICKET_STATUSES,
    TIERS,
    USER_ROLES,
)

State = dict[str, Any]
Call = dict[str, Any]

CATEGORIES_TC = ("billing", "bug", "how_to", "feature_request", "account_access")
AGENTS = ("dana", "eli", "fatima", "gus", "hana", "ivo")
SUSPEND_REASONS = ("nonpayment", "abuse", "requested")
RESOLUTIONS = ("resolved", "wont_fix", "duplicate")
# seat_limit per tier (None = unlimited), mirroring the SQL plans table's idea of a seat cap
SEAT_LIMITS: dict[str, int | None] = {"free": 3, "starter": 10, "pro": 40, "enterprise": None}
MAX_SEATS_PER_CALL = 10

_ACCOUNT_NAMES = (
    "Northwind", "Brightpath", "Cobalt Labs", "Driftwood", "Evergreen", "Fathom", "Granite Peak",
    "Harbor & Co", "Ironleaf", "Juniper", "Keystone", "Lattice", "Meridian", "Nimbus",
)  # fmt: skip
_FIRST = (
    "Ana",
    "Ben",
    "Chloe",
    "Dev",
    "Esra",
    "Finn",
    "Gita",
    "Hugo",
    "Ines",
    "Jon",
    "Kira",
    "Leo",
)
_LAST = (
    "Moss",
    "Nakamura",
    "Okafor",
    "Petrov",
    "Quinn",
    "Rossi",
    "Silva",
    "Tanaka",
    "Umar",
    "Vogel",
)


class ToolError(Exception):
    """A call violated its schema or a precondition."""


def _schema(
    props: dict[str, dict[str, Any]], required: Sequence[str] | None = None
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": props,
        "required": list(props if required is None else required),
        "additionalProperties": False,
    }


_ID = {"type": "integer", "minimum": 1}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    schema: dict[str, Any]
    mutating: bool
    handler: Callable[[State, dict[str, Any]], None]


def _check_value(path: str, spec: Mapping[str, Any], v: Any) -> None:
    t = spec["type"]
    if t == "integer":
        if isinstance(v, bool) or not isinstance(v, int):
            raise ToolError(f"{path}: expected integer, got {v!r}")
        if "minimum" in spec and v < spec["minimum"]:
            raise ToolError(f"{path}: {v} < minimum {spec['minimum']}")
        if "maximum" in spec and v > spec["maximum"]:
            raise ToolError(f"{path}: {v} > maximum {spec['maximum']}")
    elif t == "string":
        if not isinstance(v, str):
            raise ToolError(f"{path}: expected string, got {v!r}")
        if "enum" in spec and v not in spec["enum"]:
            raise ToolError(f"{path}: {v!r} not in {list(spec['enum'])}")
        if len(v) < spec.get("minLength", 0):
            raise ToolError(f"{path}: string too short")
    else:  # pragma: no cover - specs are ours
        raise ToolError(f"{path}: unsupported schema type {t}")


def validate_args(schema: Mapping[str, Any], args: Any) -> None:
    """Minimal JSON-schema check (object, required, additionalProperties=false, int/string, enum,
    minimum/maximum, minLength). Raises ``ToolError``."""
    if not isinstance(args, dict):
        raise ToolError("args must be an object")
    props = schema["properties"]
    for k in schema["required"]:
        if k not in args:
            raise ToolError(f"missing required arg {k!r}")
    for k, v in args.items():
        if k not in props:
            raise ToolError(f"unexpected arg {k!r}")
        _check_value(k, props[k], v)


# ---- state helpers -------------------------------------------------------------------------


def _get(state: State, table: str, key: str, args: Mapping[str, Any]) -> dict[str, Any]:
    row = state[table].get(args[key])
    if row is None:
        raise ToolError(f"{table[:-1]} {args[key]} does not exist")
    return row  # type: ignore[no-any-return]


def live_users(state: State, account_id: int) -> list[int]:
    return sorted(u for u, r in state["users"].items() if r["account_id"] == account_id)


def _need_open_account(acc: Mapping[str, Any], aid: int) -> None:
    if acc["status"] not in ("active", "trial"):
        raise ToolError(f"account {aid} is {acc['status']}; only active/trial accounts can change")


# ---- tools ---------------------------------------------------------------------------------


def _create_user(state: State, a: dict[str, Any]) -> None:
    acc = _get(state, "accounts", "account_id", a)
    _need_open_account(acc, a["account_id"])
    if any(
        u["email"] == a["email"] for u in state["users"].values()
    ):  # emails are unique across the whole system
        raise ToolError(f"email {a['email']} already exists")
    if len(live_users(state, a["account_id"])) >= acc["seats"]:
        raise ToolError(f"account {a['account_id']} has no free seat")
    uid = state["next_user_id"]
    state["next_user_id"] = uid + 1
    state["users"][uid] = {
        "account_id": a["account_id"], "email": a["email"],
        "full_name": a["full_name"], "role": a["role"],
    }  # fmt: skip


def _update_plan(state: State, a: dict[str, Any]) -> None:
    acc = _get(state, "accounts", "account_id", a)
    _need_open_account(acc, a["account_id"])
    if acc["plan"] == a["plan"]:
        raise ToolError(f"account {a['account_id']} is already on {a['plan']}")
    limit = SEAT_LIMITS[a["plan"]]
    if limit is not None and acc["seats"] > limit:
        raise ToolError(f"{a['plan']} allows {limit} seats; account has {acc['seats']}")
    acc["plan"] = a["plan"]


def _refund_invoice(state: State, a: dict[str, Any]) -> None:
    inv = _get(state, "invoices", "invoice_id", a)
    if inv["status"] != "paid":
        raise ToolError(f"invoice {a['invoice_id']} is {inv['status']}; only paid can be refunded")
    if inv["amount_cents"] is None:
        raise ToolError(f"invoice {a['invoice_id']} has no amount")
    inv["status"] = "void"
    inv["refunded"] = True


def _assign_ticket(state: State, a: dict[str, Any]) -> None:
    t = _get(state, "tickets", "ticket_id", a)
    if t["status"] == "closed":
        raise ToolError(f"ticket {a['ticket_id']} is closed")
    if t["assignee"] == a["agent"]:
        raise ToolError(f"ticket {a['ticket_id']} is already assigned to {a['agent']}")
    t["assignee"] = a["agent"]
    t["status"] = "pending"


def _close_ticket(state: State, a: dict[str, Any]) -> None:
    t = _get(state, "tickets", "ticket_id", a)
    if t["status"] == "closed":
        raise ToolError(f"ticket {a['ticket_id']} is already closed")
    if t["assignee"] is None:
        raise ToolError(f"ticket {a['ticket_id']} must be assigned before it is closed")
    t["status"] = "closed"
    t["resolution"] = a["resolution"]


def _add_seat(state: State, a: dict[str, Any]) -> None:
    acc = _get(state, "accounts", "account_id", a)
    _need_open_account(acc, a["account_id"])
    limit = SEAT_LIMITS[acc["plan"]]
    if acc["plan"] == "free":
        raise ToolError("free plan cannot buy extra seats")
    if limit is not None and acc["seats"] + a["count"] > limit:
        raise ToolError(f"{acc['plan']} allows at most {limit} seats")
    acc["seats"] += a["count"]


def _suspend_account(state: State, a: dict[str, Any]) -> None:
    acc = _get(state, "accounts", "account_id", a)
    _need_open_account(acc, a["account_id"])
    acc["status"] = "suspended"
    acc["suspend_reason"] = a["reason"]


def _lookup_account(state: State, a: dict[str, Any]) -> None:
    _get(state, "accounts", "account_id", a)


def _lookup_user(state: State, a: dict[str, Any]) -> None:
    if not any(u["email"] == a["email"] for u in state["users"].values()):
        raise ToolError(f"no user with email {a['email']}")


def _lookup_invoice(state: State, a: dict[str, Any]) -> None:
    _get(state, "invoices", "invoice_id", a)


def _lookup_ticket(state: State, a: dict[str, Any]) -> None:
    _get(state, "tickets", "ticket_id", a)


def _enum(values: Sequence[str]) -> dict[str, Any]:
    return {"type": "string", "enum": list(values)}


_STR = {"type": "string", "minLength": 1}
_ACC = {"account_id": _ID}

TOOL_LIST: tuple[ToolSpec, ...] = (
    ToolSpec("create_user", "Add a user to an active/trial account with a free seat; emails are globally unique.",
             _schema({**_ACC, "email": _STR, "full_name": _STR, "role": _enum(USER_ROLES)}), True, _create_user),
    ToolSpec("update_plan", "Move an active/trial account to a different tier (its seats must fit the tier's seat limit).",
             _schema({**_ACC, "plan": _enum(TIERS)}), True, _update_plan),
    ToolSpec("refund_invoice", "Refund a paid invoice (it becomes void).",
             _schema({"invoice_id": _ID}), True, _refund_invoice),
    ToolSpec("assign_ticket", "Assign a not-closed ticket to a support agent (status becomes pending).",
             _schema({"ticket_id": _ID, "agent": _enum(AGENTS)}), True, _assign_ticket),
    ToolSpec("close_ticket", "Close an assigned ticket with a resolution.",
             _schema({"ticket_id": _ID, "resolution": _enum(RESOLUTIONS)}), True, _close_ticket),
    ToolSpec("add_seat", "Buy extra seats for an active/trial account on a paid tier (within the tier limit).",
             _schema({**_ACC, "count": {"type": "integer", "minimum": 1, "maximum": MAX_SEATS_PER_CALL}}), True, _add_seat),
    ToolSpec("suspend_account", "Suspend an active/trial account.",
             _schema({**_ACC, "reason": _enum(SUSPEND_REASONS)}), True, _suspend_account),
    ToolSpec("lookup_account", "Read-only: check that an account exists.", _schema(_ACC), False, _lookup_account),
    ToolSpec("lookup_user", "Read-only: check that a user email exists.", _schema({"email": _STR}), False, _lookup_user),
    ToolSpec("lookup_invoice", "Read-only: check that an invoice exists.", _schema({"invoice_id": _ID}), False, _lookup_invoice),
    ToolSpec("lookup_ticket", "Read-only: check that a ticket exists.", _schema({"ticket_id": _ID}), False, _lookup_ticket),
)  # fmt: skip
TOOLS: dict[str, ToolSpec] = {t.name: t for t in TOOL_LIST}
SEAT_LIMIT_TEXT = ", ".join(
    f"{k}={'unlimited' if v is None else v}" for k, v in SEAT_LIMITS.items()
)


def apply_call(state: State, call: Any) -> None:
    """Validate ``call`` and apply it to ``state`` in place. Raises ``ToolError``."""
    if not isinstance(call, dict) or set(call) != {"tool", "args"}:
        raise ToolError('a call must be an object with exactly "tool" and "args"')
    spec = TOOLS.get(call["tool"]) if isinstance(call["tool"], str) else None
    if spec is None:
        raise ToolError(f"unknown tool {call['tool']!r}")
    validate_args(spec.schema, call["args"])
    spec.handler(state, call["args"])


@dataclass(frozen=True)
class ReplayResult:
    ok: bool
    state: State
    error: str = ""
    failed_index: int | None = None


def replay(state: State, calls: Sequence[Any]) -> ReplayResult:
    """Apply ``calls`` in order to a COPY of ``state``; the first invalid call fails the run."""
    cur = copy.deepcopy(state)
    for i, c in enumerate(calls):
        try:
            apply_call(cur, c)
        except ToolError as e:
            return ReplayResult(False, cur, f"call {i} ({_name(c)}): {e}", i)
    return ReplayResult(True, cur)


def _name(c: Any) -> str:
    return str(c.get("tool")) if isinstance(c, dict) else "?"


def canonical_state(state: State) -> str:
    """Stable string form of a state (sorted keys, compact). Equal states <=> equal strings."""
    return json.dumps(state, sort_keys=True, separators=(",", ":"))


def canonical_calls(calls: Sequence[Any]) -> str:
    """Answer text for a call list: one call per line (``tool`` first, args keys sorted)."""
    if not calls:
        return "[]"
    lines = [
        '{"tool": '
        + json.dumps(c["tool"])
        + ', "args": '
        + json.dumps(c["args"], sort_keys=True)
        + "}"
        for c in calls
    ]
    return "[\n" + ",\n".join(lines) + "\n]"


# ---- seeded state --------------------------------------------------------------------------


def build_state(seed: int, n_accounts: int = 10) -> State:
    """The deterministic world for ``seed``: ~10 accounts, their users, invoices and tickets.

    Deliberate traps: two accounts share a name (different countries), some accounts are full
    (users == seats) or at their tier's seat limit, some invoices are not paid or have no
    amount, tickets are open/unassigned, pending/assigned or closed.
    """
    rng = random.Random(f"toolcall-env|{seed}")
    names = rng.sample(_ACCOUNT_NAMES, n_accounts)
    names[-1] = names[0]  # duplicated-looking name
    state: State = {"accounts": {}, "users": {}, "invoices": {}, "tickets": {}, "next_user_id": 1}
    countries = rng.sample(COUNTRIES, n_accounts)
    uid = iid = tid = 1
    for aid in range(1, n_accounts + 1):
        plan = rng.choices(TIERS, weights=(2, 3, 3, 1))[0]
        limit = SEAT_LIMITS[plan]
        status = rng.choices(ACCOUNT_STATUSES, weights=(6, 2, 1, 1))[0]
        n_users = rng.randint(1, 5 if limit is None else min(5, limit))
        full = rng.random() < 0.35
        seats = n_users if full else n_users + rng.randint(1, 3)
        if limit is not None:
            seats = min(seats, limit)
            n_users = min(n_users, seats)
        acc: dict[str, Any] = {
            "name": names[aid - 1], "country": countries[aid - 1], "industry": rng.choice(INDUSTRIES),
            "status": status, "plan": plan, "seats": seats,
        }  # fmt: skip
        if status == "suspended":
            acc["suspend_reason"] = rng.choice(SUSPEND_REASONS)
        state["accounts"][aid] = acc
        for k in range(n_users):
            full_name = f"{rng.choice(_FIRST)} {rng.choice(_LAST)}"
            email = f"{full_name.lower().replace(' ', '.')}{uid}@{names[aid - 1].lower().split()[0].strip('&')}.example"
            role = "owner" if k == 0 else rng.choice(USER_ROLES[1:])
            state["users"][uid] = {
                "account_id": aid,
                "email": email,
                "full_name": full_name,
                "role": role,
            }
            uid += 1
        for _ in range(rng.randint(1, 3)):
            st = rng.choices(INVOICE_STATUSES, weights=(5, 2, 1, 1))[0]
            amount = None if rng.random() < 0.08 else rng.randrange(2_000, 90_000, 100)
            state["invoices"][iid] = {
                "account_id": aid,
                "amount_cents": amount,
                "status": st,
                "refunded": False,
            }
            iid += 1
        for _ in range(rng.randint(0, 3)):
            st = rng.choice(TICKET_STATUSES)
            t: dict[str, Any] = {
                "account_id": aid, "priority": rng.choice(PRIORITIES),
                "category": rng.choice(CATEGORIES_TC), "status": st,
                "assignee": None if st == "open" else rng.choice(AGENTS),
            }  # fmt: skip
            if st == "closed":
                t["resolution"] = rng.choice(RESOLUTIONS)
            state["tickets"][tid] = t
            tid += 1
    state["next_user_id"] = uid
    return state


def env_ref(seed: int) -> str:
    return f"toolcall:{seed}"


def parse_env_ref(ref: str) -> int:
    kind, _, seed = ref.partition(":")
    if kind != "toolcall" or not seed.lstrip("-").isdigit():
        raise ValueError(f"not a toolcall env ref: {ref!r}")
    return int(seed)


def render_context(state: State) -> str:
    """Prompt text: the tool list (with JSON schemas) and the current state, compactly."""
    out = ['Tools (call format: {"tool": name, "args": {...}}):']
    for t in TOOL_LIST:
        props = t.schema["properties"]
        sig = ", ".join(
            f"{k}: " + ("|".join(v["enum"]) if "enum" in v else v["type"]) for k, v in props.items()
        )
        out.append(f"- {t.name}({sig}): {t.description}")
    out.append(
        f"Seat limits per tier: {SEAT_LIMIT_TEXT}. add_seat count is 1-{MAX_SEATS_PER_CALL}."
    )
    out.append("An invalid call (bad args or a violated precondition) fails the whole task.")
    out.append("\nAccounts (id | name | country | status | plan | seats):")
    for i, a in state["accounts"].items():
        extra = f" ({a['suspend_reason']})" if a.get("suspend_reason") else ""
        out.append(
            f"{i} | {a['name']} | {a['country']} | {a['status']}{extra} | {a['plan']} | {a['seats']}"
        )
    out.append("Users (id | account | email | name | role):")
    for i, u in state["users"].items():
        out.append(f"{i} | {u['account_id']} | {u['email']} | {u['full_name']} | {u['role']}")
    out.append("Invoices (id | account | amount_cents | status):")
    for i, v in state["invoices"].items():
        out.append(
            f"{i} | {v['account_id']} | {v['amount_cents'] if v['amount_cents'] is not None else 'NULL'} | {v['status']}"
        )
    out.append("Tickets (id | account | priority | category | status | assignee):")
    for i, t in state["tickets"].items():
        out.append(
            f"{i} | {t['account_id']} | {t['priority']} | {t['category']} | {t['status']} | {t['assignee'] or '-'}"
        )
    out.append(f"Next user id: {state['next_user_id']}")
    return "\n".join(out)


def call_of(tool: str, **args: Any) -> Call:
    return {"tool": tool, "args": args}
