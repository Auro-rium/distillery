# ruff: noqa: E501, S311
"""Seeded generator for an original, fictional B2B SaaS company database ("Lumenstack").

Eight tables: plans, accounts, users, subscriptions, invoices, payments, support_tickets,
feature_usage. The database is deterministic for a given seed and contains deliberate traps:
nullable FKs, NULL amounts, duplicated-looking names, soft-deleted rows, enum columns and
ISO-8601 date strings. There is no dependency on the wall clock; "today" is ``AS_OF``.
"""

from __future__ import annotations

import random
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

AS_OF = "2025-12-31"
_AS_OF_DATE = date.fromisoformat(AS_OF)

TABLES: tuple[str, ...] = (
    "plans",
    "accounts",
    "users",
    "subscriptions",
    "invoices",
    "payments",
    "support_tickets",
    "feature_usage",
)

INDUSTRIES = ("fintech", "healthcare", "retail", "logistics", "education", "media", "energy")
COUNTRIES = ("US", "DE", "GB", "FR", "IN", "BR", "CA", "AU", "JP", "NL")
ACCOUNT_STATUSES = ("active", "trial", "churned", "suspended")
USER_ROLES = ("owner", "admin", "member", "viewer")
TIERS = ("free", "starter", "pro", "enterprise")
SUB_STATUSES = ("active", "cancelled", "past_due", "trialing")
INVOICE_STATUSES = ("paid", "open", "overdue", "void")
PAYMENT_METHODS = ("card", "ach", "wire", "paypal")
PAYMENT_STATUSES = ("succeeded", "failed", "refunded")
PRIORITIES = ("low", "normal", "high", "urgent")
CATEGORIES = ("billing", "bug", "how_to", "feature_request", "account_access")
TICKET_STATUSES = ("open", "pending", "closed")
FEATURES = ("dashboards", "api", "exports", "sso", "automations", "audit_log")

DDL = """\
CREATE TABLE plans (
  plan_id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  tier TEXT NOT NULL,                -- free | starter | pro | enterprise
  monthly_price_cents INTEGER NOT NULL,
  seat_limit INTEGER                 -- NULL = unlimited seats
);
CREATE TABLE accounts (
  account_id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,                -- not unique: some names repeat across countries
  industry TEXT NOT NULL,            -- fintech | healthcare | retail | logistics | education | media | energy
  country TEXT NOT NULL,             -- ISO 2-letter code
  created_at TEXT NOT NULL,          -- YYYY-MM-DD
  status TEXT NOT NULL,              -- active | trial | churned | suspended
  deleted_at TEXT,                   -- soft delete: NULL = not deleted
  parent_account_id INTEGER REFERENCES accounts(account_id),  -- NULL = top-level
  sales_rep TEXT                     -- NULL = unassigned
);
CREATE TABLE users (
  user_id INTEGER PRIMARY KEY,
  account_id INTEGER NOT NULL REFERENCES accounts(account_id),
  email TEXT NOT NULL,
  full_name TEXT NOT NULL,           -- not unique
  role TEXT NOT NULL,                -- owner | admin | member | viewer
  created_at TEXT NOT NULL,          -- YYYY-MM-DD
  last_login_at TEXT,                -- YYYY-MM-DD HH:MM:SS, NULL = never logged in
  deleted_at TEXT                    -- soft delete: NULL = not deleted
);
CREATE TABLE subscriptions (
  subscription_id INTEGER PRIMARY KEY,
  account_id INTEGER NOT NULL REFERENCES accounts(account_id),
  plan_id INTEGER NOT NULL REFERENCES plans(plan_id),
  start_date TEXT NOT NULL,          -- YYYY-MM-DD
  end_date TEXT,                     -- NULL = still running
  status TEXT NOT NULL,              -- active | cancelled | past_due | trialing
  seats INTEGER NOT NULL,
  discount_pct REAL                  -- NULL = no discount
);
CREATE TABLE invoices (
  invoice_id INTEGER PRIMARY KEY,
  account_id INTEGER NOT NULL REFERENCES accounts(account_id),
  subscription_id INTEGER REFERENCES subscriptions(subscription_id),  -- NULL = one-off invoice
  issued_date TEXT NOT NULL,         -- YYYY-MM-DD
  due_date TEXT NOT NULL,            -- YYYY-MM-DD
  amount_cents INTEGER,              -- NULL = amount not yet computed
  currency TEXT NOT NULL,            -- USD | EUR
  status TEXT NOT NULL               -- paid | open | overdue | void
);
CREATE TABLE payments (
  payment_id INTEGER PRIMARY KEY,
  invoice_id INTEGER NOT NULL REFERENCES invoices(invoice_id),
  paid_at TEXT NOT NULL,             -- YYYY-MM-DD
  amount_cents INTEGER NOT NULL,
  method TEXT NOT NULL,              -- card | ach | wire | paypal
  status TEXT NOT NULL,              -- succeeded | failed | refunded
  processor_ref TEXT                 -- NULL for some ach/wire payments
);
CREATE TABLE support_tickets (
  ticket_id INTEGER PRIMARY KEY,
  account_id INTEGER NOT NULL REFERENCES accounts(account_id),
  user_id INTEGER REFERENCES users(user_id),  -- NULL = submitted by email
  opened_at TEXT NOT NULL,           -- YYYY-MM-DD HH:MM:SS
  closed_at TEXT,                    -- NULL while not closed
  priority TEXT NOT NULL,            -- low | normal | high | urgent
  category TEXT NOT NULL,            -- billing | bug | how_to | feature_request | account_access
  status TEXT NOT NULL,              -- open | pending | closed
  csat_score INTEGER                 -- 1-5, NULL = no rating given
);
CREATE TABLE feature_usage (
  usage_id INTEGER PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(user_id),
  feature TEXT NOT NULL,             -- dashboards | api | exports | sso | automations | audit_log
  used_on TEXT NOT NULL,             -- YYYY-MM-DD
  event_count INTEGER NOT NULL
);
"""

_PLANS: tuple[tuple[int, str, str, int, int | None], ...] = (
    (1, "Free", "free", 0, 3),
    (2, "Starter", "starter", 2900, 10),
    (3, "Pro Monthly", "pro", 9900, 50),
    (4, "Pro Annual", "pro", 8500, 50),
    (5, "Enterprise", "enterprise", 29900, None),
    (6, "Enterprise Plus", "enterprise", 49900, None),
)

_NAME_A = (
    "Bluefin", "Quillon", "Harbor", "Tamarack", "Ostrava", "Pinewood", "Redwing", "Solstice",
    "Kestrel", "Mosaic", "Fjord", "Cobalt", "Lantern", "Meridian", "Sable", "Tundra",
)  # fmt: skip
_NAME_B = ("Labs", "Systems", "Group", "Works", "Analytics", "Logistics", "Health", "Media")
_FIRST = (
    "Ana", "Ben", "Chen", "Dara", "Eli", "Farah", "Gus", "Hana", "Ivan", "Jia", "Kofi", "Lena",
    "Mika", "Noor", "Omar", "Priya", "Quinn", "Rosa", "Sam", "Tara",
)  # fmt: skip
_LAST = ("Alvarez", "Berg", "Cho", "Dubois", "Evans", "Fischer", "Gupta", "Hughes", "Ito", "Khan")
_REPS = ("R. Okafor", "L. Meyer", "S. Tanaka", "J. Costa", "M. Novak", "P. Singh")


def schema_ddl(seed: int = 0) -> str:
    """CREATE TABLE text shown to the model. The schema is identical for every seed."""
    del seed  # the schema does not vary; data does
    return DDL


def _rand_date(rng: random.Random, lo: date, hi: date) -> date:
    if hi <= lo:
        return lo
    return lo + timedelta(days=rng.randint(0, (hi - lo).days))


def _slug(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum())


def _connect_memory(seed: int) -> sqlite3.Connection:
    rng = random.Random(seed)
    con = sqlite3.connect(":memory:")
    con.executescript(DDL)
    con.executemany("INSERT INTO plans VALUES (?,?,?,?,?)", _PLANS)
    plan_price = {p[0]: p[3] for p in _PLANS}
    plan_limit = {p[0]: p[4] for p in _PLANS}

    # accounts -------------------------------------------------------------------------
    accounts: list[tuple[object, ...]] = []
    acct_created: dict[int, date] = {}
    acct_status: dict[int, str] = {}
    names: list[str] = []
    n_accounts = 160
    for aid in range(1, n_accounts + 1):
        if names and aid > 20 and rng.random() < 0.06:
            name = rng.choice(
                names
            )  # duplicated-looking name (different row, likely different country)
        else:
            name = f"{rng.choice(_NAME_A)} {rng.choice(_NAME_B)}"
        names.append(name)
        created = _rand_date(rng, date(2022, 1, 1), date(2025, 10, 31))
        status = rng.choices(ACCOUNT_STATUSES, weights=(60, 10, 20, 10))[0]
        deleted = None
        if rng.random() < 0.05:
            deleted = _rand_date(rng, created, _AS_OF_DATE).isoformat()
        parent = rng.randint(1, aid - 1) if aid > 10 and rng.random() < 0.10 else None
        rep = None if rng.random() < 0.25 else rng.choice(_REPS)
        acct_created[aid] = created
        acct_status[aid] = status
        accounts.append(
            (
                aid, name, rng.choice(INDUSTRIES), rng.choice(COUNTRIES), created.isoformat(),
                status, deleted, parent, rep,
            )
        )  # fmt: skip
    con.executemany("INSERT INTO accounts VALUES (?,?,?,?,?,?,?,?,?)", accounts)

    # users ----------------------------------------------------------------------------
    users: list[tuple[object, ...]] = []
    uid = 0
    for aid in range(1, n_accounts + 1):
        for k in range(rng.randint(1, 8)):
            uid += 1
            first, last = rng.choice(_FIRST), rng.choice(_LAST)
            role = "owner" if k == 0 else rng.choices(USER_ROLES[1:], weights=(2, 6, 3))[0]
            created = _rand_date(rng, acct_created[aid], _AS_OF_DATE)
            login = None
            if rng.random() > 0.15:
                ld = _rand_date(rng, created, _AS_OF_DATE)
                login = f"{ld.isoformat()} {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:00"
            deleted = (
                _rand_date(rng, created, _AS_OF_DATE).isoformat() if rng.random() < 0.08 else None
            )
            email = f"{_slug(first)}.{_slug(last)}{uid}@acct{aid}.example.com"
            users.append(
                (uid, aid, email, f"{first} {last}", role, created.isoformat(), login, deleted)
            )
    con.executemany("INSERT INTO users VALUES (?,?,?,?,?,?,?,?)", users)

    # subscriptions --------------------------------------------------------------------
    subs: list[tuple[object, ...]] = []
    sid = 0
    for aid in range(1, n_accounts + 1):
        st = acct_status[aid]
        n_subs = 1 if rng.random() < 0.7 else 2
        cursor = acct_created[aid]
        for k in range(n_subs):
            sid += 1
            plan_id = rng.choice((1, 2, 2, 3, 3, 4, 5, 6))
            start = min(cursor + timedelta(days=rng.randint(0, 30)), _AS_OF_DATE)
            is_last = k == n_subs - 1
            end: date | None = None
            if not is_last:
                end = min(start + timedelta(days=rng.randint(60, 300)), _AS_OF_DATE)
                sub_status = "cancelled"
                cursor = end
            elif st == "churned":
                end = min(start + timedelta(days=rng.randint(30, 400)), _AS_OF_DATE)
                sub_status = "cancelled"
            elif st == "trial":
                sub_status = "trialing"
            elif st == "suspended":
                sub_status = "past_due"
            else:
                sub_status = "active"
            limit = plan_limit[plan_id]
            seats = rng.randint(1, limit if limit is not None else 200)
            discount = None if rng.random() < 0.6 else float(rng.choice((5, 10, 15, 20)))
            subs.append(
                (
                    sid, aid, plan_id, start.isoformat(), end.isoformat() if end else None,
                    sub_status, seats, discount,
                )
            )  # fmt: skip
    con.executemany("INSERT INTO subscriptions VALUES (?,?,?,?,?,?,?,?)", subs)

    # invoices & payments --------------------------------------------------------------
    invoices: list[tuple[object, ...]] = []
    payments: list[tuple[object, ...]] = []
    iid = pid = 0

    def add_invoice(aid: int, sub: int | None, issued: date, cents: int | None) -> None:
        nonlocal iid, pid
        iid += 1
        due = issued + timedelta(days=30)
        status = rng.choices(INVOICE_STATUSES, weights=(75, 8, 10, 7))[0]
        cur = "USD" if rng.random() < 0.85 else "EUR"
        invoices.append((iid, aid, sub, issued.isoformat(), due.isoformat(), cents, cur, status))
        pay_amt = cents if cents is not None else rng.randint(1000, 90000)
        if status == "paid":
            if rng.random() < 0.15:  # a failed attempt before the successful one
                pid += 1
                payments.append(
                    (pid, iid, issued.isoformat(), pay_amt, rng.choice(PAYMENT_METHODS),
                     "failed", "ref_" + str(pid))
                )  # fmt: skip
            pid += 1
            paid_on = min(issued + timedelta(days=rng.randint(0, 45)), _AS_OF_DATE)
            method = rng.choice(PAYMENT_METHODS)
            pstatus = "refunded" if rng.random() < 0.05 else "succeeded"
            ref = None if method in ("ach", "wire") and rng.random() < 0.5 else f"ref_{pid}"
            payments.append((pid, iid, paid_on.isoformat(), pay_amt, method, pstatus, ref))
        elif status == "overdue" and rng.random() < 0.4:
            pid += 1
            payments.append(
                (
                    pid,
                    iid,
                    min(due, _AS_OF_DATE).isoformat(),
                    pay_amt,
                    "card",
                    "failed",
                    f"ref_{pid}",
                )
            )

    for sub in subs:
        s_id, aid, plan_id = int(sub[0]), int(sub[1]), int(sub[2])  # type: ignore[call-overload]
        price = plan_price[plan_id]
        if price == 0:
            continue
        start = date.fromisoformat(str(sub[3]))
        end_s = sub[4]
        stop = date.fromisoformat(str(end_s)) if end_s else _AS_OF_DATE
        seats = int(sub[6])  # type: ignore[call-overload]
        disc = sub[7]
        issued = start
        for _ in range(8):
            if issued > stop:
                break
            amount: int | None = round(price * seats * (1 - (float(disc) if disc else 0.0) / 100))  # type: ignore[arg-type]
            if rng.random() < 0.03:
                amount = None
            add_invoice(aid, s_id, issued, amount)
            issued += timedelta(days=30)
    for _ in range(60):  # one-off invoices without a subscription
        add_invoice(
            rng.randint(1, n_accounts), None,
            _rand_date(rng, date(2023, 1, 1), _AS_OF_DATE), rng.randint(5, 400) * 100,
        )  # fmt: skip
    con.executemany("INSERT INTO invoices VALUES (?,?,?,?,?,?,?,?)", invoices)
    con.executemany("INSERT INTO payments VALUES (?,?,?,?,?,?,?)", payments)

    # support tickets ------------------------------------------------------------------
    tickets: list[tuple[object, ...]] = []
    tid = 0
    users_by_acct: dict[int, list[int]] = {}
    for u in users:
        users_by_acct.setdefault(int(u[1]), []).append(int(u[0]))  # type: ignore[call-overload]
    for aid in range(1, n_accounts + 1):
        if rng.random() < 0.2:
            continue  # accounts that never contacted support
        for _ in range(rng.randint(1, 8)):
            tid += 1
            opened_d = _rand_date(rng, acct_created[aid], _AS_OF_DATE)
            opened = datetime(
                opened_d.year, opened_d.month, opened_d.day, rng.randint(0, 23), rng.randint(0, 59)
            )
            status = rng.choices(TICKET_STATUSES, weights=(15, 10, 75))[0]
            closed = None
            csat = None
            if status == "closed":
                closed_dt = opened + timedelta(hours=rng.randint(1, 24 * 20))
                closed = closed_dt.strftime("%Y-%m-%d %H:%M:%S")
                if rng.random() < 0.5:
                    csat = rng.randint(1, 5)
            user = None if rng.random() < 0.10 else rng.choice(users_by_acct[aid])
            tickets.append(
                (
                    tid, aid, user, opened.strftime("%Y-%m-%d %H:%M:%S"), closed,
                    rng.choices(PRIORITIES, weights=(30, 45, 18, 7))[0], rng.choice(CATEGORIES),
                    status, csat,
                )
            )  # fmt: skip
    con.executemany("INSERT INTO support_tickets VALUES (?,?,?,?,?,?,?,?,?)", tickets)

    # feature usage --------------------------------------------------------------------
    usage: list[tuple[object, ...]] = []
    xid = 0
    for u in users:
        if rng.random() < 0.15:
            continue  # users who never used a tracked feature
        for _ in range(rng.randint(1, 4)):
            xid += 1
            day = _rand_date(rng, date(2025, 1, 1), _AS_OF_DATE)
            usage.append((xid, u[0], rng.choice(FEATURES), day.isoformat(), rng.randint(1, 500)))
    con.executemany("INSERT INTO feature_usage VALUES (?,?,?,?,?)", usage)
    con.commit()
    return con


def build_database(seed: int) -> bytes:
    """Return the serialized SQLite database (bytes) for ``seed``; deterministic."""
    con = _connect_memory(seed)
    try:
        return con.serialize()
    finally:
        con.close()


def write_database(seed: int, path: str | Path) -> Path:
    """Write the database for ``seed`` to ``path`` (overwriting) and return the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(build_database(seed))
    return p
