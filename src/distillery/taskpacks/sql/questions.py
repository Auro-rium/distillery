# ruff: noqa: S608, S311, E501
"""Templated text-to-SQL question generator with named families and family-aware splitting.

Every template returns ``(question, gold_sql)``. ``requires_order`` is derived from the gold SQL
(top-level ORDER BY), and templates whose gold has ORDER BY say so in the question. Gold SQL is
executed against the database; tasks that error or return no rows are dropped and counted.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict

from distillery.taskpacks.sql.runner import DbRef, run_select
from distillery.taskpacks.sql.schema import (
    CATEGORIES,
    COUNTRIES,
    INDUSTRIES,
    PRIORITIES,
    TIERS,
)
from distillery.taskpacks.sql.sqltext import has_top_level_order_by

Difficulty = Literal["easy", "medium", "hard"]


class SqlTask(BaseModel):
    """One question with verified-executable gold SQL."""

    model_config = ConfigDict(frozen=True)

    task_id: str
    family: str
    question: str
    gold_sql: str
    requires_order: bool
    difficulty: Difficulty
    tables: tuple[str, ...] = ()
    template: str = ""


@dataclass(frozen=True)
class Template:
    name: str
    family: str
    tables: tuple[str, ...]
    difficulty: Difficulty
    build: Callable[[random.Random], tuple[str, str]]


_TEMPLATES: list[Template] = []


def _register(
    family: str, tables: tuple[str, ...], difficulty: Difficulty
) -> Callable[
    [Callable[[random.Random], tuple[str, str]]], Callable[[random.Random], tuple[str, str]]
]:
    def deco(
        fn: Callable[[random.Random], tuple[str, str]],
    ) -> Callable[[random.Random], tuple[str, str]]:
        _TEMPLATES.append(Template(fn.__name__, family, tables, difficulty, fn))
        return fn

    return deco


def _pick(rng: random.Random, options: tuple[str, ...] | list[str]) -> str:
    return rng.choice(list(options))


def _date(rng: random.Random, lo: str = "2023-01-01", hi: str = "2025-10-31") -> str:
    a, b = date.fromisoformat(lo), date.fromisoformat(hi)
    return (a + timedelta(days=rng.randint(0, (b - a).days))).isoformat()


_MONTHS = (
    "January", "February", "March", "April", "May", "June", "July", "August", "September",
    "October", "November", "December",
)  # fmt: skip
_STATUS = ("active", "trial", "churned", "suspended")
_ROLES = ("owner", "admin", "member", "viewer")
_INV_STATUS = ("paid", "open", "overdue", "void")
_FEATURES = ("dashboards", "api", "exports", "sso", "automations", "audit_log")
_PAY_METHOD = ("card", "ach", "wire", "paypal")

# ==== filter ==================================================================================


@_register("filter", ("accounts",), "easy")
def accounts_by_country_status(rng: random.Random) -> tuple[str, str]:
    c, s = _pick(rng, COUNTRIES), _pick(rng, _STATUS)
    q = _pick(
        rng,
        [
            f"List the names of {s} accounts in {c}.",
            f"Which accounts based in {c} currently have the status '{s}'? Give their names.",
            f"Show the name of every account in country {c} whose status is {s}.",
        ],
    )
    return q, f"SELECT name FROM accounts WHERE country = '{c}' AND status = '{s}'"


@_register("filter", ("users",), "easy")
def users_by_role(rng: random.Random) -> tuple[str, str]:
    r = _pick(rng, _ROLES)
    q = _pick(
        rng,
        [
            f"How many users have the role '{r}'?",
            f"Count the users whose role is {r}.",
            f"What is the number of user records with role {r}?",
        ],
    )
    return q, f"SELECT COUNT(*) FROM users WHERE role = '{r}'"


@_register("filter", ("invoices",), "easy")
def invoices_over_amount(rng: random.Random) -> tuple[str, str]:
    x = rng.choice((200, 500, 1000, 2000, 5000))
    q = _pick(
        rng,
        [
            f"Which invoices are for more than ${x}? Return the invoice ids.",
            f"List the ids of invoices with an amount above ${x} (amounts are stored in cents).",
        ],
    )
    return q, f"SELECT invoice_id FROM invoices WHERE amount_cents > {x * 100}"


@_register("filter", ("support_tickets",), "easy")
def tickets_priority_category(rng: random.Random) -> tuple[str, str]:
    p, c = _pick(rng, PRIORITIES), _pick(rng, CATEGORIES)
    q = _pick(
        rng,
        [
            f"Which {p}-priority support tickets are in the {c} category? Return ticket ids.",
            f"List ticket ids of tickets with priority '{p}' and category '{c}'.",
        ],
    )
    return q, f"SELECT ticket_id FROM support_tickets WHERE priority = '{p}' AND category = '{c}'"


@_register("filter", ("accounts",), "easy")
def accounts_created_after(rng: random.Random) -> tuple[str, str]:
    i, d = _pick(rng, INDUSTRIES), _date(rng, "2022-06-01", "2025-06-01")
    q = _pick(
        rng,
        [
            f"Which {i} accounts were created after {d}? Return account name and creation date.",
            f"Show name and created_at of accounts in the {i} industry created later than {d}.",
        ],
    )
    return q, f"SELECT name, created_at FROM accounts WHERE industry = '{i}' AND created_at > '{d}'"


# ==== join ====================================================================================


@_register("join", ("accounts", "subscriptions", "plans"), "medium")
def active_subs_country(rng: random.Random) -> tuple[str, str]:
    c = _pick(rng, COUNTRIES)
    q = _pick(
        rng,
        [
            f"List each account name with its plan name for active subscriptions of accounts in {c}.",
            f"For accounts located in {c}, show account name and plan name of their active subscriptions.",
        ],
    )
    return q, (
        "SELECT a.name, p.name FROM accounts a JOIN subscriptions s ON s.account_id = a.account_id "
        f"JOIN plans p ON p.plan_id = s.plan_id WHERE s.status = 'active' AND a.country = '{c}'"
    )


@_register("join", ("subscriptions", "plans", "accounts"), "medium")
def tier_subscriptions(rng: random.Random) -> tuple[str, str]:
    t = _pick(rng, TIERS[1:])
    q = _pick(
        rng,
        [
            f"Show account name and seats for every subscription on a {t}-tier plan.",
            f"For subscriptions whose plan tier is '{t}', list the account name and number of seats.",
        ],
    )
    return q, (
        "SELECT a.name, s.seats FROM subscriptions s JOIN plans p ON p.plan_id = s.plan_id "
        f"JOIN accounts a ON a.account_id = s.account_id WHERE p.tier = '{t}'"
    )


@_register("join", ("users", "support_tickets"), "medium")
def users_with_ticket_priority(rng: random.Random) -> tuple[str, str]:
    p = _pick(rng, PRIORITIES)
    q = _pick(
        rng,
        [
            f"Which users have opened at least one {p}-priority ticket? Return distinct emails.",
            f"List the distinct emails of users who submitted a ticket with priority {p}.",
        ],
    )
    return q, (
        "SELECT DISTINCT u.email FROM users u JOIN support_tickets t ON t.user_id = u.user_id "
        f"WHERE t.priority = '{p}'"
    )


@_register("join", ("invoices", "accounts"), "easy")
def invoices_account_names(rng: random.Random) -> tuple[str, str]:
    s, y = _pick(rng, _INV_STATUS), rng.choice((2023, 2024, 2025))
    q = _pick(
        rng,
        [
            f"List invoice ids with the account name for {s} invoices issued in {y}.",
            f"Show invoice_id and the owning account's name for invoices in {y} with status '{s}'.",
        ],
    )
    return q, (
        "SELECT i.invoice_id, a.name FROM invoices i JOIN accounts a ON a.account_id = i.account_id "
        f"WHERE i.status = '{s}' AND strftime('%Y', i.issued_date) = '{y}'"
    )


# ==== aggregation =============================================================================


@_register("aggregation", ("accounts",), "easy")
def accounts_per_industry(rng: random.Random) -> tuple[str, str]:
    st = _pick(rng, _STATUS)
    q = _pick(
        rng,
        [
            f"How many {st} accounts are there in each industry?",
            f"Count accounts with status '{st}' per industry, returning the industry and the count.",
        ],
    )
    return q, f"SELECT industry, COUNT(*) FROM accounts WHERE status = '{st}' GROUP BY industry"


@_register("aggregation", ("subscriptions", "plans"), "medium")
def active_subs_per_plan(rng: random.Random) -> tuple[str, str]:
    st = rng.choice(("active", "cancelled", "past_due", "trialing"))
    q = _pick(
        rng,
        [
            f"For each plan name, how many subscriptions have status '{st}'?",
            f"Return every plan name that has {st} subscriptions along with how many.",
        ],
    )
    return q, (
        "SELECT p.name, COUNT(*) FROM subscriptions s JOIN plans p ON p.plan_id = s.plan_id "
        f"WHERE s.status = '{st}' GROUP BY p.name"
    )


@_register("aggregation", ("payments", "invoices"), "hard")
def accounts_paid_over(rng: random.Random) -> tuple[str, str]:
    x = rng.choice((500, 1000, 3000, 5000))
    q = _pick(
        rng,
        [
            f"Which accounts have more than ${x} in succeeded payments in total? Return account_id and the total in cents.",
            f"Return account_id and total succeeded payment cents for accounts whose total exceeds ${x}.",
        ],
    )
    return q, (
        "SELECT i.account_id, SUM(p.amount_cents) FROM payments p "
        "JOIN invoices i ON i.invoice_id = p.invoice_id WHERE p.status = 'succeeded' "
        f"GROUP BY i.account_id HAVING SUM(p.amount_cents) > {x * 100}"
    )


@_register("aggregation", ("support_tickets",), "easy")
def avg_csat_category(rng: random.Random) -> tuple[str, str]:
    p = _pick(rng, PRIORITIES)
    q = _pick(
        rng,
        [
            f"What is the average CSAT score per category among {p}-priority tickets, rounded to 2 decimals?",
            f"For {p}-priority tickets, give each category's average csat_score rounded to two decimal places.",
        ],
    )
    return q, (
        "SELECT category, ROUND(AVG(csat_score), 2) FROM support_tickets "
        f"WHERE priority = '{p}' GROUP BY category"
    )


@_register("aggregation", ("accounts",), "medium")
def countries_many_accounts(rng: random.Random) -> tuple[str, str]:
    n = rng.randint(5, 13)
    q = _pick(
        rng,
        [
            f"Which countries have more than {n} accounts? Return country and account count.",
            f"List countries with over {n} accounts together with how many accounts each has.",
        ],
    )
    return q, f"SELECT country, COUNT(*) FROM accounts GROUP BY country HAVING COUNT(*) > {n}"


# ==== window ==================================================================================


@_register("window", ("accounts", "users"), "hard")
def biggest_account_per_industry(rng: random.Random) -> tuple[str, str]:
    q = _pick(
        rng,
        [
            "For each industry, which account has the most users (count all user rows)? "
            "Break ties by lowest account_id. Return industry, account name and user count.",
            "Find the account with the largest number of users in every industry (ties: lowest "
            "account_id). Show industry, name, and user count.",
        ],
    )
    return q, (
        "WITH c AS (SELECT a.account_id, a.name, a.industry, COUNT(u.user_id) AS n FROM accounts a "
        "LEFT JOIN users u ON u.account_id = a.account_id GROUP BY a.account_id), "
        "r AS (SELECT *, ROW_NUMBER() OVER (PARTITION BY industry ORDER BY n DESC, account_id) AS rn "
        "FROM c) SELECT industry, name, n FROM r WHERE rn = 1"
    )


@_register("window", ("payments",), "hard")
def running_payments(rng: random.Random) -> tuple[str, str]:
    y = rng.choice((2023, 2024, 2025))
    q = _pick(
        rng,
        [
            f"For {y}, show each month (YYYY-MM) with its succeeded payment total in cents and the "
            "running total so far, ordered by month.",
            f"Give month, monthly succeeded payment cents and cumulative cents for {y}, sorted by month.",
        ],
    )
    return q, (
        "SELECT month, total, SUM(total) OVER (ORDER BY month) AS running FROM "
        "(SELECT strftime('%Y-%m', paid_at) AS month, SUM(amount_cents) AS total FROM payments "
        f"WHERE status = 'succeeded' AND strftime('%Y', paid_at) = '{y}' GROUP BY month) ORDER BY month"
    )


@_register("window", ("invoices",), "medium")
def latest_invoice_per_account(rng: random.Random) -> tuple[str, str]:
    c = _pick(rng, ("USD", "EUR"))
    q = _pick(
        rng,
        [
            f"For each account, what is its most recent {c} invoice (latest issued_date, ties by "
            "highest invoice_id)? Return account_id, invoice_id and issued_date.",
            f"Return the latest {c}-currency invoice per account (break date ties with the larger "
            "invoice_id): account_id, invoice_id, issued_date.",
        ],
    )
    return q, (
        "SELECT account_id, invoice_id, issued_date FROM (SELECT account_id, invoice_id, issued_date, "
        "ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY issued_date DESC, invoice_id DESC) AS rn "
        f"FROM invoices WHERE currency = '{c}') WHERE rn = 1"
    )


@_register("window", ("support_tickets",), "hard")
def top_ticket_accounts_rank(rng: random.Random) -> tuple[str, str]:
    k = rng.randint(2, 4)
    q = _pick(
        rng,
        [
            f"Rank accounts by how many support tickets they have (RANK, highest first) and return "
            f"account_id, ticket count and rank for accounts ranked {k} or better.",
            f"Using RANK() over ticket counts (descending), list account_id, ticket count, and rank "
            f"for every account with rank <= {k}.",
        ],
    )
    return q, (
        "SELECT account_id, n, rnk FROM (SELECT account_id, COUNT(*) AS n, "
        "RANK() OVER (ORDER BY COUNT(*) DESC) AS rnk FROM support_tickets GROUP BY account_id) "
        f"WHERE rnk <= {k}"
    )


# ==== date_math ===============================================================================


@_register("date_math", ("invoices",), "easy")
def invoices_in_month(rng: random.Random) -> tuple[str, str]:
    y, m = rng.choice((2023, 2024, 2025)), rng.randint(1, 12)
    q = _pick(
        rng,
        [
            f"How many invoices were issued in {_MONTHS[m - 1]} {y}?",
            f"Count invoices whose issued_date falls in {_MONTHS[m - 1]} of {y}.",
        ],
    )
    return q, f"SELECT COUNT(*) FROM invoices WHERE strftime('%Y-%m', issued_date) = '{y}-{m:02d}'"


@_register("date_math", ("support_tickets",), "medium")
def avg_resolution_days(rng: random.Random) -> tuple[str, str]:
    c = _pick(rng, CATEGORIES)
    q = _pick(
        rng,
        [
            f"For closed {c} tickets, what is the average number of days from opened_at to closed_at "
            "per priority, rounded to 2 decimals?",
            f"Per priority, compute the mean days between opening and closing of closed {c} tickets "
            "(use julianday, round to 2 decimals).",
        ],
    )
    return q, (
        "SELECT priority, ROUND(AVG(julianday(closed_at) - julianday(opened_at)), 2) "
        f"FROM support_tickets WHERE closed_at IS NOT NULL AND category = '{c}' GROUP BY priority"
    )


@_register("date_math", ("payments", "invoices"), "medium")
def late_payments(rng: random.Random) -> tuple[str, str]:
    y = rng.choice((2023, 2024, 2025))
    q = _pick(
        rng,
        [
            f"Which succeeded payments made in {y} arrived after the invoice due date? Return payment ids.",
            f"List payment_id for succeeded payments in {y} whose paid_at is later than the invoice's due_date.",
        ],
    )
    return q, (
        "SELECT p.payment_id FROM payments p JOIN invoices i ON i.invoice_id = p.invoice_id "
        f"WHERE p.status = 'succeeded' AND p.paid_at > i.due_date AND strftime('%Y', p.paid_at) = '{y}'"
    )


@_register("date_math", ("accounts",), "easy")
def accounts_in_quarter(rng: random.Random) -> tuple[str, str]:
    y, qn = rng.choice((2022, 2023, 2024, 2025)), rng.randint(1, 4)
    lo = f"{y}-{3 * qn - 2:02d}-01"
    hi = f"{y}-{3 * qn:02d}-31"
    q = _pick(
        rng,
        [
            f"Which accounts were created in Q{qn} {y}? Return names.",
            f"List the names of accounts whose creation date is in the {qn}-quarter of {y} (calendar quarters).",
        ],
    )
    return q, f"SELECT name FROM accounts WHERE created_at BETWEEN '{lo}' AND '{hi}'"


@_register("date_math", ("subscriptions",), "medium")
def subs_active_on(rng: random.Random) -> tuple[str, str]:
    d = _date(rng, "2023-06-01", "2025-10-01")
    q = _pick(
        rng,
        [
            f"How many subscriptions were running on {d}? A subscription runs from start_date until end_date (or forever if end_date is NULL).",
            f"Count subscriptions that had started on or before {d} and had not ended before it (NULL end_date means still running).",
        ],
    )
    return q, (
        f"SELECT COUNT(*) FROM subscriptions WHERE start_date <= '{d}' "
        f"AND (end_date IS NULL OR end_date >= '{d}')"
    )


# ==== null_handling ===========================================================================


@_register("null_handling", ("support_tickets",), "easy")
def closed_no_csat(rng: random.Random) -> tuple[str, str]:
    p = _pick(rng, PRIORITIES)
    q = _pick(
        rng,
        [
            f"How many closed {p}-priority tickets have no CSAT score?",
            f"Count closed tickets of priority {p} where csat_score is missing.",
        ],
    )
    return q, (
        f"SELECT COUNT(*) FROM support_tickets WHERE status = 'closed' AND priority = '{p}' "
        "AND csat_score IS NULL"
    )


@_register("null_handling", ("invoices",), "easy")
def invoices_missing_amount(rng: random.Random) -> tuple[str, str]:
    s = _pick(rng, _INV_STATUS)
    q = _pick(
        rng,
        [
            f"List ids of {s} invoices that have no amount recorded.",
            f"Which invoices with status '{s}' have a NULL amount? Return invoice_id.",
        ],
    )
    return q, f"SELECT invoice_id FROM invoices WHERE status = '{s}' AND amount_cents IS NULL"


@_register("null_handling", ("users", "accounts"), "medium")
def never_logged_in(rng: random.Random) -> tuple[str, str]:
    i = _pick(rng, INDUSTRIES)
    q = _pick(
        rng,
        [
            f"Which users in {i} accounts have never logged in? Return their emails.",
            f"List emails of users belonging to {i}-industry accounts whose last_login_at is empty (never logged in).",
        ],
    )
    return q, (
        "SELECT u.email FROM users u JOIN accounts a ON a.account_id = u.account_id "
        f"WHERE a.industry = '{i}' AND u.last_login_at IS NULL"
    )


@_register("null_handling", ("invoices",), "medium")
def avg_amount_missing_as_zero(rng: random.Random) -> tuple[str, str]:
    q = _pick(
        rng,
        [
            "Per invoice status, what is the average invoice amount in cents when a missing amount "
            "counts as 0? Round to 2 decimals.",
            "For each invoice status, average amount_cents treating NULL as zero, rounded to 2 decimals.",
        ],
    )
    return q, (
        "SELECT status, ROUND(AVG(COALESCE(amount_cents, 0)), 2) FROM invoices GROUP BY status"
    )


@_register("null_handling", ("accounts",), "easy")
def unassigned_by_country(rng: random.Random) -> tuple[str, str]:
    st = _pick(rng, _STATUS)
    q = _pick(
        rng,
        [
            f"How many {st} accounts have no sales rep assigned, per country?",
            f"Per country, count the accounts with status '{st}' where sales_rep is NULL.",
        ],
    )
    return q, (
        f"SELECT country, COUNT(*) FROM accounts WHERE sales_rep IS NULL AND status = '{st}' "
        "GROUP BY country"
    )


@_register("null_handling", ("invoices",), "easy")
def one_off_invoices(rng: random.Random) -> tuple[str, str]:
    c = _pick(rng, ("USD", "EUR"))
    q = _pick(
        rng,
        [
            f"How many {c} invoices are not tied to any subscription?",
            f"Count {c} invoices whose subscription_id is NULL.",
        ],
    )
    return q, f"SELECT COUNT(*) FROM invoices WHERE currency = '{c}' AND subscription_id IS NULL"


@_register("null_handling", ("support_tickets",), "medium")
def tickets_without_user(rng: random.Random) -> tuple[str, str]:
    q = _pick(
        rng,
        [
            "For each category, how many tickets were submitted without a user, and how many tickets "
            "are there in total? Return category, count without user, total count.",
            "Per ticket category, give the number of tickets with no user_id next to the total number of tickets.",
        ],
    )
    return q, (
        "SELECT category, SUM(CASE WHEN user_id IS NULL THEN 1 ELSE 0 END), COUNT(*) "
        "FROM support_tickets GROUP BY category"
    )


# ==== subquery_cte ============================================================================


@_register("subquery_cte", ("payments", "invoices"), "hard")
def accounts_above_avg_paid(rng: random.Random) -> tuple[str, str]:
    q = _pick(
        rng,
        [
            "Which accounts have a total of succeeded payments above the average total across "
            "accounts that have any succeeded payment? Return account_id.",
            "List account_ids whose succeeded-payment total exceeds the mean of per-account totals "
            "(only accounts with at least one succeeded payment count toward the mean).",
        ],
    )
    return q, (
        "WITH t AS (SELECT i.account_id, SUM(p.amount_cents) AS total FROM payments p "
        "JOIN invoices i ON i.invoice_id = p.invoice_id WHERE p.status = 'succeeded' "
        "GROUP BY i.account_id) SELECT account_id FROM t WHERE total > (SELECT AVG(total) FROM t)"
    )


@_register("subquery_cte", ("users", "subscriptions", "plans"), "medium")
def users_in_tier_accounts(rng: random.Random) -> tuple[str, str]:
    t = _pick(rng, TIERS[1:])
    q = _pick(
        rng,
        [
            f"List emails of users whose account has an active subscription on a {t}-tier plan.",
            f"Which users belong to accounts with an active {t}-tier subscription? Return email.",
        ],
    )
    return q, (
        "SELECT email FROM users WHERE account_id IN (SELECT s.account_id FROM subscriptions s "
        f"JOIN plans p ON p.plan_id = s.plan_id WHERE s.status = 'active' AND p.tier = '{t}')"
    )


@_register("subquery_cte", ("support_tickets",), "medium")
def accounts_more_tickets_than_avg(rng: random.Random) -> tuple[str, str]:
    q = _pick(
        rng,
        [
            "Which accounts have more support tickets than the average ticket count per account "
            "(averaged over accounts that have at least one ticket)? Return account_id and count.",
            "Return account_id and ticket count for accounts above the mean per-account ticket "
            "count (accounts without tickets are excluded from the mean).",
        ],
    )
    return q, (
        "WITH c AS (SELECT account_id, COUNT(*) AS n FROM support_tickets GROUP BY account_id) "
        "SELECT account_id, n FROM c WHERE n > (SELECT AVG(n) FROM c)"
    )


@_register("subquery_cte", ("invoices",), "hard")
def invoices_above_currency_avg(rng: random.Random) -> tuple[str, str]:
    y = rng.choice((2024, 2025))
    q = _pick(
        rng,
        [
            f"Which invoices issued in {y} are larger than the average invoice amount of the same "
            "currency (over all invoices with a known amount)? Return invoice_id.",
            f"List invoice_ids from {y} whose amount_cents is above the average amount_cents of all "
            "invoices in that invoice's currency (NULL amounts ignored).",
        ],
    )
    return q, (
        "SELECT i.invoice_id FROM invoices i WHERE strftime('%Y', i.issued_date) = "
        f"'{y}' AND i.amount_cents > (SELECT AVG(j.amount_cents) FROM invoices j "
        "WHERE j.currency = i.currency)"
    )


# ==== anti_join ===============================================================================


@_register("anti_join", ("accounts", "support_tickets"), "medium")
def accounts_no_tickets(rng: random.Random) -> tuple[str, str]:
    c = _pick(rng, COUNTRIES)
    q = _pick(
        rng,
        [
            f"Which accounts in {c} have never opened a support ticket? Return names.",
            f"List the names of {c} accounts with no support tickets at all.",
        ],
    )
    if rng.random() < 0.5:
        sql = (
            "SELECT a.name FROM accounts a WHERE a.country = "
            f"'{c}' AND NOT EXISTS (SELECT 1 FROM support_tickets t WHERE t.account_id = a.account_id)"
        )
    else:
        sql = (
            "SELECT a.name FROM accounts a LEFT JOIN support_tickets t ON t.account_id = a.account_id "
            f"WHERE a.country = '{c}' AND t.ticket_id IS NULL"
        )
    return q, sql


@_register("anti_join", ("users", "feature_usage"), "medium")
def users_no_usage(rng: random.Random) -> tuple[str, str]:
    r = _pick(rng, _ROLES)
    q = _pick(
        rng,
        [
            f"Which {r} users have no feature usage recorded? Return user_id.",
            f"List user_ids of users with role '{r}' that never appear in feature_usage.",
        ],
    )
    return q, (
        f"SELECT u.user_id FROM users u WHERE u.role = '{r}' AND NOT EXISTS "
        "(SELECT 1 FROM feature_usage f WHERE f.user_id = u.user_id)"
    )


@_register("anti_join", ("invoices", "payments"), "medium")
def invoices_no_success_payment(rng: random.Random) -> tuple[str, str]:
    s = _pick(rng, ("open", "overdue", "paid"))
    q = _pick(
        rng,
        [
            f"Which {s} invoices have no succeeded payment? Return invoice_id.",
            f"List invoice ids with status '{s}' for which no payment with status 'succeeded' exists.",
        ],
    )
    return q, (
        f"SELECT i.invoice_id FROM invoices i WHERE i.status = '{s}' AND NOT EXISTS "
        "(SELECT 1 FROM payments p WHERE p.invoice_id = i.invoice_id AND p.status = 'succeeded')"
    )


@_register("anti_join", ("accounts", "subscriptions"), "medium")
def accounts_no_active_sub(rng: random.Random) -> tuple[str, str]:
    i = _pick(rng, INDUSTRIES)
    q = _pick(
        rng,
        [
            f"Which {i} accounts have no active subscription? Return account_id and name.",
            f"List account_id and name of accounts in the {i} industry that do not have any subscription with status 'active'.",
        ],
    )
    return q, (
        "SELECT a.account_id, a.name FROM accounts a WHERE a.industry = "
        f"'{i}' AND NOT EXISTS (SELECT 1 FROM subscriptions s WHERE s.account_id = a.account_id "
        "AND s.status = 'active')"
    )


# ==== soft_delete =============================================================================


@_register("soft_delete", ("users", "accounts"), "medium")
def live_users_per_account(rng: random.Random) -> tuple[str, str]:
    c = _pick(rng, COUNTRIES)
    q = _pick(
        rng,
        [
            f"For accounts in {c} that are not soft-deleted, how many non-deleted users does each have? "
            "Return account_id and the count (only accounts with at least one such user).",
            f"Count non-deleted users per non-deleted account in {c}; skip accounts with none. Return account_id, count.",
        ],
    )
    return q, (
        "SELECT a.account_id, COUNT(*) FROM accounts a JOIN users u ON u.account_id = a.account_id "
        f"WHERE a.deleted_at IS NULL AND u.deleted_at IS NULL AND a.country = '{c}' "
        "GROUP BY a.account_id"
    )


@_register("soft_delete", ("accounts",), "easy")
def deleted_accounts_status(rng: random.Random) -> tuple[str, str]:
    y = rng.choice((2022, 2023, 2024, 2025))
    q = _pick(
        rng,
        [
            f"How many soft-deleted accounts created in or after {y} are there for each status?",
            f"Per account status, count the accounts that have a deleted_at value and were created in {y} or later.",
        ],
    )
    return q, (
        "SELECT status, COUNT(*) FROM accounts WHERE deleted_at IS NOT NULL "
        f"AND created_at >= '{y}-01-01' GROUP BY status"
    )


@_register("soft_delete", ("users",), "easy")
def live_users_by_role(rng: random.Random) -> tuple[str, str]:
    y = rng.choice((2022, 2023, 2024, 2025))
    q = _pick(
        rng,
        [
            f"How many users created in {y} or later that are not deleted are there for each role?",
            f"Count non-deleted users with created_at in {y} or later, per role.",
        ],
    )
    return q, (
        f"SELECT role, COUNT(*) FROM users WHERE deleted_at IS NULL AND created_at >= '{y}-01-01' "
        "GROUP BY role"
    )


# ==== conditional =============================================================================


@_register("conditional", ("invoices",), "medium")
def invoice_size_buckets(rng: random.Random) -> tuple[str, str]:
    y = rng.choice((2023, 2024, 2025))
    q = _pick(
        rng,
        [
            f"Among invoices issued in {y}, bucket invoices as 'unknown' (no amount), 'small' (under $100), 'medium' ($100 up to "
            "but not including $1000) or 'large' ($1000 and over), and count invoices per bucket.",
            f"Count {y} invoices (by issued_date year) by size: 'unknown' if amount is missing, 'small' below 10000 cents, "
            "'medium' from 10000 to 99999 cents, 'large' from 100000 cents up.",
        ],
    )
    return q, (
        "SELECT CASE WHEN amount_cents IS NULL THEN 'unknown' WHEN amount_cents < 10000 THEN 'small' "
        "WHEN amount_cents < 100000 THEN 'medium' ELSE 'large' END AS bucket, COUNT(*) "
        f"FROM invoices WHERE strftime('%Y', issued_date) = '{y}' GROUP BY bucket"
    )


@_register("conditional", ("payments", "invoices"), "hard")
def success_vs_failed(rng: random.Random) -> tuple[str, str]:
    n = rng.randint(1, 2)
    q = _pick(
        rng,
        [
            f"For each account, count succeeded and failed payments; return accounts with at least {n} failed payment(s). Columns: account_id, succeeded, failed.",
            f"Per account show number of succeeded and of failed payments, keeping only accounts with {n} or more failed ones.",
        ],
    )
    return q, (
        "SELECT i.account_id, SUM(CASE WHEN p.status = 'succeeded' THEN 1 ELSE 0 END), "
        "SUM(CASE WHEN p.status = 'failed' THEN 1 ELSE 0 END) FROM payments p "
        "JOIN invoices i ON i.invoice_id = p.invoice_id GROUP BY i.account_id "
        f"HAVING SUM(CASE WHEN p.status = 'failed' THEN 1 ELSE 0 END) >= {n}"
    )


@_register("conditional", ("support_tickets",), "medium")
def tickets_open_closed_split(rng: random.Random) -> tuple[str, str]:
    c = _pick(rng, CATEGORIES)
    q = _pick(
        rng,
        [
            f"For each priority, how many {c} tickets are closed and how many are not closed?",
            f"Per priority, give the count of closed {c} tickets and the count in any other status.",
        ],
    )
    return q, (
        "SELECT priority, SUM(status = 'closed'), SUM(status <> 'closed') FROM support_tickets "
        f"WHERE category = '{c}' GROUP BY priority"
    )


# ==== set_ops =================================================================================


@_register("set_ops", ("support_tickets", "invoices"), "hard")
def urgent_or_overdue(rng: random.Random) -> tuple[str, str]:
    q = _pick(
        rng,
        [
            "Which accounts have either an urgent ticket or an overdue invoice (or both)? Return distinct account_id.",
            "Give the union of account_ids that have an urgent-priority ticket and those that have an overdue invoice.",
        ],
    )
    return q, (
        "SELECT account_id FROM support_tickets WHERE priority = 'urgent' UNION "
        "SELECT account_id FROM invoices WHERE status = 'overdue'"
    )


@_register("set_ops", ("support_tickets", "invoices"), "hard")
def high_and_void(rng: random.Random) -> tuple[str, str]:
    p = _pick(rng, ("high", "normal", "urgent"))
    s = _pick(rng, ("void", "open"))
    q = _pick(
        rng,
        [
            f"Which accounts have both a {p}-priority ticket and a {s} invoice? Return account_id.",
            f"List account_ids appearing in both {p}-priority tickets and {s} invoices.",
        ],
    )
    return q, (
        f"SELECT account_id FROM support_tickets WHERE priority = '{p}' INTERSECT "
        f"SELECT account_id FROM invoices WHERE status = '{s}'"
    )


# ==== top_n (ordered) =========================================================================


@_register("top_n", ("payments", "invoices", "accounts"), "hard")
def top_accounts_paid(rng: random.Random) -> tuple[str, str]:
    k = rng.randint(3, 8)
    q = _pick(
        rng,
        [
            f"Top {k} accounts by total succeeded payment cents, highest first (ties by lower account_id). Return account name and total.",
            f"Which {k} accounts have paid the most in succeeded payments? Order by total descending, then account_id ascending; show name and total cents.",
        ],
    )
    return q, (
        "SELECT a.name, SUM(p.amount_cents) AS total FROM payments p "
        "JOIN invoices i ON i.invoice_id = p.invoice_id JOIN accounts a ON a.account_id = i.account_id "
        "WHERE p.status = 'succeeded' GROUP BY a.account_id "
        f"ORDER BY total DESC, a.account_id ASC LIMIT {k}"
    )


@_register("top_n", ("support_tickets",), "easy")
def latest_tickets(rng: random.Random) -> tuple[str, str]:
    k, c = rng.randint(3, 10), _pick(rng, CATEGORIES)
    q = _pick(
        rng,
        [
            f"Show the {k} most recently opened {c} tickets, newest first (ties by higher ticket_id). Return ticket_id and opened_at.",
            f"Return ticket_id and opened_at for the latest {k} tickets in the {c} category, ordered from newest to oldest, ties broken by larger ticket_id.",
        ],
    )
    return q, (
        f"SELECT ticket_id, opened_at FROM support_tickets WHERE category = '{c}' "
        f"ORDER BY opened_at DESC, ticket_id DESC LIMIT {k}"
    )


@_register("top_n", ("feature_usage",), "medium")
def top_users_usage(rng: random.Random) -> tuple[str, str]:
    k, f = rng.randint(3, 8), _pick(rng, _FEATURES)
    q = _pick(
        rng,
        [
            f"Which {k} users have the most total '{f}' events? Order by total events descending then user_id ascending; return user_id and total.",
            f"Top {k} users by summed event_count for the {f} feature (ties by smaller user_id), highest first. Return user_id, total events.",
        ],
    )
    return q, (
        "SELECT user_id, SUM(event_count) AS total FROM feature_usage "
        f"WHERE feature = '{f}' GROUP BY user_id ORDER BY total DESC, user_id ASC LIMIT {k}"
    )


# ==== high-parameter-space medium/hard templates ================================================

_SPAN_LO, _SPAN_HI = "2023-01-01", "2025-11-30"


def _span(rng: random.Random) -> tuple[str, str]:
    d1 = date.fromisoformat(_date(rng, _SPAN_LO, "2025-06-30"))
    return d1.isoformat(), (d1 + timedelta(days=rng.randint(45, 400))).isoformat()


def _cents(rng: random.Random) -> int:
    return rng.randint(1, 60) * 100


_CURRENCIES = ("USD", "EUR")


@_register("join", ("invoices", "subscriptions", "plans", "accounts"), "hard")
def invoice_plan_account(rng: random.Random) -> tuple[str, str]:
    x, d, c = _cents(rng), _date(rng, "2023-01-01", "2025-09-30"), _pick(rng, COUNTRIES)
    q = _pick(
        rng,
        [
            f"For subscription invoices over ${x / 100:.0f} issued after {d} to accounts in {c}, list invoice_id, account name and the plan tier.",
            f"Show invoice_id, account name and plan tier for invoices above ${x / 100:.0f} (issued after {d}) belonging to {c} accounts and tied to a subscription.",
        ],
    )
    return q, (
        "SELECT i.invoice_id, a.name, p.tier FROM invoices i "
        "JOIN subscriptions s ON s.subscription_id = i.subscription_id "
        "JOIN plans p ON p.plan_id = s.plan_id JOIN accounts a ON a.account_id = i.account_id "
        f"WHERE i.amount_cents > {x} AND i.issued_date > '{d}' AND a.country = '{c}'"
    )


@_register("join", ("support_tickets", "users", "accounts"), "medium")
def ticket_user_account(rng: random.Random) -> tuple[str, str]:
    p, i, d = _pick(rng, PRIORITIES), _pick(rng, INDUSTRIES), _date(rng, "2022-06-01", "2025-09-30")
    q = _pick(
        rng,
        [
            f"List ticket_id, the submitting user's full name and the account name for {p}-priority tickets opened after {d} by {i} accounts.",
            f"Which {p} tickets from {i}-industry accounts were opened after {d}? Return ticket_id, user full_name, account name.",
        ],
    )
    return q, (
        "SELECT t.ticket_id, u.full_name, a.name FROM support_tickets t "
        "JOIN users u ON u.user_id = t.user_id JOIN accounts a ON a.account_id = t.account_id "
        f"WHERE t.priority = '{p}' AND a.industry = '{i}' AND t.opened_at > '{d}'"
    )


@_register("join", ("payments", "invoices", "accounts"), "medium")
def payment_method_accounts(rng: random.Random) -> tuple[str, str]:
    m, s, x = _pick(rng, _PAY_METHOD), _pick(rng, ("succeeded", "failed", "refunded")), _cents(rng)
    q = _pick(
        rng,
        [
            f"List payment_id and account name for {s} {m} payments larger than ${x / 100:.0f}.",
            f"Which {m} payments with status '{s}' exceed ${x / 100:.0f}? Return payment_id and the account's name.",
        ],
    )
    return q, (
        "SELECT p.payment_id, a.name FROM payments p JOIN invoices i ON i.invoice_id = p.invoice_id "
        f"JOIN accounts a ON a.account_id = i.account_id WHERE p.method = '{m}' "
        f"AND p.status = '{s}' AND p.amount_cents > {x}"
    )


@_register("join", ("accounts",), "hard")
def child_parent_accounts(rng: random.Random) -> tuple[str, str]:
    c = _pick(rng, COUNTRIES)
    y = rng.choice((2022, 2023, 2024, 2025))
    q = _pick(
        rng,
        [
            f"List each child account created in or after {y} next to the name of its parent account, for parents located in {c}.",
            f"For sub-accounts (parent_account_id set) created {y} or later whose parent is in {c}, show child name and parent name.",
        ],
    )
    return q, (
        "SELECT ch.name, pa.name FROM accounts ch JOIN accounts pa ON pa.account_id = ch.parent_account_id "
        f"WHERE pa.country = '{c}' AND ch.created_at >= '{y}-01-01'"
    )


@_register("join", ("feature_usage", "users", "accounts"), "medium")
def feature_users_accounts(rng: random.Random) -> tuple[str, str]:
    f, n, i = _pick(rng, _FEATURES), rng.randint(50, 480), _pick(rng, INDUSTRIES)
    q = _pick(
        rng,
        [
            f"Which {i} accounts have users with a single '{f}' usage record of more than {n} events? Return distinct account names.",
            f"List distinct names of {i}-industry accounts where some user logged over {n} '{f}' events in one usage row.",
        ],
    )
    return q, (
        "SELECT DISTINCT a.name FROM feature_usage f JOIN users u ON u.user_id = f.user_id "
        f"JOIN accounts a ON a.account_id = u.account_id WHERE f.feature = '{f}' "
        f"AND f.event_count > {n} AND a.industry = '{i}'"
    )


@_register("aggregation", ("payments", "invoices", "subscriptions", "plans"), "hard")
def revenue_by_tier(rng: random.Random) -> tuple[str, str]:
    d1, d2 = _span(rng)
    cur = _pick(rng, _CURRENCIES)
    q = _pick(
        rng,
        [
            f"Total succeeded payment cents per plan tier for {cur} invoices paid between {d1} and {d2} (inclusive).",
            f"For {cur} invoices, sum succeeded payments made from {d1} through {d2}, grouped by the tier of the invoice's plan.",
        ],
    )
    return q, (
        "SELECT pl.tier, SUM(p.amount_cents) FROM payments p JOIN invoices i ON i.invoice_id = p.invoice_id "
        "JOIN subscriptions s ON s.subscription_id = i.subscription_id "
        "JOIN plans pl ON pl.plan_id = s.plan_id WHERE p.status = 'succeeded' "
        f"AND i.currency = '{cur}' AND p.paid_at BETWEEN '{d1}' AND '{d2}' GROUP BY pl.tier"
    )


@_register("aggregation", ("support_tickets", "accounts"), "medium")
def accounts_many_tickets(rng: random.Random) -> tuple[str, str]:
    n, p, c = rng.randint(0, 2), _pick(rng, PRIORITIES), _pick(rng, CATEGORIES)
    q = _pick(
        rng,
        [
            f"Which accounts have more than {n} {p}-priority {c} tickets? Return account name and the ticket count.",
            f"List account names with over {n} tickets of priority '{p}' in category '{c}', plus the count.",
        ],
    )
    return q, (
        "SELECT a.name, COUNT(*) FROM support_tickets t JOIN accounts a ON a.account_id = t.account_id "
        f"WHERE t.priority = '{p}' AND t.category = '{c}' GROUP BY a.account_id HAVING COUNT(*) > {n}"
    )


@_register("aggregation", ("feature_usage", "users", "accounts"), "medium")
def usage_by_feature(rng: random.Random) -> tuple[str, str]:
    r, i, d1 = _pick(rng, _ROLES), _pick(rng, INDUSTRIES), _date(rng, "2025-01-01", "2025-11-01")
    q = _pick(
        rng,
        [
            f"Total event_count per feature for {r} users of {i} accounts since {d1}.",
            f"For users with role {r} in the {i} industry, sum feature usage events per feature for usage on or after {d1}.",
        ],
    )
    return q, (
        "SELECT f.feature, SUM(f.event_count) FROM feature_usage f JOIN users u ON u.user_id = f.user_id "
        "JOIN accounts a ON a.account_id = u.account_id "
        f"WHERE u.role = '{r}' AND a.industry = '{i}' AND f.used_on >= '{d1}' GROUP BY f.feature"
    )


@_register("aggregation", ("invoices", "accounts"), "medium")
def avg_invoice_by_country(rng: random.Random) -> tuple[str, str]:
    cur, s, n = _pick(rng, _CURRENCIES), _pick(rng, _INV_STATUS), rng.randint(1, 6)
    q = _pick(
        rng,
        [
            f"Average {s} {cur} invoice amount in cents per account country (ignore missing amounts), only countries with at least {n} such invoices; round to 2 decimals.",
            f"For {s} invoices in {cur}, give each country's mean amount_cents rounded to 2 decimals, keeping countries with {n} or more of them.",
        ],
    )
    return q, (
        "SELECT a.country, ROUND(AVG(i.amount_cents), 2) FROM invoices i "
        "JOIN accounts a ON a.account_id = i.account_id "
        f"WHERE i.status = '{s}' AND i.currency = '{cur}' AND i.amount_cents IS NOT NULL "
        f"GROUP BY a.country HAVING COUNT(*) >= {n}"
    )


@_register("window", ("support_tickets",), "hard")
def first_ticket_per_account(rng: random.Random) -> tuple[str, str]:
    c, d, p = _pick(rng, CATEGORIES), _date(rng, "2022-06-01", "2025-06-30"), _pick(rng, PRIORITIES)
    q = _pick(
        rng,
        [
            f"For each account, find its earliest {c} ticket opened after {d} with priority {p} (ties: lower ticket_id). Return account_id, ticket_id and opened_at.",
            f"Return the first {p}-priority {c} ticket per account among tickets opened after {d} (break ties with the smaller ticket_id): account_id, ticket_id, opened_at.",
        ],
    )
    return q, (
        "SELECT account_id, ticket_id, opened_at FROM (SELECT account_id, ticket_id, opened_at, "
        "ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY opened_at, ticket_id) AS rn "
        f"FROM support_tickets WHERE category = '{c}' AND priority = '{p}' AND opened_at > '{d}') WHERE rn = 1"
    )


@_register("window", ("payments", "invoices"), "hard")
def payment_increase_over_previous(rng: random.Random) -> tuple[str, str]:
    m, cur, x = _pick(rng, _PAY_METHOD), _pick(rng, _CURRENCIES), rng.randint(0, 60) * 100
    q = _pick(
        rng,
        [
            f"Looking at succeeded {m} payments on {cur} invoices in chronological order per account (ties by payment_id), which payments exceed the account's previous such payment by more than {x} cents? Return account_id, payment_id and the increase.",
            f"For each account's succeeded {cur} {m} payments ordered by paid_at then payment_id, return account_id, payment_id and increase over the prior payment where the increase is above {x} cents.",
        ],
    )
    return q, (
        "SELECT account_id, payment_id, amount_cents - prev FROM (SELECT i.account_id, p.payment_id, "
        "p.amount_cents, LAG(p.amount_cents) OVER (PARTITION BY i.account_id ORDER BY p.paid_at, p.payment_id) AS prev "
        "FROM payments p JOIN invoices i ON i.invoice_id = p.invoice_id "
        f"WHERE p.status = 'succeeded' AND p.method = '{m}' AND i.currency = '{cur}') "
        f"WHERE prev IS NOT NULL AND amount_cents - prev > {x}"
    )


@_register("window", ("accounts", "users"), "hard")
def ntile_accounts_by_users(rng: random.Random) -> tuple[str, str]:
    k, st, i = rng.randint(2, 5), _pick(rng, _STATUS), rng.choice(INDUSTRIES)
    q = _pick(
        rng,
        [
            f"Among {st} {i} accounts, split accounts within each country into {k} equal-sized buckets by user count (most users first, ties by account_id) using NTILE; return account_id and country for those in bucket 1.",
            f"For {i} accounts with status '{st}', use NTILE({k}) per country ordered by number of users descending then account_id, and list account_id, country of bucket 1.",
        ],
    )
    return q, (
        "SELECT account_id, country FROM (SELECT a.account_id, a.country, NTILE("
        f"{k}) OVER (PARTITION BY a.country ORDER BY COUNT(u.user_id) DESC, a.account_id) AS b "
        "FROM accounts a LEFT JOIN users u ON u.account_id = a.account_id "
        f"WHERE a.status = '{st}' AND a.industry = '{i}' GROUP BY a.account_id) WHERE b = 1"
    )


@_register("window", ("feature_usage",), "hard")
def running_usage_for_user(rng: random.Random) -> tuple[str, str]:
    u = rng.randint(1, 739)
    q = _pick(
        rng,
        [
            f"For user {u}, list each usage record (usage_id, used_on, event_count) with the running total of events, ordered by used_on then usage_id.",
            f"Show user_id {u}'s feature usage in date order (ties by usage_id) with usage_id, used_on, event_count and cumulative event_count.",
        ],
    )
    return q, (
        "SELECT usage_id, used_on, event_count, SUM(event_count) OVER (ORDER BY used_on, usage_id) "
        f"FROM feature_usage WHERE user_id = {u} ORDER BY used_on, usage_id"
    )


@_register("window", ("accounts", "users"), "hard")
def rank_accounts_in_country(rng: random.Random) -> tuple[str, str]:
    c, k = _pick(rng, COUNTRIES), rng.randint(1, 4)
    r = rng.choice(("all", "non-deleted"))
    cond = "" if r == "all" else " AND u.deleted_at IS NULL"
    q = _pick(
        rng,
        [
            f"In {c}, rank accounts by their number of {r} users with DENSE_RANK (most first) and return account_id, user count and rank for ranks up to {k}.",
            f"Using DENSE_RANK over {r} user counts (descending) among accounts in {c}, list account_id, count and rank where rank <= {k}.",
        ],
    )
    return q, (
        "SELECT account_id, n, r FROM (SELECT a.account_id, COUNT(u.user_id) AS n, "
        "DENSE_RANK() OVER (ORDER BY COUNT(u.user_id) DESC) AS r FROM accounts a "
        f"LEFT JOIN users u ON u.account_id = a.account_id{cond} "
        f"WHERE a.country = '{c}' GROUP BY a.account_id) WHERE r <= {k}"
    )


@_register("subquery_cte", ("accounts", "users"), "hard")
def accounts_above_industry_avg_users(rng: random.Random) -> tuple[str, str]:
    st, y = _pick(rng, _STATUS), rng.choice((2022, 2023, 2024, 2025))
    c = _pick(rng, COUNTRIES)
    q = _pick(
        rng,
        [
            f"Which {st} accounts in {c} created in or before {y} have more users than the average user count of accounts in their industry (over all accounts in that industry)? Return account_id and user count.",
            f"List account_id and user count for {st} accounts (created up to {y}) whose number of users is above their industry's average per-account user count.",
        ],
    )
    return q, (
        "WITH c AS (SELECT a.account_id, a.industry, a.status, a.country, a.created_at, COUNT(u.user_id) AS n "
        "FROM accounts a LEFT JOIN users u ON u.account_id = a.account_id GROUP BY a.account_id) "
        f"SELECT account_id, n FROM c WHERE status = '{st}' AND country = '{c}' AND created_at < '{y + 1}-01-01' "
        "AND n > (SELECT AVG(n) FROM c c2 WHERE c2.industry = c.industry)"
    )


@_register("subquery_cte", ("invoices",), "hard")
def invoices_above_account_avg(rng: random.Random) -> tuple[str, str]:
    cur, s = _pick(rng, _CURRENCIES), _pick(rng, _INV_STATUS)
    d1, d2 = _span(rng)
    q = _pick(
        rng,
        [
            f"Which {s} {cur} invoices issued between {d1} and {d2} are larger than the average amount of all {cur} invoices of the same account (NULL amounts ignored)? Return invoice_id.",
            f"List invoice_ids of {s} invoices in {cur}, issued {d1} to {d2}, whose amount_cents exceeds the average amount_cents over that account's {cur} invoices.",
        ],
    )
    return q, (
        "SELECT i.invoice_id FROM invoices i WHERE i.status = "
        f"'{s}' AND i.currency = '{cur}' AND i.issued_date BETWEEN '{d1}' AND '{d2}' "
        "AND i.amount_cents > (SELECT AVG(j.amount_cents) FROM invoices j "
        f"WHERE j.account_id = i.account_id AND j.currency = '{cur}')"
    )


@_register("subquery_cte", ("users", "support_tickets"), "medium")
def users_in_ticket_heavy_accounts(rng: random.Random) -> tuple[str, str]:
    n, p, r = rng.randint(1, 4), _pick(rng, PRIORITIES), _pick(rng, _ROLES)
    q = _pick(
        rng,
        [
            f"List emails of {r} users whose account has at least {n} {p}-priority tickets.",
            f"Which users with role {r} belong to accounts that raised {n} or more tickets of priority {p}? Return email.",
        ],
    )
    return q, (
        f"SELECT email FROM users WHERE role = '{r}' AND account_id IN (SELECT account_id FROM "
        f"support_tickets WHERE priority = '{p}' GROUP BY account_id HAVING COUNT(*) >= {n})"
    )


@_register("subquery_cte", ("invoices",), "hard")
def busy_months(rng: random.Random) -> tuple[str, str]:
    y, cur, t = rng.choice((2023, 2024, 2025)), _pick(rng, _CURRENCIES), rng.randint(0, 3)
    s = _pick(rng, _INV_STATUS)
    q = _pick(
        rng,
        [
            f"In {y}, which months (YYYY-MM) had more than {t} above the average monthly number of {s} {cur} invoices (average over months that have any)? Return month and count.",
            f"Return month and count of {s} {cur} invoices in {y} for months whose count exceeds the average monthly count for that year by more than {t}.",
        ],
    )
    return q, (
        "WITH m AS (SELECT strftime('%Y-%m', issued_date) AS month, COUNT(*) AS n FROM invoices "
        f"WHERE status = '{s}' AND currency = '{cur}' AND strftime('%Y', issued_date) = '{y}' GROUP BY month) "
        f"SELECT month, n FROM m WHERE n > (SELECT AVG(n) FROM m) + {t}"
    )


@_register("subquery_cte", ("accounts", "subscriptions"), "hard")
def accounts_latest_sub_plan(rng: random.Random) -> tuple[str, str]:
    pid, st, c = rng.randint(1, 6), _pick(rng, ("active", "cancelled", "past_due", "trialing")), _pick(rng, COUNTRIES)
    q = _pick(
        rng,
        [
            f"Which {c} accounts have a most recent subscription (latest start_date) on plan_id {pid} with status '{st}'? Return account_id.",
            f"List account_ids in {c} whose newest subscription by start_date is on plan {pid} and has status {st}.",
        ],
    )
    return q, (
        "SELECT a.account_id FROM accounts a JOIN subscriptions s ON s.account_id = a.account_id "
        f"WHERE a.country = '{c}' AND s.plan_id = {pid} AND s.status = '{st}' AND s.start_date = "
        "(SELECT MAX(s2.start_date) FROM subscriptions s2 WHERE s2.account_id = a.account_id)"
    )


@_register("null_handling", ("subscriptions", "plans", "accounts"), "hard")
def effective_revenue(rng: random.Random) -> tuple[str, str]:
    st, c, y = _pick(rng, ("active", "cancelled", "past_due", "trialing")), _pick(rng, COUNTRIES), rng.choice((2022, 2023, 2024, 2025))
    q = _pick(
        rng,
        [
            f"Estimated monthly revenue in cents per plan name: seats * monthly price * (1 - discount/100), treating a missing discount as 0, for {st} subscriptions of {c} accounts started in or after {y}. Round each total to 2 decimals.",
            f"Per plan name, sum seats * monthly_price_cents * (1 - COALESCE(discount_pct,0)/100.0) over {st} subscriptions started {y} or later by {c} accounts; round to 2 decimals.",
        ],
    )
    return q, (
        "SELECT p.name, ROUND(SUM(s.seats * p.monthly_price_cents * (1 - COALESCE(s.discount_pct, 0) / 100.0)), 2) "
        "FROM subscriptions s JOIN plans p ON p.plan_id = s.plan_id JOIN accounts a ON a.account_id = s.account_id "
        f"WHERE s.status = '{st}' AND a.country = '{c}' AND s.start_date >= '{y}-01-01' GROUP BY p.name"
    )


@_register("null_handling", ("invoices",), "medium")
def accounts_missing_amounts(rng: random.Random) -> tuple[str, str]:
    n, s, cur = rng.randint(1, 3), _pick(rng, _INV_STATUS), _pick(rng, _CURRENCIES)
    y = rng.choice((2023, 2024, 2025))
    q = _pick(
        rng,
        [
            f"Which accounts have at least {n} {s} {cur} invoices from {y} with a missing amount? Return account_id and how many.",
            f"Return account_id and the number of NULL-amount {s} {cur} invoices issued in {y}, for accounts with {n} or more.",
        ],
    )
    return q, (
        "SELECT account_id, COUNT(*) FROM invoices WHERE amount_cents IS NULL "
        f"AND status = '{s}' AND currency = '{cur}' AND strftime('%Y', issued_date) = '{y}' "
        f"GROUP BY account_id HAVING COUNT(*) >= {n}"
    )


@_register("null_handling", ("support_tickets", "accounts"), "medium")
def accounts_rated_tickets(rng: random.Random) -> tuple[str, str]:
    n, c, p = rng.randint(1, 2), _pick(rng, CATEGORIES), _pick(rng, PRIORITIES)
    q = _pick(
        rng,
        [
            f"Average CSAT (rounded to 2 decimals) per account name for {p}-priority {c} tickets, only accounts with at least {n} rated tickets (missing scores don't count).",
            f"For {p} {c} tickets, give each account's average csat_score rounded to 2 decimals if it has {n} or more non-NULL scores.",
        ],
    )
    return q, (
        "SELECT a.name, ROUND(AVG(t.csat_score), 2) FROM support_tickets t "
        "JOIN accounts a ON a.account_id = t.account_id "
        f"WHERE t.priority = '{p}' AND t.category = '{c}' GROUP BY a.account_id "
        f"HAVING COUNT(t.csat_score) >= {n}"
    )


@_register("null_handling", ("invoices", "subscriptions"), "medium")
def invoices_low_or_no_discount(rng: random.Random) -> tuple[str, str]:
    x, cur, s = rng.choice((6, 11, 16, 21)), _pick(rng, _CURRENCIES), _pick(rng, _INV_STATUS)
    y = rng.choice((2023, 2024, 2025))
    q = _pick(
        rng,
        [
            f"List {s} {cur} invoices from {y} whose subscription has no discount or a discount below {x}%. Return invoice_id.",
            f"Which {s} invoices in {cur} issued in {y} belong to subscriptions with a NULL discount or discount_pct under {x}? Return invoice_id.",
        ],
    )
    return q, (
        "SELECT i.invoice_id FROM invoices i JOIN subscriptions s ON s.subscription_id = i.subscription_id "
        f"WHERE i.status = '{s}' AND i.currency = '{cur}' AND strftime('%Y', i.issued_date) = '{y}' "
        f"AND (s.discount_pct IS NULL OR s.discount_pct < {x})"
    )


@_register("anti_join", ("accounts", "payments", "invoices"), "hard")
def accounts_no_payment_in_range(rng: random.Random) -> tuple[str, str]:
    d1, d2 = _span(rng)
    i, st = _pick(rng, INDUSTRIES), _pick(rng, _STATUS)
    q = _pick(
        rng,
        [
            f"Which {st} {i} accounts made no succeeded payment between {d1} and {d2}? Return account_id.",
            f"List account_ids of {i}-industry accounts with status '{st}' that have no succeeded payment dated {d1} to {d2}.",
        ],
    )
    return q, (
        f"SELECT a.account_id FROM accounts a WHERE a.industry = '{i}' AND a.status = '{st}' "
        "AND NOT EXISTS (SELECT 1 FROM payments p JOIN invoices i ON i.invoice_id = p.invoice_id "
        "WHERE i.account_id = a.account_id AND p.status = 'succeeded' "
        f"AND p.paid_at BETWEEN '{d1}' AND '{d2}')"
    )


@_register("anti_join", ("users", "support_tickets", "accounts"), "medium")
def users_never_ticketed(rng: random.Random) -> tuple[str, str]:
    r, c = _pick(rng, _ROLES), _pick(rng, COUNTRIES)
    live = rng.random() < 0.5
    cond = " AND u.deleted_at IS NULL" if live else ""
    q = _pick(
        rng,
        [
            f"Which {'non-deleted ' if live else ''}{r} users in {c} accounts never submitted a support ticket? Return user_id.",
            f"List user_ids of {r} users{' who are not deleted' if live else ''} from {c} accounts that do not appear as the submitter of any ticket.",
        ],
    )
    return q, (
        "SELECT u.user_id FROM users u JOIN accounts a ON a.account_id = u.account_id "
        f"WHERE u.role = '{r}' AND a.country = '{c}'{cond} AND NOT EXISTS "
        "(SELECT 1 FROM support_tickets t WHERE t.user_id = u.user_id)"
    )


@_register("anti_join", ("accounts", "feature_usage", "users"), "hard")
def uses_one_not_other(rng: random.Random) -> tuple[str, str]:
    f1, f2 = rng.sample(_FEATURES, 2)
    i = _pick(rng, INDUSTRIES)
    q = _pick(
        rng,
        [
            f"Which {i} accounts have users who used '{f1}' but no user who ever used '{f2}'? Return account_id.",
            f"List account_ids in the {i} industry with some '{f1}' usage among their users and no '{f2}' usage at all.",
        ],
    )
    return q, (
        f"SELECT DISTINCT a.account_id FROM accounts a JOIN users u ON u.account_id = a.account_id "
        f"JOIN feature_usage f ON f.user_id = u.user_id WHERE a.industry = '{i}' AND f.feature = '{f1}' "
        "AND NOT EXISTS (SELECT 1 FROM users u2 JOIN feature_usage f2 ON f2.user_id = u2.user_id "
        f"WHERE u2.account_id = a.account_id AND f2.feature = '{f2}')"
    )


@_register("anti_join", ("invoices", "payments"), "medium")
def invoices_never_paid_month(rng: random.Random) -> tuple[str, str]:
    y, mth, s = rng.choice((2023, 2024, 2025)), rng.randint(1, 12), _pick(rng, _INV_STATUS)
    q = _pick(
        rng,
        [
            f"Which {s} invoices issued in {_MONTHS[mth - 1]} {y} have no payment record of any kind? Return invoice_id.",
            f"List invoice_ids with status '{s}' issued during {_MONTHS[mth - 1]} {y} that never appear in payments.",
        ],
    )
    return q, (
        "SELECT i.invoice_id FROM invoices i LEFT JOIN payments p ON p.invoice_id = i.invoice_id "
        f"WHERE i.status = '{s}' AND strftime('%Y-%m', i.issued_date) = '{y}-{mth:02d}' AND p.payment_id IS NULL"
    )


@_register("set_ops", ("support_tickets", "invoices"), "hard")
def ticket_union_invoice(rng: random.Random) -> tuple[str, str]:
    p, s, cur = _pick(rng, PRIORITIES), _pick(rng, _INV_STATUS), _pick(rng, _CURRENCIES)
    y = rng.choice((2023, 2024, 2025))
    q = _pick(
        rng,
        [
            f"Which accounts either opened a {p}-priority ticket in {y} or received a {s} {cur} invoice in {y}? Return distinct account_id.",
            f"Union of account_ids with a {p} ticket opened in {y} and account_ids with a {s} {cur} invoice issued in {y}.",
        ],
    )
    return q, (
        f"SELECT account_id FROM support_tickets WHERE priority = '{p}' AND strftime('%Y', opened_at) = '{y}' "
        f"UNION SELECT account_id FROM invoices WHERE status = '{s}' AND currency = '{cur}' "
        f"AND strftime('%Y', issued_date) = '{y}'"
    )


@_register("set_ops", ("subscriptions", "invoices"), "hard")
def plan_except_void(rng: random.Random) -> tuple[str, str]:
    pid, s, cur = rng.randint(1, 6), _pick(rng, _INV_STATUS), _pick(rng, _CURRENCIES)
    st = _pick(rng, ("active", "cancelled", "past_due", "trialing"))
    q = _pick(
        rng,
        [
            f"Accounts with a {st} subscription on plan {pid} that do NOT have any {s} {cur} invoice. Return account_id.",
            f"Which account_ids have a subscription on plan_id {pid} with status {st} but no {s} invoice in {cur}?",
        ],
    )
    return q, (
        f"SELECT account_id FROM subscriptions WHERE plan_id = {pid} AND status = '{st}' EXCEPT "
        f"SELECT account_id FROM invoices WHERE status = '{s}' AND currency = '{cur}'"
    )


@_register("set_ops", ("feature_usage", "users"), "hard")
def users_both_features(rng: random.Random) -> tuple[str, str]:
    f1, f2 = rng.sample(_FEATURES, 2)
    n, r = rng.randint(1, 250), _pick(rng, _ROLES)
    q = _pick(
        rng,
        [
            f"Which {r} users have at least one '{f1}' record with over {n} events AND at least one '{f2}' record? Return user_id.",
            f"List user_ids of {r} users appearing both in '{f1}' usage with event_count above {n} and in any '{f2}' usage.",
        ],
    )
    return q, (
        f"SELECT f.user_id FROM feature_usage f JOIN users u ON u.user_id = f.user_id WHERE u.role = '{r}' "
        f"AND f.feature = '{f1}' AND f.event_count > {n} INTERSECT "
        f"SELECT user_id FROM feature_usage WHERE feature = '{f2}'"
    )


@_register("conditional", ("subscriptions", "plans"), "medium")
def seat_size_labels(rng: random.Random) -> tuple[str, str]:
    a = rng.randint(3, 20)
    b = a + rng.randint(5, 60)
    st = _pick(rng, ("active", "cancelled", "past_due", "trialing"))
    q = _pick(
        rng,
        [
            f"For {st} subscriptions, label seat counts 'small' (up to {a}), 'mid' (more than {a} up to {b}) or 'large' (over {b}) and count subscriptions per plan tier and label.",
            f"Count {st} subscriptions per plan tier and size label, where size is small if seats <= {a}, mid if seats <= {b}, else large.",
        ],
    )
    return q, (
        "SELECT p.tier, CASE WHEN s.seats <= "
        f"{a} THEN 'small' WHEN s.seats <= {b} THEN 'mid' ELSE 'large' END AS size, COUNT(*) "
        "FROM subscriptions s JOIN plans p ON p.plan_id = s.plan_id "
        f"WHERE s.status = '{st}' GROUP BY p.tier, size"
    )


@_register("conditional", ("payments", "invoices"), "hard")
def payment_status_pivot(rng: random.Random) -> tuple[str, str]:
    y, cur, x = rng.choice((2023, 2024, 2025)), _pick(rng, _CURRENCIES), rng.randint(0, 50) * 100
    q = _pick(
        rng,
        [
            f"For {cur} invoices and payments made in {y} over {x} cents, give per method the number of succeeded, failed and refunded payments.",
            f"Per payment method: counts of succeeded, failed and refunded payments in {y} with amount above {x} cents on {cur} invoices.",
        ],
    )
    return q, (
        "SELECT p.method, SUM(p.status = 'succeeded'), SUM(p.status = 'failed'), SUM(p.status = 'refunded') "
        "FROM payments p JOIN invoices i ON i.invoice_id = p.invoice_id "
        f"WHERE i.currency = '{cur}' AND strftime('%Y', p.paid_at) = '{y}' AND p.amount_cents > {x} "
        "GROUP BY p.method"
    )


@_register("date_math", ("payments", "invoices"), "medium")
def avg_days_to_pay(rng: random.Random) -> tuple[str, str]:
    y, cur, x = rng.choice((2023, 2024, 2025)), _pick(rng, _CURRENCIES), rng.randint(0, 30) * 100
    q = _pick(
        rng,
        [
            f"Average days between issued_date and paid_at per payment method for succeeded {cur} payments of invoices issued in {y} with amount over {x} cents; round to 2 decimals.",
            f"Per method, mean days from invoice issue to payment for succeeded payments (invoices in {cur}, issued {y}, payment amount > {x}); round to 2 decimals.",
        ],
    )
    return q, (
        "SELECT p.method, ROUND(AVG(julianday(p.paid_at) - julianday(i.issued_date)), 2) FROM payments p "
        "JOIN invoices i ON i.invoice_id = p.invoice_id WHERE p.status = 'succeeded' "
        f"AND i.currency = '{cur}' AND strftime('%Y', i.issued_date) = '{y}' AND p.amount_cents > {x} "
        "GROUP BY p.method"
    )


@_register("date_math", ("support_tickets",), "medium")
def slow_tickets(rng: random.Random) -> tuple[str, str]:
    n, c, p = rng.randint(1, 15), _pick(rng, CATEGORIES), _pick(rng, PRIORITIES)
    q = _pick(
        rng,
        [
            f"Which closed {p}-priority {c} tickets took more than {n} days to close? Return ticket_id.",
            f"List ticket_ids of closed tickets (priority {p}, category {c}) where closed_at is more than {n} days after opened_at.",
        ],
    )
    return q, (
        "SELECT ticket_id FROM support_tickets WHERE closed_at IS NOT NULL "
        f"AND priority = '{p}' AND category = '{c}' AND julianday(closed_at) - julianday(opened_at) > {n}"
    )


@_register("date_math", ("accounts", "invoices"), "hard")
def slow_first_invoice(rng: random.Random) -> tuple[str, str]:
    n, c = rng.randint(0, 150), _pick(rng, COUNTRIES)
    q = _pick(
        rng,
        [
            f"For accounts in {c}, which had their first invoice issued more than {n} days after the account was created? Return account_id and the gap in days.",
            f"List account_id and days between created_at and the earliest invoice issued_date for {c} accounts where that gap exceeds {n}.",
        ],
    )
    return q, (
        "SELECT a.account_id, CAST(julianday(MIN(i.issued_date)) - julianday(a.created_at) AS INTEGER) "
        "FROM accounts a JOIN invoices i ON i.account_id = a.account_id "
        f"WHERE a.country = '{c}' GROUP BY a.account_id "
        f"HAVING julianday(MIN(i.issued_date)) - julianday(a.created_at) > {n}"
    )


@_register("date_math", ("payments",), "medium")
def payments_in_window(rng: random.Random) -> tuple[str, str]:
    d1, d2 = _span(rng)
    m = _pick(rng, _PAY_METHOD)
    q = _pick(
        rng,
        [
            f"How many succeeded {m} payments were made from {d1} to {d2} inclusive, and what is their total in cents?",
            f"Count and total cents of succeeded {m} payments with paid_at between {d1} and {d2}.",
        ],
    )
    return q, (
        "SELECT COUNT(*), SUM(amount_cents) FROM payments WHERE status = 'succeeded' "
        f"AND method = '{m}' AND paid_at BETWEEN '{d1}' AND '{d2}'"
    )


@_register("top_n", ("accounts", "users"), "medium")
def top_accounts_by_users(rng: random.Random) -> tuple[str, str]:
    k, c, st = rng.randint(3, 8), _pick(rng, COUNTRIES), _pick(rng, _STATUS)
    q = _pick(
        rng,
        [
            f"The {k} {st} accounts in {c} with the most users, most first (ties by lower account_id). Return account_id and user count.",
            f"Top {k} accounts from {c} with status '{st}' by number of users; order by user count descending, then account_id ascending; show account_id and count.",
        ],
    )
    return q, (
        "SELECT a.account_id, COUNT(u.user_id) AS n FROM accounts a JOIN users u ON u.account_id = a.account_id "
        f"WHERE a.country = '{c}' AND a.status = '{st}' GROUP BY a.account_id ORDER BY n DESC, a.account_id ASC LIMIT {k}"
    )


@_register("top_n", ("invoices",), "hard")
def largest_invoices(rng: random.Random) -> tuple[str, str]:
    k, s, cur = rng.randint(3, 10), _pick(rng, _INV_STATUS), _pick(rng, _CURRENCIES)
    d1, d2 = _span(rng)
    q = _pick(
        rng,
        [
            f"The {k} largest {s} {cur} invoices issued between {d1} and {d2}, biggest first (ties by higher invoice_id). Return invoice_id and amount_cents.",
            f"Show the top {k} {cur} invoices with status {s} issued {d1} to {d2}, ordered by amount descending then invoice_id descending: invoice_id, amount_cents.",
        ],
    )
    return q, (
        f"SELECT invoice_id, amount_cents FROM invoices WHERE status = '{s}' AND currency = '{cur}' "
        f"AND issued_date BETWEEN '{d1}' AND '{d2}' AND amount_cents IS NOT NULL "
        f"ORDER BY amount_cents DESC, invoice_id DESC LIMIT {k}"
    )


@_register("soft_delete", ("accounts", "users"), "medium")
def accounts_with_deleted_users(rng: random.Random) -> tuple[str, str]:
    n, c = rng.randint(1, 2), _pick(rng, COUNTRIES)
    y = rng.choice((2022, 2023, 2024, 2025))
    q = _pick(
        rng,
        [
            f"Which non-deleted accounts in {c} created in or after {y} have at least {n} soft-deleted users? Return account_id and the number.",
            f"List account_id and count of deleted users for accounts in {c} (deleted_at IS NULL, created {y} or later) having {n} or more deleted users.",
        ],
    )
    return q, (
        "SELECT a.account_id, COUNT(*) FROM accounts a JOIN users u ON u.account_id = a.account_id "
        f"WHERE a.deleted_at IS NULL AND a.country = '{c}' AND a.created_at >= '{y}-01-01' AND u.deleted_at IS NOT NULL "
        f"GROUP BY a.account_id HAVING COUNT(*) >= {n}"
    )


@_register("soft_delete", ("accounts", "users"), "hard")
def last_live_login(rng: random.Random) -> tuple[str, str]:
    i, d = _pick(rng, INDUSTRIES), _date(rng, "2023-01-01", "2025-11-01")
    c = _pick(rng, COUNTRIES)
    q = _pick(
        rng,
        [
            f"For {i} accounts in {c}, the most recent last_login_at among non-deleted users, only where it is after {d}. Return account_id and that timestamp.",
            f"Per {c} {i} account, find the latest login of any non-deleted user and keep accounts whose latest login is later than {d}: account_id, latest login.",
        ],
    )
    return q, (
        "SELECT a.account_id, MAX(u.last_login_at) FROM accounts a JOIN users u ON u.account_id = a.account_id "
        f"WHERE a.industry = '{i}' AND a.country = '{c}' AND u.deleted_at IS NULL GROUP BY a.account_id "
        f"HAVING MAX(u.last_login_at) > '{d}'"
    )


# ==============================================================================================

TEMPLATES: tuple[Template, ...] = tuple(_TEMPLATES)
FAMILIES: tuple[str, ...] = tuple(sorted({t.family for t in TEMPLATES}))


def _task_id(family: str, gold_sql: str) -> str:
    return "sql-" + hashlib.sha256(f"{family}|{gold_sql}".encode()).hexdigest()[:12]


@dataclass
class GenerationReport:
    """Generated tasks plus a count of everything that was dropped, and why."""

    tasks: list[SqlTask]
    dropped_error: int = 0
    dropped_empty: int = 0
    dropped_duplicate: int = 0
    attempts: int = 0
    dropped_by_template: Counter[str] = field(default_factory=Counter)

    @property
    def dropped_total(self) -> int:
        return self.dropped_error + self.dropped_empty + self.dropped_duplicate


def generate_tasks(
    db: DbRef,
    n: int,
    seed: int,
    *,
    families: tuple[str, ...] | None = None,
    max_attempt_factor: int = 40,
) -> GenerationReport:
    """Generate up to ``n`` distinct tasks whose gold SQL executes and returns >= 1 row.

    Templates are cycled round-robin (with randomised parameters) so families are balanced.
    Tasks that error, return an empty result, or repeat an existing gold SQL are dropped and
    counted in the report. Deterministic for (db contents, n, seed, families).
    """
    rng = random.Random(seed)
    pool = [t for t in TEMPLATES if families is None or t.family in families]
    if not pool:
        raise ValueError(f"no templates for families {families!r}")
    report = GenerationReport(tasks=[])
    seen: set[str] = set()
    dup_streak: Counter[str] = Counter()
    i = 0
    while pool and len(report.tasks) < n and report.attempts < n * max_attempt_factor:
        tpl = pool[i % len(pool)]
        i += 1
        report.attempts += 1
        question, sql = tpl.build(rng)
        if sql in seen:
            report.dropped_duplicate += 1
            report.dropped_by_template[tpl.name] += 1
            dup_streak[tpl.name] += 1
            if dup_streak[tpl.name] >= _EXHAUSTED_AFTER:  # parameter space used up
                pool = [t for t in pool if t.name != tpl.name]
            continue
        dup_streak[tpl.name] = 0
        outcome = run_select(db, sql)
        if not outcome.ok:
            report.dropped_error += 1
            report.dropped_by_template[tpl.name] += 1
            continue
        if not outcome.rows:
            report.dropped_empty += 1
            report.dropped_by_template[tpl.name] += 1
            continue
        seen.add(sql)
        report.tasks.append(
            SqlTask(
                task_id=_task_id(tpl.family, sql),
                family=tpl.family,
                question=question,
                gold_sql=sql,
                requires_order=has_top_level_order_by(sql),
                difficulty=tpl.difficulty,
                tables=tuple(sorted(tpl.tables)),
                template=tpl.name,
            )
        )
    return report


# ---- splitting -------------------------------------------------------------------------------

UNSEEN_FAMILY = "unseen_family"
UNSEEN_TABLE_COMBO = "unseen_table_combo"
SEEN = "seen"
MIN_UNSEEN_FRACTION = 0.2
_EXHAUSTED_AFTER = 6


@dataclass(frozen=True)
class SplitResult:
    train: list[SqlTask]
    heldout: list[SqlTask]


def split_by_family(
    tasks: list[SqlTask],
    heldout_families: set[str] | frozenset[str],
    seed: int,
    *,
    in_dist_fraction: float = 0.15,
) -> SplitResult:
    """Split tasks so that whole ``heldout_families`` never appear in train.

    Held-out = every task of the held-out families + a random ``in_dist_fraction`` of the
    remaining tasks. The in-distribution share is capped so that at least 20% of held-out tasks
    come from unseen families. Raises ValueError if no held-out family has tasks. Deterministic.
    """
    rng = random.Random(seed)
    unseen = [t for t in tasks if t.family in heldout_families]
    rest = [t for t in tasks if t.family not in heldout_families]
    if not unseen:
        raise ValueError("no tasks belong to the held-out families")
    n_in = min(round(len(rest) * in_dist_fraction), 4 * len(unseen))
    order = list(range(len(rest)))
    rng.shuffle(order)
    picked = set(order[:n_in])
    train = [t for i, t in enumerate(rest) if i not in picked]
    heldout = unseen + [t for i, t in enumerate(rest) if i in picked]
    return SplitResult(train=train, heldout=heldout)


def classify_heldout(train: list[SqlTask], heldout: list[SqlTask]) -> dict[str, str]:
    """Map each held-out task_id to 'unseen_family', 'unseen_table_combo' or 'seen'."""
    fams = {t.family for t in train}
    combos = {frozenset(t.tables) for t in train}
    out: dict[str, str] = {}
    for t in heldout:
        if t.family not in fams:
            out[t.task_id] = UNSEEN_FAMILY
        elif frozenset(t.tables) not in combos:
            out[t.task_id] = UNSEEN_TABLE_COMBO
        else:
            out[t.task_id] = SEEN
    return out


def unseen_family_ids(train: list[SqlTask], heldout: list[SqlTask]) -> set[str]:
    """task_ids of held-out items whose family never occurs in train."""
    return {k for k, v in classify_heldout(train, heldout).items() if v == UNSEEN_FAMILY}


def unseen_fraction(train: list[SqlTask], heldout: list[SqlTask]) -> float:
    """Share of held-out tasks that are unseen-family or unseen-table-combination."""
    if not heldout:
        return 0.0
    labels = classify_heldout(train, heldout)
    return sum(1 for v in labels.values() if v != SEEN) / len(heldout)
