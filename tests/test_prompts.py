# ruff: noqa: S101, E501, S108, S608
from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from distillery.prompts import (
    NO_THINK_MARKER,
    ROLES,
    SYSTEM_PROMPT,
    build_messages,
    extract_sql,
    strip_thinking,
    to_training_row,
)
from distillery.taskpacks.sql.schema import schema_ddl

DDL = schema_ddl(0)
Q = "How many users have the role 'admin'?"


@dataclass
class _T:
    question: str


def test_prefix_identical_across_roles() -> None:
    prefixes = [json.dumps(build_messages(Q, DDL, role=r)) for r in ROLES]
    assert len(set(prefixes)) == 1


def test_train_row_prefix_is_byte_identical_to_eval_request() -> None:
    row = to_training_row(_T(Q), "SELECT COUNT(*) FROM users WHERE role = 'admin'", schema_ddl=DDL)
    msgs = row["messages"]
    eval_msgs = build_messages(Q, DDL, role="eval_student")
    assert msgs[:-1] == eval_msgs
    assert json.dumps(msgs[:-1]).encode() == json.dumps(eval_msgs).encode()
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[-1]["content"] == "SELECT COUNT(*) FROM users WHERE role = 'admin'"


def test_marker_and_content() -> None:
    msgs = build_messages(Q, DDL, role="train")
    assert msgs[0]["content"] == SYSTEM_PROMPT
    assert msgs[1]["content"].endswith(NO_THINK_MARKER)
    assert Q in msgs[1]["content"] and "CREATE TABLE accounts" in msgs[1]["content"]
    assert NO_THINK_MARKER == "/no_think"


def test_unknown_role_rejected() -> None:
    with pytest.raises(ValueError):
        build_messages(Q, DDL, role="nope")  # type: ignore[arg-type]


def test_training_row_is_json_serialisable_and_last_is_assistant() -> None:
    row = to_training_row(_T(Q), "  SELECT 1  \n", schema_ddl=DDL)
    parsed = json.loads(json.dumps(row))
    assert list(parsed) == ["messages"]
    assert parsed["messages"][-1] == {"role": "assistant", "content": "SELECT 1"}
    with pytest.raises(ValueError):
        to_training_row(_T(Q), "  ", schema_ddl=DDL)


def test_ddl_fits_student_context() -> None:
    # Default fine-tune context_length is 8192 tokens; ~4 chars/token is a rough upper bound.
    prompt = "".join(m["content"] for m in build_messages(Q, DDL, role="train"))
    assert len(prompt) / 3 < 4000


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("SELECT 1", "SELECT 1"),
        ("SELECT 1;", "SELECT 1"),
        ("```sql\nSELECT a FROM t;\n```", "SELECT a FROM t"),
        ("```\nSELECT a\nFROM t\n```", "SELECT a\nFROM t"),
        ("Here you go:\n```sql\nSELECT 1\n```\nHope it helps!", "SELECT 1"),
        ("```sql\nSELECT 1\n```\nActually:\n```sql\nSELECT 2\n```", "SELECT 2"),
        ("<think>maybe SELECT 9</think>\nSELECT 3", "SELECT 3"),
        (
            "<think>\nSELECT 9;\n</think>\n\n```sql\nWITH c AS (SELECT 1) SELECT * FROM c\n```",
            "WITH c AS (SELECT 1) SELECT * FROM c",
        ),
        (
            "Sure.\n\nSELECT x FROM y WHERE z = 'a;b';\n\nThis returns x.",
            "SELECT x FROM y WHERE z = 'a;b'",
        ),
        ("Query: SELECT COUNT(*) FROM users", "SELECT COUNT(*) FROM users"),
        ("reasoning...\n</think>\nSELECT 4", "SELECT 4"),
        ("```sql\nSELECT 5\n", "SELECT 5"),
        ("SELECT 1; SELECT 2;", "SELECT 2"),
        ("```sql\n-- comment\nSELECT 6\n```", "SELECT 6"),
        (
            "WITH RECURSIVE r(n) AS (SELECT 1) SELECT n FROM r",
            "WITH RECURSIVE r(n) AS (SELECT 1) SELECT n FROM r",
        ),
    ],
)
def test_extract_sql(raw: str, expected: str) -> None:
    assert extract_sql(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "I cannot answer that.",
        "<think>SELECT 1</think>",
        "<think>SELECT 1",
        "```\nnothing\n```",
    ],
)
def test_extract_sql_none(raw: str) -> None:
    assert extract_sql(raw) is None


def test_extract_roundtrips_training_completion() -> None:
    sql = "SELECT a.name FROM accounts a WHERE a.name = 'x; y'"
    assert extract_sql(sql) == sql


def test_strip_thinking() -> None:
    assert strip_thinking("<think>a</think>b<think>c</think>d").strip() == "bd"


# ---- few-shot (B2)
from distillery.prompts import build_fewshot_messages, select_fewshot_examples  # noqa: E402

_EX = [
    {"question": "How many plans?", "sql": "SELECT COUNT(*) FROM plans;"},
    {"question": "How many users?", "sql": " SELECT COUNT(*) FROM users "},
]


def test_fewshot_zero_examples_is_exactly_zero_shot() -> None:
    assert build_fewshot_messages(Q, DDL, []) == build_messages(Q, DDL, role="eval_teacher")


def test_fewshot_layout_and_final_turn_identical_to_zero_shot() -> None:
    msgs = build_fewshot_messages(Q, DDL, _EX)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert msgs[-1] == build_messages(Q, DDL, role="eval_teacher")[1]
    assert msgs[1] == build_messages(_EX[0]["question"], DDL, role="eval_teacher")[1]
    assert msgs[2]["content"] == "SELECT COUNT(*) FROM plans;" and msgs[4]["content"].endswith(
        "users"
    )


def test_fewshot_is_byte_stable() -> None:
    a = json.dumps(build_fewshot_messages(Q, DDL, _EX), sort_keys=True).encode()
    b = json.dumps(build_fewshot_messages(Q, DDL, [dict(e) for e in _EX]), sort_keys=True).encode()
    assert a == b
    import hashlib

    # golden digest: any change to the few-shot wire format must be a deliberate, visible edit
    assert hashlib.sha256(a).hexdigest() == _FEWSHOT_GOLDEN


def test_fewshot_rejects_empty_sql() -> None:
    with pytest.raises(ValueError):
        build_fewshot_messages(Q, DDL, [{"question": "q", "sql": "  "}])


def _train(n_families: int = 10, per: int = 5) -> list[dict[str, str]]:
    return [
        {
            "task_id": f"f{f}-{i}",
            "family": f"fam{f}",
            "question": f"q{f}-{i}",
            "gold_sql": f"S{f}{i}",
        }
        for f in range(n_families)
        for i in range(per)
    ]


def test_select_examples_deterministic_stratified_order_independent() -> None:
    train = _train()
    a = select_fewshot_examples(train)
    assert a == select_fewshot_examples(train)
    assert a == select_fewshot_examples(list(reversed(train)))
    assert len(a) == 8 and len({e["family"] for e in a}) == 8  # 8 distinct families of 10
    assert {e["task_id"] for e in a} <= {t["task_id"] for t in train}
    assert select_fewshot_examples(train, seed=1) != a


def test_select_examples_wraps_families_when_k_exceeds_them() -> None:
    few = select_fewshot_examples(_train(3, 4), k=8)
    assert len(few) == 8 and len({e["task_id"] for e in few}) == 8
    assert (
        select_fewshot_examples(_train(1, 2), k=8)
        and len(select_fewshot_examples(_train(1, 2), k=8)) == 2
    )
    with pytest.raises(ValueError):
        select_fewshot_examples([])


_FEWSHOT_GOLDEN = "483082e45b85f5937e5b35df50973edd6f4abc3c2b5834cdd159334c47fcb18a"
