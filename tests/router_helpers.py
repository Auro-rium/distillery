# ruff: noqa: S101, S311
"""Synthetic N1-like run store for the router data tests (no network, no .distillery access)."""

from __future__ import annotations

import json
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from distillery.prompts import to_training_row
from distillery.store import Store, atomic_write_bytes, sha256_hex
from distillery.taskpacks.base import get_pack
from distillery.taskpacks.sql import schema as sql_schema
from distillery.taskpacks.sql.executor import LocalExecutor

RUN = "r1"
PACK = get_pack("sql")
DB = sql_schema.build_database(0)
DB_SHA = sha256_hex(DB)
DDL = sql_schema.schema_ddl(0)


def item(t: Any, cls: str) -> dict[str, Any]:
    d: dict[str, Any] = t.model_dump(mode="json")
    d["heldout_class"] = cls
    return d


def confirmed_file(
    path: Path, pairs: Sequence[tuple[str, str]], *, db_sha: str = DB_SHA, tag: str = "h"
) -> Path:
    """A ``human_confirmed.json`` with (question, gold_sql) pairs."""
    items = [
        {
            "family": "human", "gold_sql": sql, "heldout_class": "human", "question": q,
            "requires_order": False, "source": "human", "tables": [],
            "task_id": f"{tag}-{i}-{zlib.crc32(q.encode()) % 10**6}", "template": "",
        }
        for i, (q, sql) in enumerate(pairs)
    ]  # fmt: skip
    doc = {
        "format": "distillery-human-confirmed-v1", "items": items, "db_sha256": db_sha,
        "question_file_sha256": "0" * 64, "teacher_model": "TEST", "counts": {"confirmed": len(items)},
    }  # fmt: skip
    atomic_write_bytes(path, json.dumps(doc, sort_keys=True).encode("utf-8"))
    return path


@dataclass
class SynthRun:
    store: Store
    run_id: str
    db_ref: str
    executor: LocalExecutor
    train: list[Any]
    dev: list[Any]
    heldout: list[Any]
    stress: list[Any]
    human_pairs: list[tuple[str, str]]


def make_run(
    root: Path,
    *,
    n_train: int = 150,
    natural_rows: int = 5,
    extra_train_q: Sequence[str] = (),
    extra_dev_q: Sequence[str] = (),
    extra_gate: Sequence[Any] = (),
    extra_stress: Sequence[Any] = (),
) -> SynthRun:
    """A run store shaped like the N1 one: sealed gate/stress/human sets, a ``split`` stage with
    train+dev, and ``round1[-nXXXX]/train.jsonl`` rows (paraphrase-free, plus natural rows)."""
    store = Store(root / "store")
    store.create_run(RUN)
    run_dir = store.run_dir(RUN)
    db_path = run_dir / "db.sqlite"
    atomic_write_bytes(db_path, DB)
    reserved = PACK.stress_families()
    tasks = PACK.generate_tasks(
        str(db_path), n_train + 70, 1, exclude_families=reserved, skeleton_cap=PACK.default_skeleton_cap
    ).tasks  # fmt: skip
    train, dev = tasks[:n_train], tasks[n_train : n_train + 20]
    heldout = tasks[n_train + 20 : n_train + 50]
    stress = PACK.generate_tasks(
        str(db_path), 25, 2, families=reserved, template_share=0.1, family_share=1.0
    ).tasks  # fmt: skip
    store.seal_heldout(RUN, [item(t, "in_distribution") for t in [*heldout, *extra_gate]])
    store.seal_stress(RUN, [item(t, "stress") for t in [*stress, *extra_stress]])
    human_pairs = [
        ("How many plans have unlimited seats?", "SELECT COUNT(*) FROM plans WHERE seat_limit IS NULL"),
        ("Count all accounts please", "SELECT COUNT(*) FROM accounts"),
    ]  # fmt: skip
    store.seal_human(
        RUN,
        [
            {
                "family": "human", "gold_sql": s, "heldout_class": "human", "question": q,
                "requires_order": False, "source": "human", "tables": [],
                "task_id": f"h-{i}-sealed", "template": "",
            }
            for i, (q, s) in enumerate(human_pairs)
        ],
    )  # fmt: skip
    store.get_or_run(
        RUN, "split", {"k": 1},
        lambda: {
            "train": [item(t, "x") for t in train],
            "dev": [item(t, "x") for t in dev]
            + [{"question": q, "task_id": f"dev-extra-{i}"} for i, q in enumerate(extra_dev_q)],
        },
    )  # fmt: skip
    rows = [to_training_row(t, t.gold_sql, schema_ddl=DDL) for t in train]
    natural = [
        to_training_row(
            type("Q", (), {"question": q}),
            "SELECT COUNT(*) FROM users", schema_ddl=DDL,
        )
        for q in [f"natural training question number {i}?" for i in range(natural_rows)]
        + list(extra_train_q)
    ]  # fmt: skip
    for name, rs in (("round1", rows), ("round1-nabcd1234", rows + natural)):
        atomic_write_bytes(
            run_dir / name / "train.jsonl",
            ("\n".join(json.dumps(r, sort_keys=True) for r in rs) + "\n").encode("utf-8"),
        )
    return SynthRun(
        store, RUN, str(db_path), LocalExecutor(), train, dev, heldout, stress, human_pairs
    )
