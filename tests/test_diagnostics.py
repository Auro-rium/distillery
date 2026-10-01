# ruff: noqa: S101
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from distillery import diagnostics as d
from distillery.prompts import build_messages, to_training_row
from distillery.store import Store
from distillery.taskpacks.sql import questions as q
from distillery.taskpacks.sql import schema as s
from distillery.taskpacks.sql.executor import LocalExecutor

TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n"
    "{% endif %}{% endif %}"
)


class _Task:
    def __init__(self, question: str) -> None:
        self.question = question


def test_identical_prompt_comparison() -> None:
    ddl = "CREATE TABLE t(a INT);"
    row = to_training_row(_Task("How many rows?"), "SELECT 1", schema_ddl=ddl)
    cmp = d.compare_prompts(row, ddl)
    assert cmp["byte_identical"] is True and cmp["question"] == "How many rows?"
    assert cmp["train_assistant"] == "SELECT 1"
    assert cmp["train_prompt"] == build_messages("How many rows?", ddl, role="eval_student")


def test_prompt_comparison_detects_a_difference() -> None:
    ddl = "CREATE TABLE t(a INT);"
    row = to_training_row(_Task("How many rows?"), "SELECT 1", schema_ddl=ddl)
    row["messages"][0]["content"] += " "  # a trailing space in the system prompt
    assert d.compare_prompts(row, ddl)["byte_identical"] is False
    # a different schema than the one trained on is also a mismatch
    row2 = to_training_row(_Task("How many rows?"), "SELECT 1", schema_ddl=ddl)
    assert d.compare_prompts(row2, "CREATE TABLE u(b INT);")["byte_identical"] is False


def test_render_all_through_template() -> None:
    pytest.importorskip("jinja2")
    ddl = "CREATE TABLE t(a INT);"
    row = to_training_row(_Task("Q?"), "SELECT 1", schema_ddl=ddl)
    out = d.render_all(TEMPLATE, d.compare_prompts(row, ddl))
    assert out["eval_no_thinking"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert out["eval_thinking"].endswith("<|im_start|>assistant\n")
    assert "<|im_start|>assistant\nSELECT 1<|im_end|>" in out["train_with_assistant"]


def test_render_all_degrades_on_bad_template() -> None:
    pytest.importorskip("jinja2")
    ddl = "CREATE TABLE t(a INT);"
    row = to_training_row(_Task("Q?"), "SELECT 1", schema_ddl=ddl)
    out = d.render_all("{% if %}", d.compare_prompts(row, ddl))
    assert "error" in out


def test_failure_overlap() -> None:
    a = {"failures": [{"task_id": "t1"}, {"task_id": "t2"}, {"task_id": "t3"}]}
    b = {"failures": [{"task_id": "t2"}, {"task_id": "t4"}]}
    ov = d.failure_overlap(a, b)
    assert ov["both"] == ["t2"] and ov["only_a"] == ["t1", "t3"] and ov["only_b"] == ["t4"]
    assert ov["n_a"] == 3 and ov["n_b"] == 2 and ov["n_both"] == 1


def test_stage_artifact_by_name_and_hash(tmp_path: Path) -> None:
    store = Store(tmp_path)
    store.get_or_run("r", "dev_eval_r1", {"x": 1}, lambda: {"failures": [{"task_id": "t1"}]})
    by_name = d.stage_artifact(store, "r", "dev_eval_r1")
    assert d.failed_ids(by_name) == ["t1"]
    digest = store.put_artifact("r", json.dumps(by_name, sort_keys=True).encode())
    assert d.stage_artifact(store, "r", digest) == by_name
    with pytest.raises(LookupError):
        d.stage_artifact(store, "r", "nope")


class _Server:
    def __init__(self, answers: dict[str, str], junk_every: int = 0) -> None:
        self.answers, self.junk_every = answers, junk_every
        self.calls: list[int] = []
        self.closed = False

    def generate(self, messages_batch: Any) -> list[str]:
        self.calls.append(len(messages_batch))
        out = []
        for i, msgs in enumerate(messages_batch):
            sql = next(v for k, v in self.answers.items() if k in msgs[-1]["content"])
            out.append("SELECT 1" if self.junk_every and i % self.junk_every == 0 else sql)
        return [f"```sql\n{o}\n```" for o in out]

    def close(self) -> None:
        self.closed = True


def test_run_student_diagnosis_with_fake_servers(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    s.write_database(0, str(db))
    ddl = s.schema_ddl(0)
    tasks = q.generate_tasks(str(db), 30, 0).tasks[:30]
    train, dev, held = tasks[:10], tasks[10:16], tasks[16:30]
    store = Store(tmp_path / "root")
    dump = lambda ts: [t.model_dump(mode="json") for t in ts]  # noqa: E731
    store.get_or_run("r1", "split", {"a": 1}, lambda: {"train": dump(train), "dev": dump(dev)})
    store.seal_heldout("r1", dump(held))
    rdir = store.run_dir("r1") / "round1"
    rdir.mkdir(parents=True)
    rows = [to_training_row(t, t.gold_sql, schema_ddl=ddl) for t in train]
    (rdir / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    answers = {t.question: t.gold_sql for t in tasks}

    made: list[tuple[tuple[Path, ...], _Server]] = []

    def make_server(files: Any) -> _Server:
        srv = _Server(answers, junk_every=2 if files else 0)
        made.append((tuple(files), srv))
        return srv

    out = d.run_student_diagnosis(
        store, "r1", {"round1": (1, [tmp_path / "adapter_model.safetensors"])},
        make_server=make_server, executor=LocalExecutor(), db_ref=str(db), schema_ddl=ddl,
        out_path=tmp_path / "out.json",
    )  # fmt: skip
    assert len(made) == 2 and all(srv.closed for _f, srv in made)
    assert all(len(srv.calls) == 1 for _f, srv in made)  # each model loaded and called once
    assert out["accuracy"]["base"]["dev"] == 1.0 and out["accuracy"]["base"]["heldout"] == 1.0
    assert out["accuracy"]["round1"]["train"] == 0.5
    assert out["accuracy"]["base"]["train[round1]"] == 1.0
    ident = out["identical_to_base"]["round1"]
    assert ident["train"]["raw"] == 0.5 and ident["dev"]["sql"] == 0.5 and "heldout" in ident
    saved = json.loads((tmp_path / "out.json").read_text())
    item = saved["models"]["round1"]["train"]["items"][0]
    assert {"raw", "sql", "task_id", "correct"} <= set(item)
    assert item["raw"] == "```sql\nSELECT 1\n```" and item["sql"] == "SELECT 1"
