# ruff: noqa: S101, S102
"""H3: per-request mean answer log-probability. The pure think-block/EOS rule, the script embedding,
the result types, the fakes and the vectors. All hermetic (the stubbed-sandbox path is in
test_student_sandbox.py, the pipeline path in test_orchestrator.py)."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Any

import pytest

from distillery.config import GateThresholds
from distillery.evaluator import Setup, evaluate, score_model, score_setups
from distillery.prompts import build_messages
from distillery.sandbox_student import (
    GEN_SCRIPT,
    OUT_MARKER,
    answer_logprob_stats,
    parse_output,
    student_output_from_item,
)
from distillery.store import Store
from distillery.student import (
    DetailedStudentServer,
    FakeStudent,
    StudentOutput,
    fake_logprobs,
    generate_detailed,
    text_sha256,
)
from distillery.taskpacks.sql import questions as q
from distillery.taskpacks.sql import schema as s
from distillery.taskpacks.sql.executor import LocalExecutor
from tests.test_evaluator import Lookup

# ---- the pure rule -----------------------------------------------------------------------------

STOP = {151645, 151643}


def stats(pieces: list[str], lps: list[float], ids: list[int] | None = None) -> dict[str, Any]:
    return answer_logprob_stats(ids or list(range(len(pieces))), lps, pieces, STOP)


def test_think_block_and_following_whitespace_are_excluded() -> None:
    pieces = ["<think>", "\n\n", "</think>", "\n\n", "SELECT", " 1"]
    out = stats(pieces, [-9.0, -9.0, -9.0, -9.0, -1.0, -3.0])
    assert out["n_answer_tokens"] == 2
    assert out["mean_logprob"] == pytest.approx(-2.0) and out["min_logprob"] == -3.0
    assert out["n_total_tokens"] == 6


def test_no_think_block_means_every_token_before_eos_counts_and_eos_is_dropped() -> None:
    ids = [10, 11, 12, 151645, 151643]
    out = stats(["SELECT", " a", ";", "<|im_end|>", "<|endoftext|>"], [-1, -2, -3, -9, -9], ids)
    assert out["n_answer_tokens"] == 3 and out["mean_logprob"] == pytest.approx(-2.0)


def test_think_tag_split_over_tokens_and_case_and_straddling_token() -> None:
    pieces = ["</", "think", ">", "\n", "\nSELECT", " 1"]  # "\nSELECT" straddles the boundary
    out = stats(pieces, [-5, -5, -5, -5, -1.0, -1.0])
    assert out["n_answer_tokens"] == 2
    upper = stats(["</THINK>", "\n\n", "x"], [-8, -8, -0.5])
    assert upper["n_answer_tokens"] == 1 and upper["mean_logprob"] == -0.5


def test_last_close_tag_wins_and_unclosed_think_has_no_answer() -> None:
    twice = stats(["a", "</think>", "b", "</think>", "\n", "c"], [-9, -9, -9, -9, -9, -0.25])
    assert twice["n_answer_tokens"] == 1 and twice["mean_logprob"] == -0.25
    unclosed = stats(["<think>", "\n", "hmm"], [-1, -1, -1])
    assert unclosed["n_answer_tokens"] == 0 and unclosed["mean_logprob"] is None
    assert unclosed["min_logprob"] is None
    # an answer followed by a fresh unclosed think block keeps only the answer part
    tail = stats(["</think>", "\n", "ok", "<think>", "x"], [-9, -9, -0.5, -9, -9])
    assert tail["n_answer_tokens"] == 1 and tail["mean_logprob"] == -0.5


def test_zero_answer_tokens_gives_none() -> None:
    assert stats([], [])["mean_logprob"] is None
    only_eos = answer_logprob_stats([151645], [-0.1], ["<|im_end|>"], STOP)
    assert only_eos["n_answer_tokens"] == 0 and only_eos["mean_logprob"] is None
    only_think = stats(["</think>", "\n\n"], [-1, -1])
    assert only_think["n_answer_tokens"] == 0 and only_think["min_logprob"] is None


# ---- the script embeds exactly that function ---------------------------------------------------


def _function_source(script: str, name: str) -> str:
    tree = ast.parse(script)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(script, node) or ""


def test_gen_script_embeds_the_tested_helper_and_is_valid_python() -> None:
    ast.parse(GEN_SCRIPT)
    compile(GEN_SCRIPT, "gen.py", "exec")
    embedded = _function_source(GEN_SCRIPT, "answer_logprob_stats")
    assert embedded.strip() == inspect.getsource(answer_logprob_stats).strip()
    ns: dict[str, Any] = {}
    exec("from __future__ import annotations\n" + embedded, ns)  # the sandbox's copy runs alone
    pieces = ["<think>", "\n\n", "</think>", "\n\n", "SELECT", " 1", "<|im_end|>"]
    args = (list(range(6)) + [151645], [-9, -9, -9, -9, -1.0, -2.0, -7.0], pieces, STOP)
    assert ns["answer_logprob_stats"](*args) == answer_logprob_stats(*args)
    assert GEN_SCRIPT.startswith("from __future__ import annotations")  # annotations never run


def test_gen_script_requests_scores_greedily_and_does_not_stack_them() -> None:
    assert "do_sample=False" in GEN_SCRIPT and "output_scores=True" in GEN_SCRIPT
    assert "return_dict_in_generate=True" in GEN_SCRIPT
    assert "torch.stack" not in GEN_SCRIPT and "torch.cat" not in GEN_SCRIPT
    assert "merge_and_unload()" in GEN_SCRIPT and "--max-new-tokens" in GEN_SCRIPT


# ---- parse_output and the result type ----------------------------------------------------------


def _stdout(items: list[dict[str, Any]]) -> str:
    return "noise\n" + OUT_MARKER + json.dumps({"results": items, "n": len(items)}) + "\n"


def test_parse_output_new_fields_round_trip() -> None:
    sha = hashlib.sha256(b"SELECT 1").hexdigest()
    item = {
        "text": "SELECT 1", "error": None, "prompt_tokens": 10, "new_tokens": 4, "seconds": 1.5,
        "mean_logprob": -0.25, "min_logprob": -1.0, "n_answer_tokens": 3, "output_sha256": sha,
    }  # fmt: skip
    parsed = parse_output(_stdout([item]))
    out = student_output_from_item(parsed["results"][0])
    assert out == StudentOutput("SELECT 1", -0.25, -1.0, 3, sha, 1.5)


def test_old_format_output_still_parses_with_no_signals() -> None:
    item = {"text": "SELECT 2", "error": None, "prompt_tokens": 1, "new_tokens": 2, "seconds": 0.5}
    out = student_output_from_item(parse_output(_stdout([item]))["results"][0])
    assert out.text == "SELECT 2" and out.mean_logprob is None and out.n_answer_tokens is None
    assert out.output_sha256 == text_sha256("SELECT 2")  # recomputed when the script gave none


def test_item_edge_cases_none_error_and_non_finite() -> None:
    zero = student_output_from_item({"text": "", "mean_logprob": None, "n_answer_tokens": 0})
    assert zero.mean_logprob is None and zero.n_answer_tokens == 0
    bad = student_output_from_item({"text": "x", "mean_logprob": float("nan"), "min_logprob": "a"})
    assert bad.mean_logprob is None and bad.min_logprob is None
    err = student_output_from_item({"text": "ignored", "error": "RuntimeError: boom", "seconds": 2})
    assert err.text == "" and err.error == "RuntimeError: boom" and err.mean_logprob is None


# ---- fakes -------------------------------------------------------------------------------------


def test_fake_student_produces_deterministic_plausible_logprobs() -> None:
    fs = FakeStudent(lambda m: f"answer {m[-1]['content']}")
    batch = [[{"role": "user", "content": "a"}], [{"role": "user", "content": "b"}]]
    outs = fs.generate_detailed(batch)
    again = FakeStudent(lambda m: f"answer {m[-1]['content']}").generate_detailed(batch)
    assert outs == again and isinstance(fs, DetailedStudentServer)
    for o in outs:
        assert o.mean_logprob is not None and -2.0 < o.mean_logprob < 0
        assert o.min_logprob is not None and o.min_logprob <= o.mean_logprob
        assert o.n_answer_tokens == 2 and o.output_sha256 == text_sha256(o.text)
    assert outs[0].mean_logprob != outs[1].mean_logprob
    assert (outs[0].mean_logprob, outs[0].min_logprob, outs[0].n_answer_tokens) == fake_logprobs(
        "answer a"
    )
    assert fs.generate(batch) == ["answer a", "answer b"] and fs.calls == 2  # one call each


def test_fake_student_custom_logprob_and_closed() -> None:
    fs = FakeStudent(lambda m: "x", lambda m, text: -0.5)
    assert fs.generate_detailed([[]])[0].mean_logprob == -0.5
    fs.close()
    with pytest.raises(RuntimeError):
        fs.generate_detailed([[]])


def test_generate_detailed_falls_back_for_text_only_servers() -> None:
    assert generate_detailed(Lookup({}), []) == []

    class TextOnly:
        def generate(self, batch: Any) -> list[str]:
            return ["u", "v"]

    got = generate_detailed(TextOnly(), [[], []])
    assert [(o.text, o.mean_logprob) for o in got] == [("u", None), ("v", None)]
    assert got[0].output_sha256 == text_sha256("u")


# ---- evaluator / vectors -----------------------------------------------------------------------


@pytest.fixture
def env(tmp_path: Path):  # type: ignore[no-untyped-def]
    db = tmp_path / "db.sqlite"
    s.write_database(0, str(db))
    tasks = q.generate_tasks(str(db), 12, 0).tasks[:12]
    items = [
        {"task_id": t.task_id, "family": t.family, "question": t.question,
         "gold_sql": t.gold_sql, "requires_order": t.requires_order, "heldout_class": "seen"}
        for t in tasks
    ]  # fmt: skip
    store = Store(tmp_path / "root")
    store.seal_heldout("r1", items)
    answers = {it["question"]: it["gold_sql"] for it in items}
    return store, str(db), s.schema_ddl(0), items, answers


class LpLookup(FakeStudent):
    """A detailed generator: answers via the lookup, with a fixed mean log-probability."""

    def __init__(self, answers: dict[str, str]) -> None:
        inner = Lookup(answers)
        super().__init__(lambda m: inner.generate([list(m)])[0], lambda m, t: -0.125)


def test_score_model_carries_logprobs_and_hashes(env) -> None:  # type: ignore[no-untyped-def]
    _store, db, ddl, items, answers = env
    gold = LocalExecutor().run_batch(db, [str(it["gold_sql"]) for it in items])
    kw: dict[str, Any] = {"schema_ddl": ddl, "db_ref": db, "executor": LocalExecutor()}
    sc = score_model(LpLookup(answers), items, gold, role="student", **kw)
    assert sc.mean_logprob == [-0.125] * len(items) and sc.accuracy == 1.0
    assert sc.output_sha256 == [text_sha256(o) for o in sc.outputs]
    plain = score_model(Lookup(answers), items, gold, role="teacher", **kw)  # type: ignore[arg-type]
    assert plain.mean_logprob == [None] * len(items)  # text-only generator: not measured
    assert plain.output_sha256 == sc.output_sha256


def test_score_setups_vectors_carry_signals_and_round_trip(env) -> None:  # type: ignore[no-untyped-def]
    store, db, ddl, items, answers = env

    def prompt(it: Any) -> Any:
        return build_messages(str(it["question"]), ddl, role="eval_student")

    setups = [Setup("student", LpLookup(answers), prompt), Setup("S0", Lookup(answers), prompt)]
    payload = score_setups(store, "r1", setups, LocalExecutor(), db_ref=db, schema_ddl=ddl)
    assert payload["format"] == "distillery-setup-vectors-v1"  # unchanged: new keys are additive
    blocks = payload["sets"]["heldout"]["setups"]
    assert blocks["student"]["mean_logprob"] == [-0.125] * len(items)
    assert blocks["S0"]["mean_logprob"] == [None] * len(items)
    hashes = [text_sha256(o) for o in blocks["student"]["outputs"]]
    assert blocks["student"]["output_sha256"] == hashes
    # JSON round trip, then a re-score from the cache: no model call, signals preserved
    cached = json.loads(json.dumps(payload))
    nogen = [Setup(st.name, None, st.prompt) for st in setups]
    kw: dict[str, Any] = {"db_ref": db, "schema_ddl": ddl}
    again = score_setups(store, "r1", nogen, LocalExecutor(), cached=cached, **kw)
    assert again == payload
    # a v1 payload written before this change (no new keys) still re-scores
    for b in cached["sets"]["heldout"]["setups"].values():
        del b["mean_logprob"], b["output_sha256"]
    old = score_setups(store, "r1", nogen, LocalExecutor(), cached=cached, **kw)
    old_blocks = old["sets"]["heldout"]["setups"]
    assert old_blocks["student"]["correct"] == blocks["student"]["correct"]
    assert old_blocks["student"]["mean_logprob"] == [None] * len(items)
    assert old_blocks["student"]["output_sha256"] == hashes


def test_eval_report_vectors_json_is_full_and_separate_from_the_capped_report(env) -> None:  # type: ignore[no-untyped-def]
    store, db, ddl, items, answers = env
    gens: dict[str, Any] = {
        "base": LpLookup(answers),
        "student": LpLookup(answers),
        "teacher": Lookup(answers),
    }
    rep = evaluate(
        store, "r1", gens, LocalExecutor(), db_ref=db, schema_ddl=ddl, gate_cfg=GateThresholds()
    )
    v = rep.vectors_json()
    json.dumps(v)
    assert v["format"] == "distillery-final-eval-vectors-v1"
    assert v["task_ids"] == [it["task_id"] for it in items] and v["n"] == len(items)
    assert set(v["setups"]) == {"base", "student", "teacher"}
    for m, blk in v["setups"].items():
        assert len(blk["correct"]) == len(blk["sqls"]) == len(blk["output_sha256"]) == len(items)
        assert len(blk["mean_logprob"]) == len(items)
        assert blk["accuracy"] == rep.scores[m].accuracy
    assert v["setups"]["teacher"]["mean_logprob"] == [None] * len(items)
    assert math.isclose(v["setups"]["student"]["mean_logprob"][0], -0.125)
    assert "setups" not in rep.to_json() and "task_ids" not in rep.to_json()  # report unchanged
