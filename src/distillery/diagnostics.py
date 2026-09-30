"""Read-only diagnostics helpers for the scripts/diagnose_*.py tools.

Pure functions (prompt comparison, chat-template rendering, failure overlap, training-row items)
plus one orchestration function whose sandbox/serving client is injected, so tests use fakes.
Nothing here spends money or touches the network by itself.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from distillery import evaluator as evaluator_mod
from distillery.prompts import build_messages
from distillery.store import Store, canonical_json
from distillery.student import StudentServer
from distillery.taskpacks.sql.executor import Executor

_QUESTION_RE = re.compile(r"\n\nQuestion: (.*?)\n\n", re.DOTALL)


# ---------------------------------------------------------------- (e) prompt check


def question_of(user_content: str) -> str | None:
    m = _QUESTION_RE.search(user_content)
    return m.group(1) if m else None


def compare_prompts(train_row: Mapping[str, Any], schema_ddl: str) -> dict[str, Any]:
    """Training prompt (row minus its assistant turn) vs the eval prompt rebuilt with
    ``build_messages`` for the same question and the run's real schema."""
    msgs = [dict(m) for m in train_row["messages"]]
    train_prompt = msgs[:-1] if msgs and msgs[-1]["role"] == "assistant" else msgs
    question = next((question_of(m["content"]) for m in train_prompt if m["role"] == "user"), None)
    if question is None:
        raise ValueError("cannot find the question in the training row's user message")
    eval_prompt = build_messages(question, schema_ddl, role="eval_student")
    return {
        "question": question,
        "train_prompt": train_prompt,
        "eval_prompt": eval_prompt,
        "byte_identical": canonical_json(train_prompt) == canonical_json(eval_prompt),
        "train_assistant": msgs[-1]["content"]
        if msgs and msgs[-1]["role"] == "assistant"
        else None,
    }


def render_chat_template(
    template_text: str,
    messages: Sequence[Mapping[str, str]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool | None = None,
) -> str:
    """Render messages through a HF ``chat_template.jinja`` with jinja2 (raises ImportError if
    jinja2 is missing; callers degrade)."""
    import jinja2  # type: ignore[import-not-found,unused-ignore]
    from jinja2.ext import loopcontrols  # type: ignore[import-not-found,unused-ignore]
    from jinja2.sandbox import (  # type: ignore[import-not-found,unused-ignore]
        ImmutableSandboxedEnvironment,
    )

    def raise_exception(msg: str) -> None:
        raise jinja2.exceptions.TemplateError(msg)

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols]
    )
    env.globals["raise_exception"] = raise_exception
    env.filters["tojson"] = lambda v, **_k: json.dumps(v, ensure_ascii=False)
    ctx: dict[str, Any] = {
        "messages": [dict(m) for m in messages],
        "add_generation_prompt": add_generation_prompt,
        "bos_token": "",
        "eos_token": "<|im_end|>",
    }
    if enable_thinking is not None:
        ctx["enable_thinking"] = enable_thinking
    return str(env.from_string(template_text).render(**ctx))


def render_all(template_text: str, cmp: Mapping[str, Any]) -> dict[str, Any]:
    """The four renders a human compares token for token; degrades to an error note."""
    try:
        train_full = [
            *cmp["train_prompt"],
            {"role": "assistant", "content": cmp["train_assistant"]},
        ]
        return {
            "eval_no_thinking": render_chat_template(
                template_text, cmp["eval_prompt"], add_generation_prompt=True, enable_thinking=False
            ),
            "eval_thinking": render_chat_template(
                template_text, cmp["eval_prompt"], add_generation_prompt=True, enable_thinking=True
            ),
            "train_with_assistant": render_chat_template(
                template_text, train_full, add_generation_prompt=False
            ),
        }
    except ImportError as exc:
        return {"error": f"jinja2 unavailable: {exc}"}
    except Exception as exc:  # noqa: BLE001 - template problems are findings, not crashes
        return {"error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------- (f) dev failure overlap


def failed_ids(dev_eval_artifact: Mapping[str, Any]) -> list[str]:
    return [str(f["task_id"]) for f in dev_eval_artifact.get("failures", [])]


def failure_overlap(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    fa, fb = set(failed_ids(a)), set(failed_ids(b))
    return {
        "failed_a": sorted(fa),
        "failed_b": sorted(fb),
        "both": sorted(fa & fb),
        "only_a": sorted(fa - fb),
        "only_b": sorted(fb - fa),
        "n_a": len(fa),
        "n_b": len(fb),
        "n_both": len(fa & fb),
    }


def stage_artifact(store: Store, run_id: str, stage_or_hash: str) -> dict[str, Any]:
    """A stage's stored output, by artifact hash or by stage name (latest complete attempt)."""
    if re.fullmatch(r"[0-9a-f]{64}", stage_or_hash):
        digest = stage_or_hash
    else:
        con = sqlite3.connect(f"file:{store.root / 'index.sqlite'}?mode=ro", uri=True)
        try:
            row = con.execute(
                "SELECT output_sha256 FROM stages WHERE run_id=? AND stage=? AND status='complete' "
                "ORDER BY updated_at DESC LIMIT 1",
                (run_id, stage_or_hash),
            ).fetchone()
        finally:
            con.close()
        if row is None:
            raise LookupError(f"no complete stage {stage_or_hash!r} in run {run_id!r}")
        digest = row[0]
    data = json.loads(store.get_artifact(run_id, digest))
    if not isinstance(data, dict):
        raise ValueError("stage artifact is not an object")
    return data


# ---------------------------------------------------------------- student diagnosis


def train_items(
    rows: Sequence[Mapping[str, Any]], tasks: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Items for an adapter's OWN training rows: the exact trained prompt (assistant turn removed)
    and the gold SQL of the matching task (by question); falls back to the row's completion."""
    by_question = {str(t["question"]).strip(): t for t in tasks}
    out: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        msgs = [dict(m) for m in row["messages"]]
        completion = msgs[-1]["content"] if msgs[-1]["role"] == "assistant" else ""
        prompt = msgs[:-1] if msgs[-1]["role"] == "assistant" else msgs
        q = next((question_of(m["content"]) for m in prompt if m["role"] == "user"), None)
        task = by_question.get((q or "").strip())
        out.append(
            {
                "task_id": task["task_id"] if task else f"train-row-{i}",
                "messages": prompt,
                "gold_sql": task["gold_sql"] if task else completion,
                "requires_order": bool(task["requires_order"]) if task else False,
            }
        )
    return out


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def adapter_round(name: str) -> int | None:
    m = re.search(r"round(\d+)", name)
    return int(m.group(1)) if m else None


def run_student_diagnosis(
    store: Store,
    run_id: str,
    adapters: Mapping[str, tuple[int, Sequence[Path]]],
    *,
    make_server: Callable[[Sequence[Path]], StudentServer],
    executor: Executor,
    db_ref: str,
    schema_ddl: str,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Generate with the base model and each named adapter (``name -> (round, adapter files)``)
    over: that adapter's own training rows, the dev tasks and the sealed held-out.

    ``make_server(files)`` returns a ``StudentServer`` (empty files = the base model); it is
    injected so tests pass fakes and the script passes ``SandboxCpuStudent``. Each model is
    created once, closed after use, and gets all its prompts in one ``generate`` call. Result
    (also written to ``out_path``) holds raw and extracted SQL per task, accuracy per set and the
    identical-string rate of each adapter vs base on raw text and on extracted SQL.
    """
    run_dir = store.run_dir(run_id)
    split = stage_artifact(store, run_id, "split")
    dev = [dict(t) for t in split["dev"]]
    tasks = [dict(t) for t in split["train"]]
    per_model_sets: dict[str, dict[str, list[dict[str, Any]]]] = {"base": {"dev": dev}}
    for name, (rnd, _files) in adapters.items():
        rows = read_jsonl(run_dir / f"round{rnd}" / "train.jsonl")
        per_model_sets[name] = {"train": train_items(rows, tasks), "dev": dev}

    models: dict[str, Any] = {}
    identical: dict[str, Any] = {}
    # Base runs over the union of every adapter's train items too, so identical rates line up.
    base_sets: dict[str, list[dict[str, Any]]] = {"dev": dev}
    for name in adapters:
        base_sets[f"train[{name}]"] = per_model_sets[name]["train"]

    def one(name: str, files: Sequence[Path], sets: dict[str, list[dict[str, Any]]]) -> Any:
        server = make_server(files)
        try:
            return evaluator_mod.diagnose_with_heldout(
                store, run_id, {name: server}, sets, executor, db_ref=db_ref, schema_ddl=schema_ddl
            )["models"][name]
        finally:
            server.close()

    models["base"] = one("base", (), base_sets)
    for name, (_rnd, files) in adapters.items():
        res = one(name, files, per_model_sets[name])
        models[name] = res
        base_train = models["base"][f"train[{name}]"]["items"]
        ident: dict[str, Any] = {}
        for set_name, base_set in (
            ("train", base_train),
            ("dev", models["base"]["dev"]["items"]),
            ("heldout", models["base"]["heldout"]["items"]),
        ):
            mine = res[set_name]["items"] if set_name in res else None
            if mine is None:
                continue
            ident[set_name] = {
                f: evaluator_mod.identical_rate([r[f] for r in base_set], [r[f] for r in mine])
                for f in ("raw", "sql")
            }
        identical[name] = ident
    summary = {m: {s: v["accuracy"] for s, v in sets.items()} for m, sets in models.items()}
    result = {
        "run_id": run_id,
        "accuracy": summary,
        "identical_to_base": identical,
        "models": models,
    }
    if out_path is not None:
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result
