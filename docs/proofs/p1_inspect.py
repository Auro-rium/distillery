"""P1 evidence (free, local, read-only): token-level train/eval parity, lengths, SQL style.

Run:  .venv/bin/python docs/proofs/p1_inspect.py TOKENIZER_DIR CKPT_TEMPLATE [--root .distillery/live]
TOKENIZER_DIR holds the Qwen3-1.7B tokenizer.json + tokenizer_config.json (HF, pinned sha).
CKPT_TEMPLATE is a provider-returned chat_template.jinja from a real fine-tune checkpoint.
Prints JSON. Only lengths/counts of sealed gold SQL are read (P1.5); no text leaves this process.
"""

import json
import re
import statistics
import sys
from pathlib import Path

from tokenizers import Tokenizer

from distillery.diagnostics import read_jsonl, render_chat_template, stage_artifact
from distillery.prompts import build_messages, extract_sql, to_training_row
from distillery.store import Store
from distillery.taskpacks.sql.schema import schema_ddl

tok_dir, ckpt_template = Path(sys.argv[1]), Path(sys.argv[2])
root = Path(sys.argv[sys.argv.index("--root") + 1]) if "--root" in sys.argv else Path(".distillery/live")
tok = Tokenizer.from_file(str(tok_dir / "tokenizer.json"))
hf_template = json.loads((tok_dir / "tokenizer_config.json").read_text())["chat_template"]
provider_template = ckpt_template.read_text()
ids = lambda s: tok.encode(s, add_special_tokens=False).ids  # noqa: E731


def pct(xs: list[int]) -> dict[str, int]:
    xs = sorted(xs)
    return {"n": len(xs), "p50": xs[len(xs) // 2], "p95": xs[int(len(xs) * 0.95) - 1], "max": xs[-1]}


store = Store(root)
split = stage_artifact(store, "gated-1p7b-r1", "split")
allq = stage_artifact(store, "gated-1p7b-r1", "questions")
ddl = schema_ddl(0)
out: dict[str, object] = {}

# P1.2: token-ID parity. Training row rendered as the provider's template renders a chat row
# (no generation prompt); eval prompt rendered as the sandbox does (generation prompt, thinking off).
out["templates_identical_hf_vs_provider"] = hf_template.strip() == provider_template.strip()
parity = {"checked": 0, "prefix_identical": 0, "completion_decodes_to_sql": 0, "example": None}
for t in split["train"][:200]:
    row = to_training_row(type("T", (), {"question": t["question"]})(), t["gold_sql"], schema_ddl=ddl)
    train_txt = render_chat_template(provider_template, row["messages"], add_generation_prompt=False)
    eval_txt = render_chat_template(
        hf_template, build_messages(t["question"], ddl, role="eval_student"),
        add_generation_prompt=True, enable_thinking=False,
    )
    a, b = ids(train_txt), ids(eval_txt)
    parity["checked"] += 1
    if a[: len(b)] == b:
        parity["prefix_identical"] += 1
        completion = tok.decode(a[len(b):], skip_special_tokens=False)
        if completion.strip() == t["gold_sql"].strip() + "<|im_end|>":
            parity["completion_decodes_to_sql"] += 1
        if parity["example"] is None:
            parity["example"] = {"eval_tail": eval_txt[-60:], "train_completion": completion[:200],
                                 "prompt_tokens": len(b), "row_tokens": len(a)}
out["P1.2_token_parity"] = parity

# P1.4: full training-row length (prompt + completion) for real smoke rows and gated-scale rows.
lens: dict[str, list[int]] = {}
for run in ("sql-tiny-live1", "sql-mini-live1"):
    for f in sorted((root / "runs" / run).glob("round*/train.jsonl")):
        lens.setdefault(run, []).extend(
            len(ids(render_chat_template(provider_template, r["messages"], add_generation_prompt=False)))
            for r in read_jsonl(f)
        )
lens["gated_train_with_gold_as_completion"] = [
    len(ids(render_chat_template(provider_template, to_training_row(
        type("T", (), {"question": t["question"]})(), t["gold_sql"], schema_ddl=ddl)["messages"],
        add_generation_prompt=False))) for t in split["train"]
]
out["P1.4_train_row_tokens"] = {k: pct(v) for k, v in lens.items()}

# P1.5: gold SQL completion length in tokens (all generated tasks incl. sealed; lengths only)
# against the student's max_new_tokens, plus the recorded raw outputs that may have hit the cap.
gold_lens = [len(ids(t["gold_sql"])) + 1 for t in allq["tasks"] + allq["stress_tasks"]]  # +<|im_end|>
out["P1.5_gold_sql_tokens"] = pct(gold_lens) | {
    "over_160": sum(x > 160 for x in gold_lens), "over_256": sum(x > 256 for x in gold_lens)}
diag = json.loads(Path(".distillery/proofs/diag_sql-mini-live1_2026-10-01.json").read_text())
raw_stats = {"outputs": 0, "at_or_near_cap_158plus": 0, "extraction_changed_text": 0,
             "extraction_none": 0, "think_tag_seen": 0, "fence_seen": 0}
for model, sets in diag["models"].items():
    for s in sets.values():
        for it in s["items"]:
            raw = it["raw"] or ""
            raw_stats["outputs"] += 1
            raw_stats["at_or_near_cap_158plus"] += len(ids(raw)) >= 158
            sql = extract_sql(raw)
            raw_stats["extraction_none"] += sql is None
            raw_stats["extraction_changed_text"] += sql is not None and sql != raw.strip().rstrip(";").strip()
            raw_stats["think_tag_seen"] += "<think>" in raw.lower()
            raw_stats["fence_seen"] += "```" in raw
out["P1.5_recorded_raw_outputs"] = raw_stats

# P1.9: style variance of the SQL completions actually trained on (smoke rows).
sqls = [r["messages"][-1]["content"] for run in ("sql-tiny-live1", "sql-mini-live1")
        for f in (root / "runs" / run).glob("round*/train.jsonl") for r in read_jsonl(f)]
kw = re.compile(r"\b(select|from|where|join|group by|order by)\b", re.I)
out["P1.9_style"] = {
    "rows": len(sqls),
    "keywords_upper": sum(all(m.group(0).isupper() for m in kw.finditer(s)) for s in sqls),
    "keywords_lower": sum(all(m.group(0).islower() for m in kw.finditer(s)) for s in sqls),
    "uses_table_alias": sum(bool(re.search(r"\b(?:from|join)\s+\w+\s+(?:as\s+)?[a-z]{1,3}\b(?!\s*on)", s, re.I))
                            for s in sqls),
    "multiline": sum("\n" in s for s in sqls),
    "trailing_semicolon": sum(s.rstrip().endswith(";") for s in sqls),
    "median_chars": int(statistics.median(len(s) for s in sqls)),
}
print(json.dumps(out, indent=1))
