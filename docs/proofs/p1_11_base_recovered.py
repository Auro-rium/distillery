# ruff: noqa: E501, E731, E702, E401, I001, S310, E741, S101, B905, E402  (evidence script)
"""Build P1.11 score_base.json from recovered sandbox stdout (no new generation).

Sources: 30 base batches recovered from the 08:31-08:40 UTC ops (recovered_base_outputs.json) plus
the 2 previously timed-out batches completed in the 13:10 stall experiment. Same image, same command
(`gen.py --base /models/base --max-new-tokens 160`), greedy. Prompts are matched by exact messages.
"""
import json, os, sys, urllib.request
from pathlib import Path
from distillery.evaluator import diagnose_generate
from distillery.taskpacks.sql.executor import LocalExecutor
from distillery.taskpacks.sql.schema import schema_ddl
from distillery.sandbox_student import OUT_MARKER

T = Path("/home/lenovo/.claude/jobs/958b3c06/tmp")
BASE = "https://api.tokenfactory.nebius.com/sandboxes"
HDR = {"Authorization": f"Bearer {os.environ['NEBIUS_API_KEY']}", "Project": os.environ["NEBIUS_AI_PROJECT"]}
get = lambda p: json.load(urllib.request.urlopen(urllib.request.Request(BASE + p, headers=HDR), timeout=60))

batches = []  # (stdin, stdout, op_uuid)
for e in json.load(open(T / "recovered_base_outputs.json")):
    batches.append((get(f"/v1/operations/{e['operation_uuid']}")["metadata"]["stdin"]["value"], e["stdout"], e["operation_uuid"]))
for o in get("/v1/operations?since=2026-10-04T08:50:00Z&limit=50"):
    if o["status"] == "SUCCESS" and (o["consumed_cpu"] or 0) > 1:
        md = get(f"/v1/operations/{o['uuid']}")["metadata"]
        if "--adapter" in md["command"]:
            continue
        sp = get(f"/v1/operations/{o['uuid']}/subprocesses/1")
        batches.append((md["stdin"]["value"].rstrip("\n"), sp["stdout"]["value"], o["uuid"]))
lookup, prov = {}, {}
for stdin, stdout, uuid in batches:
    line = next((l for l in stdout.splitlines() if l.startswith(OUT_MARKER)), None)
    if line is None:
        print("no marker in", uuid, repr(stdout[:80]), file=sys.stderr); continue
    res = json.loads(line[len(OUT_MARKER):])["results"]
    msgs = json.loads(stdin)["messages_batch"]
    assert len(res) == len(msgs)
    for m, r in zip(msgs, res):
        k = json.dumps(m, sort_keys=True)
        assert r.get("error") is None, r
        if k in lookup and lookup[k] != r["text"]:
            print("NONDETERMINISTIC duplicate", uuid, file=sys.stderr)
        lookup.setdefault(k, r["text"]); prov.setdefault(k, uuid)

class Recorded:
    def generate(self, prompts):
        return [lookup[json.dumps(p, sort_keys=True)] for p in prompts]

sys.path.insert(0, "docs/proofs")
import argparse
import p1_overfit as P

train_sel = P.prepare(argparse.Namespace(root=".distillery/live", out_dir=".distillery/proofs/p1_11"))[0]
items = [{"task_id": t["task_id"], "gold_sql": t["gold_sql"], "requires_order": bool(t["requires_order"]),
          "question": t["question"]} for t in train_sel]
db = ".distillery/live/runs/gated-1p7b-r1/db.sqlite"
r = diagnose_generate({"base": Recorded()}, {"train64": items}, LocalExecutor(), db_ref=db, schema_ddl=schema_ddl(0))
out = r["models"]["base"]
out["provenance"] = {"source": "recovered stdout of sandbox ops (GET /v1/operations/{id}/subprocesses/1)",
                     "ops": sorted(set(prov.values())), "batches": len(batches), "prompts_matched": len(items)}
Path(".distillery/proofs/p1_11/score_base.json").write_text(json.dumps(out))
print("base accuracy", out["train64"]["accuracy"], "n", out["train64"]["n"], "batches", len(batches))
