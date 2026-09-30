"""S1b: tiny chat + json_schema calls on the three Nemotron models. Saves raw responses."""
import json
import os
import time

from openai import OpenAI

c = OpenAI(base_url=os.environ["NEBIUS_BASE_URL"], api_key=os.environ["NEBIUS_API_KEY"])
MODELS = [
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/Nemotron-3-Ultra-550b-a55b",
]
SCHEMA = {
    "type": "object",
    "properties": {"sql": {"type": "string"}},
    "required": ["sql"],
    "additionalProperties": False,
}
out = []
for m in MODELS:
    for kind in ("plain", "json_schema"):
        kw = {}
        msgs = [{"role": "user", "content": "Return the SQL `SELECT 1`."}]
        if kind == "json_schema":
            kw["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "sql_out", "schema": SCHEMA, "strict": True},
            }
            msgs[0]["content"] += ' Answer as JSON {"sql": "..."}.'
        t = time.time()
        try:
            r = c.chat.completions.create(model=m, messages=msgs, max_tokens=400, temperature=0, **kw)
            d = r.model_dump()
            rec = {"model": m, "kind": kind, "latency_s": round(time.time() - t, 2), "resp": d}
            msg = d["choices"][0]["message"]
            print(m.split("/")[-1], kind, f"{rec['latency_s']}s", "msg_keys=", sorted(k for k, v in msg.items() if v),
                  "usage=", d.get("usage"), "finish=", d["choices"][0]["finish_reason"])
            print("   content:", (msg.get("content") or "")[:160].replace("\n", " "))
        except Exception as e:  # spike script: report and continue
            rec = {"model": m, "kind": kind, "error": f"{type(e).__name__}: {str(e)[:400]}"}
            print(m.split("/")[-1], kind, "ERROR", rec["error"])
        out.append(rec)
json.dump(out, open("spikes/out/s1_chat.json", "w"), indent=1, default=str)
