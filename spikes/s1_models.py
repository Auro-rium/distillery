"""S1a: list models (no inference cost). Prints ids only; never prints the key."""
import json
import os
import sys

from openai import OpenAI

c = OpenAI(base_url=os.environ["NEBIUS_BASE_URL"], api_key=os.environ["NEBIUS_API_KEY"])
models = c.models.list()
rows = []
for m in models.data:
    d = m.model_dump()
    rows.append(d)
json.dump(rows, open("spikes/out/models.json", "w"), indent=1)
print(len(rows), "models; keys on first entry:", sorted(rows[0].keys()) if rows else None)
pat = sys.argv[1:] or ["nemotron", "qwen3-1.7", "qwen3-0.6"]
for d in rows:
    if any(p in d["id"].lower() for p in pat):
        print(d["id"])
