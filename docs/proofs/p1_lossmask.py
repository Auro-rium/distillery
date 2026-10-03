"""P1.3 evidence (free, local CPU): infer the provider's loss masking for conversational rows.

The provider does not document whether conversational rows train on the completion only or the
whole sequence. At lr 1e-5 one optimizer step barely moves the weights, so the job's step-1
train_loss is close to the BASE model's loss on the same rows. We compute the base loss both ways
on the exact rows the job trained on; whichever matches the reported step-1 loss is the masking.

Run (needs torch + transformers; not project deps):
  python docs/proofs/p1_lossmask.py RUN_DIR TEMPLATE [--model Qwen/Qwen3-0.6B]
"""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

run_dir, template = Path(sys.argv[1]), Path(sys.argv[2]).read_text()
model_id = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).eval()
rows = [
    json.loads(x)
    for f in sorted(run_dir.glob("round1/train.jsonl"))
    for x in f.read_text().splitlines()
]

tot = {"full": [0.0, 0], "completion": [0.0, 0], "completion_no_think_block": [0.0, 0]}
for r in rows:
    msgs = r["messages"]
    full = tok.apply_chat_template(msgs, chat_template=template, tokenize=False)
    prefix = tok.apply_chat_template(
        msgs[:-1],
        chat_template=template,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prefix_bare = tok.apply_chat_template(
        msgs[:-1], chat_template=template, tokenize=False, add_generation_prompt=True
    )
    ids = tok(full, add_special_tokens=False, return_tensors="pt").input_ids
    with torch.no_grad():
        logits = model(ids).logits[0, :-1].float()
    nll = torch.nn.functional.cross_entropy(logits, ids[0, 1:], reduction="none")
    for key, p in (
        ("full", ""),
        ("completion", prefix),
        ("completion_no_think_block", prefix_bare),
    ):
        start = len(tok(p, add_special_tokens=False).input_ids) if p else 1
        seg = nll[start - 1 :]
        tot[key][0] += float(seg.sum())
        tot[key][1] += seg.numel()
print(
    json.dumps(
        {
            "model": model_id,
            "rows": len(rows),
            "base_mean_token_loss": {k: round(s / n, 4) for k, (s, n) in tot.items()},
            "tokens": {k: n for k, (_s, n) in tot.items()},
        },
        indent=1,
    )
)
