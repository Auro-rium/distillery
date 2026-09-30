# STATUS

**Date:** 2026-09-30 · **Phase:** live plumbing verified end to end · **Result claims:** none yet (see "What the first live run does and does not show")

## Verified (commands run 2026-09-30, real output)
- `.venv/bin/python -m pytest -q`: 402 passed (~73 s). `ruff check .` and `ruff format --check .` clean (82 files). `mypy src`: no issues in 37 source files.
- `python -m distillery run --pack sql --scale tiny --dry-run` runs the pipeline offline with fake models and ends in a labelled "DRY RUN, numbers are NOT results" report. It proves plumbing only.
- Live against Nebius: S1 inference, S2 Sandboxes (4 CPU / ~4 GB, open egress, branching, 40 parallel), S4 student on sandbox CPU (Qwen3-0.6B bf16: 2.09 GB peak RSS, ~7.8 s/sample, ~7.8 tok/s), LoRA fine-tune job on Qwen3-0.6B (files, job, checkpoint, adapter pulled into a sandbox), and one complete end-to-end pipeline run (`sql-tiny-live1`).

## What the first live run does and does not show
`sql-tiny-live1` is a **plumbing smoke test at n=20 held-out tasks, 40 training rows, 1 round**. It is not a result.
- Decision: **REJECT** (the gate refused to promote): ratio lower bound 0.000 < 0.85 (point estimate 0.158); student does not beat base (student-only wins 0, base-only wins 0).
- Held-out accuracy: base 3/20 (0.15), student 3/20 (0.15), teacher 19/20 (0.95). Base dev accuracy 0.10; student dev accuracy 0.30 (n is tiny).
- The adapter is applied: student SQL differs from base SQL on 14 of 17 report examples. It is undertrained, not a no-op.
- Estimated spend **$3.07** = $1.07 LLM (measured tokens x ASSUMED ceiling prices: Ultra $0.71, Super $0.34, Nano $0.01) + $2.00 fine-tune ceiling (operator-supplied, not a billed amount). Sandbox compute is not priced. The console is the only source for the billed amount.
- Resumability was exercised live: the first attempt died on a transient `APIConnectionError` while polling the fine-tune job (and the cancel attempt failed for the same reason); the rerun adopted the still-running job instead of paying for a second one.
- `sandbox_lineage` is empty because branching happens only after failure analysis in round 2+; a 1-round run has no sandbox branch. The experiment tree needs a run with `--max-rounds >= 2`.
- Final state check: 0 sandbox operations executing; the only fine-tune job is `succeeded`.

## NOT verified
- Any accuracy claim at a meaningful sample size (needs a larger, multi-round run).
- Real (console) prices for Ultra/Super/Nano and the fine-tune price; the sandbox price. Third-party aggregator listings (not console) show Ultra ~$1/$3, Super ~$0.30/$0.90, Nano ~$0.06/$0.24 per Mtok on Nebius; all below our ceilings, so the guard is conservative.
- Round 2+ live path (targeted generation, sandbox branch) has only run against fakes.
- Backend public over HTTPS on Nebius (see docs/DEPLOY.md); the Vercel page currently cannot reach an API.

## Config
- Student: `Qwen/Qwen3-0.6B` (LoRA), served on Nebius Sandbox CPU. Budget: $19.50 project cap, $10 of it reserved for the demo.
- Live home (holds the spend ledger, keep using it so the project cap accounting stays correct): `.distillery/live` (gitignored).
- Prices: `.distillery/prices.json`, labelled "ASSUMED UPPER BOUND for budget guard, NOT the console price".

## Open risks
- Headroom (base <= 60%, teacher >= 85%) held in the smoke test (base 15%, teacher 95%) but the student did not move with 40 rows; a real run needs more rows and 2+ rounds.
- Remaining estimated budget for working runs: $19.50 - $10 reserve - $3.07 = ~$6.4 by ceiling estimates (real charge unknown).

## Blockers (user)
- Console usage/charge for `sql-tiny-live1` and real per-Mtok prices.
- Demo video, final Devpost submit, making the repo public.
