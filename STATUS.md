# STATUS

**Date:** 2026-09-29 · **Phase:** 0 (kill the risks), code-first while waiting for credentials · **Gate 0:** NOT met

## Verified (commands run, offline)
- `pytest`: 259 passed (~32s). `ruff check`, `ruff format --check`, `mypy --strict src` (25 files): clean. CI green on the last push before the orchestrator.
- `python -m distillery run --pack sql --scale tiny --dry-run` runs the whole pipeline offline with fake models (~9s) and ends in a labelled "DRY RUN — numbers are NOT results" report. This proves plumbing only (resume, cancel-on-failure, budget refusal, sealed held-out, artifact check), not model quality.
- Offline modules written and unit-tested: config, stats (paired bootstrap, exact McNemar, Wilson), gate (pure function), budget (ledger, `paid_resource` cancels on any exit path), store (SQLite + content-addressed artifacts, resumable stages, sealed held-out), llm (routing by role, retry classification, structured output, usage logging), sandbox (Contree wrapper with lazy import + `FakeSandbox` with lineage), finetune (files/jobs/checkpoints, cancel-on-failure), evaluator (only module that loads held-out; artifact identity check; AST test enforces it), prompts (single message builder), SQL task pack (schema generator, 49 templates in 12 families, execution-match verifier, corruption helpers).

## NOT verified
- No Nebius API call has been made. Everything in `docs/NEBIUS_NOTES.md` is from docs only. Spikes S1–S5 have not run.
- Model IDs, prices, `response_format` wrapper, reasoning-content field, Qwen3 `/no_think` behaviour, adapter file names, Contree PyPI package names and base URL are all unconfirmed.
- Whether the fine-tuned LoRA student can be served at all: docs say custom weights on dedicated endpoints are "beta, on request, contact Support" and the linked deploy page 404s. This is spike S4 and is the biggest risk.

## Known problems
- SQL generator was 75% easy `filter` tasks (29 hard at n=3000); now capped per template/family: at n=1500, 635 hard / 696 medium / 169 easy. It may now be too hard: headroom (base <= 60%, teacher >= 85%) is unproven until real models run.
- Live runs refuse to start until a student serving path exists (S4); live wiring in `cli.py` (`make_live_deps`) is untested against real APIs.
- The gold cross-check sends all generated questions (including future held-out ones) to the planner model, for label quality; the "planner never sees held-out" rule is enforced and tested only for failure analysis onward.

## Next actions
1. With a Nebius key in `.env`: run S1 (inference: model IDs, JSON mode, `usage`, thinking output) and S2 (Sandboxes: image with python+sqlite, 40 parallel ops, branching).
2. Ask for approval before S3 (first fine-tune spend); run S4 (student serving) before building anything around it.

## Blockers
- Nebius Token Factory API key and Sandbox (Contree) credentials in `.env`.
