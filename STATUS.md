# STATUS

**Date:** 2026-09-29 · **Phase:** 0 (kill the risks), code-first while waiting for credentials · **Gate 0:** NOT met

## Verified (commands run, offline)
- `pytest`: all tests pass (last full run before the evaluator: 206 passed; evaluator adds 9; core 42, clients 55 are included in that total of the modules that existed at the time). Re-run for current counts.
- `ruff check`, `ruff format --check`, `mypy --strict src`: clean at the last full run.
- Offline modules written and unit-tested: config, stats (paired bootstrap, exact McNemar, Wilson), gate (pure function), budget (ledger, `paid_resource` cancels on any exit path), store (SQLite + content-addressed artifacts, resumable stages, sealed held-out), llm (routing by role, retry classification, structured output, usage logging), sandbox (Contree wrapper with lazy import + `FakeSandbox` with lineage), finetune (files/jobs/checkpoints, cancel-on-failure), evaluator (only module that loads held-out; artifact identity check; AST test enforces it), prompts (single message builder), SQL task pack (schema generator, 49 templates in 12 families, execution-match verifier, corruption helpers).

## NOT verified
- No Nebius API call has been made. Everything in `docs/NEBIUS_NOTES.md` is from docs only. Spikes S1–S5 have not run.
- Model IDs, prices, `response_format` wrapper, reasoning-content field, Qwen3 `/no_think` behaviour, adapter file names, Contree PyPI package names and base URL are all unconfirmed.
- Whether the fine-tuned LoRA student can be served at all: docs say custom weights on dedicated endpoints are "beta, on request, contact Support" and the linked deploy page 404s. This is spike S4 and is the biggest risk.

## Known problems being worked
- SQL generator skew: at n=3000, seed 0, 75% of tasks are the easy `filter` family and only 29 are hard (two date templates dominate). Being rebalanced with per-template/family caps; headroom (base <= 60%, teacher >= 85%) is unproven until real models run.
- Orchestrator/CLI with a fully offline `--dry-run` (fake models, clearly labelled not-results) is in progress.

## Next 3 actions
1. Finish rebalance + orchestrator; run the dry-run end to end and record the output here.
2. With a Nebius key in `.env`: run S1 (inference: model IDs, JSON mode, `usage`, thinking output) and S2 (Sandboxes: image with python+sqlite, 40 parallel ops, branching).
3. Ask for approval before S3 (first fine-tune spend); run S4 (student serving) before building anything around it.

## Blockers
- Nebius Token Factory API key and Sandbox (Contree) credentials in `.env`.
