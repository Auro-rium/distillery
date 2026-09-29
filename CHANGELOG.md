# CHANGELOG

## 2026-09-29
- Repo skeleton: MIT LICENSE, .gitignore, .env.example, pyproject.toml (ruff/mypy/pytest config), pre-commit (ruff, gitleaks), CI workflow, tracking docs.
- Core logic: config, stats, gate, budget, store (offline, unit-tested).
- Nebius clients written against the saved docs (not yet run live): llm, sandbox (+FakeSandbox), finetune, student protocol; `docs/NEBIUS_NOTES.md` rewritten from docs.
- SQL task pack: schema generator, question templates, execution-match verifier, executor protocol, prompt builder.
- evaluator: sealed held-out scoring, artifact identity check, AST test that only evaluator.py loads held-out.
- Orchestrator (resumable stages, cancel-on-failure, budget preflight, dev-only failure analysis), CLI with admin-token gating, offline dry-run pipeline with fakes, sandbox-backed SQL executor.
