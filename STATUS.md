# STATUS

**Date:** 2026-09-30 · **Phase:** 0 (kill the risks) · **Gate 0:** NOT met

## Verified (commands run 2026-09-30, real output)
- `.venv/bin/python -m pytest -q`: 363 passed (~74s; the server agent was adding tests concurrently, so this count moves). `ruff check .`: all checks passed. `ruff format --check .`: 76 files already formatted. `mypy src`: no issues in 37 source files.
- `python -m distillery run --pack sql --scale tiny --dry-run` runs the whole pipeline offline with fake models and ends in a labelled "DRY RUN, numbers are NOT results" report. This proves plumbing only (resume, cleanup on failure, budget refusal, sealed held-out, artifact check), not model quality.
- Live-verified against Nebius: S1 inference (model ids, `json_schema` mode, usage, reasoning field) and `LLMClient` live (see `spikes/s1_*.py`, `spikes/out/`).
- Cleanup: fine-tune jobs and endpoints are cancelled/deleted in `finally` blocks; the final-eval closes every serving endpoint even if one `close()` raises (tested with fakes only).
- Endpoint identity: `student_serving="endpoint"` refuses (ConfigRefusal) if `EndpointSpec.base_model_name` differs from the configured student, or the trained `base_model` / checkpoint name differs; the report's artifact section records the served model names. File hashes still only prove the adapter on disk, not what an endpoint serves.

## NOT verified (no live call has been made for these)
- Sandboxes (Contree): blocked on the beta access form / project id. Image build, parallelism, branching and the CPU student path are untested live.
- Fine-tune job (files, LoRA training, checkpoint download, cancel): never run; duration and price unmeasured.
- Student serving (sandbox-CPU or dedicated endpoint): never run. Docs say custom weights on dedicated endpoints are "beta, on request, contact Support"; how a LoRA is named on an endpoint is unknown (`EndpointSpec.model_name` is explicit input).
- Prices used by the budget ledger for anything other than S1 models are unconfirmed.

## Config
- Student: `Qwen/Qwen3-1.7B` (LoRA). Budget: $19.50 project cap, of which $10 is reserved for the demo.

## Open risks
- Student serving (S4) is the biggest risk; live runs refuse to start without a serving path.
- Sandbox access is blocked externally.
- Headroom (base <= 60%, teacher >= 85%) is unproven until real models run; the SQL generator may now be too hard (n=1500: 635 hard / 696 medium / 169 easy).
- `make_live_deps` in `cli.py` is untested against the real fine-tune/sandbox/endpoint APIs.

## Known problems
- SQL generator was 75% easy `filter` tasks (29 hard at n=3000); now capped per template/family: at n=1500, 635 hard / 696 medium / 169 easy. It may now be too hard: headroom (base <= 60%, teacher >= 85%) is unproven until real models run.
- The gold cross-check sends all generated questions (including future held-out ones) to the planner model, for label quality; the "planner never sees held-out" rule is enforced and tested only for failure analysis onward.

## Next actions
1. Get the Sandbox beta access / project id; then run the Sandboxes spike (image with python+sqlite, parallel ops, branching).
2. Ask for approval before the first fine-tune spend; run the student-serving spike before building anything around it.

## Blockers
- Sandbox (Contree) beta form / project id.
