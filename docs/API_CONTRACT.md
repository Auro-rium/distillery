# API CONTRACT (server <-> frontend). Source of truth for both sides; change it here first.

Base path `/api`. JSON. Errors: `{"error": "<code>", "message": "<human text>"}` with proper HTTP status.
Every run/report object carries `dry_run: bool`; the UI MUST show a persistent banner "DRY RUN — fake models, numbers are NOT results"
when true, and "Recorded run · <recorded_at> · real Token Factory jobs" for replay bundles of real runs. Never show numbers without one of these labels.

## Read endpoints (public)
- `GET /api/health` -> `{ok: true, mode: "live"|"replay-only", version}`
- `GET /api/config` -> `{models: {planner, teacher, triage, student}, thresholds: {ratio_lower_bound_min, mcnemar_alpha, bootstrap_resamples}, run_cap_usd, playground: {enabled, per_ip_per_hour, daily_cap_usd, spent_today_usd}}` (never secrets)
- `GET /api/runs` -> `[{run_id, dry_run, recorded: bool, recorded_at|null, status: "pending"|"running"|"complete"|"failed", decision: "PROMOTE"|"REJECT"|null, created_at}]`
- `GET /api/runs/{id}` -> `{run_id, dry_run, recorded, recorded_at, status, error|null, stages: [{name, status: "pending"|"running"|"done"|"failed", started_at|null, ended_at|null}], spend: {total_usd, cap_usd, by_model: {<model_id>: {usd, calls, input_tokens, output_tokens}}, finetune_usd_estimate|null}, sandbox: {operations: int, concurrency_peak: int|null}, verifier: {language: "python"|"sql", code: string|null, selftest: {accepted_gold: int, rejected_corruptions: int, failures: int}|null}}`
- `GET /api/runs/{id}/events` -> Server-Sent Events. `id:` monotonic int (support `Last-Event-ID`). `event:` one of `stage`, `spend`, `log`, `done`. `data:` JSON `{ts, ...}` (`stage`: `{name, status}`, `spend`: same as `spend` above, `log`: `{level, message}`, `done`: `{status, decision}`). Heartbeat comment every 15 s.
- `GET /api/runs/{id}/report` -> the run's `report.json` verbatim (see `docs/fixtures/sample-dry-run-report.json` for the real shape) plus top-level `dry_run`, `recorded`, `recorded_at`. 404 `{error:"report_not_ready"}` until complete.
- `GET /api/runs/{id}/tree` -> `{nodes: [{id, parent_id|null, label, round: int|null, hypothesis: string|null, data_delta: {added_rows: int, families: string[]}|null, dev_score: number|null, cost_usd: number|null, sandbox_image: string|null, selected: bool}]}` built from real sandbox lineage (`(uuid, parent_uuid, label)`) joined with per-round metrics. Never invent nodes.
- `GET /api/runs/{id}/examples?kind=fixed|still_wrong|regressed|all&limit=20` -> `[{task_id, family, heldout_class, question, gold_sql, base_sql, student_sql, teacher_sql, base_ok, student_ok, teacher_ok}]` — only after the run is complete (held-out items are revealed only post-evaluation). `fixed` = student ok & base wrong; `still_wrong` = student wrong; `regressed` = base ok & student wrong.
- `GET /api/replay` -> list of replay bundles (same item shape as `/api/runs`). Default replay content until a real run is exported is the dry-run sample, labelled `dry_run: true`.

## Write endpoints
- `POST /api/runs` header `X-Admin-Token` (constant-time compare; missing/wrong -> 401/403; live runs also require body `approve_spend: true`) body `{pack: "sql", scale: "tiny"|"small"|"full", dry_run: bool, budget_usd?: number, run_id?: string}` -> `202 {run_id}`. `dry_run: true` needs no token but is rate-limited per IP. Runs execute in ONE background worker; a second live run while one is active -> 409.
- `POST /api/runs/{id}/cancel` (admin) -> cancels provider jobs, `{status}`.
- `POST /api/playground` body `{question: string (<=500 chars)}` -> `{results: {teacher|base|student: {available: bool, reason|null, sql|null, verified: bool|null, rows_preview|null, error|null}}, cost_usd, note}`. Per-IP rate limit (429 with `Retry-After`) and the global daily $ cap (`503 {error:"demo_budget_exhausted"}` -> UI says "demo budget exhausted, see replay"). Models that cannot be served are `available:false` with an honest `reason` (e.g. "student serving path not deployed"); never fake a result. SQL is verified by executing it read-only against the demo DB and comparing with a gold answer only when the question matches a known task; otherwise `verified: null`.

## Frontend rules
Four screens: New run, Live run, Experiment tree, Report; plus Replay (default landing, public) and Playground. Dark/light via `prefers-color-scheme` + toggle. Responsive to 400 px. No new numbers computed in the UI beyond formatting: every number comes from the API.
