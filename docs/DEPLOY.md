# Deploying Distillery: one service on Render

The Docker image (see `Dockerfile`) contains the API and the built frontend, so one Render web service serves the whole app from one origin: no CORS, no second host, `VITE_API_BASE` stays empty. Model inference, fine-tuning and sandbox work run on Nebius Token Factory; the web service only calls them.

Why Render and not Nebius: the hackathon credit is for Token Factory only. Nebius AI Cloud (serverless endpoints, VMs, registries) would bill a separate account, and Token Factory itself cannot host a web service.

## Deploy
1. Render dashboard -> New -> Blueprint -> connect the GitHub repo. It reads `render.yaml` (service `distillery-api`, Docker, free plan, health check `/api/health`).
2. Set the secrets Render prompts for: `NEBIUS_API_KEY`, `NEBIUS_AI_PROJECT`. Secrets live only in Render.
3. **Do not set `DISTILLERY_ADMIN_TOKEN`.** Without it every New run attempt is refused (401), which is what a public demo wants. Live runs are started from a trusted machine with the CLI (`python -m distillery run ...`), not from the website.
4. Check `https://<service>.onrender.com/api/health` (`"mode":"live"`), `/api/runs` (both recorded runs listed, `recorded: true`) and `/api/config` (models set, playground teacher `ok`).

## Dashboard checklist: the certified router (2026-10-16)
The live service `distillery` takes its environment from the dashboard, which overrides both `render.yaml` and the image defaults. Before the Manual Deploy that ships `deploy/router` and `deploy/student`, check Settings -> Environment (names and values only, no secret is read here):

| Variable | Set to | Why |
|---|---|---|
| `DISTILLERY_PLAYGROUND_DAILY_CAP_USD` | `0.02` (or delete it: the image default is `0.02`) | the live service reported `0.5`, which over the judged window could spend far more than the $1.50 hosted reservation |
| `DISTILLERY_MODEL_STUDENT` | `Qwen/Qwen3-1.7B` (or delete it) | the live service reported `Qwen/Qwen3-0.6B`; the shipped checkpoint fixes its own base either way, but the dashboard should not disagree |
| `DISTILLERY_PLAYGROUND_SANDBOX_MODELS` | delete it, or `1` | the image default `1` turns the student on; a dashboard `0` would keep it off |
| `DISTILLERY_ROUTER_DIR`, `DISTILLERY_PLAYGROUND_STUDENT_DIR`, `DISTILLERY_GATEWAY_STUDENT_DAILY_MAX` | delete them if present | image defaults: `/app/deploy/router`, `/app/deploy/student`, `60` |
| `DISTILLERY_GLOBAL_LEDGER` | `none` | unchanged |
| `NEBIUS_AI_PROJECT` | must exist (secret; check the name only) | without it the sandbox student cannot start and every request goes to the teacher |
| `DISTILLERY_DRIFT_*` | must NOT be set | they override `deploy/router/drift.json` (the pre-registered W 20, τ 0.90); a half-set group stops the server |

Then Manual Deploy -> "Deploy latest commit", and `uv run python scripts/smoke_live.py https://distillery.onrender.com --no-run`: it fails if the student model, the daily cap, the certificate or the checkpoint pin is wrong.

**After the deploy, in this order:** `smoke_live.py` (its playground call also warms the sandbox serving image), then the hosted check before any other traffic: `uv run python scripts/w_e2e.py --base-url https://distillery.onrender.com --hosted --prereg-commit 705876d --out docs/proofs/evidence/w_e2e_hosted.json` (about $0.01 of that day's $0.02 hosted cap), then record the deployed commit in `DECISIONS.md`.

**Drift rollback on the hosted demo.** The guard rolls the gateway back to teacher-only when 18 of the last 20 routed requests went to the teacher (τ 0.90, W 20, pre-registered). Visitors mostly type naturally phrased questions, and on calibration the router defers about 93% of those, so a run of natural questions is likely to trip it: that is the guard doing its job (the certificate does not cover that traffic), and it is safe, but afterwards the student serves nothing. There is no automatic return; re-certifying needs the admin token, which the public service does not have. The state lives in the service's store under `/data`, which is the free plan's ephemeral disk, so **Restart service** (or a redeploy) in the dashboard should bring a fresh instance back to CERTIFIED; check `GET /api/gateway/drift` afterwards. Changing the drift parameters is possible but is a change to pre-registered values and must be disclosed.

## Settings (all in `render.yaml` except the two secrets)
| Variable | Value |
|---|---|
| `NEBIUS_BASE_URL` | `https://api.tokenfactory.nebius.com/v1/` |
| `DISTILLERY_PRICES_FILE` | `/app/deploy/prices.json` (LLM prices read from the API's own listing; no fine-tune or sandbox price yet) |
| `DISTILLERY_MODEL_PLANNER` / `_TEACHER` / `_TRIAGE` / `_STUDENT` | the four model ids in `render.yaml` |
| `DISTILLERY_PROJECT_CAP_USD`, `DISTILLERY_RUN_CAP_USD` | spend caps |
| `DISTILLERY_PLAYGROUND_DAILY_CAP_USD` | daily ceiling on public playground teacher calls |
| `DISTILLERY_GLOBAL_LEDGER` | `none` (see "Global spend ledger" below) |

## Global spend ledger (laptop) and why Render turns it off
On a trusted machine every spend (all `.distillery/*` store copies, every pipeline child process and the local playground) is also written to one SQLite file, `~/.distillery/global_ledger.sqlite` (override with `DISTILLERY_GLOBAL_LEDGER=<path>`). The project cap (`DISTILLERY_PROJECT_CAP_USD`) is checked against that file's committed total (spend + active reservations), read fresh on every check, so it holds across store copies and processes. `python -m distillery ledger status` prints the total, cap, headroom and reservations; `ledger seed --file deploy/ledger_seed.json` loads the verified historical spend once (idempotent).

**Render sets `DISTILLERY_GLOBAL_LEDGER=none`** (in `render.yaml` and as an image default in the `Dockerfile`; because the live service may not follow the blueprint, also check Settings -> Environment in the dashboard). The hosted service has an ephemeral disk and cannot see the laptop's ledger, and the laptop cannot see hosted spend. The hosted playground therefore relies on its own caps (`DISTILLERY_PLAYGROUND_DAILY_CAP_USD`, `DISTILLERY_PROJECT_CAP_USD`, the per-IP limits). The money it may spend is accounted for the other way round: the $1.50 hosted-demo allocation (Oct 30 - Dec 15 2026) is a permanent reservation (`hosted-demo`) in the laptop ledger, so it is already counted against the project cap. **Hosted daily cap: $0.02.** The site is live from now until judging ends, so at the cap the hosted playground spends at most 64 days (Oct 12 - Dec 15) x $0.02 = $1.28, inside the $1.50 reservation (whose note names the judged window, Oct 30 - Dec 15). Hence `render.yaml` sets `DISTILLERY_PLAYGROUND_DAILY_CAP_USD=0.02` (about 40 teacher calls a day). The counter lives on the ephemeral disk and resets when the instance restarts, so the worst case is $0.02 x (days + restarts); keep-alive keeps restarts rare. Set the same value in the dashboard if the live service does not follow the blueprint. **Limit of the laptop check:** `Ledger.preflight` reads a fresh committed total but does not hold a reservation between the check and the spend, so processes running at the same moment can each pass and overshoot by at most one call's estimate per process; `GlobalLedger.reserve` (atomic, tested with 8 processes) is the primitive for paid steps that must be held. Paid steps are run one at a time after approval. With the variable set to `none`, `off` or an empty string the ledger behaves exactly as before (per-store totals). Dry runs never write to the global ledger.

## Limits of the free plan
The service sleeps when idle (the first request after a sleep takes about a minute) and its disk is ephemeral: the playground spend counter resets on restart, so the daily cap is a soft guard (teacher calls cost about $0.0005 each). Recorded runs are baked into the image (`replay/`), so they survive restarts. To refresh them after a new run: `python -m distillery export-replay ...`, commit `replay/`, push (auto-deploy). A later round that reuses an earlier run id (round N1 reuses P4's) is exported with `--bundle-id <new id>`: the bundle is listed and labelled under that id, the run id stays in `manifest.source_run_id`, and an existing bundle that holds a different trained artifact is never overwritten.

## Unattended runs on the free plan (plan A4)
- **Auto-deploy is off.** `render.yaml` sets `autoDeploy: false`, so a push cannot restart the instance under a running pipeline. Deploys are manual (dashboard -> Manual Deploy). **The live Render service is named `distillery` (`srv-dauuokk1nsns73fqcv70`), while `render.yaml` names the blueprint service `distillery-api`, so the blueprint may not govern the live service: also set Settings -> Build & Deploy -> Auto-Deploy to Off in the dashboard by hand.** (Not done from here: no Render API call was made.)
- **Keep-alive.** `.github/workflows/keepalive.yml` calls `GET /api/health` every 10 minutes (`workflow_dispatch` too) so the free instance does not spin down mid-run. The URL is the repository variable `DISTILLERY_HEALTH_URL`, default `https://distillery.onrender.com/api/health`. GitHub schedules are best effort (runs can be delayed or skipped).
- **Supervisor in-process.** The worker thread supervises the CLI child inside the server (Dockerfile CMD unchanged): exit 75 or death by signal restarts it, SIGTERM on shutdown suspends without cancelling paid jobs, and `create_app` re-submits unfinished runs on start. Every restart, start and cancel is in `GET /api/runs/{id}/audit`.
- **Stated limit (honest scope of the autonomy claim).** The free instance's disk is wiped when the **instance** restarts. Autonomy is therefore proven for worker crashes, provider and network faults and sandbox failures **within one instance lifetime**. A host restart is out of scope unless a persistent disk is added later.
- **Memory headroom.** The free instance has 512 MB. A local dry run (`distillery run --dry-run --scale tiny`) under `/usr/bin/time -v` is recorded in `docs/proofs/evidence/a4_memory.json` (max RSS vs 512 MB). A dry run uses fakes, so it bounds the pipeline's own memory (it does not include the live OpenAI client, only the fake transport).
- **Dashboard actions left to the operator.** Set `DISTILLERY_ADMIN_TOKEN` (and student model / run cap if not Blueprint-managed) in the Render dashboard; this reverses step 3 above, which is needed to start a run from the hosted UI.
- **Chaos injection** (`DISTILLERY_CHAOS`, `src/distillery/chaos.py`) is honoured only for run ids starting `chaos-` (dry rehearsals: `dry-chaos-`); leave it unset on the service except for the A5 proof.

## Observability
- **Logs:** one JSON line per request on stderr (Render's log stream): `{"evt":"http","method","route","status","ms","stream","request_id"}`. The route is the template (`/api/runs/{run_id}`), never the raw path or query string; no headers, bodies, tokens or IPs are logged. Every response has an `X-Request-ID` (a safe inbound id is kept) so a log line can be matched to a browser request.
- **`GET /api/metrics`:** Prometheus text (requests by method/route/status, latency histogram, in-flight, open event streams, playground and run-start counters). **`GET /api/telemetry`:** the same as JSON with p50/p95 bucket bounds, uptime, run counts by status and today's playground spend. In-memory: they reset on restart, and say so.
- **Post-deploy check (free, about $0.0005):** `python scripts/smoke_live.py https://<service>.onrender.com`. It verifies health, models, recorded runs and their labels, that New run is locked, request ids, metrics, the frontend, one playground answer, and that a dry run's values really change (stage events advance, spend rises, it completes). Exit status 1 on any failure.

## Local image check
```bash
docker build -t distillery:prod .
docker run --rm -p 8000:8000 -e PORT=8000 distillery:prod   # replay-only without secrets
curl -s localhost:8000/api/health
```

## Rollback
Render dashboard -> the service -> Events -> redeploy a previous commit.
