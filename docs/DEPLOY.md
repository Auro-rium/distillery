# Deploying Distillery: one service on Render

The Docker image (see `Dockerfile`) contains the API and the built frontend, so one Render web service serves the whole app from one origin: no CORS, no second host, `VITE_API_BASE` stays empty. Model inference, fine-tuning and sandbox work run on Nebius Token Factory; the web service only calls them.

Why Render and not Nebius: the hackathon credit is for Token Factory only. Nebius AI Cloud (serverless endpoints, VMs, registries) would bill a separate account, and Token Factory itself cannot host a web service.

## Deploy
1. Render dashboard -> New -> Blueprint -> connect the GitHub repo. It reads `render.yaml` (service `distillery-api`, Docker, free plan, health check `/api/health`).
2. Set the secrets Render prompts for: `NEBIUS_API_KEY`, `NEBIUS_AI_PROJECT`. Secrets live only in Render.
3. **Do not set `DISTILLERY_ADMIN_TOKEN`.** Without it every New run attempt is refused (401), which is what a public demo wants. Live runs are started from a trusted machine with the CLI (`python -m distillery run ...`), not from the website.
4. Check `https://<service>.onrender.com/api/health` (`"mode":"live"`), `/api/runs` (both recorded runs listed, `recorded: true`) and `/api/config` (models set, playground teacher `ok`).

## Settings (all in `render.yaml` except the two secrets)
| Variable | Value |
|---|---|
| `NEBIUS_BASE_URL` | `https://api.tokenfactory.nebius.com/v1/` |
| `DISTILLERY_PRICES_FILE` | `/app/deploy/prices.json` (LLM prices read from the API's own listing; no fine-tune or sandbox price yet) |
| `DISTILLERY_MODEL_PLANNER` / `_TEACHER` / `_TRIAGE` / `_STUDENT` | the four model ids in `render.yaml` |
| `DISTILLERY_PROJECT_CAP_USD`, `DISTILLERY_RUN_CAP_USD` | spend caps |
| `DISTILLERY_PLAYGROUND_DAILY_CAP_USD` | daily ceiling on public playground teacher calls |

## Limits of the free plan
The service sleeps when idle (the first request after a sleep takes about a minute) and its disk is ephemeral: the playground spend counter resets on restart, so the daily cap is a soft guard (teacher calls cost about $0.0005 each). Recorded runs are baked into the image (`replay/`), so they survive restarts. To refresh them after a new run: `python -m distillery export-replay ...`, commit `replay/`, push (auto-deploy).

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
