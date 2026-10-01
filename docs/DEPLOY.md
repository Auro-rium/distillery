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
