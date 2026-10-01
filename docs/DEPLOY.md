# Deploying Distillery: frontend on Vercel, backend on Nebius

Layout: the static frontend (`frontend/`) on Vercel, the API (`src/distillery/server`) in a container on
Nebius. Core functionality uses no other cloud.

Status legend: **VERIFIED** = done in this repo. **UNVERIFIED** = written from the code, never run against real
Nebius or Vercel. Check each UNVERIFIED step against the vendors' current docs before relying on it.

## What was verified locally

- `docker build .` succeeds and the container answers `/api/health`, serves the SPA at `/`, and serves the
  sample replay at `/api/replay` (replay-only mode, no keys). VERIFIED (local Docker, 2026-09-30).
- The Docker frontend stage (only `frontend/` present) builds; the same build was reproduced in a temp
  copy of `frontend/` with `npm ci && npm run build`. VERIFIED.
- CORS behaviour (allowed origin echoed, other origins get no header, preflight for `POST` with
  `X-Admin-Token`, exposed headers) is covered by `tests/test_server_security.py`. VERIFIED with the
  FastAPI test client; browser behaviour against a real Vercel origin is UNVERIFIED.

## 1. Backend on Nebius

The image comes from the root `Dockerfile`. It serves the API on `$PORT` (default 8000), runs as a non-root
user, and has a `/api/health` health check. No secrets are baked in.

### 1.1 Build and push the image (UNVERIFIED for Nebius)

```bash
docker build -t distillery:prod .
# tag and push to a registry your Nebius endpoint/VM can pull from (Nebius Container Registry or another).
```

### 1.2 Run it

Either target works; pick one. Both are UNVERIFIED: no Nebius deployment was made from this repo.

- **Nebius Serverless Endpoints (UNVERIFIED):** create an endpoint from the pushed image, container port
  `8000`, environment variables and secrets as in 1.3, health check path `/api/health`. Confirm in the Nebius
  console that it gives you a public HTTPS URL and how it injects `PORT`.
- **Nebius VM (UNVERIFIED):** install Docker, then

  ```bash
  docker run -d --restart unless-stopped -p 127.0.0.1:8000:8000 \
    -v distillery-data:/data --env-file /etc/distillery.env distillery:prod
  ```

  and put a TLS-terminating reverse proxy (Caddy or nginx with a certificate for your domain) in front of it.
  For SSE (`/api/runs/<id>/events`) turn response buffering off in the proxy.

### 1.3 Environment variables (set as secrets where sensitive)

| Variable | Value |
| --- | --- |
| `DISTILLERY_ALLOWED_ORIGINS` | the exact Vercel **production** URL, e.g. `https://your-app.vercel.app` |
| `DISTILLERY_ADMIN_TOKEN` | a long random string; required for anything that spends money. Keep it in the secret store only |
| `NEBIUS_API_KEY`, `NEBIUS_BASE_URL`, `NEBIUS_AI_PROJECT` | Token Factory and sandboxes. Without the key the server starts in `replay-only` mode |
| `DISTILLERY_RUN_CAP_USD`, `DISTILLERY_PROJECT_CAP_USD`, `DISTILLERY_PLAYGROUND_DAILY_CAP_USD` | budget caps in USD; set them deliberately before exposing the endpoint |
| `DISTILLERY_PRICES_FILE` | path (inside the container) to the price table, e.g. `/app/deploy/prices.json`; without it live runs have no price table. See "Price file" below |
| `DISTILLERY_HOME` | `/data`, on a persistent volume |
| `DISTILLERY_TRUSTED_PROXIES` | the address(es) of your reverse proxy, so per-IP rate limits use `X-Forwarded-For`. Default: header ignored |

`DISTILLERY_ALLOWED_ORIGINS` is a comma-separated list of exact origins: no wildcards, no paths, `https`
required except for `localhost`. A bad value stops the server at startup. `DISTILLERY_DEV_ORIGIN` still works
as a deprecated alias, only when the new variable is unset. Credentials are never allowed cross-origin. The
allowed request headers are `X-Admin-Token`, `Content-Type`, `Last-Event-ID`; the exposed response headers are
`X-Examples-Available`, `X-Examples-Cap-Per-Kind`, `X-Examples-Totals`, `Retry-After`.

Allow only the **production** Vercel origin. Preview deployments get a new URL each time, so they cannot use
the live backend; that is intended. Do not add wildcards to work around it.

### Price file

`deploy/prices.json` is committed and contains no secrets, so the container can read it. **Its numbers are
ASSUMED UPPER BOUNDS for the budget guard, not console prices** (each entry says so in `source`). Before
trusting any cost figure, replace them with the prices shown in the Nebius console. Every entry needs
`source` and `date`. Beyond the flat per-model LLM entries (`input_per_mtok`, `output_per_mtok`) there are two
optional typed sections:

```json
"finetune": {"Qwen/Qwen3-0.6B": {"usd_per_mtok_trained_tokens": <console>, "source": "...", "date": "..."}},
"sandbox": {"usd_per_cpu_second": <console>, "source": "...", "date": "..."}
```

Without a `finetune` price the fine-tune is costed at the operator ceiling and labelled `ceiling estimate`;
without a `sandbox` price sandbox compute stays "unavailable". Nothing is ever priced at zero silently. With
prices present, reports say `billed-basis (measured tokens x console price)`. Check a scale before running
with `python scripts/estimate_run_cost.py --scale mini --prices deploy/prices.json`; recompute past spend with
`python scripts/rebase_ledger.py --root .distillery/live --prices deploy/prices.json` (report-only; `--apply`
appends corrective rows).

### 1.4 HTTPS is required

A page served from Vercel over HTTPS cannot call a plain-HTTP API: browsers block it as mixed content. The
backend URL you give the frontend must be `https://`.

### 1.5 Check it (UNVERIFIED against your deployment)

```bash
curl -s https://<api-origin>/api/health                  # {"ok":true,"mode":...}
curl -si -H "Origin: https://<your-app>.vercel.app" https://<api-origin>/api/health | grep -i access-control
curl -si -H "Origin: https://evil.example" https://<api-origin>/api/health | grep -ci access-control   # 0
```

## 2. Frontend on Vercel (UNVERIFIED; no Vercel project was created from this repo)

1. Import the repository in Vercel.
2. Project settings: **Root Directory** `frontend`, **Install Command** `npm ci`, **Build Command**
   `npm run build`, **Output Directory** `dist`. These are also in `frontend/vercel.json`.
3. Environment variable `VITE_API_BASE` = `https://<api-origin>` (no trailing slash), for the Production
   environment. It is public: it goes into the JS bundle. **Never put the admin token, an API key, or any
   secret in a `VITE_*` variable.** Users type the admin token into the page; it is kept in memory only.
4. Edit `frontend/vercel.json`: in the `Content-Security-Policy` header replace
   `https://REPLACE-WITH-API-ORIGIN.example` in `connect-src` with the same `https://<api-origin>`. JSON has no
   comments, so the guidance lives here. If the API origin is missing from `connect-src`, the browser blocks
   every API call and the page shows "Cannot reach the API server."
   - `script-src` contains the sha256 of the inline theme script in `frontend/index.html`. If you edit that
     script, recompute the hash (base64 of the SHA-256 of the text between `<script>` and `</script>`),
     otherwise the browser blocks it and the theme flashes.
   - `style-src` and `font-src` allow Google Fonts, which `index.html` loads. `'unsafe-inline'` for styles is
     needed for React inline `style` attributes.
5. Deploy. Then set `DISTILLERY_ALLOWED_ORIGINS` on the backend to the Production URL (1.3), including any custom
   domain you attach later.
6. `vercel.json` also: rewrites non-file routes to `/index.html` (client-side routing), caches `/assets/*` as
   immutable for a year, and serves everything else `no-cache`. Security headers: `X-Content-Type-Options`,
   `Referrer-Policy`, `X-Frame-Options`, `Permissions-Policy`, CSP. No secrets are in this file.

## 3. What the page must show

The deployed UI must keep the honesty labels: sample and replay data is shown as **recorded** or **dry run**
(fake models, fake prices, no quality claim), never as a live result, and a missing flag is shown as "Label
unknown". After deploying, open the replay screen and the sample report and confirm the labels are visible. Do
not present replay-only mode as a real run.

## 4. Rollback

- **Frontend:** in Vercel, promote the previous production deployment (or `vercel rollback`). No backend
  change is needed unless the API contract changed.
- **Backend:** redeploy the previous image tag (keep tags immutable, do not deploy `latest`). Data lives in the
  `/data` volume and is not touched by an image change.
- **Emergency stop for spending:** unset `DISTILLERY_ADMIN_TOKEN` or `NEBIUS_API_KEY` and restart; the server
  falls back to replay-only or refuses live runs. Rotate the admin token if it may have leaked.
