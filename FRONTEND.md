# FRONTEND.md: Distillery for the frontend collaborator (and their Claude)

Read this first. It explains what the system is, what the UI is allowed to say, where the data comes from, and how not to break the guarantees. Everything here was checked against the repo on 2026-10-22. When this file and the code disagree, the code wins; fix this file.

## 1. What the product is

Distillery is an autonomous agent that distils a narrow, repetitive job (today: text-to-SQL over a synthetic SQLite database) from a big Nemotron model (the *teacher*) into a small fine-tuned model (the *student*, `Qwen/Qwen3-1.7B` + LoRA). Every training example and every evaluation is executed and verified in Nebius Token Factory Sandboxes. A pure-Python **gate** (no LLM) decides PROMOTE or REJECT. A **certified router** then serves the student only where its error rate is certified, and defers everything else to the teacher. A **drift guard** rolls the gateway back to the teacher-only mode on its own.

Pipeline (see `README.md` and `docs/ARCHITECTURE.md` for detail):

```
generate tasks -> cross-check gold -> verifier self-test -> split + seal held-out
  -> teacher data (execution-verified) -> fine-tune student -> dev eval -> failure analysis
  -> next round (a sandbox branch) -> sealed held-out scored once -> GATE: PROMOTE / REJECT
  -> certified router -> gateway (student if certified, else teacher) -> drift guard -> receipts
```

The real results are mixed, and the UI must keep showing them honestly (`README.md`, Results table):
- The student **PROMOTEd on templated questions** (P4) and **REJECTed on naturally phrased ones** (Gate B, N1).
- The router serves the student on 49% of sealed-test traffic with a certified selective error; on natural phrasing it mostly defers to the teacher.
- The student's **serving cost per request is not measured**. The UI must not show a per-request cost or saving for the student beyond what the API returns (it labels the basis).

## 2. Architecture in one picture

```
Browser (this SPA)  --REST /api/*, SSE-->  FastAPI (src/distillery/server/app.py)
                                              |-- reader.py: derives progress from the SQLite store
                                              |-- gateway.py: POST /v1/chat/completions (OpenAI-compatible)
                                              `-- replay.py: recorded real runs baked into the image (replay/)
```

- One process, one SQLite store, one background worker. No queue, no auth service.
- The built SPA is served by the same FastAPI image (`GET /{path:path}` fallback), so production is **same-origin**: no CORS, `VITE_API_BASE` stays empty. Dockerfile stage 1 builds `frontend/`, stage 2 copies `frontend/dist` into the Python image.
- Production: https://distillery.onrender.com (Render free plan, manual deploys, commit shown in `DECISIONS.md`). Locally: backend on :8000, Vite dev server proxies `/api` to it.

## 3. Run it

```bash
cd frontend && npm install
npm run dev:mock     # no backend needed: dev-only mock API (every response labelled dry_run: true)
npm run dev          # needs the backend: from the repo root `uv run python -m distillery serve --port 8000`
npm test             # vitest (jsdom): 43 files, 435 tests passed on 2026-10-22
npm run typecheck    # tsc --noEmit: clean on 2026-10-22
npm run build        # tsc on tsconfig.build.json, then vite build -> frontend/dist
npm run e2e          # Playwright (uses system Chrome; see playwright.config.ts)
```

Full offline backend for the UI (fake models, labelled DRY RUN; it proves plumbing, not quality): `uv run python -m distillery run --pack sql --scale tiny --dry-run`. The backend needs no key for replay-only mode (stored runs). Gateway and playground need `NEBIUS_API_KEY`; do not ask for it, use mock or replay.

Stack: React 18, Vite 5, TypeScript, react-router-dom 6, Radix (dialog, tabs, tooltip), hand-written SVG charts. No state library, no CSS framework. Dark is the default theme, light is a toggle (`data-theme` on `<html>`, set before paint in `index.html`; tokens in `src/styles.css`).

## 4. Routes and screens (`src/App.tsx`, `src/screens/*`)

| Route | Screen dir | What it shows | Main endpoints |
|---|---|---|---|
| `/` | `home` | Product overview: the pitch, gateway status and drift state, headline evidence | `/api/config`, `/api/gateway/status`, `/api/gateway/drift`, `/api/evidence` |
| `/mission` | `replay` | How it works, proof ladder, stored (recorded) runs, limits | `/api/replay`, `/api/runs`, `/api/runs/{id}/report` |
| `/runs/:id` | `live` | Live run: stage timeline, spend, ledger, sandbox, verifier, event log, heartbeat (SSE) | `/api/runs/{id}`, `/api/runs/{id}/events`, `/api/runs/{id}/health` |
| `/runs/:id/tree` | `tree` | Experiment tree drawn from real sandbox-branch lineage, pan/zoom | `/api/runs/{id}/tree`, `/experiments` |
| `/runs/:id/report` | `report` | Gate statistics, accuracy tables, charts, per-item examples | `/api/runs/{id}/report`, `/examples` |
| `/new` | `newrun` | Start a run (locked on the public demo: admin token required) | `POST /api/runs` |
| `/playground` | `playground` | "Acme analyst": ask a question, see who answered (student/teacher), SQL, rows, cost, savings, drift panel | `POST /api/acme/ask`, `/api/gateway/status|savings|drift`, `/api/config` |
| `/receipts` | `receipts` | Every claim: value, artifact link, hashes, pre-registration commit | `/api/evidence` |
| `/humanset` | `humanset` | Human-review tool for natural-question drafts | `/api/humanset/drafts`, `POST /api/humanset/decide` |

Shared code: `src/api/` (typed client, SSE hook, `format.ts`, `types.ts`), `src/charts/`, `src/components/` (Shell, RunShell, banners), `src/motion/` (reduced-motion aware), `src/lib/`, `src/testutil/` (contract fixtures, provenance helper). The authoritative API description is `docs/API_CONTRACT.md` (response shapes, error shapes, SSE event types). Read it before touching any fetch.

## 5. The truthfulness rules (non-negotiable, enforced by tests)

This project's whole value is that numbers are real. The UI must never invent, round up, or hide anything. Rules (also in `frontend/README.md`):

1. **Every number, label and string comes from a backend response.** No literals, no `Math.random`, no placeholder text. `src/truth.test.ts` greps the source.
2. **Absent data is shown as absent.** `null`/`undefined`/`NaN` render as `not measured` (via `src/api/format.ts`), never `0`, `0%`, `$0.00`, blank or `-`.
3. **No arithmetic on backend numbers** beyond display rounding (and x100 for percentages). Savings, spend and rates are computed server-side.
4. **Errors are not empty states.** Every screen distinguishes loading, error (show the API `code` + message) and empty.
5. **Label banners are mandatory** on every screen showing run or report data: "DRY RUN — fake models, numbers are NOT results" when `dry_run` is true; the "Recorded run" banner when `recorded` is true; "Label unknown" when the flag is missing. A missing flag is never rendered as a real run.
6. **Live screen:** if the SSE stream disconnects or is silent for 30 s, show "Disconnected / stale" and mark values as last known. Reconnect with `?last_event_id=`. `observed_at` is "seen", never "when it happened"; only `stage.at` is an event time. Never compute durations.
7. **The production bundle contains no mock, no `synthetic`, no `docs/fixtures` values.** `truth.test.ts` builds without `VITE_MOCK` and inspects the output.
8. **Provenance:** `src/screens/provenance.test.tsx` renders every screen from real API shapes and checks every visible number against the payload. A number that is not in a payload must be allow-listed in that test with a reason.
9. **Pending means pending.** Anything without an evidence file shows "pending: not yet measured". Do not "fill in" a number from the docs. Allowed numbers live in `docs/proofs/evidence/*.json` and the README Results table.
10. **Never put a secret in the bundle.** Anything `VITE_*` ends up in public JS. The admin token is typed by the operator at runtime and sent as `X-Admin-Token` only by the typed client (`src/api/client.ts`, admin calls); it is held in memory only (a module variable) and is not stored in source, env or browser storage.

## 6. API contract and how drift is caught

`src/testutil/contract/*.json` are real responses of the FastAPI app, written by `tests/test_frontend_truth.py` at the repo root. CI fails if the server's key sets or value types drift from those files, and the frontend tests render screens from them. After an **intentional** backend API change:

```bash
npm run contract:update     # runs the Python test with UPDATE_CONTRACT=1 (needs ../.venv)
```

Review and commit the JSON diff. If a frontend change needs a new field, the backend owner changes `docs/API_CONTRACT.md` and the server first; do not guess shapes. `mock/plugin.ts` (dev only) must keep covering every endpoint in the contract and must label everything `dry_run: true`.

## 7. The backend surface the UI may rely on

Public, read-only: `GET /api/health`, `/api/config`, `/api/runs`, `/api/runs/{id}` (+ `/events` SSE, `/health`, `/report`, `/tree`, `/examples`, `/experiments`, `/audit`), `/api/replay`, `/api/evidence`, `/api/gateway/status|savings|drift`. Spending or privileged: `POST /api/runs` (202; admin token required, **returns 401 on the public demo by design**), `POST /api/runs/{id}/cancel`, `POST /api/playground` and `POST /api/acme/ask` (public but capped per address and per day; they spend real money), `POST /v1/chat/completions` (OpenAI-compatible gateway; no streaming, no tools: `400`). Error shape on `/api`: see the contract; on `/v1` it is the OpenAI shape.

Because the playground spends money, the hosted daily cap is $0.02 (about 40 teacher calls). Do not add auto-refresh, polling loops or "try all examples" buttons that call paid endpoints. Student answers take roughly 30-70 s (sandbox CPU), so the UI needs a patient waiting state with elapsed time and must not time out early. The first request after the free host sleeps can take about a minute.

## 8. Do and do not

Do:
- Keep the tests green (`npm test`, `npm run typecheck`, `npm run build`) and add a test for any new screen state, including a provenance case.
- Keep accessibility: skip link, per-route document titles, focus to `<main>` after navigation, reduced-motion respect, axe checks in e2e.
- Show who answered (student vs teacher), the confidence against the threshold, and cost, as the playground does. This is the product's core claim.

Do not:
- Hard-code any metric, price, count, model id or date. Read them from the API (`/api/config` has the model ids).
- Change the meaning of PROMOTE, REJECT, "pending", or the banners.
- Add a UI path that starts a paid run from the public site, or that exposes the admin token.
- Touch `deploy/`, `replay/`, `docs/proofs/evidence/` or the Python side as part of a UI change. Those are evidence and shipped artifacts; ask the backend owner.
- Promise anything the README lists under Limitations (the win is in-distribution; the database is synthetic; student serving is slow; one run per result).

## 9. Backend map (for context, not for UI work)

`src/distillery/`: `orchestrator.py` (stage machine), `gate.py` + `stats.py` (decision), `evaluator.py` (only module allowed to read the sealed held-out set), `router*.py` (certified router), `drift.py`, `server/` (API, gateway, SSE, replay), `taskpacks/sql/` (schema, questions, verifier). Docs: `README.md`, `docs/ARCHITECTURE.md`, `docs/API_CONTRACT.md`, `docs/DEPLOY.md`, `DECISIONS.md` (pre-registrations and results), `docs/proofs/evidence/` (the evidence files).

## 10. Open items the frontend owner should know

- The demo video link is not in the README yet; add it when the video is public (owner: the repo author).
- Submission deadline: Devpost 2026-10-30 10:00 PT (internal date 2026-10-27). Judging runs 2026-12-01 to 12-15, so the hosted site must stay up and stable.
