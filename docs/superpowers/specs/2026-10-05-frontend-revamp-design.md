# Plan: Distillery frontend revamp ("mission control"), a winnable hosted demo

(The earlier "stop building, start proving" plan is complete through P4/P5-local/P5.6-read; its record lives in
GROUND_TRUTH.md / STATUS.md / DECISIONS.md. This plan supersedes this file.)

## Context
The system is now proven: P4 PROMOTE on the sealed held-out (student Qwen3-1.7B 92.3% vs teacher Nemotron-3-Super
90.0% vs base 25.7%, ratio CI [0.993, 1.061], McNemar p 1.6e-57, $4.26), kill-mid-fine-tune adopted the job, and the
hosted site serves the run. But the site does not sell it: `/` opens on a generic headline ("Teach a small model one
task...") and a static diagram, and the PROMOTE run sits below the fold as one of three cards. The fine-tune record
(loss curve, trained tokens), the job ledger (started/adopted/closed) and the proof ladder (P1.11 64/64, P1.12
142/150, P3 power, P2 labels) are never shown. Audience (user): judges skimming, live demo, and technical reviewers,
all three. Decisions (user): full redesign, fast (days, not 2 weeks), dark "mission control" look, rebuild in place.

Success = a first-time visitor understands "a 1.7B student matched its 70B-class teacher on this task, proven on a
sealed set, for $4" within ~10 s; can drill from any headline number to its evidence in <= 2 clicks; the live
control room looks alive during long stages; every number still comes from the API (existing guards stay green).

## Non-negotiables (keep)
- Data layer as is: `frontend/src/api/client.ts`, `types.ts`, `useSSE.ts` (heartbeat liveness), `format.ts`.
- Guards stay green and are extended, never loosened without a stated reason: `src/truth.test.ts` (no hardcoded
  numbers/random/fixtures in shipped code), `screens/provenance.test.tsx` (every on-screen number matches a contract
  payload), `screens/honesty.test.tsx` (exact dry/recorded/unknown labels, stale banner, API errors never look empty),
  Playwright e2e matrix (3 widths x 2 schemes + axe).
- No new runtime dependencies (plain CSS tokens, Radix primitives already present, HTML/SVG charts).
- Sealed held-out text is never added anywhere new (the report examples list stays the existing capped design).

## Information architecture
Nav: **Mission** (`/`) · **Evidence** (`/runs/:featured/report`) · **Live** (`/runs/:id`, latest active or featured) ·
**Playground** · **New run** (admin). GitHub link right. "Featured run" = newest recorded non-dry PROMOTE run from
`/api/replay` (fallback: newest recorded run), computed client-side; no hardcoded run id.

1. **Mission `/`** (replaces `screens/replay/index.tsx` layout; reuse its data hooks)
   - Hero verdict: PROMOTE badge + one static sentence (no digits) + three big CountUp numerals student/teacher/base
     (from report `evaluation.gate`) + a ratio-CI number line with the threshold tick (`gate.thresholds`), n, McNemar.
   - Proof ladder W1-W5: cards fed by new `GET /api/evidence` (P1.11 overfit, P1.12 pilot, P2 data, P3 gate power,
     P4 run, P5 kill/adopt), each with PASS state, its key number(s), and a link to the evidence file on GitHub.
   - How it works: existing `Pipeline.tsx` restyled + animated flow; per-stage durations of the featured run from its
     `events.json` (served via existing run/replay payloads).
   - Training: loss curve (see Graphs) + trained tokens, steps, config chips.
   - Limits & cost (one section): stress set accuracies (unseen families) + in-distribution overlap rate; run total
     + basis + fine-tune $/1M tokens (report `cost`, `finetune_lines`).
   - CTAs: "Open the evidence" (featured report), "Watch a live demo" (existing dry DemoButton, label kept),
     "Start a live run" (admin), GitHub.
   - Below: stored runs + live runs lists (existing components, restyled, compact).
2. **Live control room `/runs/:id`** (rebuild `screens/live/*` layout; same SSE data)
   - Left stage rail with status + elapsed per stage; centre terminal-style log; right spend meter (billed vs ceiling
     split already in the API) + **job ledger** (new `GET /api/runs/:id/experiments`: finetune_job_started/adopted/
     closed, serving images) so the kill->adopt story is visible; header pulse "LIVE" driven by heartbeat, stale banner
     unchanged.
3. **Evidence dossier `/runs/:id/report`** (rebuild `screens/report/*` layout; keep sections' data logic)
   - Sticky verdict header; Gate: ratio CI number line + threshold, discordant pairs (student-only vs base-only) as a
     2-cell matrix, accuracy bars (existing AccuracyChart restyled); Stress; **Fine-tune card** (new: trained tokens,
     steps, hyperparameters, loss-curve sparkline from `report.finetune[].loss_curve`); Artifact IDs with copy;
     Cost lines with basis; Examples diff viewer (restyle); Drops/errors.
4. Tree, New run, Playground: restyle only (tokens), no logic change.

## Graphs (user: "add graph, loss curve, cards, not too much clutter")
Plain SVG, axes + labels, values from the API only, hover tooltips, colour = model (student green, teacher amber,
base cyan), every chart has a one-line caption saying what it measures:
1. **Accuracy bars** (held-out: base/student/teacher with CI whiskers): Mission hero + dossier (restyled AccuracyChart).
2. **Ratio CI number line** (student/teacher, point + 95% interval, threshold tick from `gate.thresholds`): Mission
   hero + dossier. This is the single "why PROMOTE" graphic.
3. **Loss curve** (`LossChart`): train + validation loss vs step from `report.finetune[0].loss_curve` (P4 has 3
   checkpoints: train 0.094 -> 0.042, valid 0.353 -> 0.550). Markers per checkpoint, no smoothing, honest caption:
   validation loss is token-level vs gold SQL text while the student trained on teacher SQL; execution accuracy is
   the gate metric (dev 0.907, held-out 0.923), labelled as interpretation. Shown on Mission ("Training") + dossier.
   P1.11's 10-point overfit curve appears as a sparkline in its proof card (from `/api/evidence`).
4. **Stress bars** (unseen families, student vs teacher): Mission "Limits" + dossier.
5. **Discordant matrix** (student-only 202 vs base-only 2): dossier only.

**Clutter budget:** Mission = hero + 4 sections max (Proof ladder, Training, How it works, Limits & cost), at most
6 proof cards and 3 charts; one primary number per card, details behind "Evidence ->" links or a collapsed
"details" row; generous spacing, no more than 2 columns on desktop. Dossier keeps all sections but groups them under
3 tabs: Verdict (gate, CI, accuracy, discordant), Training (loss, hyperparameters, artifact, cost), Examples & errors.

## Design system ("mission control")
- Dark default, light kept (e2e runs both schemes; toggle remains). New tokens in `frontend/src/styles.css`:
  near-black panels, hairline grid, phosphor green = pass/student, amber = warn/teacher, red = fail, cyan = info/base;
  mono tabular numerals for every metric. Fonts via Google Fonts: Space Grotesk (headings), IBM Plex Sans/Mono (kept).
- Restyle `frontend/src/ui/*` primitives (Card -> panel with header strip, Badge -> status chip, Stat -> big numeral).
- New small components (HTML/SVG, no deps): `CiNumberLine`, `DiscordantMatrix`, `Sparkline`, `ProofCard`,
  `JobLedger`, `StageRail`. Motion: existing `motion/CountUp`, `useReducedMotion` respected.

## Backend additions (small, tested)
- `scripts/build_evidence.py` -> `deploy/evidence.json`: curated numbers + file paths + commit-pinned GitHub URLs
  from `docs/proofs/evidence/{p1_11_overfit,p1_12_decision,p2_data,p3_calibrate,p4_run_summary,p5_live}.json`
  (numbers only; no sealed text). Shipped by the existing `COPY deploy/`.
- `GET /api/evidence` (serves that file) and `GET /api/runs/{id}/experiments` (read-only rows from `index.sqlite`;
  recorded runs: from the bundle if exported, else empty) in `src/distillery/server/app.py`; types in `api/types.ts`;
  contract fixtures regenerated with `npm run contract:update`.

## Execution (after approval)
1. Save this design as `docs/superpowers/specs/2026-10-05-frontend-revamp-design.md`, then writing-plans.
2. Foundation first, sequential: tokens + fonts + ui primitives + nav/Shell + backend endpoints + contract update.
3. Then parallel subagents, one per screen dir (Mission, Live, Report; Tree/New/Playground restyle), each with its own
   tests; I review and integrate (SDD), one commit per screen, `feat(ui): ...` (freeze waived by the user for this).
4. Re-export the featured replay bundle if new fields are needed; push to main (Render redeploys) with approval.

## Verification
- `npm run typecheck`, `npx vitest run` (truth/provenance/honesty green), `npm run build`, Playwright e2e matrix
  with axe; backend `pytest -q`, `ruff`, `mypy`.
- Manual: local `distillery serve` with `replay/`, then hosted after push: fresh browser, the 10-second test on
  `/` (verdict + 3 numerals + proof ladder visible above the fold at 1366x768 and 390x844), every headline number links
  to evidence in <= 2 clicks, live control room shows LIVE pulse during a dry run, screenshots to
  `docs/proofs/evidence/hosted/`.
