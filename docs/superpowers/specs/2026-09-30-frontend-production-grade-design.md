# Frontend production-grade pass (Approach 2 + headless Radix primitives)

Status: approved in chat 2026-09-30 ("yes go with approach 2 and 3 fuse them"). Classified architectural.

## Intent
Judges and visitors see a fast, polished, trustworthy UI: desktop + phone, light + dark, accessible, and honest about
what is real. Screens double as pictures for the tweet. Non-negotiables carried over from `frontend/README.md`
(Truthfulness rules): every number traces to the API payload; null/absent renders "not measured"; errors are not empty
states; DRY RUN / recorded banners come from the payload; no invented content.

## Decisions
- **D1 (approved, option A):** numbers always show their exact final value. Only bars, whiskers, reveals, connectors animate.
  A test samples rendered text at every animation frame and fails if any displayed number differs from the payload.
- **D2:** adopt headless Radix primitives (`@radix-ui/react-tabs|dialog|tooltip|switch|...`) wrapped in a small internal
  kit `src/ui/` styled with existing CSS variables. No Tailwind, no component-library theme. Bundle growth budget: +25 kB gzip.
- **D3:** one `RunShell` (sticky header: run id, status, label banner, verdict; tabs Run | Report | Tree) wraps all run screens.
  Interface (fixed, other work depends on it):
  `RunShell({ runId: string; tab: "live"|"report"|"tree"; run?: RunDetail; children })` in `src/components/RunShell.tsx`.
- **D4:** Live = two columns (stage timeline | spend by model, sandbox counters, verifier, log); Tree = master-detail with larger nodes;
  Report = sticky summary + sections. Wasted-space fixes.
- **D5:** fixes for observed bugs: new-run form label/field wiring; phone nav (no mid-word wraps, collapses to a menu <= 560 px);
  Replay pipeline diagram (no clipped text, no struck-through labels, fits without cropping at 400 px and 1440 px).
- **D6 (safety net):** Playwright (dev-only, using installed Chrome via `channel: "chrome"`) + axe-core + Lighthouse budget.
  Screenshots of every screen at 400/768/1440 px x light/dark saved as CI artifacts; zero axe violations (serious/critical);
  Lighthouse performance/a11y/best-practices budgets in CI.

## Out of scope
Re-theming beyond the existing token system, new backend endpoints, new data. Screens keep the API contract in `docs/API_CONTRACT.md`.

## Verification
`npm run typecheck`, `npm run build`, `npm test` (all existing truth/provenance tests untouched and green), the new e2e/a11y suite,
plus screenshots inspected by a reviewer agent. Not verified without a real run: anything that needs real Nebius data.
