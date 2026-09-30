# Distillery frontend

React + Vite + TypeScript. `npm run dev` (proxies `/api` to :8000), `npm run dev:mock` (dev-only mock API,
never part of a production build), `npm test`, `npm run build`.

## Truthfulness rules

The UI must never lie. Each rule is enforced by a test.

1. Every number, label and string shown comes from the backend response. No literals, no `Math.random`,
   no placeholder text (`src/truth.test.ts` greps the source).
2. Absent data is shown as absent: `null`/`undefined`/`NaN` render as the text `not measured`
   (`src/api/format.ts`), never 0, 0%, $0.00, blank or `-`. Backend strings such as
   `unavailable: ...` are shown verbatim.
3. No arithmetic on backend numbers beyond rounding for display (and x100 for percentages).
4. Errors are not empty states: every screen distinguishes loading / error (API code + message) / empty.
5. Every screen showing run or report data carries the label banner: `DRY RUN — fake models, numbers are NOT
   results` when `dry_run` is true, the recorded banner when `recorded` is true, and "Label unknown" when the
   flag is missing (a missing flag is never rendered as a real run).
6. Live screen: if the SSE stream disconnects or is silent for 30 s it shows `Disconnected / stale` and marks
   the values as last known. Reconnects use `?last_event_id=`. `observed_at` is shown as "seen", never as
   when something happened; only `stage.at` is an event time. Durations are never computed.
7. The production bundle never contains the mock, `synthetic`, `docs/fixtures` or fixture values
   (`src/truth.test.ts` builds without `VITE_MOCK` and inspects the output).
8. Render provenance: `src/screens/provenance.test.tsx` renders every screen from the real API shapes and
   checks each visible number against the payload (`src/testutil/provenance.ts`). Numbers that are not in a
   payload must be allow-listed in the test with a reason.

## API contract fixtures

`src/testutil/contract/*.json` are real responses of the FastAPI app (fakes / dry run), written by
`tests/test_frontend_truth.py`. In CI that Python test fails when the server's key sets or value types drift
from these files; the frontend tests render the screens from them. After an intentional API change:

    npm run contract:update     # runs the Python test with UPDATE_CONTRACT=1 (needs ../.venv)

then review and commit the JSON diff.
