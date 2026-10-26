import { lazy, Suspense, useState } from "react";
import { ApiError, api } from "../../api/client";
import { ErrorState, Spinner } from "../../components";
import { Button } from "../../ui";
import { LazyDialog } from "../../ui/LazyDialog";
import { describeCreateError } from "../newrun/errors";

// The run view is the Live screen itself, loaded only when a demo is started.
const LiveRun = lazy(() => import("../live"));

/**
 * Starts a dry run (fake models, no spend) and shows it right here, in a dialog on the Mission page: the same live
 * view as the run page, without leaving. The run carries its own DRY RUN label; the numbers are not results.
 */
export function DemoButton() {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  async function start() {
    setBusy(true);
    setErr(null);
    try {
      const r = await api.createRun({ pack: "sql", scale: "tiny", dry_run: true });
      setRunId(r.run_id);
    } catch (e) {
      setErr(e instanceof ApiError && e.message ? `${e.message} (HTTP ${e.status})` : describeCreateError(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="demo stack">
      <div className="demo-row">
        <Button variant="primary" type="button" loading={busy} disabled={busy} onClick={start}>Watch a live demo run</Button>
        <span className="muted">Fake models, no spend, results are NOT real (dry run)</span>
      </div>
      {err && <ErrorState title="Could not start the demo run" message={err} />}
      <LazyDialog
        open={runId !== null}
        onOpenChange={(o) => { if (!o) setRunId(null); }}
        title="Demo run"
        description="A dry run: fake models, no spend. Its numbers are NOT results."
        className="demo-dialog"
      >
        {runId && (
          <Suspense fallback={<Spinner label="Loading the run" />}>
            <LiveRun runId={runId} />
          </Suspense>
        )}
      </LazyDialog>
    </div>
  );
}
