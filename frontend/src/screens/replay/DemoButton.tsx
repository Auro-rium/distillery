import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { ApiError, api } from "../../api/client";
import { ErrorState } from "../../components";
import { Button } from "../../ui";
import { describeCreateError } from "../newrun/errors";

export function DemoButton() {
  const nav = useNavigate();
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  async function start() {
    setBusy(true);
    setErr(null);
    try {
      const r = await api.createRun({ pack: "sql", scale: "tiny", dry_run: true });
      nav(`/runs/${encodeURIComponent(r.run_id)}`);
    } catch (e) {
      setErr(e instanceof ApiError && e.message ? `${e.message} (HTTP ${e.status})` : describeCreateError(e));
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
    </div>
  );
}
