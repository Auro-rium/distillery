import { Link } from "react-router-dom";
import { api } from "../../api/client";
import { fmtText } from "../../api/format";
import type { RunSummary } from "../../api/types";
import { Badge, Card, ApiErrorState, EmptyState, LabelBanner, Spinner } from "../../components";
import { useAsync } from "../report/useAsync";

export function Bundle({ b }: { b: RunSummary }) {
  const id = encodeURIComponent(b.run_id);
  const done = b.status === "complete";
  return (
    <Card>
      <LabelBanner dry_run={b.dry_run} recorded={b.recorded} recorded_at={b.recorded_at} />
      <div className="row" style={{ marginTop: 12, justifyContent: "space-between" }}>
        <h3 style={{ margin: 0 }} className="mono">{b.run_id}</h3>
        <span className="row" style={{ gap: 6 }}>
          <Badge>{b.status}</Badge>
          {b.decision && <Badge tone={b.decision === "PROMOTE" ? "ok" : "bad"}>{b.decision}</Badge>}
        </span>
      </div>
      <p className="muted">Created {fmtText(b.created_at)}</p>
      <div className="row">
        {done ? (
          <>
            <Link className="btn primary" to={`/runs/${id}/report`}>Report</Link>
            <Link className="btn" to={`/runs/${id}/tree`}>Experiment tree</Link>
          </>
        ) : (
          <Link className="btn" to={`/runs/${id}`}>Watch run (no report yet)</Link>
        )}
      </div>
    </Card>
  );
}

export default function Replay() {
  const [res, retry] = useAsync(() => api.replay(), "replay");
  return (
    <div className="stack">
      <div>
        <h1>Replay</h1>
        <p className="muted">
          Distillery distils a narrow task (text-to-SQL) into a small fine-tuned model. Every example and
          evaluation is verified by execution and a fixed gate decides PROMOTE or REJECT. These are stored
          runs you can open without spending anything.
        </p>
      </div>
      {res.state === "loading" && <Spinner label="Loading replay bundles" />}
      {res.state === "error" && <ApiErrorState error={res.error} onRetry={retry} />}
      {res.state === "ok" && res.data.length === 0 && (
        <EmptyState title="No replay bundles available">Nothing has been exported yet.</EmptyState>
      )}
      {res.state === "ok" && res.data.map((b) => <Bundle key={b.run_id} b={b} />)}
    </div>
  );
}
