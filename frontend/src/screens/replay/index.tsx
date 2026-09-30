import { Link } from "react-router-dom";
import { api } from "../../api/client";
import { NOT_MEASURED } from "../../api/format";
import type { RunSummary } from "../../api/types";
import { ApiErrorState, Badge, Card, EmptyState, LabelBanner, Spinner } from "../../components";
import { LabelTag, labelKey } from "../report/label";
import { useAsync } from "../report/useAsync";
import { Pipeline } from "./Pipeline";
import "./replay.css";

/** One banner per distinct label among the listed runs, above everything else on the page. */
function LabelBanners({ runs }: { runs: RunSummary[] }) {
  const seen = new Map<string, RunSummary>();
  for (const r of runs) if (!seen.has(labelKey(r))) seen.set(labelKey(r), r);
  return (
    <div className="banners">
      {[...seen.values()].map((r) => <LabelBanner key={labelKey(r)} dry_run={r.dry_run} recorded={r.recorded} recorded_at={r.recorded_at} />)}
    </div>
  );
}

/** Label badge, status, decision and date: each only from the payload; the date is shown as the API wrote it. */
export function Bundle({ b }: { b: RunSummary }) {
  const id = encodeURIComponent(b.run_id);
  const done = b.status === "complete";
  return (
    <Card className="bundle lift">
      <div className="bundle-head">
        <h3 className="mono">{b.run_id}</h3>
        <div className="bundle-tags">
          <LabelTag b={b} />
          <Badge>{b.status}</Badge>
          {b.decision && <Badge tone={b.decision === "PROMOTE" ? "ok" : "bad"}>{b.decision}</Badge>}
        </div>
      </div>
      <p className="muted bundle-date">
        Created {b.created_at ? <time dateTime={b.created_at}>{b.created_at}</time> : NOT_MEASURED}
      </p>
      <div className="bundle-actions">
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
      {/* Label banners come first, above the hero. The slot is reserved while loading so nothing shifts. */}
      {res.state === "loading" && <div className="banners"><div className="banner-skeleton" aria-hidden="true" /></div>}
      {res.state === "ok" && res.data.length > 0 && <LabelBanners runs={res.data} />}
      <section className="hero">
        <div className="eyebrow">Replay</div>
        <h1>Teach a small model one task, and check whether it worked.</h1>
        <p className="muted hero-copy">
          Distillery distils a narrow task (text-to-SQL) into a small fine-tuned model. Every example and
          evaluation is verified by execution and a fixed gate decides PROMOTE or REJECT. These are stored
          runs you can open without spending anything.
        </p>
        <Pipeline />
      </section>
      {res.state === "loading" && <Spinner label="Loading replay bundles" />}
      {res.state === "error" && <ApiErrorState error={res.error} onRetry={retry} />}
      {res.state === "ok" && res.data.length === 0 && (
        <EmptyState title="No replay bundles available">Nothing has been exported yet.</EmptyState>
      )}
      {res.state === "ok" && res.data.length > 0 && (
        <ul className="bundles" role="list" aria-label="Stored runs">
          {res.data.map((b) => <li key={b.run_id}><Bundle b={b} /></li>)}
        </ul>
      )}
    </div>
  );
}
