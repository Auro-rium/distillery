import { Link } from "react-router-dom";
import { api } from "../../api/client";
import { fmtText } from "../../api/format";
import type { RunSummary } from "../../api/types";
import { Badge, Card, ApiErrorState, EmptyState, LabelBanner, Spinner } from "../../components";
import { useAsync } from "../report/useAsync";
import { Pipeline } from "./Pipeline";

type Flags = Pick<RunSummary, "dry_run" | "recorded" | "recorded_at">;

/** What the label says, by the same rule as LabelBanner: a missing flag is never read as real. */
function labelKey(b: Flags): string {
  if (b.dry_run === true) return "dry";
  if (b.dry_run === false && b.recorded === true) return `recorded:${b.recorded_at ?? ""}`;
  if (b.dry_run === false && b.recorded === false) return "live";
  return "unknown";
}

function LabelTag({ b }: { b: Flags }) {
  const k = labelKey(b);
  if (k === "dry") return <Badge tone="warn">dry run</Badge>;
  if (k.startsWith("recorded")) return <Badge tone="info">recorded</Badge>;
  if (k === "live") return <Badge tone="info">live run</Badge>;
  return <Badge tone="warn">label unknown</Badge>;
}

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

export function Bundle({ b }: { b: RunSummary }) {
  const id = encodeURIComponent(b.run_id);
  const done = b.status === "complete";
  return (
    <Card className="lift">
      <div className="row" style={{ justifyContent: "space-between" }}>
        <h3 style={{ margin: 0 }} className="mono">{b.run_id}</h3>
        <span className="row" style={{ gap: 6 }}>
          <LabelTag b={b} />
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
      {res.state === "ok" && res.data.map((b) => <Bundle key={b.run_id} b={b} />)}
    </div>
  );
}
