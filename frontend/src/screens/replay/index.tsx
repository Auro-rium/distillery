// Mission (`/`): the featured run's verdict first, then the proof ladder, training, the method, and
// limits & cost, then the stored and live run lists. Every number comes from /api/replay, the featured
// run's /api/runs/:id/report, or /api/evidence, formatted by api/format.ts.
import { Link } from "react-router-dom";
import { api } from "../../api/client";
import { NOT_MEASURED } from "../../api/format";
import type { RunSummary } from "../../api/types";
import { pickFeatured } from "../../api/useFeaturedRun";
import { ApiErrorState, Badge, Card, EmptyState, LabelBanner, Spinner } from "../../components";
import { GITHUB_URL } from "../../components/Shell";
import { LabelTag, labelKey } from "../report/label";
import { useAsync, type Async } from "../report/useAsync";
import { DemoButton } from "./DemoButton";
import { Hero } from "./Hero";
import { Limits } from "./Limits";
import { LiveRuns } from "./LiveRuns";
import { Pipeline } from "./Pipeline";
import { ProofLadder } from "./ProofLadder";
import { Training } from "./Training";
import "./replay.css";

/** One banner per distinct label among the listed runs, except the featured run's (the hero shows that one). */
function LabelBanners({ runs, skip }: { runs: RunSummary[]; skip: string | null }) {
  const seen = new Map<string, RunSummary>();
  for (const r of runs) if (!seen.has(labelKey(r)) && labelKey(r) !== skip) seen.set(labelKey(r), r);
  if (seen.size === 0) return null;
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
            <Link className="btn primary sm" to={`/runs/${id}/report`}>Report</Link>
            <Link className="btn sm" to={`/runs/${id}/tree`}>Experiment tree</Link>
          </>
        ) : (
          <Link className="btn sm" to={`/runs/${id}`}>Watch run (no report yet)</Link>
        )}
      </div>
    </Card>
  );
}

/** A Mission section that needs the featured run's report; says why when it cannot show it. */
function ReportSection(props: {
  id: string;
  title: string;
  lead: string;
  replay: Async<unknown>;
  featured: RunSummary | null;
  report: Async<unknown>;
  children: (report: unknown) => JSX.Element;
}) {
  const { replay, featured, report } = props;
  let body: JSX.Element;
  if (replay.state === "error") body = <p className="muted">Unavailable: the run list did not load (the error is shown above).</p>;
  else if (replay.state === "loading" || (featured && report.state === "loading")) body = <Spinner label={`Loading ${props.title.toLowerCase()}`} lines={2} />;
  else if (!featured) body = <p className="muted">Shown once a recorded run is available to feature.</p>;
  else if (report.state === "error") body = <p className="muted">Unavailable: the featured run&rsquo;s report did not load (the error is shown above).</p>;
  else body = props.children(report.state === "ok" ? report.data : null);
  return (
    <section className="m-section" aria-labelledby={props.id}>
      <div className="m-section-head">
        <h2 id={props.id}>{props.title}</h2>
        <p className="muted">{props.lead}</p>
      </div>
      {body}
    </section>
  );
}

export default function Mission() {
  const [replay, retryReplay] = useAsync<RunSummary[]>(() => api.replay(), "replay");
  const featured = replay.state === "ok" ? pickFeatured(replay.data) : null;
  const fid = featured?.run_id ?? null;
  // Each answer carries the id it was fetched for, so a stale answer (from before the featured run was
  // known) is never shown as this run's report: until the matching one arrives the state is "loading".
  const [tagged, retryReport] = useAsync<{ id: string | null; data: unknown }>(
    () => (fid ? api.report(fid).then((data) => ({ id: fid, data })) : Promise.resolve({ id: null, data: null })),
    `report:${fid ?? ""}`,
  );
  const report: Async<unknown> =
    tagged.state === "ok" ? (tagged.data.id === fid ? { state: "ok", data: tagged.data.data } : { state: "loading" }) : tagged;
  const [evidence, retryEvidence] = useAsync<unknown>(() => api.evidence(), "evidence");
  const runs = replay.state === "ok" && Array.isArray(replay.data) ? replay.data : [];

  return (
    <div className="stack mission">
      <Hero replay={replay} featured={featured} report={report} onRetry={retryReplay} onRetryReport={retryReport} />

      <ProofLadder res={evidence} onRetry={retryEvidence} />

      <ReportSection id="training-h" title="Training" lead="What the fine-tune recorded for the featured run." replay={replay} featured={featured} report={report}>
        {(r) => <Training report={r} />}
      </ReportSection>

      <section className="m-section" aria-labelledby="how-h">
        <div className="m-section-head">
          <h2 id="how-h">How it works</h2>
          <p className="muted">Every example and every evaluation is checked by running the SQL.</p>
        </div>
        <Pipeline />
      </section>

      <ReportSection id="limits-h" title="Limits & cost" lead="Where the student is weak, and what the run cost." replay={replay} featured={featured} report={report}>
        {(r) => <Limits report={r} />}
      </ReportSection>

      <section className="m-section m-ctas" aria-labelledby="try-h">
        <h2 id="try-h">Try it</h2>
        <DemoButton />
        <div className="m-cta-row">
          <Link className="btn" to="/new">Start a live run</Link>
          <a className="btn ghost" href={GITHUB_URL} target="_blank" rel="noreferrer noopener">
            GitHub<span className="sr-only"> (opens in a new tab)</span>
          </a>
        </div>
      </section>

      <div className="m-lists">
        <section className="stack stored" aria-labelledby="stored-h">
          <h2 id="stored-h">Stored runs</h2>
          {runs.length > 0 && <LabelBanners runs={runs} skip={featured ? labelKey(featured) : null} />}
          {replay.state === "loading" && <Spinner label="Loading replay bundles" lines={2} />}
          {replay.state === "error" && <ApiErrorState error={replay.error} onRetry={retryReplay} />}
          {replay.state === "ok" && runs.length === 0 && (
            <EmptyState title="No replay bundles available">Nothing has been exported yet.</EmptyState>
          )}
          {runs.length > 0 && (
            <ul className="bundles" role="list" aria-label="Stored runs">
              {runs.map((b) => <li key={b.run_id}><Bundle b={b} /></li>)}
            </ul>
          )}
        </section>
        <LiveRuns />
      </div>
    </div>
  );
}
