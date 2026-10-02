import { Link, useParams } from "react-router-dom";
import { ApiError, api } from "../../api/client";
import { fmtInt } from "../../api/format";
import type { Report } from "../../api/types";
import { ApiErrorState, EmptyState, LabelBanner, Spinner } from "../../components";
import { RunShell } from "../../components/RunShell";
import { ExamplesCard } from "./Examples";
import { Summary } from "./Summary";
import { Accuracy, ByClass, GateCaveat, HumanSet, Stress, Clusters, Counters, CostLatency, GateDetails } from "./sections";
import { useAsync } from "./useAsync";
import "./report.css";

/** The report body: a sticky summary, then sections in a two-column grid on wide screens. */
export function ReportView({ r }: { r: Report }) {
  return (
    <div className="rp">
      <h2 className="sr-only">Report</h2>
      <Summary r={r} />
      <p className="muted rp-meta">Pack {r.pack} · candidate round {fmtInt(r.candidate_round)}</p>
      <GateCaveat r={r} />
      <div className="rp-grid">
        <Accuracy r={r} />
        <GateDetails r={r} />
        <ByClass r={r} />
        <HumanSet r={r} />
        <Stress r={r} />
        <ExamplesCard id={r.run_id} />
        <CostLatency r={r} />
        <Clusters r={r} />
        <Counters r={r} />
      </div>
    </div>
  );
}

export default function ReportScreen() {
  const { id = "" } = useParams();
  const [res, retry] = useAsync(() => api.report(id), id);
  const [run] = useAsync(() => api.run(id), id);
  let body;
  if (res.state === "loading") body = <Spinner label="Loading report" />;
  else if (res.state === "error") {
    const e = res.error;
    body = e instanceof ApiError && e.code === "report_not_ready"
      ? <EmptyState title="Report not ready">{e.code}: {e.message} <Link to={`/runs/${encodeURIComponent(id)}`}>Watch it live</Link></EmptyState>
      : <ApiErrorState error={e} onRetry={retry} />;
  } else {
    const r = res.data;
    body = (
      <>
        {/* The shell's banner comes from the run detail. If that could not be loaded, the numbers still carry a label, from the report's own flags. */}
        {run.state === "error" && <div className="rp-banner"><LabelBanner dry_run={r.dry_run} recorded={r.recorded} recorded_at={r.recorded_at} /></div>}
        <ReportView r={r} />
      </>
    );
  }
  return <RunShell runId={id} tab="report" run={run.state === "ok" ? run.data : undefined}>{body}</RunShell>;
}
