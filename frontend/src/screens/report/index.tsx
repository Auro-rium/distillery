import { Link, useParams, useSearchParams } from "react-router-dom";
import { ApiError, api } from "../../api/client";
import { fmtInt } from "../../api/format";
import type { Report } from "../../api/types";
import { ApiErrorState, EmptyState, LabelBanner, Spinner } from "../../components";
import { RunShell } from "../../components/RunShell";
import { answerLanguage } from "../../lib/packs";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../../ui";
import { ExamplesCard } from "./Examples";
import { Summary } from "./Summary";
import { Accuracy, ByClass, Clusters, CostLatency, Counters, FineTune, GateCaveat, GateDetails, HumanSet, Stress } from "./sections";
import { useAsync } from "./useAsync";
import "./report.css";

const TABS = [
  { value: "verdict", label: "Verdict" },
  { value: "training", label: "Training" },
  { value: "examples", label: "Examples & errors" },
] as const;
type Tab = (typeof TABS)[number]["value"];
const isTab = (v: string | null): v is Tab => TABS.some((t) => t.value === v);

/** The open tab lives in `?tab=` so a section can be linked to; anything unknown opens the verdict. */
function useTab(): [Tab, (v: string) => void] {
  const [params, setParams] = useSearchParams();
  const raw = params.get("tab");
  const tab: Tab = isTab(raw) ? raw : "verdict";
  const set = (v: string) =>
    setParams((p) => {
      const next = new URLSearchParams(p);
      if (v === "verdict") next.delete("tab");
      else next.set("tab", v);
      return next;
    }, { replace: true });
  return [tab, set];
}

/**
 * The evidence dossier: a sticky verdict header, the in-distribution caveat, then three tabs. Every tab
 * stays mounted (inactive ones are hidden by CSS), so switching is instant, examples load once, and
 * every number of the report is on the page for the provenance checks.
 */
export function ReportView({ r }: { r: Report }) {
  const [tab, setTab] = useTab();
  return (
    <div className="rp">
      <h2 className="sr-only">Report</h2>
      <Summary r={r} />
      <p className="muted rp-meta">Pack {r.pack} · candidate round {fmtInt(r.candidate_round)}</p>
      <GateCaveat r={r} />
      <Tabs value={tab} onValueChange={setTab} className="rp-tabs">
        <TabsList aria-label="Report sections" className="rp-tablist">
          {TABS.map((t) => <TabsTrigger key={t.value} value={t.value}>{t.label}</TabsTrigger>)}
        </TabsList>
        <TabsContent value="verdict" forceMount className="rp-tab">
          <div className="rp-grid">
            <GateDetails r={r} />
            <Accuracy r={r} />
            <ByClass r={r} />
            <HumanSet r={r} />
          </div>
        </TabsContent>
        <TabsContent value="training" forceMount className="rp-tab">
          <div className="rp-grid">
            <FineTune r={r} />
            <CostLatency r={r} />
            <Stress r={r} />
          </div>
        </TabsContent>
        <TabsContent value="examples" forceMount className="rp-tab">
          <div className="rp-grid">
            <ExamplesCard id={r.run_id} language={answerLanguage(r.pack)} />
            <Clusters r={r} />
            <Counters r={r} />
          </div>
        </TabsContent>
      </Tabs>
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
