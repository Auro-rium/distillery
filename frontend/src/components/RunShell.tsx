import { useEffect, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { Decision, RunDetail } from "../api/types";
import { LabelBanner } from ".";
import { Badge, Tabs, TabsContent, TabsList, TabsTrigger, type Tone } from "../ui";

export type RunTab = "live" | "report" | "tree";

const TABS: { value: RunTab; label: string; path: string }[] = [
  { value: "live", label: "Run", path: "" },
  { value: "report", label: "Report", path: "/report" },
  { value: "tree", label: "Tree", path: "/tree" },
];

const STATUS_TONE: Record<RunDetail["status"], Tone> = { pending: "info", running: "info", complete: "ok", failed: "bad" };

/**
 * Verdict source: `RunDetail` carries no decision (API contract), so once the run is complete the
 * decision is read from GET /api/runs/{id}/report, the same source the Live screen uses. While that is
 * loading, or if it fails, nothing is shown: no badge, no alert, no guess. The error itself is reported
 * by the Report screen, which owns that request.
 */
function useVerdict(runId: string, complete: boolean): Decision | null {
  const [decision, setDecision] = useState<Decision | null>(null);
  useEffect(() => {
    setDecision(null);
    if (!complete) return;
    let live = true;
    api.report(runId).then((r) => { if (live) setDecision(r.decision); }, () => undefined);
    return () => { live = false; };
  }, [runId, complete]);
  return decision;
}

/**
 * Header and tab bar shared by the run screens, one sticky block below the site header. Every header slot is
 * always rendered (empty while `run` is still loading) so nothing moves when the data arrives; the
 * label slot then holds the LabelBanner built from the payload's dry_run / recorded flags, and says
 * "Label unknown" only when a loaded payload lacks them.
 */
export function RunShell(props: { runId: string; tab: RunTab; run?: RunDetail; children: ReactNode }) {
  const { runId, tab, run } = props;
  const verdict = useVerdict(runId, run?.status === "complete");
  const base = `/runs/${encodeURIComponent(runId)}`;
  return (
    <Tabs value={tab} activationMode="manual" className="run-shell">
      <header className="run-sticky">
        <div className="run-head">
          <div className="run-head-row">
            <h1 className="run-title">Run <span className="mono" data-slot="id">{runId}</span></h1>
            <span className="run-status" data-slot="status">
              {run ? <Badge tone={STATUS_TONE[run.status] ?? "neutral"}>{run.status}</Badge> : null}
            </span>
            <span className="run-verdict" data-slot="verdict">
              {verdict ? <Badge tone={verdict === "PROMOTE" ? "ok" : "bad"}>{verdict}</Badge> : null}
            </span>
          </div>
          <div className="run-label" data-slot="label" data-state={run ? "ready" : "pending"}>
            {run ? <LabelBanner dry_run={run.dry_run} recorded={run.recorded} recorded_at={run.recorded_at} status={run.status} /> : null}
          </div>
        </div>
        <div className="run-tabbar">
          <TabsList aria-label="Run sections">
            {TABS.map((t) => (
              <TabsTrigger
                key={t.value}
                value={t.value}
                asChild
                // Only the mounted panel exists; the inactive tabs must not point at a panel that is not in the DOM.
                aria-controls={t.value === tab ? undefined : (undefined as unknown as string)}
              >
                <Link to={`${base}${t.path}`} aria-current={t.value === tab ? "page" : undefined}>{t.label}</Link>
              </TabsTrigger>
            ))}
          </TabsList>
        </div>
      </header>
      <TabsContent value={tab} className="run-panel">{props.children}</TabsContent>
    </Tabs>
  );
}
