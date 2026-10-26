import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { Decision, RunDetail } from "../api/types";
import { useFeaturedRun } from "../api/useFeaturedRun";
import { prefersReducedMotion } from "../motion/useReducedMotion";
import { rememberTabOrigin, takeTabOrigin } from "../motion/tabOpen";
import { LabelBanner } from ".";
import { runSentence, runTitle } from "./runName";
import { Badge, Tabs, TabsContent, TabsList, TabsTrigger, type Tone } from "../ui";

export type RunTab = "live" | "report" | "tree";

const TABS: { value: RunTab; label: string; path: string }[] = [
  { value: "live", label: "Live run", path: "" },
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
 * Header and tab bar shared by the run screens. Only the header (run, status, label) stays in view while scrolling; the tab bar scrolls away. Every header slot is
 * always rendered (empty while `run` is still loading) so nothing moves when the data arrives; the
 * label slot then holds the LabelBanner built from the payload's dry_run / recorded flags, and says
 * "Label unknown" only when a loaded payload lacks them.
 */
export function RunShell(props: { runId: string; tab: RunTab; run?: RunDetail; children: ReactNode }) {
  const { runId, tab, run } = props;
  const verdict = useVerdict(runId, run?.status === "complete");
  // "Latest" is the run the Evidence link points at: the newest recorded run (the one Mission features).
  const { run: featured } = useFeaturedRun();
  const latest = featured !== null && featured.run_id === runId;
  const base = `/runs/${encodeURIComponent(runId)}`;
  const name = runTitle(runId);
  const panel = useRef<HTMLDivElement>(null);
  // Arriving from a tab click: the page grows out of that tab. Reads the click position once; nothing otherwise.
  useLayoutEffect(() => {
    const o = takeTabOrigin();
    const el = panel.current;
    if (!o || !el || prefersReducedMotion()) return;
    const r = el.getBoundingClientRect();
    el.style.setProperty("--ox", `${o.x - r.left}px`);
    el.style.setProperty("--oy", `${o.y - r.top}px`);
    // the first frame shows only a strip under the tab (its width, from the top of the page); it opens to the full page
    el.style.setProperty("--cl", `${Math.max(0, o.left - r.left)}px`);
    el.style.setProperty("--cr", `${Math.max(0, r.right - o.right)}px`);
    el.style.setProperty("--cb", `${Math.max(0, r.height - 44)}px`);
    el.dataset.opening = "true";
    const clear = () => { delete el.dataset.opening; };
    // animationend bubbles: only this animation's own end counts, not a child's (a child ending would cut the opening short)
    const onEnd = (e: AnimationEvent) => { if (e.target === el && e.animationName === "run-open") clear(); };
    el.addEventListener("animationend", onEnd);
    const t = setTimeout(clear, 900);
    return () => { clearTimeout(t); el.removeEventListener("animationend", onEnd); };
  }, []);
  return (
    <Tabs value={tab} activationMode="manual" className="run-shell">
      <header className="run-sticky">
        <div className="run-head">
          <div className="run-latest" data-slot="latest">
            {latest ? <Badge tone="ok" size="lg" className="run-latest-tag">LATEST RUN</Badge> : null}
          </div>
          <div className="run-head-row">
            <h1 className="run-title">{name.pack && <span className="run-pack">{name.pack}</span>}{name.rest}</h1>
            <span className="run-status" data-slot="status">
              {run ? <Badge tone={STATUS_TONE[run.status] ?? "neutral"}>{run.status}</Badge> : null}
            </span>
            <span className="run-verdict" data-slot="verdict">
              {verdict ? <Badge tone={verdict === "PROMOTE" ? "ok" : "bad"}>{verdict}</Badge> : null}
            </span>
          </div>
          <p className="run-about" data-slot="about">{runSentence(runId, latest)}</p>
          <p className="run-idline"><span className="eyebrow">Run ID</span> <code className="mono run-id" data-slot="id">{runId}</code></p>
          <div className="run-label" data-slot="label" data-state={run ? "ready" : "pending"}>
            {run ? <LabelBanner dry_run={run.dry_run} recorded={run.recorded} recorded_at={run.recorded_at} status={run.status} /> : null}
          </div>
        </div>
      </header>
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
                <Link to={`${base}${t.path}`} aria-current={t.value === tab ? "page" : undefined} onClick={(e) => rememberTabOrigin(e.currentTarget)}>{t.label}</Link>
              </TabsTrigger>
            ))}
        </TabsList>
      </div>
      {/* The opening animation runs on an inner wrapper: Radix sets animation-duration: 0s on the panel itself when it mounts. */}
      <TabsContent value={tab} className="run-panel"><div ref={panel} className="run-body">{props.children}</div></TabsContent>
    </Tabs>
  );
}
