import { Link } from "react-router-dom";
import { fmtNumber, fmtPercent } from "../../api/format";
import type { Report, RunSummary } from "../../api/types";
import { ApiErrorState, Badge, LabelBanner } from "../../components";
import { CountUp } from "../../motion/CountUp";
import { Skeleton } from "../../ui";
import type { Async } from "../report/useAsync";
import { gateOf } from "./mission";
import { MissionHint } from "./MissionHint";
import { RunBox } from "./RunBox";

const pct = (v: number | null | undefined) => fmtPercent(v, 1);

/** The one sentence under the verdict chip. It is chosen by the decision only and carries no number. */
function sentence(decision: unknown): string {
  if (decision === "PROMOTE")
    return "The small fine-tuned student matched its teacher on a sealed held-out set it never saw, so the fixed gate promoted it.";
  if (decision === "REJECT") return "The student did not clear the fixed gate on the sealed held-out set, so it was rejected.";
  return "The report carries no gate decision.";
}

function Numeral({ tone, label, value }: { tone: "student" | "teacher" | "base"; label: string; value: unknown }) {
  return (
    <div className={`stat ${tone} m-num`}>
      <div className="v num"><CountUp value={typeof value === "number" ? value : null} format={pct} fromZero /></div>
      <div className="l eyebrow">{label}</div>
    </div>
  );
}

function Verdict({ run, report }: { run: RunSummary; report: unknown }) {
  const g = gateOf(report);
  const id = encodeURIComponent(run.run_id);
  if (!g) {
    return (
      <div className="m-verdict">
        <p className="muted">The report for <span className="mono">{run.run_id}</span> carries no gate result.</p>
        <Link className="btn" to={`/runs/${id}/report`}>Open the report</Link>
      </div>
    );
  }
  const decision = g.decision ?? (report as Partial<Report>).decision;
  const tone = decision === "PROMOTE" ? "ok" : decision === "REJECT" ? "bad" : "neutral";
  return (
    <div className="m-verdict">
      <div className="m-left">
        <div className="m-chiprow" data-tone={tone}>
          <Badge tone={tone} size="lg" className="m-chip">{decision ?? fmtNumber(null)}</Badge>
          <span className="m-arrow" aria-hidden="true">&gt;</span>
          <RunBox run={run} tone={tone} />
        </div>
        <p className="m-sentence">{sentence(decision)}</p>
        <div className="m-nums" role="group" aria-label="Held-out accuracy">
          <Numeral tone="student" label="Student" value={g.student_acc} />
          <Numeral tone="teacher" label="Teacher" value={g.teacher_acc} />
          <Numeral tone="base" label="Base" value={g.base_acc} />
        </div>
        <p className="muted m-numcap">Execution accuracy on the sealed held-out set (gate set).</p>
      </div>
      <div className="m-right">
        <Link className="m-cta" to={`/runs/${id}/report`}>
          <span>Open the evidence</span>
          <span className="m-cta-arrow" aria-hidden="true">→</span>
        </Link>
      </div>
    </div>
  );
}

/**
 * Hero verdict for the featured run. `replay` is the page's own replay list (so an API error shows here as an
 * error, not as "nothing to feature"); `featured` is picked from it; `report` is that run's report.
 */
export function Hero(props: {
  replay: Async<unknown>;
  featured: RunSummary | null;
  report: Async<unknown>;
  onRetry: () => void;
  onRetryReport: () => void;
}) {
  const { replay, featured, report } = props;
  const busy = replay.state === "loading" || (featured !== null && report.state === "loading");
  return (
    <section className="hero mission-hero" aria-labelledby="mission-h" aria-busy={busy || undefined}>
      <MissionHint runId={featured?.run_id ?? null} />
      <h1 id="mission-h">Distil one task into a small model, then prove it on a sealed set.</h1>
      {/* The featured run's label sits first in the hero; the slot is reserved while loading. */}
      <div className="banners">
        {replay.state === "loading" && <div className="banner-skeleton" aria-hidden="true" />}
        {/* A recorded run is labelled inside its run box (open the arrow); dry, live and unknown labels stay as banners. */}
        {featured && !(featured.dry_run === false && featured.recorded === true) && (
          <LabelBanner dry_run={featured.dry_run} recorded={featured.recorded} recorded_at={featured.recorded_at} status={featured.status} />
        )}
      </div>
      {busy && (
        <div className="m-loading" role="status" aria-live="polite">
          <span className="sr-only">Loading the featured run</span>
          <div className="m-nums" aria-hidden="true">{[0, 1, 2].map((i) => <Skeleton key={i} className="m-sk" />)}</div>
        </div>
      )}
      {replay.state === "error" && <ApiErrorState title="Could not load the featured run" error={replay.error} onRetry={props.onRetry} />}
      {replay.state === "ok" && !featured && (
        <p className="muted m-none">
          There is no recorded run to feature on this server yet. Start a demo below, or open a stored run.
        </p>
      )}
      {featured && report.state === "error" && (
        <ApiErrorState title={`Could not load the report for ${featured.run_id}`} error={report.error} onRetry={props.onRetryReport} />
      )}
      {featured && report.state === "ok" && <Verdict run={featured} report={report.data} />}
    </section>
  );
}
