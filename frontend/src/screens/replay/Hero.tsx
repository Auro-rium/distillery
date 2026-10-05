import { Link } from "react-router-dom";
import { fmtInt, fmtNumber, fmtPercent } from "../../api/format";
import type { Report, RunSummary } from "../../api/types";
import { CiNumberLine } from "../../charts";
import { ApiErrorState, Badge, LabelBanner } from "../../components";
import { CountUp } from "../../motion/CountUp";
import { Skeleton } from "../../ui";
import type { Async } from "../report/useAsync";
import { gateOf, P_SHOW_MIN } from "./mission";

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

function McNemar({ p, alpha }: { p: unknown; alpha: unknown }) {
  if (typeof p !== "number" || !Number.isFinite(p)) return <>McNemar p {fmtNumber(null)}</>;
  const below = typeof alpha === "number" && p < alpha;
  const shown = p >= P_SHOW_MIN ? `p = ${fmtNumber(p, 3)}` : below ? "p far below α" : `p ${fmtNumber(p, 3)}`;
  return <>McNemar vs base: {shown} (α = {fmtNumber(alpha as number | null, 2)})</>;
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
  const pass = decision === "PROMOTE";
  return (
    <div className="m-verdict">
      <div className="m-left">
        <div className="m-chiprow">
          <Badge tone={pass ? "ok" : decision === "REJECT" ? "bad" : "neutral"} size="lg" className="m-chip">{decision ?? fmtNumber(null)}</Badge>
          <span className="mono m-runid">{run.run_id}</span>
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
        <CiNumberLine
          className="m-ci"
          title="Student / teacher accuracy ratio, confidence interval"
          point={g.ratio_point}
          lo={g.ratio_lo}
          hi={g.ratio_hi}
          threshold={g.thresholds?.ratio_lower_bound_min}
          thresholdLabel="gate minimum"
          caption="The gate needs the interval's lower bound at or above the minimum."
        />
        <ul className="m-facts" role="list">
          <li><span className="num">n = {fmtInt(g.n)}</span> sealed held-out tasks</li>
          <li><McNemar p={g.mcnemar_p} alpha={g.thresholds?.mcnemar_alpha} /></li>
          <li>Student right, base wrong: <span className="num">{fmtInt(g.student_only_vs_base)}</span> · base right, student wrong: <span className="num">{fmtInt(g.base_only_vs_student)}</span></li>
        </ul>
        <Link className="btn primary m-cta" to={`/runs/${id}/report`}>Open the evidence <span aria-hidden="true">→</span></Link>
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
      <div className="eyebrow">Mission</div>
      <h1 id="mission-h">Distil one task into a small model, then prove it on a sealed set.</h1>
      {/* The featured run's label sits first in the hero; the slot is reserved while loading. */}
      <div className="banners">
        {replay.state === "loading" && <div className="banner-skeleton" aria-hidden="true" />}
        {featured && <LabelBanner dry_run={featured.dry_run} recorded={featured.recorded} recorded_at={featured.recorded_at} status={featured.status} />}
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
