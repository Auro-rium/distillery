import { fmtInt, fmtNumber, fmtUsd } from "../../api/format";
import type { GatewaySavings } from "../../api/types";
import { CountUp } from "../../motion/CountUp";
import { Badge, Card, Stat } from "../../ui";

/**
 * Animate toward a number, but finish on the server's own display string: the last frame is `display`,
 * never a string worked out here. While counting, the in-between values are only formatted.
 */
function money(value: number, display: string, digits: number) {
  return (v: number | null | undefined) => (v === value ? display : fmtUsd(v, digits));
}

/**
 * The savings counter: GET /api/gateway/savings, drawn as it came. Money is the server's `*_display` string;
 * the counts are its integers; the basis of every figure is printed next to it, including "pending H1" for the
 * student's serving cost. Nothing here is added, subtracted or divided.
 */
export function Savings({ s }: { s: GatewaySavings }) {
  const pending = s.student_usd_per_1k === null;
  return (
    <Card className="pg-savings" aria-labelledby="pg-savings-h">
      <div className="pg-st-head">
        <h2 id="pg-savings-h" className="pg-h">Savings so far</h2>
        <Badge tone={pending ? "warn" : "ok"} title="Basis of the student's serving cost">
          student serving cost: {s.student_cost_basis}
        </Badge>
      </div>
      <div className="pg-stats">
        <Stat
          tone="student"
          size="lg"
          label="saved vs teacher-only"
          value={<CountUp value={s.saved_usd} format={money(s.saved_usd, s.saved_display, 4)} />}
          hint={s.saved_pct_display ? <><CountUp value={s.saved_pct} format={(v) => (v === s.saved_pct ? s.saved_pct_display! : `${fmtNumber(v, 1)}%`)} /> of the teacher-only cost</> : "percentage not available yet"}
        />
        <Stat label="actually spent" value={<CountUp value={s.actual_usd} format={money(s.actual_usd, s.actual_display, 4)} />} />
        <Stat tone="teacher" label="teacher-only would have cost" value={<CountUp value={s.teacher_only_usd} format={money(s.teacher_only_usd, s.teacher_only_display, 4)} />} />
        <Stat label="requests" value={<CountUp value={s.requests} format={fmtInt} />} hint={`${fmtInt(s.served_by.student)} student · ${fmtInt(s.served_by.teacher)} teacher · ${fmtInt(s.refused)} refused`} />
      </div>
      {pending && (
        <p className="pg-pending" role="note">
          The student's own serving cost is not measured yet ({s.student_cost_basis}), so it is not in these figures.
        </p>
      )}
      <details className="pg-basis">
        <summary>How these figures are computed</summary>
        <dl>
          <dt>Actually spent</dt><dd>{s.basis.actual}</dd>
          <dt>Teacher-only</dt><dd>{s.basis.teacher_only}</dd>
          <dt>Saved</dt><dd>{s.basis.saved}</dd>
          <dt>Window</dt><dd>{s.window}</dd>
          <dt>Counted</dt>
          <dd>{fmtInt(s.requests_in_savings)} requests in the savings, {fmtInt(s.requests_excluded)} excluded (no measured teacher output length)</dd>
        </dl>
      </details>
    </Card>
  );
}
