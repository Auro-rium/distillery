import type { CSSProperties } from "react";
import { fmtInt, fmtPercent } from "../../api/format";
import { CountUp } from "../../motion/CountUp";
import { useInView } from "../../motion/useInView";
// Hand-written SVG accuracy bars with optional confidence-interval whiskers.
// Draws only values it is given: it computes no statistics. Position = value on a 0..1 axis.
export interface BarRow {
  key: "base" | "student" | "teacher";
  label: string;
  value: number | null;
  ci: [number, number] | null;
  n: number | null;
}

// Bar geometry only: position on a 0..1 axis scaled to the 100-unit viewBox. Not a statistic.
const x = (v: number): number => Math.min(1, Math.max(0, v)) * 100;
const TICKS = [0, 0.5, 1] as const;

export function rowDescription(r: BarRow): string {
  const ci = r.ci ? `confidence interval ${fmtPercent(r.ci[0])} to ${fmtPercent(r.ci[1])}` : "no confidence interval in report";
  return `${r.label} ${fmtPercent(r.value)}, ${ci}, n=${fmtInt(r.n)}`;
}

/**
 * The DOM always carries the final values (bar widths, CI positions, labels). Once, when the chart
 * first scrolls into view, CSS grows the bars and whiskers from zero to those same values with
 * transform/opacity; reduced motion or no IntersectionObserver shows them at once.
 */
export function AccuracyChart({ rows, title }: { rows: BarRow[]; title: string }) {
  const [ref, seen] = useInView<HTMLElement>();
  return (
    <figure className="acc" aria-label={title} ref={ref} data-reveal={seen ? "done" : "pending"}>
      <div className="acc-rows">
        {rows.map((r, i) => (
          <div className="acc-row" key={r.key} style={{ "--i": i } as CSSProperties}>
            <span className="acc-label">{r.label}</span>
            <svg
              className={`acc-svg ${r.key}`}
              viewBox="0 0 100 16"
              preserveAspectRatio="none"
              role="img"
              aria-label={rowDescription(r)}
            >
              <rect className="track" x="0" y="0" width="100" height="16" />
              {[25, 50, 75].map((g) => (
                <line key={g} className="grid" x1={g} x2={g} y1="0" y2="16" />
              ))}
              {r.value !== null && Number.isFinite(r.value) && (
                <rect className="bar" x="0" y="3" width={x(r.value)} height="10" />
              )}
              {r.ci && (
                <g className="ci">
                  <line x1={x(r.ci[0])} x2={x(r.ci[1])} y1="8" y2="8" />
                  <line x1={x(r.ci[0])} x2={x(r.ci[0])} y1="4" y2="12" />
                  <line x1={x(r.ci[1])} x2={x(r.ci[1])} y1="4" y2="12" />
                </g>
              )}
            </svg>
            <span className="acc-val">
              <strong><CountUp value={r.value} format={(v) => fmtPercent(v)} fromZero /></strong>
              <span className="muted">
                {r.ci ? ` CI ${fmtPercent(r.ci[0])} to ${fmtPercent(r.ci[1])}` : " no CI in report"} · n={fmtInt(r.n)}
              </span>
            </span>
          </div>
        ))}
      </div>
      <figcaption className="acc-axis muted" aria-hidden="true">
        {TICKS.map((t) => <span key={t}>{fmtPercent(t, 0)}</span>)}
      </figcaption>
    </figure>
  );
}
