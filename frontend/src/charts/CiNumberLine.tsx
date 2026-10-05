// Ratio confidence interval on a number line: the point estimate, its interval, and the threshold the
// lower bound must clear. It is the single "why PROMOTE" graphic. Labels are exactly the four values
// passed in (lo, point, hi, threshold); the axis has no other ticks, so nothing on it is invented.
import { fmtNumber } from "../api/format";
import { cx } from "../ui/cx";
import { NL_AXIS_Y, NL_DOMAIN_PAD, NL_HEIGHT, NL_MIN_SPAN, PAD } from "./constants";
import { ChartFigure, DataPoint, finite, scale, useChartWidth, type Formatter } from "./core";

export interface CiNumberLineProps {
  point: number | null | undefined;
  lo: number | null | undefined;
  hi: number | null | undefined;
  threshold: number | null | undefined;
  /** What the line measures, e.g. "Student / teacher accuracy ratio". Also the figure's accessible name. */
  title: string;
  caption?: string;
  /** Verdict, when the caller has it from the API (the gate decision). Default: lower bound >= threshold. */
  pass?: boolean | null;
  format?: Formatter;
  thresholdLabel?: string;
  className?: string;
}

export function CiNumberLine(props: CiNumberLineProps) {
  const fmt = props.format ?? ((v) => fmtNumber(v, 3));
  const [ref, width] = useChartWidth<HTMLDivElement>();
  const { point, lo, hi, threshold } = props;
  const ok = finite(lo) && finite(hi);
  const pass = props.pass ?? (ok && finite(threshold) ? lo >= threshold : null);
  const state = pass === true ? "pass" : pass === false ? "fail" : "unknown";

  const vals = [lo, hi, point, threshold].filter(finite);
  const summary = `${props.title}: point ${fmt(point)}, interval ${fmt(lo)} to ${fmt(hi)}, threshold ${fmt(threshold)}`;

  if (!vals.length) {
    return (
      <ChartFigure title={props.title} caption={props.caption} className={cx("ci-line", props.className)}>
        <p className="chart-empty">{fmt(null)}</p>
      </ChartFigure>
    );
  }

  const min = Math.min(...vals);
  const max = Math.max(...vals);
  const span = Math.max(max - min, NL_MIN_SPAN);
  const x = scale(min - span * NL_DOMAIN_PAD, max + span * NL_DOMAIN_PAD, PAD.left / 2, width - PAD.right);
  const y = NL_AXIS_Y;

  return (
    <ChartFigure title={props.title} caption={props.caption} className={cx("ci-line", `is-${state}`, props.className)}>
      <div ref={ref} className="chart-box">
        <svg className="chart-svg" width={width} height={NL_HEIGHT} viewBox={`0 0 ${width} ${NL_HEIGHT}`}>
          <title>{summary}</title>
          <line className="axis" x1={PAD.left / 2} x2={width - PAD.right} y1={y} y2={y} />
          {finite(threshold) && (
            <g className="ci-threshold">
              <rect className="ci-fail-zone" x={PAD.left / 2} y={y - 22} width={Math.max(0, x(threshold) - PAD.left / 2)} height={44} />
              <line x1={x(threshold)} x2={x(threshold)} y1={y - 26} y2={y + 26} />
              <text x={x(threshold)} y={y + 42} textAnchor="middle" className="ci-lab thr">{fmt(threshold)}</text>
              <text x={x(threshold)} y={y - 32} textAnchor="middle" className="ci-lab thr-name">{props.thresholdLabel ?? "threshold"}</text>
            </g>
          )}
          {ok && (
            <g className="ci-interval">
              <line className="ci-bar" x1={x(lo)} x2={x(hi)} y1={y} y2={y} />
              <line className="ci-cap" x1={x(lo)} x2={x(lo)} y1={y - 9} y2={y + 9} />
              <line className="ci-cap" x1={x(hi)} x2={x(hi)} y1={y - 9} y2={y + 9} />
              <text x={x(lo)} y={y + 24} textAnchor="end" className="ci-lab lo">{fmt(lo)}</text>
              <text x={x(hi)} y={y + 24} textAnchor="start" className="ci-lab hi">{fmt(hi)}</text>
            </g>
          )}
          {finite(point) && (
            <g className="ci-point">
              <DataPoint label={summary} cx={x(point)} cy={y} r={7} className="ci-dot" />
              <text x={x(point)} y={y - 14} textAnchor="middle" className="ci-lab pt">{fmt(point)}</text>
            </g>
          )}
        </svg>
      </div>
    </ChartFigure>
  );
}
