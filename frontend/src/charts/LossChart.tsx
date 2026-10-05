// Train and validation loss vs step. Straight segments between the recorded checkpoints (no smoothing,
// no interpolation), a marker on every checkpoint, and a break in the line wherever a value is absent:
// a missing loss is never drawn as zero. Axis labels are the recorded steps and the observed min/max.
import { fmtInt, fmtNumber } from "../api/format";
import { cx } from "../ui/cx";
import { LOSS_HEIGHT, LOSS_Y_PAD, MARKER_R, PAD } from "./constants";
import { ChartFigure, ChartLegend, DataPoint, finite, scale, useChartWidth, type Formatter } from "./core";

export interface LossPoint {
  step: number | null;
  train_loss: number | null;
  valid_loss: number | null;
}

export interface LossChartProps {
  points: readonly LossPoint[];
  title: string;
  caption?: string;
  format?: Formatter;
  className?: string;
}

type SeriesKey = "train_loss" | "valid_loss";
const SERIES: { key: SeriesKey; label: string; cls: string }[] = [
  { key: "train_loss", label: "Train loss", cls: "s-train" },
  { key: "valid_loss", label: "Validation loss", cls: "s-valid" },
];

/** Splits a series into runs of consecutive present values (a gap ends a run). */
function runs(pts: { x: number; y: number | null }[]): { x: number; y: number }[][] {
  const out: { x: number; y: number }[][] = [];
  let cur: { x: number; y: number }[] = [];
  for (const p of pts) {
    if (finite(p.y)) cur.push({ x: p.x, y: p.y });
    else if (cur.length) { out.push(cur); cur = []; }
  }
  if (cur.length) out.push(cur);
  return out;
}

export function LossChart(props: LossChartProps) {
  const fmt = props.format ?? ((v) => fmtNumber(v, 3));
  const [ref, width] = useChartWidth<HTMLDivElement>();
  const pts = props.points.filter((p) => finite(p.step)) as (LossPoint & { step: number })[];
  pts.sort((a, b) => a.step - b.step);
  const ys = pts.flatMap((p) => [p.train_loss, p.valid_loss]).filter(finite);
  const legend = <ChartLegend items={SERIES.map((s) => ({ key: s.key, label: s.label, className: s.cls }))} />;

  if (!pts.length || !ys.length) {
    return (
      <ChartFigure title={props.title} caption={props.caption} className={cx("loss", props.className)}>
        <p className="chart-empty">{fmt(null)}</p>
      </ChartFigure>
    );
  }

  const h = LOSS_HEIGHT;
  const x0 = pts[0].step;
  const x1 = pts[pts.length - 1].step;
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const ySpan = yMax - yMin || Math.abs(yMax) || 1;
  const sx = scale(x0, x1, PAD.left, width - PAD.right);
  const sy = scale(yMin - ySpan * LOSS_Y_PAD, yMax + ySpan * LOSS_Y_PAD, h - PAD.bottom, PAD.top);
  const steps = [...new Set(pts.map((p) => p.step))];

  return (
    <ChartFigure title={props.title} caption={props.caption} legend={legend} className={cx("loss", props.className)}>
      <div ref={ref} className="chart-box">
        <svg className="chart-svg" width={width} height={h} viewBox={`0 0 ${width} ${h}`}>
          {/* y axis: observed min and max only */}
          <line className="axis" x1={PAD.left} x2={PAD.left} y1={PAD.top} y2={h - PAD.bottom} />
          <line className="axis" x1={PAD.left} x2={width - PAD.right} y1={h - PAD.bottom} y2={h - PAD.bottom} />
          {[yMin, yMax].map((v, i) => (
            <g key={`y${i}`} className="tick">
              <line className="grid" x1={PAD.left} x2={width - PAD.right} y1={sy(v)} y2={sy(v)} />
              <text x={PAD.left - 6} y={sy(v)} dy="0.32em" textAnchor="end">{fmt(v)}</text>
            </g>
          ))}
          {steps.map((s) => (
            <g key={`x${s}`} className="tick">
              <line className="grid" x1={sx(s)} x2={sx(s)} y1={PAD.top} y2={h - PAD.bottom} />
              <text x={sx(s)} y={h - PAD.bottom + 16} textAnchor="middle">{fmtInt(s)}</text>
            </g>
          ))}
          <text className="axis-name" x={width - PAD.right} y={h - 4} textAnchor="end">step</text>
          {SERIES.map((s) => {
            const series = pts.map((p) => ({ x: p.step, y: p[s.key] }));
            return (
              <g key={s.key} className={cx("series", s.cls)}>
                {runs(series).map((r, i) => (
                  <polyline key={i} className="line" points={r.map((p) => `${sx(p.x)},${sy(p.y)}`).join(" ")} />
                ))}
                {series.filter((p): p is { x: number; y: number } => finite(p.y)).map((p) => (
                  <DataPoint
                    key={p.x}
                    className="marker"
                    cx={sx(p.x)}
                    cy={sy(p.y)}
                    r={MARKER_R}
                    label={`${s.label} at step ${fmtInt(p.x)}: ${fmt(p.y)}`}
                  />
                ))}
              </g>
            );
          })}
        </svg>
      </div>
    </ChartFigure>
  );
}
