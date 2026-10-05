// A word-sized trend line (e.g. the overfit loss curve in a proof card). No axes: its accessible name
// states the first and last values it was given; optional end label shows the last value as text.
import { fmtNumber } from "../api/format";
import { cx } from "../ui/cx";
import { SPARK_H, SPARK_PAD, SPARK_W } from "./constants";
import { finite, scale, type Formatter, type ModelKey } from "./core";

export interface SparklineProps {
  values: readonly (number | null | undefined)[];
  /** What the values are, e.g. "Overfit train loss". Used in the accessible name. */
  label: string;
  tone?: ModelKey | "neutral";
  format?: Formatter;
  /** Show the last value as text after the line. */
  showLast?: boolean;
  width?: number;
  height?: number;
  className?: string;
}

export function Sparkline(props: SparklineProps) {
  const fmt = props.format ?? ((v) => fmtNumber(v, 3));
  const w = props.width ?? SPARK_W;
  const h = props.height ?? SPARK_H;
  const pts = props.values.map((v, i) => ({ i, v })).filter((p): p is { i: number; v: number } => finite(p.v));
  const first = pts[0]?.v;
  const last = pts[pts.length - 1]?.v;
  const name = pts.length ? `${props.label}: ${fmt(first)} to ${fmt(last)}` : `${props.label}: ${fmt(null)}`;
  const tone = props.tone ?? "student";

  if (!pts.length) return <span className={cx("spark", "empty", props.className)}>{name}</span>;

  const vs = pts.map((p) => p.v);
  const sx = scale(0, Math.max(1, props.values.length - 1), SPARK_PAD, w - SPARK_PAD);
  const sy = scale(Math.min(...vs), Math.max(...vs), h - SPARK_PAD, SPARK_PAD);
  // Break the line at missing values; never draw a gap as zero.
  const segs: string[] = [];
  let cur: string[] = [];
  props.values.forEach((v, i) => {
    if (finite(v)) cur.push(`${sx(i)},${sy(v)}`);
    else if (cur.length) { segs.push(cur.join(" ")); cur = []; }
  });
  if (cur.length) segs.push(cur.join(" "));
  const end = pts[pts.length - 1];

  return (
    <span className={cx("spark", `m-${tone}`, props.className)}>
      <svg role="img" aria-label={name} width={w} height={h} viewBox={`0 0 ${w} ${h}`} className="spark-svg">
        {segs.map((s, i) => <polyline key={i} points={s} className="spark-line" />)}
        <circle cx={sx(end.i)} cy={sy(end.v)} r={SPARK_PAD - 1} className="spark-end" />
      </svg>
      {props.showLast && <span className="spark-last num" aria-hidden="true">{fmt(last)}</span>}
    </span>
  );
}
