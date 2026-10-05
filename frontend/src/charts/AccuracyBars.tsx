// Held-out accuracy per model (base / student / teacher) with CI whiskers. The SVG successor of
// screens/report/AccuracyChart (which stays for the current report screen); same row shape.
import { cx } from "../ui/cx";
import { ChartFigure, ChartLegend, MODEL_LABEL, type Formatter, type ModelKey } from "./core";
import { HBars, type HBar } from "./HBars";

export type AccuracyRow = HBar;
export const MODEL_ORDER: readonly ModelKey[] = ["base", "student", "teacher"];

export function AccuracyBars(props: {
  rows: AccuracyRow[];
  title: string;
  caption?: string;
  format?: Formatter;
  showCi?: boolean;
  legend?: boolean;
  className?: string;
}) {
  const rows = props.showCi === false ? props.rows.map((r) => ({ ...r, ci: null })) : props.rows;
  const legend = props.legend ? (
    <ChartLegend items={rows.map((r) => ({ key: r.key, label: MODEL_LABEL[r.key], className: `m-${r.key}` }))} />
  ) : undefined;
  return (
    <ChartFigure title={props.title} caption={props.caption} legend={legend} className={cx("acc-bars", props.className)}>
      <HBars groups={[{ bars: rows }]} format={props.format} />
    </ChartFigure>
  );
}
