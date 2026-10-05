// Accuracy on the stress set (task families the student never trained on), one group per family,
// one bar per model. Values only from props; a family without a value shows "not measured".
import { fmtInt } from "../api/format";
import { cx } from "../ui/cx";
import { ChartFigure, ChartLegend, MODEL_LABEL, type Formatter, type ModelKey } from "./core";
import { HBars } from "./HBars";

export interface StressRow {
  name: string;
  n?: number | null;
  values: Partial<Record<ModelKey, number | null>>;
}

export function StressBars(props: {
  rows: StressRow[];
  title: string;
  caption?: string;
  models?: readonly ModelKey[];
  format?: Formatter;
  className?: string;
}) {
  const models = props.models ?? (["student", "teacher"] as const);
  const groups = props.rows.map((r) => ({
    name: r.name,
    sub: r.n === undefined || r.n === null ? undefined : `n ${fmtInt(r.n)}`,
    bars: models.map((m) => ({ key: m, label: MODEL_LABEL[m], value: r.values[m] ?? null })),
  }));
  const legend = <ChartLegend items={models.map((m) => ({ key: m, label: MODEL_LABEL[m], className: `m-${m}` }))} />;
  return (
    <ChartFigure title={props.title} caption={props.caption} legend={legend} className={cx("stress-bars", props.className)}>
      <HBars groups={groups} format={props.format} />
    </ChartFigure>
  );
}
