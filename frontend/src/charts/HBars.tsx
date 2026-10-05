// Horizontal bars on a fixed 0..1 axis, optionally grouped and with CI whiskers. Shared by AccuracyBars
// and StressBars. The axis has gridlines only (no invented tick numbers); every value is text at the end
// of its bar, and each bar is focusable with its exact value (and CI) as name and tooltip.
import { fmtPercent } from "../api/format";
import { Tooltip } from "../ui/Tooltip";
import { cx } from "../ui/cx";
import { BAR_GAP, BAR_H, BAR_LABEL_W, BAR_VALUE_W, GROUP_GAP, PAD_COMPACT, UNIT_GRID, WHISKER_CAP } from "./constants";
import { finite, scale, unit, useChartWidth, type Formatter, type ModelKey } from "./core";

export interface HBar {
  key: ModelKey;
  label: string;
  value: number | null | undefined;
  ci?: [number, number] | null;
}
export interface HBarGroup {
  name?: string;
  sub?: string;
  bars: HBar[];
}

const GROUP_HEAD = 20;

export function HBars(props: { groups: HBarGroup[]; format?: Formatter; ciLabel?: string }) {
  const fmt = props.format ?? ((v) => fmtPercent(v));
  const [ref, width] = useChartWidth<HTMLDivElement>();
  const x0 = PAD_COMPACT.left + BAR_LABEL_W;
  const x1 = Math.max(x0 + BAR_H, width - BAR_VALUE_W - PAD_COMPACT.right);
  const sx = scale(0, 1, x0, x1);

  let y = PAD_COMPACT.top;
  const layout = props.groups.map((g) => {
    const head = g.name ? y : null;
    if (g.name) y += GROUP_HEAD;
    const bars = g.bars.map((b) => {
      const top = y;
      y += BAR_H + BAR_GAP;
      return { b, top };
    });
    y += GROUP_GAP - BAR_GAP;
    return { g, head, bars };
  });
  const height = y + PAD_COMPACT.bottom - GROUP_GAP;
  const plotBottom = height - PAD_COMPACT.bottom;

  return (
    <div ref={ref} className="chart-box">
      <svg className="chart-svg" width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        {UNIT_GRID.map((t) => (
          <line key={t} className={t === 0 ? "axis" : "grid"} x1={sx(t)} x2={sx(t)} y1={PAD_COMPACT.top} y2={plotBottom} />
        ))}
        {layout.map(({ g, head, bars }, gi) => (
          <g key={g.name ?? gi} className="bar-group">
            {head !== null && (
              <text className="group-name" x={PAD_COMPACT.left} y={head + GROUP_HEAD / 2} dy="0.32em">
                {g.name}
                {g.sub && <tspan className="group-sub">{` ${g.sub}`}</tspan>}
              </text>
            )}
            {bars.map(({ b, top }) => {
              const ci = b.ci && finite(b.ci[0]) && finite(b.ci[1]) ? b.ci : null;
              const name = `${g.name ? `${g.name}, ` : ""}${b.label}: ${fmt(b.value)}${ci ? `, ${props.ciLabel ?? "CI"} ${fmt(ci[0])} to ${fmt(ci[1])}` : ""}`;
              const mid = top + BAR_H / 2;
              return (
                <g key={b.key} className={cx("bar", `m-${b.key}`)}>
                  <text className="bar-label" x={x0 - 8} y={mid} dy="0.32em" textAnchor="end">{b.label}</text>
                  <rect className="bar-track" x={x0} y={top} width={x1 - x0} height={BAR_H} />
                  <Tooltip content={name} side="top">
                    <rect
                      className="bar-fill"
                      tabIndex={0}
                      role="img"
                      aria-label={name}
                      x={x0}
                      y={top}
                      width={finite(b.value) ? Math.max(0, sx(unit(b.value)) - x0) : 0}
                      height={BAR_H}
                    />
                  </Tooltip>
                  {ci && (
                    <g className="whisker" aria-hidden="true">
                      <line x1={sx(unit(ci[0]))} x2={sx(unit(ci[1]))} y1={mid} y2={mid} />
                      <line x1={sx(unit(ci[0]))} x2={sx(unit(ci[0]))} y1={mid - WHISKER_CAP} y2={mid + WHISKER_CAP} />
                      <line x1={sx(unit(ci[1]))} x2={sx(unit(ci[1]))} y1={mid - WHISKER_CAP} y2={mid + WHISKER_CAP} />
                    </g>
                  )}
                  <text className="bar-value" x={x1 + 8} y={mid} dy="0.32em">{fmt(b.value)}</text>
                </g>
              );
            })}
          </g>
        ))}
      </svg>
    </div>
  );
}
