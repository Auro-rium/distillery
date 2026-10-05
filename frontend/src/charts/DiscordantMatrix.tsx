// Discordant pairs from the McNemar test: items only the student got right vs items only the base got
// right. These two counts are what the test compares. Optional concordant cells make it a full 2x2.
// Shading strength is proportional to each cell's count (geometry only; the count itself is text).
import type { CSSProperties } from "react";
import { fmtInt } from "../api/format";
import { cx } from "../ui/cx";
import { ChartFigure, finite, type Formatter } from "./core";

export interface DiscordantMatrixProps {
  /** Items the student got right and the base got wrong. */
  studentOnly: number | null | undefined;
  /** Items the base got right and the student got wrong. */
  baseOnly: number | null | undefined;
  bothRight?: number | null;
  bothWrong?: number | null;
  title: string;
  caption?: string;
  aLabel?: string;
  bLabel?: string;
  format?: Formatter;
  className?: string;
}

export function DiscordantMatrix(props: DiscordantMatrixProps) {
  const fmt = props.format ?? fmtInt;
  const a = props.aLabel ?? "Student";
  const b = props.bLabel ?? "Base";
  const full = props.bothRight !== undefined || props.bothWrong !== undefined;
  const cells = [
    { k: "a-only", label: `${a} right, ${b} wrong`, v: props.studentOnly, cls: "m-student" },
    { k: "b-only", label: `${b} right, ${a} wrong`, v: props.baseOnly, cls: "m-base" },
    ...(full
      ? [
          { k: "both-right", label: "Both right", v: props.bothRight ?? null, cls: "concordant" },
          { k: "both-wrong", label: "Both wrong", v: props.bothWrong ?? null, cls: "concordant" },
        ]
      : []),
  ];
  const max = Math.max(0, ...cells.map((c) => c.v).filter(finite));
  return (
    <ChartFigure title={props.title} caption={props.caption} className={cx("discordant", full && "full", props.className)}>
      <dl className="dm-grid">
        {cells.map((c) => (
          <div key={c.k} className={cx("dm-cell", c.cls)} style={{ "--share": finite(c.v) && max > 0 ? c.v / max : 0 } as CSSProperties}>
            <dt>{c.label}</dt>
            <dd className="num">{fmt(c.v)}</dd>
          </div>
        ))}
      </dl>
    </ChartFigure>
  );
}
