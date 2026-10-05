// Shared pieces for the charts: measured width, linear scales, the figure frame and a focusable data
// point with a tooltip. Charts draw only values they are given; every label is a prop formatted by
// api/format.ts (or a caller-supplied formatter), never a computed "nice" tick.
import { useLayoutEffect, useRef, useState, type ReactNode, type SVGProps } from "react";
import { Tooltip } from "../ui/Tooltip";
import { cx } from "../ui/cx";
import { DEFAULT_WIDTH, MIN_WIDTH } from "./constants";

export type ModelKey = "student" | "teacher" | "base";
export type Formatter = (v: number | null | undefined) => string;

export const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

/** Linear map from [d0, d1] to [r0, r1]. Geometry only. */
export function scale(d0: number, d1: number, r0: number, r1: number): (v: number) => number {
  const span = d1 - d0;
  return (v) => (span === 0 ? (r0 + r1) / 2 : r0 + ((v - d0) / span) * (r1 - r0));
}

/** Clamp a fraction to the 0..1 axis. */
export const unit = (v: number): number => Math.min(1, Math.max(0, v));

/** Width of the element in px, following resizes. jsdom has no layout, so it reports DEFAULT_WIDTH. */
export function useChartWidth<T extends HTMLElement>(): [React.RefObject<T>, number] {
  const ref = useRef<T>(null);
  const [w, setW] = useState(DEFAULT_WIDTH);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const read = () => {
      const cw = el.clientWidth;
      if (cw > 0) setW(Math.max(MIN_WIDTH, Math.floor(cw)));
    };
    read();
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(read);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

/**
 * A chart's frame: a <figure> labelled by its title, with the one-line caption saying what it measures.
 * `legend` (optional) renders between plot and caption.
 */
export function ChartFigure(props: {
  title: string;
  caption?: ReactNode;
  className: string;
  children: ReactNode;
  legend?: ReactNode;
  figureRef?: React.Ref<HTMLElement>;
  showTitle?: boolean;
}) {
  return (
    <figure className={cx("chart", props.className)} aria-label={props.title} ref={props.figureRef}>
      {props.showTitle && <div className="chart-title">{props.title}</div>}
      {props.children}
      {props.legend}
      {props.caption && <figcaption className="chart-cap">{props.caption}</figcaption>}
    </figure>
  );
}

/** Colour key: a swatch (colour) beside a name (text), so colour is never the only carrier. */
export function ChartLegend(props: { items: { key: string; label: string; className: string }[] }) {
  return (
    <ul className="chart-legend" aria-label="Chart key">
      {props.items.map((i) => (
        <li key={i.key} className={i.className}>
          <i className="swatch" aria-hidden="true" />
          {i.label}
        </li>
      ))}
    </ul>
  );
}

export const MODEL_LABEL: Record<ModelKey, string> = { student: "Student", teacher: "Teacher", base: "Base" };

/**
 * A data point that keyboard and pointer users can inspect: focusable, named by `label` (which carries
 * the exact formatted values), with the same text in a tooltip.
 */
export function DataPoint({ label, ...rest }: { label: string } & SVGProps<SVGCircleElement>) {
  return (
    <Tooltip content={label} side="top">
      <circle tabIndex={0} role="img" aria-label={label} {...rest} />
    </Tooltip>
  );
}
