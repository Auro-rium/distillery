import type { ReactNode } from "react";
import { cx } from "./cx";

export type Tone = "neutral" | "ok" | "bad" | "warn" | "info";
/** The three models; each has one fixed colour on every chart and stat (student green, teacher amber, base cyan). */
export type ModelTone = "student" | "teacher" | "base";

/**
 * A panel. With a `title` its first child is an h3 drawn as the header strip; `meta` (a badge, a link)
 * sits at the right end of that strip. `tone` tints the border for a pass/fail/warn panel.
 */
export function Card(props: {
  title?: string;
  children: ReactNode;
  className?: string;
  meta?: ReactNode;
  tone?: Tone;
  id?: string;
  "aria-labelledby"?: string;
}) {
  const tone = props.tone && props.tone !== "neutral" ? `tone-${props.tone}` : null;
  return (
    <section id={props.id} aria-labelledby={props["aria-labelledby"]} className={cx("card", tone, props.className)}>
      {props.title && props.meta ? (
        <div className="card-head">
          <h3>{props.title}</h3>
          {props.meta}
        </div>
      ) : (
        props.title && <h3>{props.title}</h3>
      )}
      {props.children}
    </section>
  );
}

/** A big numeral over a small label. Displays a value formatted by the caller; never computes numbers. */
export function Stat(props: { label: string; value: ReactNode; hint?: ReactNode; tone?: Tone | ModelTone; size?: "md" | "lg"; className?: string }) {
  const tone = props.tone && props.tone !== "neutral" ? props.tone : null;
  return (
    <div className={cx("stat", tone, props.size === "lg" && "lg", props.className)}>
      <div className="v">{props.value}</div>
      <div className="l eyebrow">{props.label}</div>
      {props.hint && <div className="h">{props.hint}</div>}
    </div>
  );
}

/** A status chip. Toned chips get a leading LED dot (CSS only, so the text stays exactly `children`). */
export function Badge(props: { tone?: Tone; children: ReactNode; size?: "md" | "lg"; className?: string; title?: string }) {
  const t = props.tone && props.tone !== "neutral" ? props.tone : null;
  return <span className={cx("badge", t, props.size === "lg" && "lg", props.className)} title={props.title}>{props.children}</span>;
}
