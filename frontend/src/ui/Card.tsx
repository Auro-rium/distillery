import type { ReactNode } from "react";
import { cx } from "./cx";

export function Card(props: { title?: string; children: ReactNode; className?: string }) {
  return (
    <section className={cx("card", props.className)}>
      {props.title && <h3>{props.title}</h3>}
      {props.children}
    </section>
  );
}

/** Displays a value formatted by the caller. Never computes numbers. */
export function Stat(props: { label: string; value: ReactNode; hint?: ReactNode }) {
  return (
    <div className="stat">
      <div className="v">{props.value}</div>
      <div className="l eyebrow">{props.label}</div>
      {props.hint && <div className="h">{props.hint}</div>}
    </div>
  );
}

export type Tone = "neutral" | "ok" | "bad" | "warn" | "info";
export function Badge(props: { tone?: Tone; children: ReactNode }) {
  const t = props.tone && props.tone !== "neutral" ? ` ${props.tone}` : "";
  return <span className={`badge${t}`}>{props.children}</span>;
}
