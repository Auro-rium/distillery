// One rung of the proof ladder: an id ("W1"), what was proven, its PASS/FAIL state, one primary number
// and a link to the evidence file. Everything shown is a prop; the card formats nothing itself.
import type { ReactNode } from "react";
import { Badge, type Tone } from "../ui/Card";
import { cx } from "../ui/cx";

export type ProofStatus = "pass" | "partial" | "fail" | "pending" | "unknown";

const STATUS: Record<ProofStatus, { text: string; tone: Tone }> = {
  pass: { text: "PASS", tone: "ok" },
  partial: { text: "PARTIAL", tone: "warn" },
  fail: { text: "FAIL", tone: "bad" },
  pending: { text: "PENDING", tone: "warn" },
  unknown: { text: "No verdict", tone: "neutral" },
};

export interface ProofCardProps {
  /** Short rung id shown in the corner, e.g. "W1". */
  rung: string;
  title: string;
  status: ProofStatus;
  /** The one primary number, already formatted by the caller (api/format.ts). */
  metric?: ReactNode;
  metricLabel?: string;
  /** One line of supporting text. */
  detail?: ReactNode;
  /** A small visual (e.g. a Sparkline). */
  visual?: ReactNode;
  href?: string | null;
  linkLabel?: string;
  className?: string;
  /** Heading level for the title (default 3). */
  level?: 2 | 3 | 4;
}

export function ProofCard(props: ProofCardProps) {
  const s = STATUS[props.status];
  const H = `h${props.level ?? 3}` as "h2" | "h3" | "h4";
  return (
    <article className={cx("proof-card", `is-${props.status}`, props.className)}>
      <header className="proof-head">
        <span className="proof-rung num">{props.rung}</span>
        <Badge tone={s.tone}>{s.text}</Badge>
      </header>
      <H className="proof-title">{props.title}</H>
      {props.metric !== undefined && (
        <div className="proof-metric">
          <span className="proof-value num">{props.metric}</span>
          {props.metricLabel && <span className="proof-mlabel eyebrow">{props.metricLabel}</span>}
        </div>
      )}
      {props.visual && <div className="proof-visual">{props.visual}</div>}
      {props.detail && <p className="proof-detail muted">{props.detail}</p>}
      {props.href && (
        <a className="proof-link" href={props.href} target="_blank" rel="noreferrer noopener">
          {props.linkLabel ?? "Evidence"}
          <span aria-hidden="true"> →</span>
          <span className="sr-only"> (opens in a new tab)</span>
        </a>
      )}
    </article>
  );
}
