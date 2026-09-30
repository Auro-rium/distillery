import type { ReactNode } from "react";
import { ApiError } from "../api/client";

export const DRY_RUN_TEXT = "DRY RUN — fake models, numbers are NOT results";

export const UNLABELLED_TEXT =
  "Label unknown: the API response carried no dry_run flag. Do not treat these numbers as real results.";
export const LIVE_TEXT = "Live run, not a recorded replay";

/**
 * Persistent label for anything that shows run/report data. A missing flag is never read as
 * "real": only an explicit dry_run === false (and recorded === false) is labelled a live run.
 */
export function LabelBanner(props: {
  dry_run?: boolean | null;
  recorded?: boolean | null;
  recorded_at?: string | null;
}) {
  if (props.dry_run === true) {
    return <div className="label-banner dry" role="status">{DRY_RUN_TEXT}</div>;
  }
  if (props.dry_run === false && props.recorded === true) {
    return (
      <div className="label-banner recorded" role="status">
        Recorded run · {props.recorded_at ?? "time not recorded"} · real Token Factory jobs
      </div>
    );
  }
  if (props.dry_run === false && props.recorded === false) {
    return <div className="label-banner recorded" role="status">{LIVE_TEXT}</div>;
  }
  return <div className="label-banner dry" role="status">{UNLABELLED_TEXT}</div>;
}

export function Card(props: { title?: string; children: ReactNode; className?: string }) {
  return (
    <section className={`card ${props.className ?? ""}`.trim()}>
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

export function Spinner(props: { label?: string }) {
  return (
    <span role="status" aria-live="polite">
      <span className="spinner" aria-hidden="true" />
      <span className={props.label ? "muted" : "sr-only"} style={{ marginLeft: 8 }}>
        {props.label ?? ""}
      </span>
    </span>
  );
}

export function ErrorState(props: { title?: string; message: string; code?: string | null; onRetry?: () => void }) {
  return (
    <div className="state error" role="alert">
      <h3>{props.title ?? "Something went wrong"}</h3>
      <p>{props.code ? <><span className="mono">{props.code}</span>: </> : null}{props.message}</p>
      {props.onRetry && <button className="btn" onClick={props.onRetry}>Retry</button>}
    </div>
  );
}

export function EmptyState(props: { title: string; children?: ReactNode }) {
  return (
    <div className="state">
      <h3>{props.title}</h3>
      {props.children && <div className="muted">{props.children}</div>}
    </div>
  );
}

export function CodeBlock(props: { code: string; caption?: string }) {
  return (
    <div>
      {props.caption && <p className="code-cap">{props.caption}</p>}
      <pre className="code"><code>{props.code}</code></pre>
    </div>
  );
}

/** An API/network failure: shows the error code and message the API gave. Never an empty state. */
export function ApiErrorState(props: { error: unknown; title?: string; onRetry?: () => void }) {
  const e = props.error;
  const code = e instanceof ApiError ? e.code : null;
  const message = e instanceof Error ? e.message : "Request failed.";
  return <ErrorState title={props.title} code={code} message={message} onRetry={props.onRetry} />;
}
