import { useEffect, useRef, useState, type ReactNode } from "react";
import { ApiError } from "../api/client";
import type { Seg } from "../lib/diff";
import { COPY_FEEDBACK_MS } from "../motion/timing";

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

/**
 * Loading state. By default a shimmering skeleton block (decorative; it shows no data-shaped
 * content) with the label as text. `inline` is the small dot pulse used next to a button.
 * Distinct from the error and empty states, which have their own components.
 */
export function Spinner(props: { label?: string; inline?: boolean; lines?: number }) {
  const label = <span className={props.label ? "muted skeleton-label" : "sr-only"}>{props.label ?? "Loading"}</span>;
  if (props.inline) {
    return (
      <span role="status" aria-live="polite" className="dots-wrap">
        <span className="dots" aria-hidden="true"><i /><i /><i /></span>
        {label}
      </span>
    );
  }
  return (
    <div role="status" aria-live="polite" aria-busy="true" className="skeleton-wrap">
      {label}
      <div className="skeleton" aria-hidden="true">
        {Array.from({ length: props.lines ?? 3 }, (_, i) => <i key={i} className={`sk-line sk-${i % 3}`} />)}
      </div>
    </div>
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

/** Copy-to-clipboard button. "Copied" is shown only after the write actually succeeded. */
export function CopyButton({ text }: { text: string }) {
  const [st, setSt] = useState<"idle" | "copied" | "failed">("idle");
  const timer = useRef<ReturnType<typeof setTimeout>>();
  useEffect(() => () => clearTimeout(timer.current), []);
  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setSt("copied");
    } catch {
      setSt("failed");
    }
    clearTimeout(timer.current);
    timer.current = setTimeout(() => setSt("idle"), COPY_FEEDBACK_MS);
  }
  const msg = st === "copied" ? "Copied" : st === "failed" ? "Copy failed" : "Copy";
  return (
    <>
      <button type="button" className={`copy-btn ${st}`} onClick={copy} aria-label="Copy to clipboard">{msg}</button>
      <span className="sr-only" aria-live="polite">{st === "idle" ? "" : msg}</span>
    </>
  );
}

/** Code with a copy button. `segments` (from lib/diff) highlight tokens; their text joins back to `code`. */
export function CodeBlock(props: { code: string; caption?: string; segments?: Seg[]; side?: "gold" | "other" }) {
  return (
    <div className="codeblock">
      {props.caption && <p className="code-cap">{props.caption}</p>}
      <div className="code-wrap">
        <pre className="code"><code>
          {props.segments
            ? props.segments.map((g, i) => (g.changed ? <mark key={i} className={`diff-${props.side ?? "other"}`}>{g.text}</mark> : g.text))
            : props.code}
        </code></pre>
        <CopyButton text={props.code} />
      </div>
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
