import { useEffect, useRef, useState, type ReactNode } from "react";
import { ApiError } from "../api/client";
import type { Seg } from "../lib/diff";
import { COPY_FEEDBACK_MS } from "../motion/timing";
import { Button } from "../ui";

// The kit owns these; the names screens import stay available from here.
export { Badge, Card, Spinner, Stat, type Tone } from "../ui";

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

export function ErrorState(props: { title?: string; message: string; code?: string | null; onRetry?: () => void }) {
  return (
    <div className="state error" role="alert">
      <h3>{props.title ?? "Something went wrong"}</h3>
      <p>{props.code ? <><span className="mono">{props.code}</span>: </> : null}{props.message}</p>
      {props.onRetry && <Button onClick={props.onRetry}>Retry</Button>}
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
