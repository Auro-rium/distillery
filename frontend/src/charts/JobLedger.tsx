// The fine-tune job ledger: started / adopted / closed rows (and serving images), in the order given.
// It makes the kill -> adopt story visible. Shows exactly the rows passed in; empty means "no rows".
import type { ReactNode } from "react";
import { fmtText } from "../api/format";
import { cx } from "../ui/cx";

export interface LedgerRow {
  /** Stable key (event id); falls back to the index. */
  id?: string | number;
  /** Event kind as the API names it, e.g. "finetune_job_started". */
  kind: string;
  /** Short display name for the kind (default: `kind`). */
  label?: string;
  /** Timestamp text as the API sent it. */
  at?: string | null;
  jobId?: string | null;
  detail?: ReactNode;
}

/** Colour class for well-known kinds: started = info, adopted = warn (a recovered job), closed = ok. */
function toneOf(kind: string): string {
  if (/adopt/i.test(kind)) return "warn";
  if (/clos|complete|succeed|done/i.test(kind)) return "ok";
  if (/fail|cancel|error/i.test(kind)) return "bad";
  if (/start|creat|submit/i.test(kind)) return "info";
  return "neutral";
}

export function JobLedger(props: { rows: LedgerRow[]; title?: string; empty?: ReactNode; className?: string }) {
  if (!props.rows.length) {
    return <p className={cx("ledger-empty muted", props.className)}>{props.empty ?? "No job events recorded for this run."}</p>;
  }
  return (
    <ol className={cx("job-ledger", props.className)} aria-label={props.title ?? "Job ledger"}>
      {props.rows.map((r, i) => (
        <li key={r.id ?? i} className={cx("ledger-row", `tone-${toneOf(r.kind)}`)}>
          <span className="ledger-dot" aria-hidden="true" />
          <span className="ledger-kind">{r.label ?? r.kind}</span>
          {r.at !== undefined && <time className="ledger-at num">{fmtText(r.at)}</time>}
          {r.jobId && <code className="ledger-job">{r.jobId}</code>}
          {r.detail && <span className="ledger-detail muted">{r.detail}</span>}
        </li>
      ))}
    </ol>
  );
}
