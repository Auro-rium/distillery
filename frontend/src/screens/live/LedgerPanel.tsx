import { JobLedger, type LedgerRow } from "../../charts";
import { fmtText } from "../../api/format";
import type { ExperimentRow, ExperimentsResult } from "../../api/types";
import { Panel } from "./Panel";

const JOB_PREFIX = "finetune_job_";

/** "finetune_job_adopted" -> "adopted"; "serving_image" -> "serving image". Only strips/rewrites the API's own name. */
export function ledgerLabel(name: string): string {
  return (name.startsWith(JOB_PREFIX) ? name.slice(JOB_PREFIX.length) : name).replace(/_/g, " ");
}

const text = (v: unknown): string | null => (typeof v === "string" && v ? v : typeof v === "number" ? String(v) : null);

/** Detail line: the payload's own fields, named as the API names them (round, outcome, checkpoint, student...). */
function detailOf(data: Record<string, unknown> | null | undefined): string | undefined {
  if (!data) return undefined;
  const parts: string[] = [];
  for (const k of ["round", "outcome", "kind", "student", "image", "checkpoint_id"]) {
    const v = text(data[k]);
    if (v !== null) parts.push(`${k.replace(/_/g, " ")} ${v}`);
  }
  return parts.length ? parts.join(" · ") : undefined;
}

export function ledgerRows(rows: ExperimentRow[]): LedgerRow[] {
  return rows.map((r, i) => ({
    id: i,
    kind: r.name,
    label: ledgerLabel(r.name),
    at: r.created_at ?? null,
    jobId: text(r.data?.job_id),
    detail: detailOf(r.data),
  }));
}

/**
 * Fine-tune job ledger: started / adopted / closed rows (and serving images) as stored, in order, so a job that was
 * killed and then adopted by a later run shows both events. `result` is null until loaded, `error` is the load failure.
 */
export function LedgerPanel({ result, error }: { result: ExperimentsResult | null; error: string | null }) {
  let body;
  if (error !== null) body = <p className="muted lp-none" role="alert">Job ledger unavailable: {error}</p>;
  else if (result === null) body = <p className="muted lp-none">Loading job ledger.</p>;
  else {
    const empty = result.source === "unavailable"
      ? "This recording carries no job ledger."
      : "No job events recorded for this run.";
    body = <JobLedger rows={ledgerRows(result.experiments)} title="Fine-tune jobs" empty={empty} />;
  }
  return (
    <Panel title="Job ledger" className="lp-ledger" aside={result ? <span className="eyebrow">{fmtText(result.source)}</span> : undefined}>
      {body}
    </Panel>
  );
}
