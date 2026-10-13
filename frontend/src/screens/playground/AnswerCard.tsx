import { fmtNumber, fmtText, fmtUsd } from "../../api/format";
import type { AcmeAskResponse } from "../../api/types";
import { Badge, Card } from "../../ui";
import { CodeBlock } from "../report/CodeBlock";
import { RowsPreview } from "./Pane";

/** Who answered, in words. `served_by` is the gateway's own field; an unknown value is shown, not mapped to one of ours. */
function served(by: string): { label: string; tone: "ok" | "warn" | "info"; model: "student" | "teacher" | null } {
  if (by === "student") return { label: "Student (certified)", tone: "ok", model: "student" };
  if (by === "teacher") return { label: "Teacher (deferred)", tone: "warn", model: "teacher" };
  return { label: by, tone: "info", model: null };
}

/**
 * One answer to one question, as POST /api/acme/ask returned it: who served it and why (confidence score against
 * the threshold lambda, in the gateway's own words), the SQL, whether it ran and returned rows, the rows, the
 * one-line summary, and what this request cost and how long it took.
 */
export function AnswerCard({ r }: { r: AcmeAskResponse }) {
  const d = r.distillery;
  const who = served(d.served_by);
  const a = r.answer;
  return (
    <Card
      title="Answer"
      className={`pg-answer pg-by-${who.model ?? "other"}`}
      tone={who.tone}
      meta={<Badge tone={who.tone} size="lg">{who.label}</Badge>}
    >
      <p className="pg-q"><span className="eyebrow">question</span> {r.question}</p>

      <div className="pg-why">
        <p className="pg-conf">
          <span className="eyebrow">confidence</span>{" "}
          <span className="num">{fmtNumber(d.score, 3)}</span>
          <span className="muted"> against threshold lambda </span>
          <span className="num">{fmtNumber(d.lambda, 3)}</span>
        </p>
        <p className="pg-reason"><span className="eyebrow">reason</span> {fmtText(d.reason)}</p>
      </div>

      {r.summary !== null ? (
        <p className="pg-summary" aria-label="Summary">{r.summary}</p>
      ) : r.summary_error ? (
        <p className="muted pg-summary-off">
          Summary unavailable (<span className="mono">{r.summary_error.code}</span>): {r.summary_error.message}
          {r.summary_error.retry_after ? ` Retry after ${r.summary_error.retry_after} seconds.` : ""}
        </p>
      ) : null}

      {a.sql ? <CodeBlock label="SQL" language="sql" code={a.sql} /> : <p className="muted">No SQL was returned.</p>}

      <div className="pg-tags">
        {a.verified ? <Badge tone="ok">verified: ran read-only and returned rows</Badge> : <Badge tone="bad">not verified: the SQL did not run or returned no rows</Badge>}
      </div>
      {a.error && <p role="alert" className="pg-err">Error: {a.error}</p>}
      {!a.error && a.sql && <RowsPreview rows={{ columns: a.columns, rows: a.rows, row_count: a.row_count }} />}

      <dl className="pg-meter-list">
        <div><dt>SQL call (gateway), cost</dt><dd className="num">{fmtUsd(d.cost_usd, 6)}</dd></div>
        <div><dt>The same call on the teacher would have cost</dt><dd className="num">{fmtUsd(d.teacher_cost_usd_counterfactual, 6)}</dd></div>
        {r.summary_cost_usd !== null && <div><dt>Summary call, cost</dt><dd className="num">{fmtUsd(r.summary_cost_usd, 6)}</dd></div>}
        <div><dt>SQL call, latency</dt><dd className="num">{fmtNumber(d.latency_ms, 1)} ms</dd></div>
        <div><dt>Request</dt><dd className="mono">{d.request_id}</dd></div>
      </dl>
      <div className="muted pg-fine">
        <p><span className="eyebrow">cost basis</span> {d.usage_basis}</p>
        <p><span className="eyebrow">teacher comparison</span> {d.counterfactual_basis}</p>
      </div>
    </Card>
  );
}
