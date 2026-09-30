import { fmtInt } from "../../api/format";
import type { PlaygroundResult } from "../../api/types";
import { Badge, Card } from "../../ui";
import { SqlBlock } from "../report/SqlBlock";
import { cellText, normalizeRows } from "./rows";

/** Longest preview drawn; the server itself only previews a few rows. */
const DRAWN_ROWS = 10;

/** `rows` is whatever the API sent as rows_preview (see rows.ts for the accepted shapes). */
export function RowsPreview({ rows }: { rows: unknown }) {
  const r = normalizeRows(rows);
  if (r === null) return <p className="muted">The rows preview came back in a format that could not be shown.</p>;
  if (r.rows.length === 0) return <p className="muted">Query returned no rows.</p>;
  const shown = r.rows.slice(0, DRAWN_ROWS);
  return (
    <div className="pg-rows">
      <div className="tbl-wrap" tabIndex={0} role="region" aria-label="Rows returned by the SQL">
        <table className="pg-tbl">
          <thead><tr>{r.columns.map((c, i) => <th key={i} scope="col">{c}</th>)}</tr></thead>
          <tbody>
            {shown.map((row, ri) => (
              <tr key={ri}>
                {r.columns.map((_, ci) => {
                  const c = cellText(row[ci]);
                  return <td key={ci} className={c.isNull ? "null" : undefined}>{c.text}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted pg-rowcount">
        {r.total === null ? `${shown.length} rows` : `showing ${shown.length} of ${fmtInt(r.total)} rows`}
      </p>
    </div>
  );
}

/** One model's answer, exactly as the API returned it: availability with its reason, the SQL, whether it was verified, the rows. */
export function Pane({ name, r }: { name: string; r: PlaygroundResult }) {
  return (
    <Card title={name} className="pg-pane">
      {!r.available ? (
        <div className="pg-na">
          <Badge tone="warn">not available</Badge>
          <p>{r.reason ?? "No reason was given."}</p>
        </div>
      ) : (
        <div className="pg-body">
          <div className="pg-tags">
            {r.verified === true && <Badge tone="ok">verified: matches the known answer</Badge>}
            {r.verified === false && <Badge tone="bad">verified: wrong result</Badge>}
            {r.verified === null && <Badge>not verified (question is not a known task)</Badge>}
          </div>
          {r.sql ? <SqlBlock label="SQL" code={r.sql} /> : <p className="muted">No SQL returned.</p>}
          {r.error && <p role="alert" className="pg-err">Error: {r.error}</p>}
          {r.rows_preview !== null && r.rows_preview !== undefined && <RowsPreview rows={r.rows_preview} />}
        </div>
      )}
    </Card>
  );
}
