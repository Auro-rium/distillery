import { fmtInt } from "../../api/format";
import { cellText, normalizeRows } from "./rows";

/** Longest preview drawn; the gateway itself returns at most twenty rows. */
const DRAWN_ROWS = 10;

/** `rows` is whatever the API sent (see rows.ts for the accepted shapes): a table of the rows the SQL returned. */
export function RowsPreview({ rows }: { rows: unknown }) {
  const r = normalizeRows(rows);
  if (r === null) return <p className="muted">The rows came back in a format that could not be shown.</p>;
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
