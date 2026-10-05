import { fmtInt, fmtPercent } from "../../api/format";

// Table twin of the report's charts (charts/AccuracyBars, charts/StressBars): every value, CI bound and
// "not measured" as plain text. It computes no statistic.

export type ModelKey = "base" | "student" | "teacher";

/** Fixed order and names; each model has one colour on every chart (student green, teacher amber, base cyan). */
export const MODELS: readonly { key: ModelKey; label: string }[] = [
  { key: "base", label: "Base" },
  { key: "student", label: "Student" },
  { key: "teacher", label: "Teacher" },
];

export interface BarRow {
  key: ModelKey;
  label: string;
  value: number | null;
  ci: [number, number] | null;
}

const finite = (v: number | null): v is number => typeof v === "number" && Number.isFinite(v);
export const ciOf = (r: BarRow): [number, number] | null => (r.ci && finite(r.ci[0]) && finite(r.ci[1]) ? r.ci : null);
export const ciText = (ci: [number, number]): string => `${fmtPercent(ci[0])} to ${fmtPercent(ci[1])}`;

export interface TableRow {
  name: string;
  n?: number | null;
  values: BarRow[];
}

/** The table twin of the charts: every value, CI bound and "not measured" as plain text. */
export function AccuracyTable(props: {
  caption: string;
  columns: readonly { key: ModelKey; label: string }[];
  rows: TableRow[];
  showN?: boolean;
  nLabel?: string;
}) {
  return (
    <details className="acc-table">
      <summary>Table view</summary>
      <div className="tbl-wrap">
        <table className="tbl">
          <caption className="sr-only">{props.caption}</caption>
          <thead>
            <tr>
              <th scope="col">Class</th>
              {props.showN && <th scope="col">{props.nLabel ?? "n"}</th>}
              {props.columns.map((c) => <th scope="col" key={c.key}>{c.label}</th>)}
            </tr>
          </thead>
          <tbody>
            {props.rows.map((row) => (
              <tr key={row.name}>
                <th scope="row">{row.name}</th>
                {props.showN && <td>{fmtInt(row.n)}</td>}
                {props.columns.map((c) => {
                  const cell = row.values.find((v) => v.key === c.key);
                  const ci = cell ? ciOf(cell) : null;
                  return (
                    <td key={c.key}>
                      {fmtPercent(cell?.value)}
                      {ci && <span className="muted"> CI {ciText(ci)}</span>}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );
}
