import { useEffect, useState, type CSSProperties } from "react";
import { fmtInt, fmtPercent } from "../../api/format";
import { useInView } from "../../motion/useInView";

// Horizontal accuracy bars with optional confidence-interval whiskers, drawn with plain elements so the
// bar ends stay crisp at every width. It draws only values it is given and computes no statistic. The
// exact value is always text beside its bar; only the bars and whiskers animate (transform / opacity),
// once, when the chart first scrolls into view. Reduced motion shows the final state at once.

export type ModelKey = "base" | "student" | "teacher";

/** Fixed order and names; the colour of each model is the same on every chart (see report.css). */
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

// Position on the fixed 0..1 axis, as a CSS percentage. Geometry only: it is never shown as text.
const pct = (v: number): string => `${Math.min(1, Math.max(0, v)) * 100}%`;
const finite = (v: number | null): v is number => typeof v === "number" && Number.isFinite(v);
const ciOf = (r: BarRow): [number, number] | null => (r.ci && finite(r.ci[0]) && finite(r.ci[1]) ? r.ci : null);
const TICKS = [0, 0.5, 1] as const;

export const ciText = (ci: [number, number]): string => `${fmtPercent(ci[0])} to ${fmtPercent(ci[1])}`;

/** One key for the three models: a swatch beside the name (the name is text, the swatch carries the colour). */
export function Legend() {
  return (
    <ul className="legend" role="list" aria-label="Chart key">
      {MODELS.map((m) => (
        <li key={m.key} className={`legend-item ${m.key}`}>
          <i className="swatch" aria-hidden="true" />
          {m.label}
        </li>
      ))}
    </ul>
  );
}

/**
 * Bars start at zero width until the chart has been seen. If the browser never reports it (print, a
 * full-page capture, a background tab) the bars are drawn anyway after this delay, so the geometry can
 * never stay hidden. The text values are on screen the whole time.
 */
const REVEAL_FALLBACK_MS = 1500;

export function AccuracyChart({ rows, title, showCi = true }: { rows: BarRow[]; title: string; showCi?: boolean }) {
  const [ref, seen] = useInView<HTMLElement>();
  const [late, setLate] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setLate(true), REVEAL_FALLBACK_MS);
    return () => clearTimeout(t);
  }, []);
  return (
    <figure className={showCi ? "acc" : "acc compact"} ref={ref} data-reveal={seen || late ? "done" : "pending"}>
      <ul className="acc-rows" role="list" aria-label={title}>
        {rows.map((r, i) => {
          const ci = showCi ? ciOf(r) : null;
          return (
            <li className={`acc-row ${r.key}`} key={r.key} style={{ "--i": i } as CSSProperties}>
              <span className="acc-label">{r.label}</span>
              <span className="acc-plot" aria-hidden="true">
                {finite(r.value) && <span className="acc-bar" style={{ width: pct(r.value) }} />}
                {ci && (
                  <span className="acc-ci" style={{ left: pct(ci[0]), width: pct(ci[1] - ci[0]) }}>
                    <span className="acc-ci-lab lo">{fmtPercent(ci[0])}</span>
                    <span className={`acc-ci-lab hi${ci[1] > 0.9 ? " edge" : ""}`}>{fmtPercent(ci[1])}</span>
                  </span>
                )}
              </span>
              <span className="acc-val">
                <strong>{fmtPercent(r.value)}</strong>
                {showCi && <span className="acc-detail">{ci ? `CI ${ciText(ci)}` : "no CI in report"}</span>}
              </span>
            </li>
          );
        })}
      </ul>
      <div className="acc-axis" aria-hidden="true">
        {TICKS.map((t) => <span key={t} style={{ left: pct(t) }}>{fmtPercent(t, 0)}</span>)}
      </div>
    </figure>
  );
}

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
