import { useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { ApiError, api } from "../../api/client";
import type { PlaygroundResponse, PlaygroundResult } from "../../api/types";
import { fmtInt, fmtUsd } from "../../api/format";
import { ApiErrorState, Badge, Card, CodeBlock, Spinner } from "../../components";
import { useAsync } from "../report/useAsync";

const MAX = 500;
const PANES = [
  ["teacher", "Teacher"],
  ["base", "Base"],
  ["student", "Student"],
] as const;

export function RowsPreview({ rows }: { rows: NonNullable<PlaygroundResult["rows_preview"]> }) {
  if (rows.length === 0) return <p className="muted">Query returned no rows.</p>;
  const first = rows[0];
  const cols = Array.isArray(first) ? first.map((_, i) => `col ${i + 1}`) : Object.keys(first);
  const cell = (r: unknown[] | Record<string, unknown>, i: number, c: string) =>
    String((Array.isArray(r) ? r[i] : r[c]) ?? "NULL");
  return (
    <div className="tbl-wrap">
      <table className="tbl">
        <thead><tr>{cols.map((c) => <th key={c}>{c}</th>)}</tr></thead>
        <tbody>{(rows as (unknown[] | Record<string, unknown>)[]).slice(0, 10).map((r, ri) => (
          <tr key={ri}>{cols.map((c, ci) => <td key={c}>{cell(r, ci, c)}</td>)}</tr>
        ))}</tbody>
      </table>
    </div>
  );
}

export function Pane({ name, r }: { name: string; r: PlaygroundResult }) {
  return (
    <Card title={name}>
      {!r.available ? (
        <p><Badge tone="warn">not available</Badge> <span>{r.reason ?? "No reason was given."}</span></p>
      ) : (
        <div className="stack">
          {r.verified === true && <Badge tone="ok">verified: matches the known answer</Badge>}
          {r.verified === false && <Badge tone="bad">verified: wrong result</Badge>}
          {r.verified === null && <Badge>not verified (question is not a known task)</Badge>}
          {r.sql ? <CodeBlock code={r.sql} /> : <p className="muted">No SQL returned.</p>}
          {r.error && <p role="alert" style={{ color: "var(--bad)" }}>Error: {r.error}</p>}
          {r.rows_preview && <RowsPreview rows={r.rows_preview} />}
        </div>
      )}
    </Card>
  );
}

export function Failure({ e }: { e: unknown }) {
  if (e instanceof ApiError && e.code === "demo_budget_exhausted")
    return (
      <div className="state" role="alert">
        <h3>Demo budget exhausted</h3>
        <p>The daily demo budget is used up. <Link to="/">See replay</Link> for stored runs.</p>
      </div>
    );
  if (e instanceof ApiError && e.status === 429)
    return (
      <div className="state" role="alert">
        <h3>Rate limit reached</h3>
        <p>{e.retryAfter ? `Try again in ${e.retryAfter} seconds.` : "Try again later."}</p>
      </div>
    );
  return <ApiErrorState error={e} />;
}

export default function Playground() {
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const [out, setOut] = useState<PlaygroundResponse | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [cfg] = useAsync(() => api.config(), "config");
  const pg = cfg.state === "ok" ? cfg.data.playground : null;

  async function submit(ev: FormEvent) {
    ev.preventDefault();
    setBusy(true);
    setErr(null);
    try {
      setOut(await api.playground(q.trim()));
    } catch (e) {
      setOut(null);
      setErr(e);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="stack">
      <h1>Playground</h1>
      <p className="muted">Ask one question. Each model that can be served answers; the rest say why not.</p>
      {cfg.state === "error" && <ApiErrorState title="Could not load the playground limits" error={cfg.error} />}
      {pg && !pg.enabled && <p role="status"><Badge tone="warn">playground disabled</Badge> <Link to="/">See replay</Link></p>}
      {pg && (
        <p className="muted">
          Limit {fmtInt(pg.per_ip_per_hour)} requests per hour per IP · daily budget spent {fmtUsd(pg.spent_today_usd, 2)} of {fmtUsd(pg.daily_cap_usd, 2)}
        </p>
      )}
      <form onSubmit={submit} className="stack">
        <label htmlFor="q" className="eyebrow">Question</label>
        <textarea
          id="q" rows={3} maxLength={MAX} value={q} onChange={(e) => setQ(e.target.value)}
          style={{ width: "100%", padding: 10, background: "var(--surface)", border: "1px solid var(--line)", borderRadius: "var(--radius)" }}
        />
        <div className="row">
          <button className="btn primary" type="submit" disabled={busy || q.trim() === "" || (pg !== null && !pg.enabled)}>Ask</button>
          <span className="muted">{q.length}/{MAX}</span>
          {busy && <Spinner label="Asking the models" />}
        </div>
      </form>
      {err !== null && <Failure e={err} />}
      {out && (
        <>
          <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))" }}>
            {PANES.map(([k, label]) => <Pane key={k} name={label} r={out.results[k]} />)}
          </div>
          <p className="muted">Cost of this request: {fmtUsd(out.cost_usd, 6)} · {out.note}</p>
        </>
      )}
    </div>
  );
}
