import { useState, type FormEvent, type KeyboardEvent } from "react";
import { Link } from "react-router-dom";
import { api } from "../../api/client";
import { fmtUsd } from "../../api/format";
import type { PlaygroundResponse } from "../../api/types";
import { ApiErrorState } from "../../components";
import { Badge, Button, Card, Field, Spinner, Textarea } from "../../ui";
import { Allowance } from "./Allowance";
import { Failure } from "./Failure";
import { Pane } from "./Pane";
import { useLimits } from "./useLimits";
import "./playground.css";

export { Pane, RowsPreview } from "./Pane";
export { Failure } from "./Failure";

/** The API's own limit on the question (docs/API_CONTRACT.md: question <= 500 chars); the input enforces it. */
const MAX = 500;
const MODELS = [
  ["teacher", "Teacher"],
  ["base", "Base"],
  ["student", "Student"],
] as const;

/**
 * Ask one question of the served models. Everything shown is what the API returned: the caps and today's
 * spend (GET /api/config), each model's availability with its own reason, its SQL, whether it was verified,
 * its rows, the request's cost and note. Nothing is generated here, and a request that failed is never drawn
 * as an empty answer.
 */
export default function Playground() {
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const [out, setOut] = useState<PlaygroundResponse | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [limits, refreshLimits] = useLimits();
  const pg = limits.state === "ok" ? limits.pg : null;
  const disabled = pg !== null && !pg.enabled;

  async function submit(ev?: FormEvent) {
    ev?.preventDefault();
    if (busy || disabled || q.trim() === "") return;
    setBusy(true);
    setErr(null);
    setOut(null);
    try {
      setOut(await api.playground(q.trim()));
    } catch (e) {
      setErr(e);
    } finally {
      setBusy(false);
      refreshLimits(); // the day's spend has moved, or the limit was hit: show the server's current figures
    }
  }
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); void submit(); }
  };
  const nobody = out !== null && MODELS.every(([k]) => !out.results[k].available);

  return (
    <div className="pg">
      <header className="pg-head">
        <h1>Playground</h1>
        <p className="muted pg-lead">Ask one question. Each model that can be served answers; the rest say why not.</p>
      </header>

      <div className="pg-top">
        <Card className="pg-ask">
          <h2 className="pg-h">Ask a question</h2>
          {disabled && (
            <div role="status" className="pg-off">
              <Badge tone="warn">playground disabled</Badge>
              <p>The server reports the playground as disabled, so no question can be sent. <Link to="/">See replay</Link></p>
            </div>
          )}
          <form onSubmit={submit} className="pg-form">
            <Field label="Question" help="One question about the demo database, in plain English.">
              <Textarea rows={4} maxLength={MAX} value={q} disabled={disabled} onChange={(e) => setQ(e.target.value)} onKeyDown={onKey} />
            </Field>
            <div className="pg-actions">
              <Button type="submit" variant="primary" disabled={disabled || q.trim() === ""} loading={busy}>Ask</Button>
              <span className="muted pg-count">{q.length} / {MAX}</span>
              <span className="muted pg-hint">Ctrl+Enter also sends</span>
            </div>
          </form>
        </Card>
        <div className="pg-side">
          {limits.state === "loading" && <Spinner lines={2} label="Loading the playground limits" />}
          {limits.state === "error" && <ApiErrorState title="Could not load the playground limits" error={limits.error} />}
          {pg && <Allowance pg={pg} />}
        </div>
      </div>

      <section className="pg-out">
        <span className="sr-only" role="status">{out && !busy ? "Answers are ready." : ""}</span>
        {busy && <Spinner label="Asking the models" />}
        {err !== null && <Failure e={err} />}
        {out && !busy && (
          <div className="pg-answers" role="region" aria-label="Answers">
            <h2 className="pg-h">Answers</h2>
            {nobody && <p className="pg-none">None of the models could answer this request.</p>}
            <div className="pg-panes">
              {MODELS.map(([k, label]) => <Pane key={k} name={label} r={out.results[k]} />)}
            </div>
            <p className="muted pg-cost">Cost of this request: {fmtUsd(out.cost_usd, 6)} · {out.note}</p>
          </div>
        )}
      </section>
    </div>
  );
}
