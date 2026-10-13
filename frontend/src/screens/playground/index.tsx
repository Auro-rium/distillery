import { useState, type FormEvent, type KeyboardEvent } from "react";
import { api } from "../../api/client";
import type { AcmeAskResponse } from "../../api/types";
import { ApiErrorState } from "../../components";
import { Button, Card, Field, Spinner, Textarea } from "../../ui";
import { AnswerCard } from "./AnswerCard";
import { DriftPanel } from "./DriftPanel";
import { Failure } from "./Failure";
import { Savings } from "./Savings";
import { StatusStrip } from "./StatusStrip";
import { useGateway } from "./useGateway";
import "./playground.css";

export { RowsPreview } from "./Pane";
export { Failure } from "./Failure";

/** The API's own limit on the question (docs/API_CONTRACT.md: question <= 500 chars); the input enforces it. */
const MAX = 500;

/**
 * Questions to start from. They are plain text to put in the box: some are phrased the way people talk, and the last
 * one asks about something the demo database does not contain, so the specialist should hand it back.
 */
const EXAMPLES = [
  "How many accounts are there?",
  "Which industry has the most active accounts?",
  "how many ppl are on the enterprise plan rn",
  "gimme the overdue invoices by currency, biggest first",
  "users who never logged in, newest signups on top",
  "What is our net revenue retention by customer cohort?",
] as const;

/**
 * The Acme analyst. A question goes to POST /api/acme/ask, which sends it through this site's own gateway: the
 * certified specialist answers when it can guarantee the answer, and the request is handed back to the teacher
 * when it cannot. Everything shown is what the API returned: who answered and why, the SQL, the verified rows,
 * the cost, and the gateway's certificate and savings. Nothing is generated or calculated here, and a request
 * that failed is never drawn as an empty answer.
 */
export default function Playground() {
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const [out, setOut] = useState<AcmeAskResponse | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const { status, savings, drift, refresh } = useGateway();

  async function submit(ev?: FormEvent) {
    ev?.preventDefault();
    if (busy || q.trim() === "") return;
    setBusy(true);
    setErr(null);
    setOut(null);
    try {
      setOut(await api.acmeAsk(q.trim()));
    } catch (e) {
      setErr(e);
    } finally {
      setBusy(false);
      refresh(); // the call log has moved (or the limit was hit): show the server's current figures
    }
  }
  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); void submit(); }
  };

  return (
    <div className="pg">
      <header className="pg-head">
        <h1>Acme analyst</h1>
        <p className="muted pg-lead">
          Change one base URL. Distillery replaces your big model with a certified specialist and hands back every
          request it can't guarantee. Ask the Acme analyst a question about its billing database and see who answered.
        </p>
      </header>

      <div className="pg-top">
        <div className="pg-main">
          <Card className="pg-ask">
            <h2 className="pg-h">Ask a question</h2>
            <form onSubmit={submit} className="pg-form">
              <Field label="Question" help="One question about the Acme billing database, in your own words.">
                <Textarea rows={3} maxLength={MAX} value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={onKey} />
              </Field>
              <div className="pg-actions">
                <Button type="submit" variant="primary" disabled={q.trim() === ""} loading={busy}>Ask</Button>
                <span className="muted pg-count">{q.length} / {MAX}</span>
                <span className="muted pg-hint">Ctrl+Enter also sends</span>
              </div>
            </form>
            <div className="pg-examples">
              <h3 className="pg-sub" id="pg-ex-h">Examples</h3>
              <ul aria-labelledby="pg-ex-h" className="pg-chips">
                {EXAMPLES.map((ex) => (
                  <li key={ex}>
                    <button type="button" className="pg-chip" disabled={busy} onClick={() => setQ(ex)}>{ex}</button>
                  </li>
                ))}
              </ul>
              <p className="muted pg-break">
                Try to break it: misspell things, use slang, change the word order, or ask for something the database
                has never held. When the specialist is not sure, the request goes to the teacher and the answer says why.
              </p>
            </div>
          </Card>

          <section className="pg-out" aria-label="Result">
            <span className="sr-only" role="status">{out && !busy ? "Answer is ready." : ""}</span>
            {busy && <Spinner label="Asking the Acme analyst" />}
            {err !== null && <Failure e={err} />}
            {out && !busy && <AnswerCard r={out} />}
          </section>
        </div>

        <div className="pg-side">
          {status.state === "loading" && <Spinner lines={3} label="Loading the gateway status" />}
          {status.state === "error" && <ApiErrorState title="Could not load the gateway status" error={status.error} />}
          {status.state === "ok" && <StatusStrip s={status.data} />}
          {drift.state === "loading" && <Spinner lines={3} label="Loading the drift guard" />}
          {drift.state === "error" && <ApiErrorState title="Could not load the drift guard" error={drift.error} />}
          {drift.state === "ok" && <DriftPanel d={drift.data} />}
          {savings.state === "loading" && <Spinner lines={3} label="Loading the savings" />}
          {savings.state === "error" && <ApiErrorState title="Could not load the savings" error={savings.error} />}
          {savings.state === "ok" && <Savings s={savings.data} />}
        </div>
      </div>
    </div>
  );
}
