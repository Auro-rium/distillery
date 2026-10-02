import { useState, type FormEvent } from "react";
import { api, setAdminToken } from "../../api/client";
import type { HumansetDrafts, HumansetItem } from "../../api/types";
import { ApiErrorState } from "../../components";
import { Button, Card, Field, Input } from "../../ui";

type Decision = "confirm" | "reject" | "skip";

/**
 * Plain, admin-only screen for confirming drafted gold SQL for the human held-out set. No polish on
 * purpose. It needs the admin token (held in memory only) and shows nothing without it: the server
 * answers 401/403 and that error is what is rendered. Never linked from the nav, never part of replay.
 */
export default function HumansetScreen() {
  const [token, setToken] = useState("");
  const [data, setData] = useState<HumansetDrafts | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);

  async function load(ev?: FormEvent, set?: string) {
    ev?.preventDefault();
    setAdminToken(token);
    setBusy(true);
    setErr(null);
    try {
      setData(await api.humansetDrafts(set ?? data?.set));
    } catch (e) {
      setErr(e);
      setData(null);
    } finally {
      setBusy(false);
    }
  }

  async function decide(item: HumansetItem, decision: Decision) {
    if (!data) return;
    setBusy(true);
    setErr(null);
    try {
      await api.humansetDecide({ set: data.set, task_id: item.task_id, decision });
      setData(await api.humansetDrafts(data.set));
    } catch (e) {
      setErr(e);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <h2>Human held-out set: confirm gold SQL</h2>
      <p className="muted">
        Admin only. Each item is a human question with SQL drafted by the teacher. Mark it correct only if the SQL
        answers the question and the rows look right. The token is held in memory only.
      </p>
      <form onSubmit={(e) => void load(e)}>
        <Field label="Admin token">
          <Input type="password" value={token} onChange={(e) => setToken(e.target.value)} autoComplete="off" />
        </Field>
        <Button type="submit" disabled={busy || token === ""}>Load drafts</Button>
      </form>
      {err !== null && <ApiErrorState error={err} />}
      {data && (
        <>
          <p role="status">
            Set {data.set}, teacher {data.teacher_model}: confirmed {data.tally.confirmed}, rejected {data.tally.rejected},
            skipped {data.tally.skipped}, undecided {data.tally.undecided}. Discarded before review:{" "}
            {Object.entries(data.discarded_by_reason).map(([k, v]) => `${k} ${v}`).join(", ")}.
          </p>
          <p className="muted">When done, run <code>distillery confirm-heldout --set {data.set} --finalize</code>.</p>
          {data.items.map((it) => (
            <Card key={it.task_id} title={it.task_id}>
              <p><strong>{it.question}</strong></p>
              <pre>{it.gold_sql}</pre>
              <table className="tbl">
                <thead>
                  <tr>{it.preview.columns.map((c) => <th scope="col" key={c}>{c}</th>)}</tr>
                </thead>
                <tbody>
                  {it.preview.rows.map((r, i) => (
                    <tr key={i}>{r.map((c, j) => <td key={j}>{c === null ? "NULL" : String(c)}</td>)}</tr>
                  ))}
                </tbody>
              </table>
              <p className="muted">{it.preview.row_count} rows; current decision: {it.decision ?? "none"}</p>
              <Button onClick={() => void decide(it, "confirm")} disabled={busy}>Correct</Button>{" "}
              <Button onClick={() => void decide(it, "reject")} disabled={busy}>Wrong</Button>{" "}
              <Button onClick={() => void decide(it, "skip")} disabled={busy}>Skip</Button>
            </Card>
          ))}
        </>
      )}
    </div>
  );
}
