import { fmtInt, fmtText, fmtUsd } from "../../api/format";
import type { DriftEventRow, DriftParams, GatewayDrift } from "../../api/types";
import { Badge, Card } from "../../ui";

/** Labels of the detector parameters, in the order the server sends them. Values are shown exactly as configured. */
const PARAMS: [keyof DriftParams, string][] = [
  ["window_w", "window (requests)"],
  ["tau", "deferral threshold tau"],
  ["min_requests", "armed after (requests)"],
  ["audit_rate", "audit rate"],
  ["audit_window_m", "audit window"],
  ["audit_k", "disagreements to roll back"],
];

function Item({ label, value, className }: { label: string; value: string; className?: string }) {
  return (
    <div className={className ? `pg-st-item ${className}` : "pg-st-item"}>
      <dt className="eyebrow">{label}</dt>
      <dd className="num">{value}</dd>
    </div>
  );
}

/** One line of the event log: when, what happened, which detector. A queued re-distillation is said to be queued only. */
function Event({ e }: { e: DriftEventRow }) {
  const queued = e.kind === "redistill_queued";
  return (
    <li className="pg-dr-event">
      <span className="pg-dr-time num">{fmtText(e.created_at)}</span>
      <span className="pg-dr-kind">{queued ? "re-distill queued (not run)" : e.kind}</span>
      <span className="pg-dr-det muted">
        {e.state_from} to {e.state_to}
        {e.detector ? `, detector: ${e.detector}` : ""}
      </span>
    </li>
  );
}

function EventLog({ events }: { events: DriftEventRow[] }) {
  return (
    <div className="pg-dr-log">
      <h3 className="pg-sub" id="pg-dr-log-h">Events, newest first</h3>
      {events.length === 0 ? (
        <p className="muted pg-dr-none">No drift events recorded.</p>
      ) : (
        <ul aria-labelledby="pg-dr-log-h" className="pg-dr-events">
          {events.map((e) => <Event key={e.id} e={e} />)}
        </ul>
      )}
    </div>
  );
}

/**
 * The drift guard as GET /api/gateway/drift reports it: CERTIFIED or rolled back to the teacher, the detector
 * parameters as configured, the deferral share of the current window (a fraction, like tau), the audit counters and the event log
 * (newest first). Read-only: there is no action here, and nothing is calculated; a figure the server did not send is
 * shown as not measured. A gateway without the guard says so instead of drawing an empty panel.
 */
export function DriftPanel({ d }: { d: GatewayDrift }) {
  if (!d.enabled || d.state === null) {
    return (
      <Card className="pg-drift" aria-labelledby="pg-drift-h">
        <div className="pg-st-head">
          <h2 id="pg-drift-h" className="pg-h">Drift guard</h2>
          <Badge tone="neutral">not configured</Badge>
        </div>
        <p className="pg-dr-off" role="status">Drift guard not configured</p>
        <p className="muted pg-dr-note">{d.note}</p>
        {d.events.length > 0 && <EventLog events={d.events} />}
      </Card>
    );
  }
  const ok = d.state === "CERTIFIED";
  const w = d.window;
  return (
    <Card className="pg-drift" tone={ok ? "ok" : "warn"} aria-labelledby="pg-drift-h">
      <div className="pg-st-head">
        <h2 id="pg-drift-h" className="pg-h">Drift guard</h2>
        <Badge tone={ok ? "ok" : "warn"}>{d.state}</Badge>
      </div>
      {!ok && (
        <div className="pg-uncert" role="status">
          <p className="pg-uncert-h">Rolled back: every skill request goes to the teacher and the student is not called.</p>
          <p className="muted">{fmtText(d.reason)}</p>
        </div>
      )}
      {d.params && (
        <dl className="pg-st-list" aria-label="Detector parameters">
          {PARAMS.map(([k, label]) => <Item key={k} label={label} value={String(d.params![k])} />)}
        </dl>
      )}
      <dl className="pg-st-list pg-dr-window" aria-label="Window">
        <Item label="deferral share" value={w?.deferral_share == null ? fmtText(null) : String(w.deferral_share)} />
        <Item label="deferred in window" value={w ? `${fmtInt(w.window_deferred)} of ${fmtInt(w.window_requests)}` : fmtText(null)} />
        <Item label="requests seen" value={fmtInt(w?.requests_seen)} />
        <Item label="detector armed" value={w ? (w.armed ? "yes" : "no") : fmtText(null)} />
      </dl>
      <dl className="pg-st-list pg-dr-audits" aria-label="Audits">
        <Item label="audits" value={fmtInt(d.audits.audits)} />
        <Item label="disagreements" value={fmtInt(w?.audit_disagreements)} />
        <Item label="inconclusive" value={fmtInt(d.audits.inconclusive)} />
        <Item label="skipped (daily cap)" value={fmtInt(d.audits.skipped_cap)} />
        <Item label="skipped (error)" value={fmtInt(d.audits.skipped_error)} />
        <Item label="audit spend" value={fmtUsd(d.audits.spent_usd, 4)} />
        <Item label="certificate sha256" value={d.cert_sha256 ?? "not recorded"} className="pg-dr-sha" />
      </dl>
      <EventLog events={d.events} />
      <p className="muted pg-dr-note">{d.note}</p>
    </Card>
  );
}
