import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import { ApiError, api, apiUrl, hasAdminToken, setAdminToken } from "../../api/client";
import { useSSE } from "../../api/useSSE";
import { ApiErrorState, ErrorState, Spinner } from "../../components";
import { RunShell } from "../../components/RunShell";
import { Button, Field, Input } from "../../ui";
import { LONG_LIST } from "./constants";
import { EventLog } from "./EventLog";
import { LedgerPanel } from "./LedgerPanel";
import { LiveStatus } from "./LiveStatus";
import { Panel } from "./Panel";
import { SandboxPanel } from "./SandboxPanel";
import { SpendPanel } from "./SpendPanel";
import { StageTimeline } from "./StageTimeline";
import { applyEvents, connectionOf } from "./state";
import { useNow } from "./useNow";
import { useShellTop } from "./useShellTop";
import { VerifierPanel } from "./VerifierPanel";
import "./live.css";
import type { ExperimentsResult, RunDetail } from "../../api/types";

function ApiErrorText({ error }: { error: unknown }) {
  const code = error instanceof ApiError ? `${error.code}: ` : "";
  return <>{code}{error instanceof Error ? error.message : "request failed"}</>;
}

/**
 * Live control room inside the shared RunShell (header, label banner, verdict and tabs come from there).
 * From 1180 px: stage rail | terminal log (+ sandbox, verifier) | spend + job ledger; two columns from 1024 px,
 * one below, with a status strip (LIVE pulse from the stream heartbeat) that is sticky on phones. Everything shown
 * is the API payload, overlaid with the events received so far; recorded runs get the ledger from their bundle.
 */
export default function LiveRun() {
  const { id = "" } = useParams();
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [token, setToken] = useState("");
  const [cancelMsg, setCancelMsg] = useState<string | null>(null);
  const [ledger, setLedger] = useState<ExperimentsResult | null>(null);
  const [ledgerError, setLedgerError] = useState<string | null>(null);
  const root = useShellTop<HTMLDivElement>();

  const load = useCallback(() => {
    api.run(id).then((d) => { setDetail(d); setError(null); }).catch((e: unknown) => setError(e));
  }, [id]);
  const loadLedger = useCallback(() => {
    api.experiments(id)
      .then((r) => { setLedger(r); setLedgerError(null); })
      .catch((e: unknown) => setLedgerError(e instanceof ApiError ? `${e.code}: ${e.message}` : e instanceof Error ? e.message : "request failed"));
  }, [id]);
  useEffect(() => { setDetail(null); setLedger(null); setLedgerError(null); load(); loadLedger(); }, [load, loadLedger]);

  const active = detail !== null && (detail.status === "running" || detail.status === "pending");
  const { events, state, stale: streamStale, refusal } = useSSE(active ? apiUrl(`/api/runs/${encodeURIComponent(id)}/events`) : null);
  const view = useMemo(() => (detail ? applyEvents(detail, events) : null), [detail, events]);
  const finished = view?.done ?? false;
  // Stale = the stream is down or silent for 30 s, or the last refresh failed. The values on
  // screen are then last-known, not current, and are presented that way.
  const stale = active && !finished && (streamStale || (error !== null && detail !== null));
  // Until the stream's first effect has run its state still reads "closed"; for an active run that means "about to
  // connect", not "disconnected" (which would flash on every load).
  const conn = connectionOf(active && !finished && state === "closed" ? "connecting" : state, stale);
  // Motion that implies activity (pulse, flowing connector, ripple) runs only while truly live.
  const live = active && !finished && conn === "connected";

  // SSE carries no sandbox/verifier data: poll while active, refetch once when done.
  useEffect(() => {
    if (!active) return;
    const t = setInterval(() => { load(); loadLedger(); }, 4000);
    return () => clearInterval(t);
  }, [active, load, loadLedger]);
  useEffect(() => { if (finished) { load(); loadLedger(); } }, [finished, load, loadLedger]);
  const nowMs = useNow(live);

  async function cancel() {
    setCancelMsg(null);
    setAdminToken(token);
    try {
      const r = await api.cancelRun(id);
      setCancelMsg(`Cancel requested (status: ${r.status}).`);
    } catch (e) {
      setCancelMsg(e instanceof ApiError && e.status === 401 ? "Admin token missing." : e instanceof ApiError && e.status === 403 ? "Admin token rejected." : e instanceof ApiError ? e.message : "Cancel failed.");
    }
  }

  if (error !== null && !detail) return <RunShell runId={id} tab="live"><ApiErrorState error={error} onRetry={load} /></RunShell>;
  if (!detail || !view) {
    return (
      <RunShell runId={id} tab="live">
        <div className="live-grid"><Spinner label="Loading run" /><Spinner lines={2} /></div>
      </RunShell>
    );
  }
  const { run } = view;
  const running = run.stages.find((s) => s.status === "running")?.name ?? null;
  const showActive = active && !finished;

  return (
    // The stream's `done` event moves the header badge at once; the verdict itself is fetched by RunShell.
    <RunShell runId={id} tab="live" run={{ ...run, status: view.status }}>
      <div ref={root} className="live" data-stale={stale} data-live={live}>
        {showActive && <LiveStatus conn={conn} events={events.length} stage={running} total={run.spend?.total_usd} />}
        {run.error && <ErrorState title="Run error" message={run.error} />}
        {stale && (
          <div className="live-alert" role="alert">
            <h3>Disconnected / stale</h3>
            <p>
              {error !== null ? <>The last refresh failed (<ApiErrorText error={error} />). </> : null}
              No live updates are arriving (stream disconnected, or no event for 30 seconds). The values below are
              the last known ones and may not be current.
            </p>
          </div>
        )}
        {refusal && !finished && (
          <ErrorState title="Live stream refused" code={refusal.code} message={`${refusal.message} Falling back to refreshing every 4 seconds.`} />
        )}

        <div className="live-grid live-room">
          <div className="live-col live-col-stages">
            <StageTimeline
              title={stale ? "Stages (last known, not current)" : "Stages"}
              stages={run.stages} live={live} nowMs={nowMs} collapseDone={showActive && run.stages.length > LONG_LIST}
            />
          </div>
          <div className="live-col live-col-log">
            <Panel title="Log" className="lp-terminal"><EventLog lines={view.logs.slice(-200)} streaming={active} /></Panel>
            <SandboxPanel sandbox={run.sandbox} />
            <VerifierPanel verifier={run.verifier} />
          </div>
          <div className="live-col live-col-side">
            <SpendPanel spend={run.spend} title={stale ? "Spend (last known, not current)" : "Spend"} />
            <LedgerPanel result={ledger} error={ledgerError} />
          </div>
        </div>

        {showActive && (
          <Panel title="Cancel run">
            <div className="cancel-row">
              {!hasAdminToken() && (
                <Field label="Admin token">
                  <Input type="password" autoComplete="off" value={token} onChange={(e) => setToken(e.target.value)} />
                </Field>
              )}
              <Button variant="danger" onClick={cancel}>Cancel run</Button>
            </div>
            {cancelMsg && <p role="status" className="lp-note">{cancelMsg}</p>}
          </Panel>
        )}
      </div>
    </RunShell>
  );
}
