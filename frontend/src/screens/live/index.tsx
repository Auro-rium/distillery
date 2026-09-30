import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ApiError, api, apiUrl, hasAdminToken, setAdminToken } from "../../api/client";
import { fmtInt, fmtUsd } from "../../api/format";
import type { Decision, RunDetail } from "../../api/types";
import { useSSE } from "../../api/useSSE";
import { ApiErrorState, Badge, Card, CodeBlock, ErrorState, LabelBanner, Spinner, Stat } from "../../components";
import { CountUp } from "../../motion/CountUp";
import { EventLog } from "./EventLog";
import { Heartbeat } from "./Heartbeat";
import { SpendMeter } from "./SpendMeter";
import { StageTimeline } from "./StageTimeline";
import { applyEvents, connectionOf } from "./state";

function ApiErrorText({ error }: { error: unknown }) {
  const code = error instanceof ApiError ? `${error.code}: ` : "";
  return <>{code}{error instanceof Error ? error.message : "request failed"}</>;
}

export default function LiveRun() {
  const { id = "" } = useParams();
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [decision, setDecision] = useState<Decision | null>(null);
  const [token, setToken] = useState("");
  const [cancelMsg, setCancelMsg] = useState<string | null>(null);

  const load = useCallback(() => {
    api.run(id).then((d) => { setDetail(d); setError(null); }).catch((e: unknown) => setError(e));
  }, [id]);
  useEffect(() => { setDetail(null); load(); }, [load]);

  const active = detail !== null && (detail.status === "running" || detail.status === "pending");
  const { events, state, stale: streamStale, refusal } = useSSE(active ? apiUrl(`/api/runs/${encodeURIComponent(id)}/events`) : null);
  const view = useMemo(() => (detail ? applyEvents(detail, events) : null), [detail, events]);
  const finished = view?.done ?? false;
  // Stale = the stream is down or silent for 30 s, or the last refresh failed. The values on
  // screen are then last-known, not current, and are presented that way.
  const stale = active && !finished && (streamStale || (error !== null && detail !== null));
  const conn = connectionOf(state, stale);
  // Motion that implies activity (pulse, flowing connector, ripple) runs only while truly live.
  const live = active && !finished && conn === "connected";

  // SSE carries no sandbox/verifier data: poll while active, refetch once when done.
  useEffect(() => {
    if (!active) return;
    const t = setInterval(load, 4000);
    return () => clearInterval(t);
  }, [active, load]);
  useEffect(() => { if (finished) load(); }, [finished, load]);
  useEffect(() => {
    const status = view?.status;
    if (view?.decision) setDecision(view.decision);
    else if (status === "complete" && !decision) api.report(id).then((r) => setDecision(r.decision)).catch(() => undefined);
  }, [view?.status, view?.decision, id, decision]);

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

  if (error !== null && !detail) return <ApiErrorState error={error} onRetry={load} />;
  if (!detail || !view) return <Spinner label="Loading run" />;
  const { run } = view;
  const st = run.verifier.selftest;

  return (
    <div className="stack" data-stale={stale} data-live={live}>
      <LabelBanner dry_run={run.dry_run} recorded={run.recorded} recorded_at={run.recorded_at} />
      <div className="row">
        <h1 style={{ margin: 0 }}>Run <span className="mono">{run.run_id}</span></h1>
        <Badge tone={view.status === "failed" ? "bad" : view.status === "complete" ? "ok" : "info"}>{view.status}</Badge>
        {active && !finished && <Heartbeat conn={conn} events={events.length} />}
        <Link to={`/runs/${encodeURIComponent(id)}/tree`}>Experiment tree</Link>
      </div>
      {run.error && <ErrorState title="Run error" message={run.error} />}
      {stale && (
        <div className="state error" role="alert">
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

      {decision && (
        <Card title="Decision">
          <div className="row">
            <Badge tone={decision === "PROMOTE" ? "ok" : "bad"}>{decision}</Badge>
            <Link to={`/runs/${encodeURIComponent(id)}/report`}>Open the report</Link>
          </div>
        </Card>
      )}

      <Card title={stale ? "Stages (last known, not current)" : "Stages"}>
        <StageTimeline stages={run.stages} live={live} />
        {run.stages.length === 0 && <p className="muted">No stages reported yet.</p>}
      </Card>

      <Card title={stale ? "Spend (last known, not current)" : "Spend"}>
        <div className="grid">
          <Stat label="Total" value={<CountUp value={run.spend.total_usd} format={fmtUsd} />} hint={`cap ${fmtUsd(run.spend.cap_usd)}`} />
          <Stat label="Sandbox ops" value={fmtInt(run.sandbox.operations)} hint={`peak concurrency ${fmtInt(run.sandbox.concurrency_peak)}`} />
          <Stat label="Fine-tune (estimate)" value={<CountUp value={run.spend.finetune_usd_estimate} format={fmtUsd} />} />
        </div>
        <SpendMeter spend={run.spend} />
        {Object.keys(run.spend.by_model).length === 0 && <p className="muted">No model calls recorded yet.</p>}
      </Card>

      <Card title="Verifier">
        {st ? (
          <p>Self-test: accepted gold <strong>{fmtInt(st.accepted_gold)}</strong>, rejected corruptions <strong>{fmtInt(st.rejected_corruptions)}</strong>, failures{" "}
            <Badge tone={st.failures === 0 ? "ok" : "bad"}>{fmtInt(st.failures)}</Badge></p>
        ) : <p className="muted">Self-test not run yet.</p>}
        {run.verifier.code ? <CodeBlock code={run.verifier.code} caption={`${run.verifier.language} verifier`} /> : <p className="muted">Verifier code: not measured (the server stores none).</p>}
      </Card>

      <Card title="Log">
        <EventLog lines={view.logs.slice(-200)} />
      </Card>

      {active && !finished && (
        <Card title="Cancel run">
          <div className="row">
            {!hasAdminToken() && (
              <input type="password" autoComplete="off" placeholder="Admin token" aria-label="Admin token" value={token} onChange={(e) => setToken(e.target.value)} />
            )}
            <button className="btn" onClick={cancel}>Cancel run</button>
          </div>
          {cancelMsg && <p role="status">{cancelMsg}</p>}
        </Card>
      )}
    </div>
  );
}
