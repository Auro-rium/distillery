import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../../api/client";
import type { RunSummary } from "../../api/types";
import { ApiErrorState, Badge, Card, LabelBanner, Spinner } from "../../components";
import { LabelTag, labelKey } from "../report/label";

export const POLL_ACTIVE_MS = 3000;
export const POLL_IDLE_MS = 15000;

/** Runs that are not recorded, running first, then newest (by created_at as the API wrote it). */
export function liveRuns(all: RunSummary[]): RunSummary[] {
  const rank = (r: RunSummary) => (r.status === "running" ? 0 : 1);
  return all
    .filter((r) => r.recorded !== true)
    .sort((a, b) => rank(a) - rank(b) || (b.created_at ?? "").localeCompare(a.created_at ?? ""));
}

type State = { s: "loading" } | { s: "ok"; runs: RunSummary[] } | { s: "error"; error: Error };

function useRuns(): [State, () => void] {
  const [st, setSt] = useState<State>({ s: "loading" });
  const [tick, setTick] = useState(0);
  useEffect(() => {
    let live = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const load = () => {
      api.runs().then(
        (all) => {
          if (!live) return;
          setSt({ s: "ok", runs: liveRuns(all) });
          timer = setTimeout(load, all.some((r) => r.status === "running") ? POLL_ACTIVE_MS : POLL_IDLE_MS);
        },
        (e: unknown) => {
          if (!live) return;
          setSt({ s: "error", error: e instanceof Error ? e : new Error(String(e)) });
          timer = setTimeout(load, POLL_IDLE_MS);
        },
      );
    };
    load();
    return () => { live = false; if (timer) clearTimeout(timer); };
  }, [tick]);
  return [st, useCallback(() => { setSt({ s: "loading" }); setTick((t) => t + 1); }, [])];
}

function LiveCard({ b }: { b: RunSummary }) {
  const id = encodeURIComponent(b.run_id);
  const isRunning = b.status === "running";
  return (
    <Card className="bundle lift">
      <div className="bundle-head">
        <h3 className="mono">{b.run_id}</h3>
        <div className="bundle-tags">
          <LabelTag b={b} />
          <Badge>{b.status}</Badge>
          {b.decision && <Badge tone={b.decision === "PROMOTE" ? "ok" : "bad"}>{b.decision}</Badge>}
        </div>
      </div>
      <div className="bundle-actions">
        {b.status === "complete" ? (
          <>
            <Link className="btn primary" to={`/runs/${id}/report`}>Report</Link>
            <Link className="btn" to={`/runs/${id}/tree`}>Experiment tree</Link>
          </>
        ) : (
          <Link className={isRunning ? "btn primary" : "btn"} to={`/runs/${id}`}>{isRunning ? "Watch live" : "Open run"}</Link>
        )}
      </div>
    </Card>
  );
}

export function LiveRuns() {
  const [st, retry] = useRuns();
  const seen = new Map<string, RunSummary>();
  if (st.s === "ok") for (const r of st.runs) if (!seen.has(labelKey(r))) seen.set(labelKey(r), r);
  return (
    <section className="stack" aria-labelledby="live-runs-h">
      <h2 id="live-runs-h">Live and recent runs on this server</h2>
      {st.s === "loading" && <Spinner label="Loading live runs" />}
      {st.s === "error" && <ApiErrorState title="Could not load live runs" error={st.error} onRetry={retry} />}
      {st.s === "ok" && st.runs.length === 0 && <p className="muted">No runs are active right now.</p>}
      {st.s === "ok" && st.runs.length > 0 && (
        <>
          <div className="banners">
            {[...seen.values()].map((r) => <LabelBanner key={labelKey(r)} dry_run={r.dry_run} recorded={r.recorded} recorded_at={r.recorded_at} />)}
          </div>
          <ul className="bundles" role="list" aria-label="Live and recent runs">
            {st.runs.map((b) => <li key={b.run_id}><LiveCard b={b} /></li>)}
          </ul>
        </>
      )}
    </section>
  );
}
