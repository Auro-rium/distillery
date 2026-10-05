// The "featured run": the run the site leads with (Evidence and Live in the nav, the Mission hero).
// Picked client-side from /api/replay, never hardcoded:
//   1. the newest recorded, non-dry run whose decision is PROMOTE;
//   2. else the newest recorded run (any decision, any label);
//   3. else none (null). A dry-only or empty bundle list features nothing.
// "Newest" sorts by recorded_at, then created_at (ISO strings); API order is not trusted.
import { useEffect, useState } from "react";
import { api } from "./client";
import type { RunSummary } from "./types";

const stamp = (r: RunSummary): string => r.recorded_at ?? r.created_at ?? "";

function newest(rows: RunSummary[]): RunSummary | null {
  let best: RunSummary | null = null;
  for (const r of rows) if (!best || stamp(r) > stamp(best)) best = r;
  return best;
}

const isRun = (r: unknown): r is RunSummary =>
  !!r && typeof r === "object" && typeof (r as RunSummary).run_id === "string" && (r as RunSummary).run_id.length > 0;

/** Pure selection rule (exported for tests and for screens that already hold the replay list). */
export function pickFeatured(rows: unknown): RunSummary | null {
  if (!Array.isArray(rows)) return null;
  const runs = rows.filter(isRun);
  const recorded = runs.filter((r) => r.recorded === true);
  const promoted = recorded.filter((r) => r.dry_run === false && r.decision === "PROMOTE");
  return newest(promoted) ?? newest(recorded);
}

export interface FeaturedRun {
  run: RunSummary | null;
  /** True until the replay list has answered (or failed). */
  loading: boolean;
}

// Concurrent callers (the nav and a screen) share one in-flight request; nothing is cached after it settles.
let inflight: Promise<RunSummary | null> | null = null;

function load(): Promise<RunSummary | null> {
  if (!inflight) {
    inflight = api
      .replay()
      .then(pickFeatured, () => null) // an unreachable API features nothing; screens show their own errors
      .finally(() => { inflight = null; });
  }
  return inflight;
}

export function useFeaturedRun(): FeaturedRun {
  const [state, setState] = useState<FeaturedRun>({ run: null, loading: true });
  useEffect(() => {
    let live = true;
    load().then((run) => { if (live) setState({ run, loading: false }); });
    return () => { live = false; };
  }, []);
  return state;
}
