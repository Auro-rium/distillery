import type { Stage } from "../../api/types";

const SEC = 1000;
const MIN = 60;
const HOUR = 3600;

/** Whole seconds between two ISO timestamps, or null if either is missing/unparseable or the order is wrong. */
export function secondsBetween(from: string | null | undefined, to: string | number | null | undefined): number | null {
  if (!from || to === null || to === undefined) return null;
  const a = Date.parse(from);
  const b = typeof to === "number" ? to : Date.parse(to);
  if (Number.isNaN(a) || Number.isNaN(b) || b < a) return null;
  return Math.floor((b - a) / SEC);
}

/** "42s", "4m 12s", "1h 03m". A derived display value; the inputs are the stored stage timestamps. */
export function fmtDuration(totalSeconds: number | null): string | null {
  if (totalSeconds === null) return null;
  if (totalSeconds < MIN) return `${totalSeconds}s`;
  if (totalSeconds < HOUR) return `${Math.floor(totalSeconds / MIN)}m ${totalSeconds % MIN}s`;
  const m = Math.floor((totalSeconds % HOUR) / MIN);
  return `${Math.floor(totalSeconds / HOUR)}h ${m < 10 ? "0" : ""}${m}m`;
}

/**
 * Elapsed time of one stage from its own timestamps. A running stage counts up to `nowMs` only when the caller
 * passes it (the stream is live); a stale or recorded view never extrapolates, so it shows nothing for it.
 */
export function stageElapsed(s: Stage, nowMs: number | null): string | null {
  if (s.status === "running") return nowMs === null ? null : fmtDuration(secondsBetween(s.started_at, nowMs));
  if (s.status === "done" || s.status === "failed") return fmtDuration(secondsBetween(s.started_at, s.ended_at));
  return null;
}
