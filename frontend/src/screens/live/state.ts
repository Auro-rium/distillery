import type { Decision, RunDetail, RunEvent, RunStatus, Stage } from "../../api/types";

// observed_at = when the server noticed the line while polling, NOT when it was logged.
export interface LogLine { id: number; observed_at: string; level: string; message: string }
export interface LiveView { run: RunDetail; logs: LogLine[]; done: boolean; status: RunStatus; decision: Decision | null }

/** Overlay SSE events on the fetched run detail. Pure; events are assumed ordered by id. */
export function applyEvents(base: RunDetail, events: RunEvent[]): LiveView {
  const stages: Stage[] = base.stages.map((s) => ({ ...s }));
  let spend = base.spend;
  let status = base.status;
  let decision: Decision | null = null;
  let done = false;
  const logs: LogLine[] = [];
  for (const ev of events) {
    if (ev.type === "stage") {
      let s = stages.find((x) => x.name === ev.data.name);
      if (!s) { s = { name: ev.data.name, status: "pending", started_at: null, ended_at: null }; stages.push(s); }
      s.status = ev.data.status;
      // Only the stored transition time `at` is an event time; observed_at is never used as one.
      if (ev.data.status === "running") s.started_at = ev.data.at;
      if (ev.data.status === "done" || ev.data.status === "failed") s.ended_at = ev.data.at;
    } else if (ev.type === "spend") {
      const { observed_at: _seen, ...rest } = ev.data;
      spend = rest;
    } else if (ev.type === "log") {
      logs.push({ id: ev.id, observed_at: ev.data.observed_at, level: ev.data.level, message: ev.data.message });
    } else {
      done = true;
      status = ev.data.status;
      decision = ev.data.decision;
    }
  }
  if (events.some((e) => e.type === "stage") && status === "pending") status = "running";
  return { run: { ...base, stages, spend }, logs, done, status, decision };
}
