import { useLayoutEffect, useRef } from "react";
import type { LogLine } from "./state";

const LEVELS = new Set(["debug", "info", "warning", "warn", "error", "critical"]);
const NEAR_BOTTOM_PX = 24;

/**
 * New lines slide in (CSS, on insertion only); lines present at mount do not animate. `observed_at` is when the
 * server noticed the line while polling, so it is labelled "seen", never as when the line was logged.
 * `streaming` is whether this view subscribes to the event stream (active runs only), which decides what an empty log means.
 */
export function EventLog({ lines, streaming }: { lines: LogLine[]; streaming: boolean }) {
  const box = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  const firstId = useRef(lines.length ? lines[lines.length - 1].id : 0);
  useLayoutEffect(() => {
    const el = box.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [lines.length]);
  return (
    <div
      ref={box} className="code log" role="log" aria-live="off" aria-label="Run log" tabIndex={0}
      onScroll={(e) => { const el = e.currentTarget; stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < NEAR_BOTTOM_PX; }}
    >
      {lines.length === 0 && (
        <p className="log-empty">{streaming ? "No log lines yet." : "No log lines in this view: the log is only streamed while a run is active."}</p>
      )}
      {lines.map((l) => (
        <div key={l.id} className={`log-line${l.id > firstId.current ? " log-new" : ""}${LEVELS.has(l.level) ? ` lv-${l.level}` : ""}`}>
          <span className="log-level">{l.level}</span>
          <span className="log-msg">{l.message}</span>
          <span className="log-seen">{`seen ${l.observed_at}`}</span>
        </div>
      ))}
    </div>
  );
}
