import { useLayoutEffect, useRef } from "react";
import type { LogLine } from "./state";

const LEVELS = new Set(["debug", "info", "warning", "warn", "error", "critical"]);
const NEAR_BOTTOM_PX = 24;

/** New lines slide in (CSS, on insertion only); lines present at mount do not animate. */
export function EventLog({ lines }: { lines: LogLine[] }) {
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
      {lines.length === 0 && "No log lines yet."}
      {lines.map((l) => (
        <div key={l.id} className={`log-line${l.id > firstId.current ? " log-new" : ""}${LEVELS.has(l.level) ? ` lv-${l.level}` : ""}`}>
          {`seen ${l.observed_at} [${l.level}] ${l.message}`}
        </div>
      ))}
    </div>
  );
}
