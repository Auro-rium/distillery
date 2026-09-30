import type { CSSProperties } from "react";
import type { Stage, StageStatus } from "../../api/types";
import { Badge, type Tone } from "../../components";

const TONE: Record<StageStatus, Tone> = { pending: "neutral", running: "info", done: "ok", failed: "bad" };

/**
 * One row per reported stage. Motion follows real state only: a connector fills when the stage
 * before it is done, the running stage pulses and its incoming connector flows, but only while
 * `live` (stream healthy and run active). Stale, disconnected or finished runs are static.
 * Times are not shown: durations are never computed.
 */
export function StageTimeline({ stages, live }: { stages: Stage[]; live: boolean }) {
  return (
    <ol className="timeline" aria-label="Stage timeline" data-live={live}>
      {stages.map((s, i) => (
        <li
          key={s.name}
          className={`tl-step ${s.status}`}
          style={{ "--i": i } as CSSProperties}
        >
          <span className="tl-node" aria-hidden="true"><i key={s.status} /></span>
          {i < stages.length - 1 && (
            <span className={`tl-link${s.status === "done" ? " filled" : ""}${s.status === "done" && stages[i + 1].status === "running" ? " flow" : ""}`} aria-hidden="true">
              <i className="fill" /><b className="pulse" />
            </span>
          )}
          <span className="mono">{s.name}</span> <Badge tone={TONE[s.status]}>{s.status}</Badge>
        </li>
      ))}
    </ol>
  );
}
