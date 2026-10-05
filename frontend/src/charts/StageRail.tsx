// Vertical stage rail for the live control room: one row per pipeline stage with its status LED and
// the elapsed time the caller formatted (durations are derived values, so the caller owns them).
import type { ReactNode } from "react";
import { cx } from "../ui/cx";

export type RailStatus = "pending" | "running" | "done" | "failed";

export interface RailStage {
  name: string;
  status: RailStatus;
  /** Pre-formatted elapsed time or other short meta for the stage. */
  elapsed?: ReactNode;
  /** Optional human label (defaults to `name`). */
  label?: string;
}

const STATUS_TEXT: Record<RailStatus, string> = { pending: "pending", running: "running", done: "done", failed: "failed" };

export function StageRail(props: { stages: RailStage[]; title?: string; live?: boolean; className?: string }) {
  return (
    <ol className={cx("stage-rail", props.className)} aria-label={props.title ?? "Pipeline stages"} data-live={props.live ? "true" : "false"}>
      {props.stages.map((s) => (
        <li key={s.name} className={cx("rail-step", `is-${s.status}`)} aria-current={s.status === "running" ? "step" : undefined}>
          <span className="rail-led" aria-hidden="true" />
          <span className="rail-name">{s.label ?? s.name}</span>
          <span className="rail-status">{STATUS_TEXT[s.status]}</span>
          {s.elapsed !== undefined && s.elapsed !== null && <span className="rail-elapsed num">{s.elapsed}</span>}
        </li>
      ))}
    </ol>
  );
}
