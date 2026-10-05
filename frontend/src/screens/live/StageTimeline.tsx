import { useId, useMemo, useState } from "react";
import type { Stage, StageStatus } from "../../api/types";
import { StageRail } from "../../charts";
import { stageElapsed } from "./elapsed";
import { groupStages, type StageGroup } from "./groups";
import { Panel } from "./Panel";

const STATUS_WORD: Record<StageStatus, string> = { pending: "pending", running: "running", done: "done", failed: "failed" };

/**
 * Compact vertical timeline of the stages exactly as the API reported them, grouped by the round suffix of
 * their real names. The running stage is the current step and is emphasised; finished groups can be collapsed
 * (`collapseDone` is only the starting state, the user's own toggles always win). Motion follows real state:
 * the running dot pulses and its incoming connector breathes only while `live` (stream healthy and run
 * active); stale, disconnected or finished runs are static. Each row shows its elapsed time derived from the stage's own timestamps (a running stage counts up only while live).
 * The expand/collapse control sits in the panel header, next to the title.
 */
export function StageTimeline({ title, stages, live, collapseDone, nowMs = null }: { title: string; stages: Stage[]; live: boolean; collapseDone: boolean; nowMs?: number | null }) {
  const uid = useId();
  const groups = useMemo(() => groupStages(stages), [stages]);
  const grouped = groups.some((g) => g.round !== null);
  const [own, setOwn] = useState<Record<string, boolean>>({});

  const lastIndex = groups.length - 1;
  const startsOpen = (g: StageGroup, i: number) => !(collapseDone && g.status === "done" && i < lastIndex);
  const isOpen = (g: StageGroup, i: number) => own[g.key] ?? startsOpen(g, i);
  const canCollapse = groups.some((g, i) => g.status === "done" && i < lastIndex);
  const anyClosed = groups.some((g, i) => !isOpen(g, i));

  const collapseCompleted = () => setOwn(Object.fromEntries(groups.map((g, i) => [g.key, !(g.status === "done" && i < lastIndex)])));
  const expandAll = () => setOwn(Object.fromEntries(groups.map((g) => [g.key, true])));

  const tools = grouped && (anyClosed || canCollapse)
    ? (anyClosed
      ? <button type="button" className="btn sm ghost" onClick={expandAll}>Expand all</button>
      : <button type="button" className="btn sm ghost" onClick={collapseCompleted}>Collapse completed</button>)
    : null;

  return (
    <Panel title={title} className="lp-stages" aside={tools}>
      {groups.length === 0 ? (
        <p className="muted lp-none">No stages reported yet.</p>
      ) : (
        <div className="stg-scroll">
          <ol className="timeline" aria-label="Stage timeline" data-live={live}>
            {groups.map((g, gi) => {
              const open = !grouped || isOpen(g, gi);
              const listId = `${uid}-${g.key}`;
              return (
                <li key={g.key} className="stg-group" data-status={g.status} data-open={open}>
                  {grouped && (
                    <h3 className="stg-head">
                      <button type="button" className="stg-toggle" aria-expanded={open} aria-controls={listId} onClick={() => setOwn({ ...own, [g.key]: !open })}>
                        <span className="stg-chev" aria-hidden="true" />
                        <span className="stg-label">{g.label}</span>
                        <span className={`stg-state is-${g.status}`}>{STATUS_WORD[g.status]}</span>
                      </button>
                    </h3>
                  )}
                  <div id={listId} className="stg-steps" hidden={!open}>
                    <StageRail
                      title={`${g.label} stages`} live={live}
                      stages={g.stages.map((x) => ({ name: x.name, status: x.status, elapsed: stageElapsed(x, nowMs) ?? undefined }))}
                    />
                  </div>
                </li>
              );
            })}
          </ol>
        </div>
      )}
    </Panel>
  );
}
