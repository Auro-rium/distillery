import { useId, useMemo, useState } from "react";
import type { Stage, StageStatus } from "../../api/types";
import { Badge, type Tone } from "../../components";
import { groupStages, type StageGroup } from "./groups";
import { Panel } from "./Panel";

const TONE: Record<StageStatus, Tone> = { pending: "neutral", running: "info", done: "ok", failed: "bad" };

/**
 * Compact vertical timeline of the stages exactly as the API reported them, grouped by the round suffix of
 * their real names. The running stage is the current step and is emphasised; finished groups can be collapsed
 * (`collapseDone` is only the starting state, the user's own toggles always win). Motion follows real state:
 * the running dot pulses and its incoming connector breathes only while `live` (stream healthy and run
 * active); stale, disconnected or finished runs are static. Times are not shown: durations are never computed.
 * The expand/collapse control sits in the panel header, next to the title.
 */
export function StageTimeline({ title, stages, live, collapseDone }: { title: string; stages: Stage[]; live: boolean; collapseDone: boolean }) {
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
                        <Badge tone={TONE[g.status]}>{g.status}</Badge>
                      </button>
                    </h3>
                  )}
                  <ol id={listId} className="stg-steps" hidden={!open}>
                    {g.stages.map((s) => (
                      <li
                        key={s.name}
                        className="stg-step"
                        data-status={s.status}
                        aria-current={s.status === "running" ? "step" : undefined}
                        title={s.ended_at ? `ended ${s.ended_at}` : s.started_at ? `started ${s.started_at}` : undefined}
                      >
                        <span className="stg-dot" aria-hidden="true"><i key={s.status} /></span>
                        <span className="stg-name mono">{s.name}</span>
                        {s.status === "done" ? <span className="sr-only">done</span> : <Badge tone={TONE[s.status]}>{s.status}</Badge>}
                      </li>
                    ))}
                  </ol>
                </li>
              );
            })}
          </ol>
        </div>
      )}
    </Panel>
  );
}
