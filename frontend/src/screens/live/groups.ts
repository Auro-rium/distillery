import type { Stage, StageStatus } from "../../api/types";

// The backend names per-round stages `<step>_r<N>` (finetune_r1, dev_eval_r2, ...). The round key is read from that
// suffix of the real name; nothing else about a stage is inferred and no stage name is ever invented.
const ROUND_SUFFIX = /_(r\d+)$/;

export function roundOf(name: string): string | null {
  return ROUND_SUFFIX.exec(name)?.[1] ?? null;
}

export interface StageGroup {
  /** Unique within one list; used for state and DOM ids. */
  key: string;
  /** "r1", "r2", ... for a round group, null for the stages between or around rounds. */
  round: string | null;
  label: string;
  /** Derived only from the member stages' own statuses. */
  status: StageStatus;
  stages: Stage[];
}

function statusOf(stages: Stage[]): StageStatus {
  if (stages.some((s) => s.status === "failed")) return "failed";
  if (stages.some((s) => s.status === "running")) return "running";
  if (stages.every((s) => s.status === "done")) return "done";
  if (stages.every((s) => s.status === "pending")) return "pending";
  return "running"; // some finished and some not started: the group is in the middle of its work
}

/** Contiguous stages with the same round key form one group, in the order the API reported them. */
export function groupStages(stages: Stage[]): StageGroup[] {
  const groups: StageGroup[] = [];
  let sawRound = false;
  for (const s of stages) {
    const round = roundOf(s.name);
    const last = groups[groups.length - 1];
    if (last && last.round === round) { last.stages.push(s); continue; }
    if (round !== null) sawRound = true;
    const n = groups.length;
    const key = round === null ? `${sawRound ? "post" : "pre"}-${n}` : groups.some((g) => g.key === round) ? `${round}-${n}` : round;
    groups.push({ key, round, label: round ?? (sawRound ? "Final" : "Setup"), status: "pending", stages: [s] });
  }
  for (const g of groups) g.status = statusOf(g.stages);
  return groups;
}
