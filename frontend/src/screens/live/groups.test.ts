import { describe, expect, it } from "vitest";
import type { Stage, StageStatus } from "../../api/types";
import { groupStages, roundOf } from "./groups";

const st = (name: string, status: StageStatus = "done"): Stage => ({ name, status, started_at: null, ended_at: null });

// The stage names a real dry run reports (GET /api/runs/dry-sql-tiny), in order.
const REAL = [
  "schema", "questions", "gold_crosscheck", "verifier_selftest", "split", "headroom", "teacher_data",
  "finetune_r1", "dev_eval_r1", "analysis_r1", "targeted_r1", "sandbox_branch_r1",
  "finetune_r2", "dev_eval_r2", "final_eval",
];

describe("roundOf", () => {
  it("reads the round key from the end of a real stage name, and nothing else", () => {
    expect(roundOf("finetune_r1")).toBe("r1");
    expect(roundOf("sandbox_branch_r12")).toBe("r12");
    expect(roundOf("final_eval")).toBeNull();
    expect(roundOf("schema")).toBeNull();
    expect(roundOf("r1")).toBeNull();
    expect(roundOf("round_r1_extra")).toBeNull();
  });
});

describe("groupStages", () => {
  it("groups contiguous stages: setup, one group per round, then the rest", () => {
    const g = groupStages(REAL.map((n) => st(n)));
    expect(g.map((x) => x.key)).toEqual(["pre-0", "r1", "r2", "post-3"]);
    expect(g.map((x) => x.round)).toEqual([null, "r1", "r2", null]);
    expect(g.map((x) => x.label)).toEqual(["Setup", "r1", "r2", "Final"]);
    // every real stage appears exactly once, in the original order, under its real name
    expect(g.flatMap((x) => x.stages.map((s) => s.name))).toEqual(REAL);
  });
  it("never invents a group: without a round suffix there is one unlabelled group", () => {
    const g = groupStages([st("schema"), st("questions", "running")]);
    expect(g).toHaveLength(1);
    expect(g[0].round).toBeNull();
    expect(groupStages([])).toEqual([]);
  });
  it("derives the group status only from the member statuses", () => {
    const status = (xs: StageStatus[]) => groupStages(xs.map((s, i) => st(`s${i}_r1`, s)))[0].status;
    expect(status(["done", "done"])).toBe("done");
    expect(status(["done", "running"])).toBe("running");
    expect(status(["done", "failed", "pending"])).toBe("failed");
    expect(status(["pending", "pending"])).toBe("pending");
    expect(status(["done", "pending"])).toBe("running"); // in the middle of the round, nothing else is known
  });
});
