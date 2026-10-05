import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { Stage, StageStatus } from "../../api/types";
import { stageElapsed } from "./elapsed";
import { StageTimeline } from "./StageTimeline";

afterEach(cleanup);

const st = (name: string, status: StageStatus = "done"): Stage => ({ name, status, started_at: null, ended_at: null });
const REAL = [
  "schema", "questions", "gold_crosscheck", "verifier_selftest", "split", "headroom", "teacher_data",
  "finetune_r1", "dev_eval_r1", "analysis_r1", "targeted_r1", "sandbox_branch_r1",
  "finetune_r2", "dev_eval_r2", "final_eval",
];
const names = (c: HTMLElement) => [...c.querySelectorAll(".rail-step .rail-name")].map((e) => e.textContent);
const head = (label: string) => screen.getByRole("button", { name: new RegExp(`^${label}`) });

describe("StageTimeline", () => {
  it("shows exactly the real stage names, grouped by round, in the reported order", () => {
    const { container } = render(<StageTimeline title="Stages" stages={REAL.map((n) => st(n))} live={false} collapseDone={false} />);
    expect(names(container)).toEqual(REAL);
    for (const label of ["Setup", "r1", "r2", "Final"]) expect(head(label)).toBeTruthy();
    expect(screen.getByRole("list", { name: "Stage timeline" })).toBeTruthy();
  });

  it("a list without rounds is flat: no group headers, nothing to collapse", () => {
    const { container } = render(<StageTimeline title="Stages" stages={[st("schema"), st("questions", "running")]} live={false} collapseDone />);
    expect(screen.queryAllByRole("button")).toHaveLength(0);
    expect(names(container)).toEqual(["schema", "questions"]);
  });

  it("emphasises the running stage as the current step, once", () => {
    const stages = REAL.slice(0, 9).map((n, i, a) => st(n, i === a.length - 1 ? "running" : "done"));
    const { container } = render(<StageTimeline title="Stages" stages={stages} live collapseDone={false} />);
    const cur = container.querySelectorAll('[aria-current="step"]');
    expect(cur).toHaveLength(1);
    expect(cur[0].textContent).toContain("dev_eval_r1");
    expect(cur[0].classList.contains("is-running")).toBe(true);
  });

  it("says the status in words for every stage that is not done; done stages carry it for screen readers", () => {
    const { container } = render(<StageTimeline title="Stages" stages={[st("a_r1", "done"), st("b_r1", "running"), st("c_r1", "failed"), st("d_r1", "pending")]} live={false} collapseDone={false} />);
    const rows = [...container.querySelectorAll(".rail-step")];
    const want = ["done", "running", "failed", "pending"];
    expect(rows.map((r) => want.find((w) => r.classList.contains(`is-${w}`)))).toEqual(want);
    rows.forEach((r, i) => expect(r.textContent).toContain(want[i]));
  });

  it("collapses completed groups when asked to, keeps the active and the last group open, and lets the user reopen", () => {
    const stages = REAL.slice(0, 14).map((n, i, a) => st(n, i === a.length - 1 ? "running" : "done")); // dev_eval_r2 running
    const { container } = render(<StageTimeline title="Stages" stages={stages} live collapseDone />);
    expect(head("Setup").getAttribute("aria-expanded")).toBe("false");
    expect(head("r1").getAttribute("aria-expanded")).toBe("false");
    expect(head("r2").getAttribute("aria-expanded")).toBe("true");
    const setup = document.getElementById(head("Setup").getAttribute("aria-controls")!)!;
    expect(setup.hidden).toBe(true);
    fireEvent.click(head("Setup"));
    expect(head("Setup").getAttribute("aria-expanded")).toBe("true");
    expect(setup.hidden).toBe(false);
    expect(within(setup).getByText("schema")).toBeTruthy();
    // a collapsed group still says how it went, from its members' statuses
    expect(head("r1").textContent).toContain("done");
    expect(names(container)).toHaveLength(14); // nothing is dropped from the DOM, only hidden
  });

  it("does not collapse anything by default when not asked (a finished run shows its full record)", () => {
    render(<StageTimeline title="Stages" stages={REAL.map((n) => st(n))} live={false} collapseDone={false} />);
    for (const label of ["Setup", "r1", "r2", "Final"]) expect(head(label).getAttribute("aria-expanded")).toBe("true");
  });

  it("offers one control to collapse the finished groups and expand them all again", () => {
    render(<StageTimeline title="Stages" stages={REAL.map((n) => st(n))} live={false} collapseDone={false} />);
    fireEvent.click(screen.getByRole("button", { name: "Collapse completed" }));
    expect(head("Setup").getAttribute("aria-expanded")).toBe("false");
    expect(head("r2").getAttribute("aria-expanded")).toBe("false");
    expect(head("Final").getAttribute("aria-expanded")).toBe("true"); // the last group stays open
    fireEvent.click(screen.getByRole("button", { name: "Expand all" }));
    expect(head("Setup").getAttribute("aria-expanded")).toBe("true");
  });

  it("a failed stage keeps its group open even when collapsing was requested", () => {
    render(<StageTimeline title="Stages" stages={[st("schema"), st("finetune_r1", "failed"), st("dev_eval_r1", "pending"), st("finetune_r2", "pending")]} live={false} collapseDone />);
    expect(head("r1").getAttribute("aria-expanded")).toBe("true");
  });

  it("motion follows the live flag only", () => {
    const a = render(<StageTimeline title="Stages" stages={[st("a", "running")]} live collapseDone={false} />);
    expect(a.container.querySelector(".stage-rail")!.getAttribute("data-live")).toBe("true");
    a.unmount();
    const b = render(<StageTimeline title="Stages" stages={[st("a", "running")]} live={false} collapseDone={false} />);
    expect(b.container.querySelector(".stage-rail")!.getAttribute("data-live")).toBe("false");
  });
});

describe("stage elapsed (derived from the stage's own timestamps)", () => {
  const at = (n: string, status: StageStatus, a: string | null, b: string | null): Stage => ({ name: n, status, started_at: a, ended_at: b });
  it("formats finished stages from started/ended and never invents one when a timestamp is missing", () => {
    expect(stageElapsed(at("a", "done", "2026-01-01T00:00:00Z", "2026-01-01T00:04:12Z"), null)).toBe("4m 12s");
    expect(stageElapsed(at("a", "done", "2026-01-01T00:00:00Z", "2026-01-01T01:03:00Z"), null)).toBe("1h 03m");
    expect(stageElapsed(at("a", "done", "2026-01-01T00:00:00Z", "2026-01-01T00:00:09Z"), null)).toBe("9s");
    expect(stageElapsed(at("a", "done", null, "2026-01-01T00:00:09Z"), null)).toBeNull();
    expect(stageElapsed(at("a", "pending", null, null), null)).toBeNull();
  });
  it("a running stage counts up only when told the stream is live", () => {
    const s = at("a", "running", "2026-01-01T00:00:00Z", null);
    expect(stageElapsed(s, null)).toBeNull();
    expect(stageElapsed(s, Date.parse("2026-01-01T00:00:30Z"))).toBe("30s");
  });
  it("shows the elapsed text in the rail row", () => {
    const { container } = render(<StageTimeline title="Stages" stages={[at("a", "done", "2026-01-01T00:00:00Z", "2026-01-01T00:04:12Z")]} live={false} collapseDone={false} />);
    expect(container.querySelector(".rail-elapsed")!.textContent).toBe("4m 12s");
  });
});
