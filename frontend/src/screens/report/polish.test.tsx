import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Report } from "../../api/types";
import { diffAgainstGold } from "../../lib/diff";
import { Counters } from "./sections";
import { SqlBlock } from "./SqlBlock";

afterEach(() => { cleanup(); vi.useRealTimers(); vi.unstubAllGlobals(); });

const LONG = "discard_planner_mismatch_only";
const report = {
  counters: { gold_crosscheck: { [LONG]: 3, discarded: 10 } },
  llm_attempt_counters: { "crosscheck_planner:schema_retries": 6 },
  llm_errors_by_purpose: {},
} as unknown as Report;

describe("long counter names wrap at underscores, never inside a word", () => {
  it("splits after each underscore into parts the line breaker may separate, and leaves the text itself unchanged", () => {
    const { container } = render(<Counters r={report} />);
    const th = [...container.querySelectorAll("tbody th")].find((e) => e.textContent === LONG)!;
    expect(th).toBeTruthy();
    expect([...th.querySelectorAll(".brk")].map((p) => p.textContent)).toEqual(["discard_", "planner_", "mismatch_", "only"]);
    expect(th.textContent).toBe(LONG);
  });
  it("does the same for the purpose column of the attempts table", () => {
    const { container } = render(<Counters r={report} />);
    const th = [...container.querySelectorAll("tbody th")].find((e) => e.textContent === "crosscheck_planner")!;
    expect([...th.querySelectorAll(".brk")].map((p) => p.textContent)).toEqual(["crosscheck_", "planner"]);
  });
});

describe("SqlBlock", () => {
  it("keeps the copy button in the header, outside the code, so it can never cover SQL", () => {
    const { container, getByRole } = render(<SqlBlock label="Gold" code="SELECT 1" status={<span>correct</span>} />);
    const head = container.querySelector(".sql-head")!;
    expect(head.textContent).toContain("Gold");
    expect(head.textContent).toContain("correct");
    expect(head.contains(getByRole("button", { name: "Copy to clipboard" }))).toBe(true);
    expect(container.querySelector("pre")!.querySelector("button")).toBeNull();
    expect(container.querySelector("pre code")!.textContent).toBe("SELECT 1");
  });
  it("marks tokens from a diff and still joins back to the exact string", () => {
    const d = diffAgainstGold("SELECT a FROM t", "SELECT b FROM t")!;
    const { container } = render(<SqlBlock label="Student" code="SELECT b FROM t" segments={d.other} side="other" />);
    expect(container.querySelector("mark.diff-other")!.textContent!.trim()).toBe("b");
    expect(container.querySelector("code")!.textContent).toBe("SELECT b FROM t");
  });
  it("shows its note (why a diff is or is not drawn) under the code", () => {
    const { container } = render(<SqlBlock label="Base" code="SELECT 1" note="diff skipped: SQL too long" />);
    expect(container.querySelector(".sql-block")!.textContent).toContain("diff skipped: SQL too long");
  });
});
