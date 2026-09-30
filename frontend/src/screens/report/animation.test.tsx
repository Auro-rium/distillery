import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Example } from "../../api/types";
import { AccuracyChart, type BarRow } from "./AccuracyChart";
import { ExampleList } from ".";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

const rows: BarRow[] = [
  { key: "base", label: "Base", value: 0.5, ci: null, n: 20 },
  { key: "student", label: "Student", value: 0.7, ci: [0.481, 0.855], n: 20 },
  { key: "teacher", label: "Teacher", value: null, ci: null, n: 20 },
];

/** IntersectionObserver whose visibility the test controls. */
function stubIO() {
  const cbs: IntersectionObserverCallback[] = [];
  vi.stubGlobal("IntersectionObserver", class {
    constructor(cb: IntersectionObserverCallback) { cbs.push(cb); }
    observe() {} disconnect() {} unobserve() {}
  });
  vi.stubGlobal("matchMedia", () => ({ matches: false, addEventListener() {}, removeEventListener() {} }));
  vi.stubGlobal("requestAnimationFrame", () => 1);
  vi.stubGlobal("cancelAnimationFrame", () => undefined);
  return { show: () => act(() => { cbs.forEach((cb) => cb([{ isIntersecting: true } as IntersectionObserverEntry], {} as IntersectionObserver)); }) };
}

describe("AccuracyChart reveal", () => {
  it("reduced motion / no observer: revealed and final values are in the DOM from the first render", () => {
    const { container } = render(<AccuracyChart title="t" rows={rows} />);
    expect(container.querySelector("figure")!.getAttribute("data-reveal")).toBe("done");
    expect(container.querySelector("svg.student .bar")!.getAttribute("width")).toBe("70");
    expect(container.textContent).toContain("70.0%");
    expect(container.textContent).toContain("not measured"); // null stays absent
  });
  it("with motion: waits for the chart to be seen, and the exact reported values are in the DOM before and after", () => {
    const io = stubIO();
    const { container } = render(<AccuracyChart title="t" rows={rows} />);
    const fig = container.querySelector("figure")!;
    expect(fig.getAttribute("data-reveal")).toBe("pending");
    expect(container.querySelector("svg.student .bar")!.getAttribute("width")).toBe("70");
    expect(container.querySelector("svg.student .ci line")!.getAttribute("x1")).toBe("48.1");
    expect(container.textContent).toContain("70.0%"); // final value stays in the DOM until the count-up starts
    io.show();
    expect(fig.getAttribute("data-reveal")).toBe("done");
    expect(container.querySelector("svg.student .bar")!.getAttribute("width")).toBe("70");
    expect(container.querySelector("svg.base .ci")).toBeNull(); // no whisker is invented for a row without a CI
    expect(container.querySelector("svg.teacher .bar")).toBeNull();
  });
});

const ex = (over: Partial<Example> = {}): Example => ({
  kind: "fixed", task_id: "t1", family: "join", heldout_class: "seen", question: "Who?",
  gold_sql: "SELECT name FROM users WHERE id = 7", base_sql: "SELECT name FROM people WHERE id = 7",
  student_sql: "SELECT name FROM users WHERE id = 7", teacher_sql: "SELECT name FROM users WHERE id = 8",
  base_ok: false, student_ok: true, teacher_ok: true, ...over,
});

describe("example cards", () => {
  it("highlight only tokens from the returned SQL; the visible text is unchanged", () => {
    const { container } = render(<ExampleList title="T" items={[ex()]} total={1} />);
    const marks = [...container.querySelectorAll("mark")].map((m) => m.textContent!.trim());
    expect(marks).toContain("people"); // base vs gold
    expect(marks).toContain("8"); // teacher vs gold
    const sqlStrings = [ex().gold_sql, ex().base_sql, ex().teacher_sql];
    for (const m of marks) expect(sqlStrings.some((s) => s.includes(m))).toBe(true);
    // the student SQL is identical to gold: nothing highlighted in its panel
    expect(container.querySelectorAll("mark.diff-other").length).toBe(2);
    expect(container.querySelector(".codeblock code")!.textContent).toBe(ex().gold_sql);
    expect(container.textContent).toContain("same tokens as gold");
  });
  it("expands and collapses on click, and switches which model the gold panel is compared with", () => {
    const { container } = render(<ExampleList title="T" items={[ex(), ex({ task_id: "t2", question: "Second?" })]} total={2} />);
    const heads = screen.getAllByRole("button", { name: /Who\?|Second\?/ });
    expect(heads[0].getAttribute("aria-expanded")).toBe("true");
    expect(heads[1].getAttribute("aria-expanded")).toBe("false");
    expect(container.querySelectorAll(".ex-body").length).toBe(1);
    fireEvent.click(heads[1]);
    expect(container.querySelectorAll(".ex-body").length).toBe(2);
    fireEvent.click(heads[0]);
    expect(container.querySelectorAll(".ex-body").length).toBe(1);
    const goldMarks = () => [...container.querySelectorAll("mark.diff-gold")].map((m) => m.textContent!.trim());
    expect(goldMarks()).toEqual([]); // default: gold vs student, identical
    fireEvent.click(screen.getAllByRole("button", { name: "Base" })[0]);
    expect(goldMarks()).toEqual(["users"]);
  });
  it("copy button reports success only after the clipboard write resolved", async () => {
    const write = vi.fn(() => Promise.resolve());
    vi.stubGlobal("navigator", { clipboard: { writeText: write } });
    render(<ExampleList title="T" items={[ex()]} total={1} />);
    fireEvent.click(screen.getAllByRole("button", { name: "Copy to clipboard" })[0]);
    expect(write).toHaveBeenCalledWith(ex().gold_sql);
    await screen.findByText("Copied", { selector: "button" });
  });
  it("copy button says so when the clipboard is unavailable", async () => {
    vi.stubGlobal("navigator", {});
    render(<ExampleList title="T" items={[ex()]} total={1} />);
    fireEvent.click(screen.getAllByRole("button", { name: "Copy to clipboard" })[0]);
    await screen.findByText("Copy failed", { selector: "button" });
  });
});
