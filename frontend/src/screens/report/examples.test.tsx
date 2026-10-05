import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Example, ExamplesResult } from "../../api/types";
import ExampleBrowser from "./ExampleBrowser";
import { ExamplesCard } from "./Examples";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

const ex = (over: Partial<Example> = {}): Example => ({
  kind: "fixed", task_id: "t1", family: "join", heldout_class: "seen", question: "Who?",
  gold_sql: "SELECT name FROM users WHERE id = 7", base_sql: "SELECT name FROM people WHERE id = 7",
  student_sql: "SELECT name FROM users WHERE id = 7", teacher_sql: "SELECT name FROM users WHERE id = 8",
  base_ok: false, student_ok: true, teacher_ok: true, ...over,
});
const res = (items: Example[], over: Partial<ExamplesResult> = {}): ExamplesResult => ({
  items, available: true, capPerKind: 20, totals: { fixed: 9, still_wrong: null, regressed: 1 }, ...over,
});
const three = [
  ex(),
  ex({ task_id: "t2", question: "Second?" }),
  ex({ kind: "still_wrong", task_id: "t3", question: "Third?", student_ok: false }),
  ex({ kind: "regressed", task_id: "t4", question: "Fourth?", base_ok: true, student_ok: false }),
];

describe("ExampleBrowser: kind tabs and counts", () => {
  it("has a tab per kind; counts say 'showing N of M' only when the header gave M, otherwise 'capped list'", () => {
    render(<ExampleBrowser result={res(three)} />);
    expect(screen.getAllByRole("tab").map((t) => t.textContent)).toEqual(["Fixed", "Still wrong", "Regressed"]);
    expect(screen.getByText("showing 2 of 9")).toBeTruthy();
    fireEvent.mouseDown(screen.getByRole("tab", { name: "Still wrong" }), { button: 0 });
    expect(screen.getByText("capped list, 1 shown")).toBeTruthy();
    expect(screen.getByRole("button", { name: /Third\?/ })).toBeTruthy();
    expect(screen.queryByRole("button", { name: /Who\?/ })).toBeNull();
    fireEvent.mouseDown(screen.getByRole("tab", { name: "Regressed" }), { button: 0 });
    expect(screen.getByText("showing 1 of 1")).toBeTruthy();
  });
  it("opens on the first kind that has items", () => {
    render(<ExampleBrowser result={res([three[2]])} />);
    expect(screen.getByRole("tab", { name: "Still wrong" }).getAttribute("aria-selected")).toBe("true");
  });
  it("a kind with nothing returned says so and keeps the count line", () => {
    render(<ExampleBrowser result={res([three[0]])} />);
    fireEvent.mouseDown(screen.getByRole("tab", { name: "Regressed" }), { button: 0 });
    expect(screen.getByText("None returned for this kind.")).toBeTruthy();
    expect(screen.getByText("showing 0 of 1")).toBeTruthy();
  });
});

describe("ExampleBrowser: master list and side-by-side", () => {
  it("opens the first example, switches between examples, and can collapse", () => {
    const { container } = render(<ExampleBrowser result={res(three)} />);
    const [first, second] = screen.getAllByRole("button", { name: /Who\?|Second\?/ });
    expect(first.getAttribute("aria-expanded")).toBe("true");
    expect(second.getAttribute("aria-expanded")).toBe("false");
    expect(container.querySelectorAll(".exd")).toHaveLength(1);
    fireEvent.click(second);
    expect(first.getAttribute("aria-expanded")).toBe("false");
    expect(second.getAttribute("aria-expanded")).toBe("true");
    expect(container.querySelector(".exd")!.textContent).toContain("Second?");
    fireEvent.click(second);
    expect(container.querySelectorAll(".exd")).toHaveLength(0);
  });

  it("shows question, gold, base, student and teacher SQL side by side with their correctness from the payload", () => {
    const { container } = render(<ExampleBrowser result={res(three)} />);
    const d = container.querySelector(".exd")!;
    expect(d.textContent).toContain("Who?");
    const panes = [...d.querySelectorAll(".exd-sql")].map((p) => p.getAttribute("data-model"));
    expect(panes).toEqual(["gold", "base", "student", "teacher"]);
    expect(within(d.querySelector<HTMLElement>('[data-model="base"]')!).getByText("wrong")).toBeTruthy();
    expect(within(d.querySelector<HTMLElement>('[data-model="student"]')!).getByText("correct")).toBeTruthy();
    expect(d.querySelector('[data-model="gold"] code')!.textContent).toBe(ex().gold_sql);
  });

  it("highlights only tokens of the returned SQL, and switches which model the gold panel is compared with", () => {
    const { container } = render(<ExampleBrowser result={res(three)} />);
    const marks = () => [...container.querySelectorAll("mark.diff-other")].map((m) => m.textContent!.trim());
    expect(marks()).toContain("people"); // base vs gold
    expect(marks()).toContain("8"); // teacher vs gold
    const strings = [ex().gold_sql, ex().base_sql, ex().teacher_sql];
    for (const m of marks()) expect(strings.some((s) => s!.includes(m))).toBe(true);
    expect(container.querySelector('[data-model="student"]')!.textContent).toContain("same tokens as gold");
    const gold = () => [...container.querySelectorAll("mark.diff-gold")].map((m) => m.textContent!.trim());
    expect(gold()).toEqual([]);
    fireEvent.click(screen.getByRole("radio", { name: "Base" }));
    expect(gold()).toEqual(["users"]);
  });

  it("every SQL block has a copy button; the first is the gold SQL; success is reported only after the write resolved", async () => {
    const write = vi.fn(() => Promise.resolve());
    vi.stubGlobal("navigator", { clipboard: { writeText: write } });
    render(<ExampleBrowser result={res(three)} />);
    const copies = screen.getAllByRole("button", { name: "Copy to clipboard" });
    expect(copies).toHaveLength(4);
    fireEvent.click(copies[0]);
    expect(write).toHaveBeenCalledWith(ex().gold_sql);
    await screen.findByText("Copied", { selector: "button" });
  });
  it("says so when the clipboard is unavailable", async () => {
    vi.stubGlobal("navigator", {});
    render(<ExampleBrowser result={res(three)} />);
    fireEvent.click(screen.getAllByRole("button", { name: "Copy to clipboard" })[0]);
    await screen.findByText("Copy failed", { selector: "button" });
  });
});

function stubExamples(body: unknown, status = 200, headers: Record<string, string> = {}) {
  vi.stubGlobal("fetch", vi.fn(() => Promise.resolve(new Response(JSON.stringify(body), { status, headers }))));
}

describe("ExamplesCard: loading, error and empty are different states", () => {
  it("loads, then shows the tabs (the browser is a lazy chunk)", async () => {
    stubExamples(three, 200, { "X-Examples-Available": "true", "X-Examples-Totals": '{"fixed":9,"still_wrong":null,"regressed":1}' });
    render(<ExamplesCard id="r" />);
    expect(screen.getByRole("status").textContent).toBe("Loading examples");
    await screen.findByRole("tab", { name: "Fixed" });
    expect(screen.getByText("showing 2 of 9")).toBeTruthy();
  });
  it("an API failure is an error with the code and message, never an empty state", async () => {
    stubExamples({ error: "examples_down", message: "cannot read" }, 500);
    render(<ExamplesCard id="r" />);
    await screen.findByText("Examples unavailable");
    expect(screen.getByRole("alert").textContent).toContain("examples_down: cannot read");
    expect(screen.queryByRole("tab")).toBeNull();
  });
  it("a report without an examples field is an explained empty state", async () => {
    stubExamples([], 200, { "X-Examples-Available": "false" });
    render(<ExamplesCard id="r" />);
    await screen.findByText("This report has no examples");
    expect(screen.queryByRole("alert")).toBeNull();
  });
  it("an empty list from a report that has the field is a plain empty state", async () => {
    stubExamples([], 200, { "X-Examples-Available": "true" });
    render(<ExamplesCard id="r" />);
    await waitFor(() => screen.getByText("No examples returned for this run"));
  });
});

describe("ExampleBrowser: pack-neutral answer fields", () => {
  const calls = (over: Partial<Example> = {}): Example => ({
    kind: "fixed", task_id: "c1", family: "single_call", heldout_class: "seen", question: "Refund invoice 7",
    gold_answer: '[{"tool":"refund_invoice","args":{"id":7}}]', base_answer: "[]",
    student_answer: '[{"tool":"refund_invoice","args":{"id":7}}]', teacher_answer: "[]",
    base_ok: false, student_ok: true, teacher_ok: false, ...over,
  });
  it("reads *_answer (no *_sql at all) and tags the blocks with the pack's language", () => {
    const { container } = render(<ExampleBrowser result={res([calls()])} language="json" />);
    expect(container.querySelector('[data-model="gold"] code')!.textContent).toBe(calls().gold_answer);
    expect(container.querySelector('[data-model="gold"] .code-block')!.getAttribute("data-language")).toBe("json");
    expect(container.querySelector('[data-model="base"] code')!.textContent).toBe("[]");
    expect(screen.getByRole("region").getAttribute("aria-label")).toContain("tool calls");
  });
  it("still reads the SQL pack's *_sql fields when no *_answer is sent (older servers)", () => {
    const { container } = render(<ExampleBrowser result={res([ex()])} />);
    expect(container.querySelector('[data-model="student"] code')!.textContent).toBe(ex().student_sql);
    expect(container.querySelector('[data-model="student"] .code-block')!.getAttribute("data-language")).toBe("sql");
  });
});
