import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import fs from "node:fs";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Example, ExamplesResult } from "../../api/types";
import ExampleBrowser from "./ExampleBrowser";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

const ex = (over: Partial<Example> = {}): Example => ({
  kind: "still_wrong", task_id: "t1", family: "join", heldout_class: "seen", question: "Who is first?",
  gold_sql: "SELECT 1", base_sql: "SELECT 2", student_sql: "SELECT 3", teacher_sql: "SELECT 1",
  base_ok: false, student_ok: false, teacher_ok: true, ...over,
});
const result = (items: Example[]): ExamplesResult => ({ items, available: true, capPerKind: 20, totals: { fixed: 0, still_wrong: items.length, regressed: 0 } });
const items = [ex(), ex({ task_id: "t2", question: "Who is second?", base_ok: true, teacher_ok: false }), ex({ task_id: "t3", question: "Who is third?" })];

/** A desktop-wide window: the 900 px query matches. */
function wide(matches: boolean) {
  vi.stubGlobal("matchMedia", (q: string) => ({ matches: q.includes("min-width") ? matches : false, media: q, addEventListener() {}, removeEventListener() {} }));
}

describe("examples: each row says which models got it right, from the payload's flags", () => {
  it("shows base, student and teacher correctness in words a screen reader gets too", () => {
    render(<ExampleBrowser result={result(items)} />);
    const first = screen.getByRole("button", { name: /Who is first\?/ });
    expect(within(first).getByText("Base").closest(".mk")!.className).toContain("bad");
    expect(within(first).getByText("Teacher").closest(".mk")!.className).toContain("ok");
    expect(first.textContent).toMatch(/Base\s*wrong/);
    expect(first.textContent).toMatch(/Teacher\s*correct/);
    const second = screen.getByRole("button", { name: /Who is second\?/ });
    expect(second.textContent).toMatch(/Base\s*correct/);
    expect(second.textContent).toMatch(/Teacher\s*wrong/);
  });
});

describe("examples layout follows the window without changing what is shown", () => {
  it("narrow: the open example sits directly under its own row, inside the one list", () => {
    wide(false);
    const { container } = render(<ExampleBrowser result={result(items)} />);
    expect(container.querySelector(".exb-list")).toBeNull();
    const row = screen.getByRole("button", { name: /Who is first\?/ });
    expect(row.getAttribute("aria-expanded")).toBe("true");
    expect(row.nextElementSibling!.classList.contains("exd")).toBe(true);
  });

  it("wide: the rows are a bounded list of their own and the open example is beside it, outside the list", () => {
    wide(true);
    const { container } = render(<ExampleBrowser result={result(items)} />);
    const list = container.querySelector(".exb-list")!;
    expect(list).not.toBeNull();
    expect(within(list as HTMLElement).getAllByRole("button")).toHaveLength(3);
    expect(list.querySelector(".exd")).toBeNull();
    const d = container.querySelector(".exd")!;
    expect(d.closest(".exb-split")).toBe(list.closest(".exb-split"));
    expect(d.textContent).toContain("Who is first?");
    fireEvent.click(screen.getByRole("button", { name: /Who is third\?/ }));
    expect(container.querySelectorAll(".exd")).toHaveLength(1);
    expect(container.querySelector(".exd")!.textContent).toContain("Who is third?");
    fireEvent.click(screen.getByRole("button", { name: /Who is third\?/ }));
    expect(container.querySelectorAll(".exd")).toHaveLength(0);
    expect(screen.getByText("Select an example to compare its SQL.")).toBeTruthy();
  });

  it("both layouts show the same four SQL blocks, in the same order", () => {
    for (const w of [false, true]) {
      wide(w);
      const { container, unmount } = render(<ExampleBrowser result={result(items)} />);
      expect([...container.querySelectorAll(".exd .exd-sql")].map((p) => p.getAttribute("data-model"))).toEqual(["gold", "base", "student", "teacher"]);
      unmount();
    }
  });
});

describe("the stylesheet never crops an example", () => {
  const css = fs.readFileSync(path.resolve(__dirname, "report.css"), "utf8");
  const rules = (sel: string) => [...css.matchAll(/([^{}]+)\{([^{}]*)\}/g)].filter(([, s]) => s.split(",").some((x) => x.trim().endsWith(sel))).map(([, , body]) => body);
  it("the open example has no height limit and no inner scrolling, at any width", () => {
    const body = rules(".exd").join(" ");
    expect(body).not.toMatch(/max-height|overflow\s*:\s*(auto|scroll|hidden)|overflow-y/);
  });
  it("SQL blocks wrap long lines instead of clipping them", () => {
    const sql = fs.readFileSync(path.resolve(__dirname, "sql.css"), "utf8");
    expect(sql).toMatch(/\.sql-block \.code\s*\{[^}]*white-space:\s*pre-wrap/);
    expect(sql).toMatch(/\.sql-block \.code\s*\{[^}]*overflow-wrap:\s*anywhere/);
  });
  it("the list of rows is the region that is bounded and scrolls", () => {
    const body = rules(".exb-list").join(" ");
    expect(body).toMatch(/max-height/);
    expect(body).toMatch(/overflow-y\s*:\s*auto/);
  });
});
