import { act, cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AccuracyChart, AccuracyTable, Legend, type BarRow } from "./AccuracyChart";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

const rows: BarRow[] = [
  { key: "base", label: "Base", value: 0.5, ci: null },
  { key: "student", label: "Student", value: 0.7, ci: [0.481, 0.855] },
  { key: "teacher", label: "Teacher", value: null, ci: null },
];

/** Motion allowed; IntersectionObserver and requestAnimationFrame are driven by the test. */
function motion() {
  const cbs: IntersectionObserverCallback[] = [];
  const frames: FrameRequestCallback[] = [];
  vi.stubGlobal("IntersectionObserver", class {
    constructor(cb: IntersectionObserverCallback) { cbs.push(cb); }
    observe() {} disconnect() {} unobserve() {}
  });
  vi.stubGlobal("matchMedia", () => ({ matches: false, addEventListener() {}, removeEventListener() {} }));
  vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => frames.push(cb));
  vi.stubGlobal("cancelAnimationFrame", () => undefined);
  return {
    show: () => act(() => { cbs.forEach((cb) => cb([{ isIntersecting: true } as IntersectionObserverEntry], {} as IntersectionObserver)); }),
    frame: (t: number) => act(() => { frames.splice(0).forEach((f) => f(t)); }),
  };
}

describe("AccuracyChart: exact values, no count-up", () => {
  it("the text on screen is identical before, during and after the reveal (D1: only bars and whiskers animate)", () => {
    const m = motion();
    const { container } = render(<AccuracyChart title="t" rows={rows} />);
    const seen = [container.textContent];
    m.show();
    seen.push(container.textContent);
    for (let t = 0; t <= 1600; t += 40) { m.frame(t); seen.push(container.textContent); }
    expect(new Set(seen).size).toBe(1);
    expect(seen[0]).toContain("70.0%");
    expect(seen[0]).toContain("50.0%");
    expect(seen[0]).not.toMatch(/(?<![\d.])0\.0%/); // never a count-up starting value
  });

  it("without motion the chart is already revealed and shows the same text", () => {
    const { container } = render(<AccuracyChart title="t" rows={rows} />);
    expect(container.querySelector("figure")!.getAttribute("data-reveal")).toBe("done");
    expect(container.textContent).toContain("70.0%");
  });

  it("with motion the reveal waits for the chart to be seen; geometry is in the DOM from the start", () => {
    const m = motion();
    const { container } = render(<AccuracyChart title="t" rows={rows} />);
    const fig = container.querySelector("figure")!;
    expect(fig.getAttribute("data-reveal")).toBe("pending");
    const bar = () => container.querySelector<HTMLElement>(".acc-row.student .acc-bar")!;
    expect(bar().style.width).toBe("70%");
    m.show();
    expect(fig.getAttribute("data-reveal")).toBe("done");
    expect(bar().style.width).toBe("70%");
  });
});

describe("AccuracyChart: what is drawn", () => {
  it("draws a whisker only for a row that has a CI, and says 'no CI in report' otherwise", () => {
    const { container } = render(<AccuracyChart title="t" rows={rows} />);
    expect(container.querySelectorAll(".acc-ci")).toHaveLength(1);
    const student = container.querySelector(".acc-row.student")!;
    const ci = student.querySelector<HTMLElement>(".acc-ci")!;
    expect(parseFloat(ci.style.left)).toBeCloseTo(48.1, 1);
    expect(parseFloat(ci.style.width)).toBeCloseTo(37.4, 1);
    expect(student.textContent).toContain("CI 48.1% to 85.5%");
    expect(container.querySelector(".acc-row.base")!.textContent).toContain("no CI in report");
    expect(container.querySelector(".acc-row.base .acc-ci")).toBeNull();
  });

  it("an absent value is 'not measured' with no bar (never a zero-length bar)", () => {
    const { container } = render(<AccuracyChart title="t" rows={rows} />);
    const t = container.querySelector(".acc-row.teacher")!;
    expect(t.textContent).toContain("not measured");
    expect(t.querySelector(".acc-bar")).toBeNull();
  });

  it("bars are a list of rows with a text label and value each, and the drawing is hidden from assistive tech", () => {
    render(<AccuracyChart title="Held-out accuracy by model" rows={rows} />);
    const list = screen.getByRole("list", { name: "Held-out accuracy by model" });
    const items = within(list).getAllByRole("listitem");
    expect(items).toHaveLength(3);
    expect(items[1].textContent).toContain("Student");
    expect(items[1].querySelector(".acc-plot")!.getAttribute("aria-hidden")).toBe("true");
  });

  it("axis labels are the fixed scale ends and middle only", () => {
    const { container } = render(<AccuracyChart title="t" rows={rows} />);
    expect([...container.querySelectorAll(".acc-axis span")].map((s) => s.textContent)).toEqual(["0%", "50%", "100%"]);
  });
});

describe("Legend and table view", () => {
  it("the key names every model once", () => {
    render(<Legend />);
    const key = screen.getByRole("list", { name: "Chart key" });
    expect(within(key).getAllByRole("listitem").map((l) => l.textContent)).toEqual(["Base", "Student", "Teacher"]);
  });
  it("the table twin carries every value, CI bound and 'not measured' as text", () => {
    render(<AccuracyTable caption="Held-out accuracy" columns={rows} rows={[{ name: "all", values: rows }]} />);
    const t = screen.getByRole("table", { name: "Held-out accuracy" });
    expect(t.textContent).toContain("70.0%");
    expect(t.textContent).toContain("48.1% to 85.5%");
    expect(t.textContent).toContain("not measured");
  });
});
