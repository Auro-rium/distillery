import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { AccuracyTable, ciOf, type BarRow } from "./AccuracyTable";

// The bars themselves are charts/AccuracyBars (tested in charts/charts.test.tsx); this file covers the table twin.
afterEach(() => { cleanup(); });

const rows: BarRow[] = [
  { key: "base", label: "Base", value: 0.5, ci: null },
  { key: "student", label: "Student", value: 0.7, ci: [0.481, 0.855] },
  { key: "teacher", label: "Teacher", value: null, ci: null },
];

describe("AccuracyTable (the table twin of the report charts)", () => {
  it("the table twin carries every value, CI bound and 'not measured' as text", () => {
    render(<AccuracyTable caption="Held-out accuracy" columns={rows} rows={[{ name: "all", values: rows }]} />);
    const t = screen.getByRole("table", { name: "Held-out accuracy" });
    expect(t.textContent).toContain("70.0%");
    expect(t.textContent).toContain("48.1% to 85.5%");
    expect(t.textContent).toContain("not measured");
  });
  it("treats a CI with a missing bound as no CI", () => {
    expect(ciOf({ key: "student", label: "Student", value: 0.7, ci: [0.4, Number.NaN] })).toBeNull();
    expect(ciOf(rows[1])).toEqual([0.481, 0.855]);
  });
});
