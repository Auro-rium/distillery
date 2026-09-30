import { render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import fixture from "../../../../docs/fixtures/sample-dry-run-report.json";
import type { Report } from "../../api/types";

import { ReportView } from ".";

const base = { ...(fixture as unknown as Report), dry_run: true, recorded: false, recorded_at: null };
const view = (r: Report) => render(<MemoryRouter><ReportView r={r} /></MemoryRouter>);

const json = (body: unknown, status = 200, headers: Record<string, string> = {}) =>
  Promise.resolve(new Response(JSON.stringify(body), { status, headers }));
const stubFetch = (h: () => Promise<Response>) => vi.stubGlobal("fetch", vi.fn(h));
beforeEach(() => stubFetch(() => json([])));
afterEach(() => vi.unstubAllGlobals());


describe("Report", () => {
  it("shows label banner, decision, reasons verbatim", async () => {
    stubFetch(() => json([]));
    view(base);
    expect(screen.getAllByRole("status")[0].textContent).toBe("DRY RUN — fake models, numbers are NOT results");
    const v = screen.getByRole("region", { name: "Gate decision" });
    expect(within(v).getByText("REJECT")).toBeTruthy();
    for (const r of base.decision_reasons) expect(within(v).getByText(r)).toBeTruthy();
    await waitFor(() => screen.getByText("No examples returned for this run"));
  });

  it("renders SVG bars with exact values, CI and n", () => {
    const { container } = view(base);
    const s = container.querySelector("svg.student")!;
    expect(s.getAttribute("aria-label")).toBe(
      "Student 70.0%, confidence interval 48.1% to 85.5%, n=20",
    );
    expect(container.querySelectorAll("svg.acc-svg").length).toBe(3 + 3 * 3);
    expect(container.querySelector("svg.base .ci")).toBeNull();
    expect(container.querySelector("svg.student .bar")!.getAttribute("width")).toBe("70");
    expect(screen.getByText("unseen_family")).toBeTruthy();
  });

  it("shows student cost string verbatim and latency not measured", () => {
    view(base);
    expect(screen.getByText("unavailable: serving path undecided (spike S4)")).toBeTruthy();
    expect(screen.getByText("the report has no latency data")).toBeTruthy();
    expect(screen.getByText("$1.11195")).toBeTruthy();
  });

  it("examples: error and populated states", async () => {
    stubFetch(() => json({ error: "report_not_ready", message: "not done" }, 404));
    view(base);
    await waitFor(() => screen.getByText("Examples unavailable"));
    // an error (role=alert) carrying the API's code and message, never an empty state
    expect(screen.getByRole("alert").textContent).toContain("report_not_ready: not done");
  });

  it("examples: side by side", async () => {
    stubFetch(() => json([{
      task_id: "t1", family: "join", heldout_class: "seen", question: "Q?", gold_sql: "G",
      base_sql: "B", student_sql: "S", teacher_sql: "T", base_ok: false, student_ok: true, teacher_ok: true,
    }], 200, { "X-Examples-Available": "true", "X-Examples-Totals": '{"fixed":9,"still_wrong":null,"regressed":1}' }));
    view(base);
    await waitFor(() => expect(screen.getAllByText("Q?").length).toBe(2));
    // total only when the server gave it; otherwise it is a sample
    expect(screen.getByText("Fixed (student right, base wrong) (showing 1 of 9)")).toBeTruthy();
    expect(screen.getByText("Still wrong (base and student wrong) (capped list, 1 shown)")).toBeTruthy();
  });

  it("handles missing counters and clusters", () => {
    view({ ...base, counters: {}, rounds: [], evaluation: { ...base.evaluation, accuracy_by_class: {} } });
    expect(screen.getByText("No drop or error counters in this report")).toBeTruthy();
    expect(screen.getByText("No failure clusters in this report")).toBeTruthy();
    expect(screen.getByText("No class split in this report")).toBeTruthy();
  });
});
