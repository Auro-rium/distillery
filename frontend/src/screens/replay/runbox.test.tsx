import { cleanup, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { contract, contractFetch } from "../../testutil/contract";
import { renderApp, stubFetch } from "../../testutil/render";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

const recorded = { dry_run: false, recorded: true, recorded_at: "2026-10-05T00:04:35.588559+00:00" };
const stages = [
  { name: "split", status: "done", started_at: null, ended_at: "2026-10-04T15:53:38.689974+00:00" },
  { name: "final_eval", status: "done", started_at: null, ended_at: "2026-10-04T18:54:32.241734+00:00" },
];
function serve(decision: "PROMOTE" | "REJECT", run: Record<string, unknown> = {}) {
  const featured = { ...contract.replay[0], ...recorded, run_id: "sql-gated-20261004T155134Z-66ff347c", decision };
  const report = {
    ...contract.report, ...recorded, run_id: featured.run_id, decision,
    evaluation: { ...contract.report.evaluation, gate: { ...contract.report.evaluation.gate, decision } },
  };
  stubFetch(contractFetch({ replay: [featured], report, run: { ...contract.run, ...run } }));
}
const chiprow = () => document.querySelector<HTMLElement>(".hero .m-chiprow")!;

describe("verdict chip > run box", () => {
  it("puts the run name in a box beside the chip with an arrow between them, themed by the decision", async () => {
    serve("PROMOTE");
    renderApp("/mission");
    await waitFor(() => expect(chiprow()).not.toBeNull());
    const row = chiprow();
    expect([...row.children].map((c) => c.className.split(" ")[0])).toEqual(["badge", "m-arrow", "m-runbox"]);
    expect(row.querySelector(".m-arrow")!.textContent).toBe(">");
    expect(row.dataset.tone).toBe("ok");
    expect(row.querySelector(".m-chip")!.className).toContain("ok");
    expect(row.querySelector(".m-runbtn")!.className).toContain("ok");
    expect(row.querySelector(".m-runbtn")!.textContent).toContain("sql-gated-20261004T155134Z-66ff347c");
  });

  it("a REJECT run gets the reject theme on both the chip and the box", async () => {
    serve("REJECT");
    renderApp("/mission");
    await waitFor(() => expect(chiprow()).not.toBeNull());
    expect(chiprow().dataset.tone).toBe("bad");
    expect(chiprow().querySelector(".m-chip")!.className).toContain("bad");
    expect(chiprow().querySelector(".m-runbtn")!.className).toContain("bad");
  });

  it("the arrow opens the time the test was done (the end of the run's own final_eval stage), and closes again", async () => {
    serve("PROMOTE", { stages });
    renderApp("/mission");
    await waitFor(() => expect(chiprow()).not.toBeNull());
    const btn = chiprow().querySelector<HTMLButtonElement>(".m-runbtn")!;
    expect(btn.getAttribute("aria-expanded")).toBe("false");
    expect(screen.queryByText(/Test done/)).toBeNull();
    fireEvent.click(btn);
    expect(btn.getAttribute("aria-expanded")).toBe("true");
    const panel = await screen.findByRole("region", { name: "When the test ran" });
    // 18:54:32 on 4 Oct: when the sealed test finished. Not the export time (5 Oct 00:04) and not the run's start.
    await waitFor(() => expect(within(panel).getByText("4 Oct 2026, 18:54:32 (UTC)")).toBeTruthy());
    expect(within(panel).getByText("Test done")).toBeTruthy();
    expect(panel.textContent).not.toContain("00:04:35");
    fireEvent.keyDown(document, { key: "Escape" });
    await waitFor(() => expect(btn.getAttribute("aria-expanded")).toBe("false"));
  });

  it("a run whose test stage never finished shows the time as not measured, not a made-up time", async () => {
    serve("PROMOTE", { stages: [{ name: "final_eval", status: "running", started_at: null, ended_at: null }] });
    renderApp("/mission");
    await waitFor(() => expect(chiprow()).not.toBeNull());
    fireEvent.click(chiprow().querySelector(".m-runbtn")!);
    const panel = await screen.findByRole("region", { name: "When the test ran" });
    await waitFor(() => expect(within(panel).getByText("Test done")).toBeTruthy());
    expect(panel.textContent).toContain("not measured");
  });
});
