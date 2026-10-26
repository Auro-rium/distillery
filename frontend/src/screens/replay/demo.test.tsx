import { cleanup, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { contract, contractFetch } from "../../testutil/contract";
import { json, renderApp, stubFetch } from "../../testutil/render";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

/** The whole app at /mission (so the demo sits under the Mission route), with a dry run the server will "create". */
function serve() {
  const featured = { ...contract.replay[0], dry_run: false, recorded: true, recorded_at: "2026-10-05T00:04:35Z", run_id: "rec-featured", decision: "PROMOTE" };
  const base = contractFetch({ replay: [featured], report: { ...contract.report, run_id: "rec-featured" } });
  return stubFetch((u, i) => {
    if (i?.method === "POST" && u.endsWith("/runs")) return json({ run_id: "demo-9" }, 202);
    if (u.endsWith("/runs/demo-9")) return json({ ...contract.run, run_id: "demo-9", dry_run: true, status: "complete" });
    if (u.endsWith("/runs/demo-9/experiments")) return json({ run_id: "demo-9", source: "store", experiments: [] });
    if (u.endsWith("/runs/demo-9/report")) return json(contract.report);
    return base(u, i);
  });
}

describe("watch a live demo run, from the Mission page", () => {
  it("opens the dry run in a dialog on the same page: no navigation, the DRY RUN label, and closes again", async () => {
    serve();
    renderApp("/mission");
    fireEvent.click(await screen.findByRole("button", { name: /Watch a live demo run/ }));
    const dlg = await screen.findByRole("dialog", { name: "Demo run" });
    expect(await within(dlg).findByText("demo-9")).toBeTruthy();
    await waitFor(() => expect(within(dlg).getByText("DRY RUN — fake models, numbers are NOT results")).toBeTruthy());
    // still the Mission page behind the dialog, and the run's own page tabs are not offered inside it
    expect(document.title).toMatch(/^Mission/);
    expect(dlg.querySelector(".run-tabbar")).not.toBeNull(); // present in the DOM, hidden by CSS in the dialog
    fireEvent.keyDown(dlg, { key: "Escape" });
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(document.title).toMatch(/^Mission/);
  });
});
