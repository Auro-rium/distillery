import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { RunDetail } from "../api/types";
import { contract, contractFetch } from "../testutil/contract";
import { DRY_RUN_TEXT, UNLABELLED_TEXT, LIVE_TEXT } from ".";
import { RunShell } from "./RunShell";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

const run = contract.run as unknown as RunDetail;
const id = run.run_id;

function shell(props: Partial<Parameters<typeof RunShell>[0]> = {}, path = `/runs/${id}`) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <RunShell runId={id} tab="live" run={run} {...props}><p>panel body</p></RunShell>
    </MemoryRouter>,
  );
}
const slot = (c: HTMLElement, name: string) => c.querySelector(`[data-slot="${name}"]`);

describe("RunShell header", () => {
  it("shows the run id, the status from the payload and the dry-run label from the flags", () => {
    const { container } = shell();
    expect(slot(container, "id")!.textContent).toBe(id);
    expect(slot(container, "status")!.textContent).toBe(run.status);
    expect(screen.getByRole("status").textContent).toBe(DRY_RUN_TEXT);
  });
  it("labels a recorded run and a live run from their flags", () => {
    const rec = shell({ run: { ...run, dry_run: false, recorded: true, recorded_at: "2026-01-02T03:04:05Z" } });
    expect(screen.getByRole("status").textContent).toBe("Recorded run · 2026-01-02T03:04:05Z · real Token Factory jobs");
    rec.unmount();
    shell({ run: { ...run, status: "running", dry_run: false, recorded: false } });
    expect(screen.getByRole("status").textContent).toBe(LIVE_TEXT);
  });
  it("says Label unknown when the flags are missing, never a real-run label", () => {
    const bare = { ...run } as Partial<RunDetail>;
    delete bare.dry_run; delete bare.recorded; delete bare.recorded_at;
    shell({ run: bare as RunDetail });
    expect(screen.getByRole("status").textContent).toBe(UNLABELLED_TEXT);
    expect(screen.getByRole("status").textContent).toMatch(/^Label unknown/);
  });
  it("reads the verdict from the report once the run is complete, and shows nothing while it loads or if it fails", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new TypeError("Failed to fetch"))));
    const failed = shell();
    await new Promise((r) => setTimeout(r, 20));
    expect(slot(failed.container, "verdict")!.textContent).toBe("");
    expect(screen.queryByRole("alert")).toBeNull();
    failed.unmount();
    vi.stubGlobal("fetch", vi.fn(contractFetch()));
    const ok = shell();
    expect(slot(ok.container, "verdict")!.textContent).toBe("");
    await waitFor(() => expect(slot(ok.container, "verdict")!.textContent).toBe(contract.report.decision));
    ok.unmount();
    const f = vi.fn(contractFetch());
    vi.stubGlobal("fetch", f);
    const running = shell({ run: { ...run, status: "running" } });
    await new Promise((r) => setTimeout(r, 20));
    expect(slot(running.container, "verdict")!.textContent).toBe("");
    expect(f).not.toHaveBeenCalled();
  });
});

describe("RunShell while the run is still loading", () => {
  it("keeps every header slot, claims no label, and does not pretend the flags are missing", () => {
    const loaded = shell();
    const slotsLoaded = [...loaded.container.querySelectorAll("[data-slot]")].map((e) => e.getAttribute("data-slot"));
    loaded.unmount();
    const { container } = shell({ run: undefined });
    const slots = [...container.querySelectorAll("[data-slot]")].map((e) => e.getAttribute("data-slot"));
    expect(slots).toEqual(slotsLoaded);
    expect(slot(container, "id")!.textContent).toBe(id);
    expect(slot(container, "status")!.textContent).toBe("");
    expect(screen.queryByRole("status")).toBeNull();
    expect(container.textContent).not.toMatch(/DRY RUN|Label unknown|Live run|Recorded run/);
    expect(slot(container, "label")!.getAttribute("data-state")).toBe("pending");
    expect(within(container).getByText("panel body")).toBeTruthy();
  });
});

describe("RunShell tabs", () => {
  it("are links to the run, report and tree routes, with the current one selected", () => {
    shell({ tab: "report" }, `/runs/${id}/report`);
    const tabs = screen.getAllByRole("tab");
    expect(tabs.map((t) => t.textContent)).toEqual(["Run", "Report", "Tree"]);
    expect(tabs.map((t) => t.getAttribute("href"))).toEqual([`/runs/${id}`, `/runs/${id}/report`, `/runs/${id}/tree`]);
    expect(tabs.map((t) => t.getAttribute("aria-selected"))).toEqual(["false", "true", "false"]);
    expect(tabs.map((t) => t.getAttribute("aria-current"))).toEqual([null, "page", null]);
    expect(screen.getByRole("tablist", { name: "Run sections" })).toBeTruthy();
  });
  it("encode the run id in their URLs", () => {
    shell({ runId: "a b/c", run: undefined });
    expect(screen.getAllByRole("tab").map((t) => t.getAttribute("href"))[0]).toBe("/runs/a%20b%2Fc");
  });
  it("never point aria-controls at an element that does not exist, and the panel is named by the current tab", () => {
    shell({ tab: "tree" }, `/runs/${id}/tree`);
    for (const t of screen.getAllByRole("tab")) {
      const c = t.getAttribute("aria-controls");
      if (c !== null) expect(document.getElementById(c), `aria-controls=${c}`).not.toBeNull();
    }
    const panel = screen.getByRole("tabpanel");
    expect(panel.getAttribute("aria-labelledby")).toBe(screen.getByRole("tab", { name: "Tree" }).id);
    expect(within(panel).getByText("panel body")).toBeTruthy();
  });
});
