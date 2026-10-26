import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
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
    // a run that is still going has no report to read (the run list is fetched for the Latest tag, nothing else)
    expect(f.mock.calls.filter(([u]) => String(u).includes("/report"))).toHaveLength(0);
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
    // "Live run" is also the first tab's name, so the label is checked in its own slot
    expect(slot(container, "label")!.textContent).toBe("");
    expect(container.textContent).not.toMatch(/DRY RUN|Label unknown|Recorded run/);
    expect(slot(container, "label")!.getAttribute("data-state")).toBe("pending");
    expect(within(container).getByText("panel body")).toBeTruthy();
  });
});

describe("RunShell sticky header", () => {
  it("only the header (run, status, label) is sticky; the tab bar is outside it, and there is no white rule under the tabs", () => {
    const { container } = shell();
    const sticky = container.querySelector(".run-sticky")!;
    expect(sticky.querySelector(".run-head")).not.toBeNull();
    expect(sticky.querySelector(".run-tabbar")).toBeNull();
    expect(sticky.querySelector("[role=tablist]")).toBeNull();
    const tabbar = container.querySelector(".run-tabbar")!;
    expect(tabbar.querySelector("[role=tablist]")).not.toBeNull();
    expect(container.querySelector(".run-sep")).toBeNull();
    // header, then tabs, then the panel
    expect(sticky.compareDocumentPosition(tabbar) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(tabbar.compareDocumentPosition(container.querySelector(".run-panel")!) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });
});

describe("RunShell heading", () => {
  it("reads as a plain title and one sentence, with the technical id on its own labelled line", () => {
    const { container } = shell();
    expect(screen.getByRole("heading", { level: 1 }).textContent).toBe("SQL run");
    expect(container.querySelector(".run-about")!.textContent).toBe("One test of the small model on SQL questions.");
    const line = container.querySelector(".run-idline")!;
    expect(line.textContent).toContain("Run ID");
    expect(slot(container, "id")!.textContent).toBe(id);
    // the id is not part of the heading any more
    expect(screen.getByRole("heading", { level: 1 }).textContent).not.toContain(id);
  });
});

describe("RunShell latest-run tag", () => {
  const recorded = { ...contract.replay[0], run_id: id, dry_run: false, recorded: true, recorded_at: "2026-10-05T00:04:35Z", decision: "PROMOTE" };
  it("shows LATEST RUN in capitals at the top for the run the Evidence link points at", async () => {
    vi.stubGlobal("fetch", vi.fn(contractFetch({ replay: [recorded] })));
    const { container } = shell();
    await waitFor(() => expect(slot(container, "latest")!.textContent).toBe("LATEST RUN"));
    expect(container.querySelector(".run-about")!.textContent).toBe("The newest recorded test of the small model on SQL questions.");
    // it is the first thing in the header, above the run title
    const head = container.querySelector(".run-head")!;
    expect(head.firstElementChild).toBe(slot(container, "latest"));
  });
  it("is not shown for any other run, and the slot is still there so nothing shifts", async () => {
    vi.stubGlobal("fetch", vi.fn(contractFetch({ replay: [{ ...recorded, run_id: "some-newer-run" }] })));
    const { container } = shell();
    await new Promise((r) => setTimeout(r, 30));
    expect(slot(container, "latest")).not.toBeNull();
    expect(slot(container, "latest")!.textContent).toBe("");
  });
});

describe("RunShell opens out of the clicked tab", () => {
  const motionOk = () => vi.stubGlobal("matchMedia", (q: string) => ({ matches: false, media: q, addEventListener: () => undefined, removeEventListener: () => undefined }));
  it("a page reached by a tab click grows out of that tab: origin set once, then cleared", () => {
    motionOk();
    const first = shell();
    const tab = [...first.container.querySelectorAll<HTMLElement>(".run-tabbar a")].find((a) => a.textContent === "Report")!;
    tab.getBoundingClientRect = () => ({ left: 100, top: 10, width: 60, height: 20, right: 160, bottom: 30 }) as DOMRect;
    fireEvent.click(tab);
    first.unmount();
    const second = shell({}, `/runs/${id}/report`);
    const panel = second.container.querySelector<HTMLElement>(".run-body")!;
    expect(panel.dataset.opening).toBe("true");
    expect(panel.style.getPropertyValue("--ox")).toMatch(/px$/);
    // the opening window starts as a strip the width of the clicked tab, from the top of the page
    for (const v of ["--oy", "--cl", "--cr", "--cb"]) expect(panel.style.getPropertyValue(v), v).toMatch(/px$/);
    second.unmount();
    // the origin was used once: a page opened any other way (typed URL, link elsewhere) does not animate
    const third = shell();
    expect(third.container.querySelector<HTMLElement>(".run-body")!.dataset.opening).toBeUndefined();
  });
  it("does not animate under reduced motion", () => {
    vi.stubGlobal("matchMedia", (q: string) => ({ matches: q.includes("reduce"), media: q, addEventListener: () => undefined, removeEventListener: () => undefined }));
    const first = shell();
    fireEvent.click(first.container.querySelector<HTMLElement>(".run-tabbar a")!);
    first.unmount();
    expect(shell().container.querySelector<HTMLElement>(".run-body")!.dataset.opening).toBeUndefined();
  });
});

describe("RunShell tabs", () => {
  it("are links to the run, report and tree routes, with the current one selected", () => {
    shell({ tab: "report" }, `/runs/${id}/report`);
    const tabs = screen.getAllByRole("tab");
    expect(tabs.map((t) => t.textContent)).toEqual(["Live run", "Report", "Tree"]);
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
