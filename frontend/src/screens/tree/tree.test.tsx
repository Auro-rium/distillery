import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { NOT_MEASURED } from "../../api/format";
import Tree from ".";
import { unexplained, visibleText } from "../../testutil/provenance";
import { RUN, fakeViewport, n, pos, show } from "./treeharness";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
const detail = () => screen.getByRole("region", { name: "Node detail" });
const REAL = [
  n("base", null, { label: "base model", dev_score: 0.4 }),
  n("r1", "base", { label: "round 1", round: 1, dev_score: 0.6, hypothesis: "add joins", data_delta: { added_rows: 7, families: ["join", "null_handling"] }, sandbox_image: "img-abc123" }),
  n("r2", "r1", { label: "round 2", round: 2, dev_score: 0.9, selected: true }),
];

describe("Tree screen states", () => {
  it("shows an empty state and invents no nodes", async () => {
    show([]);
    expect(await screen.findByText(/no sandbox branches yet/i)).toBeTruthy();
    expect(screen.queryAllByRole("button")).toHaveLength(0);
    expect(screen.queryByRole("region", { name: "Node detail" })).toBeNull();
  });
  it("a failing tree request is an error with its code, never an empty state", async () => {
    vi.stubGlobal("fetch", vi.fn((u: string) => Promise.resolve(u.endsWith("/tree")
      ? new Response(JSON.stringify({ error: "tree_down", message: "cannot read lineage" }), { status: 500 })
      : new Response(JSON.stringify(RUN), { status: 200 }))));
    const { container } = render(<MemoryRouter initialEntries={["/runs/r/tree"]}><Routes><Route path="/runs/:id/tree" element={<Tree />} /></Routes></MemoryRouter>);
    await waitFor(() => expect(screen.getByRole("alert").textContent).toContain("tree_down"));
    expect(container.textContent).toContain("cannot read lineage");
    expect(container.textContent).not.toMatch(/no sandbox branches/i);
    expect(screen.queryByRole("group", { name: "Tree view controls" })).toBeNull();
  });
});

describe("Tree inside the run shell", () => {
  it("has one h1 (the run), the Tree tab selected, and the label banner from the run's flags", async () => {
    show(REAL);
    await screen.findByRole("button", { name: /^base model/ });
    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
    expect(screen.getByRole("tab", { name: "Tree" }).getAttribute("aria-selected")).toBe("true");
    expect(screen.getAllByRole("status").map((e) => e.textContent)).toContain("DRY RUN — fake models, numbers are NOT results");
  });
  it("when the run cannot be loaded the label is unknown and says so; the error is shown", async () => {
    show(REAL, RUN, { runFails: true });
    await screen.findByText(/Could not load the run label/);
    expect(screen.getAllByRole("status").some((e) => /^Label unknown/.test(e.textContent ?? ""))).toBe(true);
    expect(screen.getByRole("alert").textContent).toContain("run_down");
  });
  it("shows no tree numbers before the run label is known", async () => {
    let release: (r: Response) => void = () => undefined;
    vi.stubGlobal("fetch", vi.fn((u: string) => (u.endsWith("/tree")
      ? Promise.resolve(new Response(JSON.stringify({ nodes: REAL }), { status: 200 }))
      : new Promise<Response>((res) => { release = res; }))));
    render(<MemoryRouter initialEntries={["/runs/r/tree"]}><Routes><Route path="/runs/:id/tree" element={<Tree />} /></Routes></MemoryRouter>);
    await new Promise((r) => setTimeout(r, 30));
    expect(screen.queryByRole("button", { name: /dev score/ })).toBeNull(); // the tree has arrived, the label has not
    act(() => release(new Response(JSON.stringify(RUN), { status: 200 })));
    expect(await screen.findByRole("button", { name: /^round 1/ })).toBeTruthy();
  });
});

describe("Tree nodes", () => {
  it("each node shows its own label, round and dev score once; nothing is repeated or invented", async () => {
    show(REAL);
    const r1 = await screen.findByRole("button", { name: /^round 1/ });
    expect(r1.textContent).toContain("round 1");
    expect(r1.textContent).toContain("0.600");
    expect(r1.textContent!.split("round 1").length - 1).toBe(1); // the label is not repeated as a round tag
    const base = screen.getByRole("button", { name: /^base model/ });
    expect(base.textContent).toContain("0.400");
    expect(base.textContent).not.toContain("round"); // a node without a round has no round tag
  });
  it("puts a round tag on a node whose label does not already say the round", async () => {
    show([n("a", null, { label: "candidate-a", round: 3, dev_score: 0.5 })]);
    const a = await screen.findByRole("button", { name: /^candidate-a/ });
    expect(a.textContent).toContain("round 3");
  });
  it("cost is never repeated per node: not measured is said once, and a measured cost shows on its node", async () => {
    show(REAL);
    await screen.findByRole("button", { name: /^base model/ });
    for (const b of screen.getAllByRole("button", { name: /dev score/ })) expect(b.textContent).not.toContain(NOT_MEASURED);
    expect(screen.getAllByText(`Cost per node: ${NOT_MEASURED}`)).toHaveLength(1);
    cleanup();
    show([n("a", null, { label: "with cost", cost_usd: 0.25 }), n("b", "a", { label: "no cost" })]);
    const a = await screen.findByRole("button", { name: /^with cost/ });
    expect(a.textContent).toContain("$0.2500");
    expect(screen.queryByText(/Cost per node/)).toBeNull();
  });
  it("marks the candidate the API selected, in words and not by colour alone", async () => {
    show(REAL);
    const r2 = await screen.findByRole("button", { name: /^round 2/ });
    expect(r2.textContent).toContain("candidate");
    expect(r2.getAttribute("aria-label")).toContain("selected candidate");
    expect(screen.getByRole("button", { name: /^round 1/ }).textContent).not.toContain("candidate");
  });
  it("only the current node is a tab stop; arrows move between the others", async () => {
    show(REAL);
    await screen.findByRole("button", { name: /^round 2/ });
    const stops = screen.getAllByRole("button", { name: /dev score/ }).filter((b) => b.tabIndex === 0);
    expect(stops).toHaveLength(1);
    expect(stops[0].getAttribute("aria-label")).toMatch(/^round 2/); // the selected candidate is the default
  });
});

describe("Node detail (master-detail)", () => {
  it("defaults to the selected candidate, and to the first root when none is selected", async () => {
    show(REAL);
    await screen.findByRole("button", { name: /^round 2/ });
    await waitFor(() => expect(within(detail()).getByRole("heading", { level: 2 }).textContent).toBe("round 2"));
    cleanup();
    show([n("root", null, { label: "the root" }), n("kid", "root")]);
    await screen.findByRole("button", { name: /^the root/ });
    await waitFor(() => expect(within(detail()).getByRole("heading", { level: 2 }).textContent).toBe("the root"));
  });
  it("selects by click and by keyboard, and shows the node's real fields", async () => {
    show(REAL);
    const r1 = await screen.findByRole("button", { name: /^round 1/ });
    fireEvent.keyDown(r1, { key: "Enter" });
    expect(r1.getAttribute("aria-pressed")).toBe("true");
    const d = detail();
    for (const t of ["add joins", "7 rows added", "join", "null_handling", "0.600", "img-abc123"]) expect(d.textContent).toContain(t);
    fireEvent.click(screen.getByRole("button", { name: /^base model/ }));
    expect(screen.getByRole("button", { name: /^base model/ }).getAttribute("aria-pressed")).toBe("true");
    expect(r1.getAttribute("aria-pressed")).toBe("false");
    expect(within(detail()).getByRole("heading", { level: 2 }).textContent).toBe("base model");
  });
  it("shows every null field as 'not measured', never blank, 0 or 'none'", async () => {
    show(REAL);
    fireEvent.click(await screen.findByRole("button", { name: /^base model/ }));
    const d = detail();
    const cell = (label: string) => within(d).getByText(label).nextElementSibling!.textContent;
    for (const f of ["Round", "Hypothesis", "Data delta", "Cost", "Sandbox image"]) expect(cell(f), f).toBe(NOT_MEASURED);
    expect(cell("Dev score")).toBe("0.400");
  });
  it("shows the path from the root as real ancestors; a crumb selects that node", async () => {
    show(REAL);
    await screen.findByRole("button", { name: /^round 2/ });
    const path = screen.getByRole("navigation", { name: "Path from root" });
    expect([...path.querySelectorAll("li")].map((li) => li.textContent)).toEqual(["base model", "round 1", "round 2"]);
    expect(path.querySelector('[aria-current="location"]')!.textContent).toBe("round 2");
    fireEvent.click(within(path).getByRole("button", { name: "Go to base model" }));
    expect(within(detail()).getByRole("heading", { level: 2 }).textContent).toBe("base model");
  });
  it("names the candidate with a badge in the detail", async () => {
    show(REAL);
    await screen.findByRole("button", { name: /^round 2/ });
    expect(within(detail()).getByText("selected candidate")).toBeTruthy();
  });
});

describe("Tree adapts to the viewport", () => {
  it("lays generations out left to right in a wide viewport and top to bottom in a narrow one", async () => {
    const vp = fakeViewport({ w: 1000, h: 500 });
    try {
      const chain = [n("a", null), n("b", "a"), n("c", "b")];
      show(chain);
      const at = async (name: RegExp) => pos(await screen.findByRole("button", { name }));
      const [a, b] = [await at(/^a,/), await at(/^b,/)];
      expect(b.x).toBeGreaterThan(a.x);
      expect(b.y).toBe(a.y);
      vp.resize(360, 500);
      await waitFor(() => expect(pos(screen.getByRole("button", { name: /^b,/ })).y).toBeGreaterThan(pos(screen.getByRole("button", { name: /^a,/ })).y));
      expect(pos(screen.getByRole("button", { name: /^b,/ })).x).toBe(pos(screen.getByRole("button", { name: /^a,/ })).x);
    } finally { vp.restore(); }
  });
});

describe("Tree provenance", () => {
  it("every number on screen comes from the tree payload", async () => {
    const { container } = show(REAL);
    await screen.findByRole("button", { name: /^round 2/ });
    fireEvent.click(screen.getByRole("button", { name: /^round 1/ }));
    const text = visibleText(container);
    expect(text).toContain("0.600");
    expect(unexplained(text, [REAL, RUN], [])).toEqual([]);
  });
});
