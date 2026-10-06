import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, setAdminToken } from "../../api/client";
import NewRun from ".";
import { describeCreateError } from "./errors";

const cfg = { models: {}, thresholds: {}, run_cap_usd: 5, playground: {} };
const json = (b: unknown, status = 200, h: Record<string, string> = {}) =>
  new Response(JSON.stringify(b), { status, headers: h });

function setup(post: () => Response) {
  const f = vi.fn((_url: string, init?: RequestInit) =>
    Promise.resolve(init?.method === "POST" ? post() : json(cfg)));
  vi.stubGlobal("fetch", f);
  render(
    <MemoryRouter initialEntries={["/new"]}>
      <Routes>
        <Route path="/new" element={<NewRun />} />
        <Route path="/runs/:id" element={<div>LIVE VIEW</div>} />
      </Routes>
    </MemoryRouter>,
  );
  return f;
}
afterEach(() => { cleanup(); setAdminToken(null); vi.unstubAllGlobals(); });

describe("describeCreateError", () => {
  it.each([[401, /token missing/i], [403, /rejected/i], [409, /already active/i], [429, /30 seconds/]])(
    "maps %i", (status, re) => {
      expect(describeCreateError(new ApiError(status, "x", "m", status === 429 ? 30 : null))).toMatch(re);
    });
});

describe("NewRun form", () => {
  it("prefills budget from /api/config and submits a dry run, then navigates", async () => {
    const f = setup(() => json({ run_id: "abc" }, 202));
    await waitFor(() => expect((screen.getByLabelText("Budget cap (USD)") as HTMLInputElement).value).toBe("5"));
    fireEvent.click(screen.getByRole("button", { name: /start dry run/i }));
    await screen.findByText("LIVE VIEW");
    const post = f.mock.calls.find((c) => c[1]?.method === "POST")!;
    expect(JSON.parse(post[1]!.body as string)).toEqual({ pack: "sql", scale: "tiny", dry_run: true, budget_usd: 5 });
  });
  it("lists the packs the server offers and posts the chosen one; one pack means a fixed select", async () => {
    const two = { ...cfg, packs: [{ name: "sql", language: "sql", answer_label: "SQL" }, { name: "toolcall", language: "json", answer_label: "tool calls" }] };
    const f = vi.fn((_url: string, init?: RequestInit) =>
      Promise.resolve(init?.method === "POST" ? json({ run_id: "abc" }, 202) : json(two)));
    vi.stubGlobal("fetch", f);
    render(<MemoryRouter initialEntries={["/new"]}><Routes><Route path="/new" element={<NewRun />} /><Route path="/runs/:id" element={<div>LIVE VIEW</div>} /></Routes></MemoryRouter>);
    const sel = (await screen.findByLabelText("Pack")) as HTMLSelectElement;
    await waitFor(() => expect(sel.disabled).toBe(false));
    expect([...sel.options].map((o) => o.value)).toEqual(["sql", "toolcall"]);
    fireEvent.change(sel, { target: { value: "toolcall" } });
    fireEvent.click(screen.getByRole("button", { name: /start dry run/i }));
    await screen.findByText("LIVE VIEW");
    const post = f.mock.calls.find((c) => c[1]?.method === "POST")!;
    expect(JSON.parse(post[1]!.body as string).pack).toBe("toolcall");
  });
  it("posts an optional run id (trimmed) and omits it when blank", async () => {
    const f = setup(() => json({ run_id: "dry-chaos-x" }, 202));
    await waitFor(() => expect((screen.getByLabelText("Budget cap (USD)") as HTMLInputElement).value).toBe("5"));
    fireEvent.change(screen.getByLabelText("Run id (optional)"), { target: { value: " dry-chaos-x " } });
    fireEvent.click(screen.getByRole("button", { name: /start dry run/i }));
    await screen.findByText("LIVE VIEW");
    const post = f.mock.calls.find((c) => c[1]?.method === "POST")!;
    expect(JSON.parse(post[1]!.body as string).run_id).toBe("dry-chaos-x");
  });
  it("blocks a live run without token or approval, without calling the API", async () => {
    const f = setup(() => json({}, 202));
    fireEvent.click(screen.getByLabelText(/^\s*Dry run \(fake/));
    expect(screen.getByText(/spend real money/i)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: /start live run/i }));
    expect((await screen.findByRole("alert")).textContent).toMatch(/token missing/i);
    fireEvent.change(screen.getByLabelText("Admin token"), { target: { value: "t" } });
    fireEvent.click(screen.getByRole("button", { name: /start live run/i }));
    await waitFor(() => expect(screen.getByRole("alert").textContent).toMatch(/approval/i));
    expect(f.mock.calls.some((c) => c[1]?.method === "POST")).toBe(false);
  });
  it("shows the 409 message from the server and stays on the form", async () => {
    setup(() => json({ error: "run_active", message: "busy" }, 409));
    fireEvent.click(screen.getByRole("button", { name: /start dry run/i }));
    expect((await screen.findByRole("alert")).textContent).toMatch(/already active/i);
    expect(screen.queryByText("LIVE VIEW")).toBeNull();
  });
  it("never persists the admin token", async () => {
    setup(() => json({ error: "forbidden", message: "no" }, 403));
    fireEvent.change(screen.getByLabelText("Admin token"), { target: { value: "sekrit-token" } });
    fireEvent.click(screen.getByRole("button", { name: /start dry run/i }));
    await screen.findByRole("alert");
    expect(JSON.stringify([...Object.entries(localStorage), ...Object.entries(sessionStorage)])).not.toContain("sekrit");
  });
});

describe("NewRun form structure", () => {
  const form = () => document.querySelector("form")!;
  const named = (el: Element) => Array.from((el as HTMLInputElement).labels ?? []).map((l) => l.textContent?.trim());

  it("every control has a real <label>, and ids are unique", async () => {
    setup(() => json({}, 202));
    await waitFor(() => expect((screen.getByLabelText("Budget cap (USD)") as HTMLInputElement).value).toBe("5"));
    const controls = Array.from(form().querySelectorAll("input, select, textarea"));
    expect(controls.length).toBeGreaterThanOrEqual(6);
    for (const c of controls) expect(named(c).length, `${c.outerHTML}`).toBeGreaterThan(0);
    // radios are named by the <label> that wraps them; every other control is named through for/id
    const others = controls.filter((c) => (c as HTMLInputElement).type !== "radio");
    const ids = others.map((c) => c.id);
    expect(ids.every(Boolean)).toBe(true);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("Scale is a fieldset with a legend and four radios; only one is checked", () => {
    setup(() => json({}, 202));
    const group = screen.getByRole("group", { name: "Scale" });
    const radios = within(group).getAllByRole("radio") as HTMLInputElement[];
    expect(radios.map((r) => r.value)).toEqual(["tiny", "small", "full", "gated"]);
    expect(radios.filter((r) => r.checked).map((r) => r.value)).toEqual(["tiny"]);
    fireEvent.click(radios[1]);
    expect(radios.filter((r) => r.checked).map((r) => r.value)).toEqual(["small"]);
  });

  it("Dry run is a switch (on by default) with helper text wired by aria-describedby", () => {
    setup(() => json({}, 202));
    const sw = screen.getByRole("switch", { name: /^Dry run \(fake/ }) as HTMLInputElement;
    expect(sw.checked).toBe(true);
    const help = document.getElementById(sw.getAttribute("aria-describedby")!);
    expect(help?.textContent ?? "").not.toBe("");
    fireEvent.click(sw);
    expect(sw.checked).toBe(false);
  });

  it("Budget shows its unit and the server default exactly as the payload gives it", async () => {
    setup(() => json({}, 202));
    const budget = screen.getByLabelText("Budget cap (USD)");
    await waitFor(() => expect(document.getElementById(budget.getAttribute("aria-describedby")!.split(" ")[0])?.textContent).toMatch(/server default of \$5\.00/));
    expect(within(budget.closest(".field")!).getByText("USD")).toBeTruthy();
  });

  it("Admin token is a password field with memory-only help text, and says when it is needed", () => {
    setup(() => json({}, 202));
    const tok = screen.getByLabelText("Admin token") as HTMLInputElement;
    expect(tok.type).toBe("password");
    expect(tok.getAttribute("autocomplete")).toBe("off");
    const help = tok.getAttribute("aria-describedby")!.split(" ").map((i) => document.getElementById(i)?.textContent ?? "").join(" ");
    expect(help).toMatch(/memory only/i);
    expect(help).toMatch(/optional for dry runs/i);
    fireEvent.click(screen.getByRole("switch"));
    const live = tok.getAttribute("aria-describedby")!.split(" ").map((i) => document.getElementById(i)?.textContent ?? "").join(" ");
    expect(live).toMatch(/required for live runs/i);
  });

  it("marks the budget invalid (aria-invalid) when it is not a positive number, without calling the API", async () => {
    const f = setup(() => json({}, 202));
    await waitFor(() => expect((screen.getByLabelText("Budget cap (USD)") as HTMLInputElement).value).toBe("5"));
    fireEvent.change(screen.getByLabelText("Budget cap (USD)"), { target: { value: "abc" } });
    fireEvent.click(screen.getByRole("button", { name: /start dry run/i }));
    expect((await screen.findByRole("alert")).textContent).toMatch(/positive number/i);
    expect(screen.getByLabelText("Budget cap (USD)").getAttribute("aria-invalid")).toBe("true");
    expect(f.mock.calls.some((c) => c[1]?.method === "POST")).toBe(false);
  });

  it("submit is clickable while the config is still loading, and shows Starting… while busy", async () => {
    vi.stubGlobal("fetch", vi.fn((_u: string, init?: RequestInit) => (init?.method === "POST" ? new Promise(() => undefined) : new Promise(() => undefined))));
    render(<MemoryRouter initialEntries={["/new"]}><Routes><Route path="/new" element={<NewRun />} /></Routes></MemoryRouter>);
    const btn = screen.getByRole("button", { name: /start dry run/i }) as HTMLButtonElement;
    expect(btn.disabled).toBe(false);
    fireEvent.click(btn);
    await waitFor(() => expect((screen.getByRole("button", { name: /starting/i }) as HTMLButtonElement).disabled).toBe(true));
  });

  it("live mode shows the spend warning and the approval checkbox as a labelled checkbox", () => {
    setup(() => json({}, 202));
    fireEvent.click(screen.getByRole("switch"));
    expect(screen.getByRole("checkbox", { name: /I approve spending up to the budget cap/ })).toBeTruthy();
    expect(screen.getByRole("button", { name: /start live run/i })).toBeTruthy();
  });
});
