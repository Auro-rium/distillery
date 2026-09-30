import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
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
