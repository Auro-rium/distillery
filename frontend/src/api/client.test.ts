import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, api, setAdminToken } from "./client";

afterEach(() => { setAdminToken(null); vi.unstubAllGlobals(); });

describe("client", () => {
  it("sends the admin token header only on admin calls, never touching storage", async () => {
    const f = vi.fn().mockResolvedValue(new Response(JSON.stringify({ run_id: "x" }), { status: 202 }));
    vi.stubGlobal("fetch", f);
    setAdminToken("tok");
    await api.createRun({ pack: "sql", scale: "tiny", dry_run: true });
    expect(new Headers(f.mock.calls[0][1].headers).get("X-Admin-Token")).toBe("tok");
    f.mockResolvedValue(new Response("[]"));
    await api.runs();
    expect(new Headers(f.mock.calls[1][1].headers).get("X-Admin-Token")).toBeNull();
    expect(JSON.stringify(Object.entries(localStorage))).not.toContain("tok");
  });
  it("maps error bodies to ApiError", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ error: "demo_budget_exhausted", message: "m" }), { status: 503, headers: { "Retry-After": "7" } })));
    await expect(api.playground("q")).rejects.toMatchObject({ status: 503, code: "demo_budget_exhausted", retryAfter: 7 });
    expect(new ApiError(0, "network", "x")).toBeInstanceOf(Error);
  });
  it("treats a 200 that is not JSON (static host answering /api with index.html) as an error, not data", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(
      new Response("<!doctype html><html></html>", { status: 200, headers: { "Content-Type": "text/html" } })));
    await expect(api.runs()).rejects.toMatchObject({ status: 200, code: "invalid_response" });
  });
});
