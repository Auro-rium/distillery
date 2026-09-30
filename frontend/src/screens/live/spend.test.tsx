import { act, cleanup, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { contract, contractFetch } from "../../testutil/contract";
import { FakeES, renderApp, stubFetch } from "../../testutil/render";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
const rid = contract.run.run_id;
const fill = (el: Element | null) => (el as HTMLElement).style.getPropertyValue("--f");

describe("spend meter and per-model bars follow real spend events", () => {
  beforeEach(() => { FakeES.all = []; vi.stubGlobal("EventSource", FakeES); });
  const running = { ...contract.run, status: "running", spend: { total_usd: 2.5, cap_usd: 10, finetune_usd_estimate: null, by_model: { "m-1": { usd: 0.5, calls: 1, input_tokens: 100, output_tokens: 10 } } } };

  it("bar lengths come from payload value / cap; text is the exact payload value; a new event moves both", async () => {
    stubFetch(contractFetch({ run: running }));
    const { container } = renderApp(`/runs/${rid}`);
    await waitFor(() => expect(FakeES.all).toHaveLength(1));
    await screen.findByRole("meter", { name: "Spend against cap" });
    const meter = () => screen.getByRole("meter", { name: "Spend against cap" });
    expect(fill(meter().firstElementChild)).toBe("0.25");
    expect(fill(container.querySelector(".mbar-fill"))).toBe("0.05");
    expect(meter().getAttribute("aria-valuenow")).toBe("2.5");
    expect(container.textContent).toContain("$2.5000");

    act(() => {
      FakeES.all[0].onopen!();
      FakeES.all[0].emit("spend", 1, { total_usd: 5, cap_usd: 10, finetune_usd_estimate: null, by_model: { "m-1": { usd: 4, calls: 3, input_tokens: 300, output_tokens: 30 }, "m-2": { usd: 1, calls: 1, input_tokens: 5, output_tokens: 5 } } });
    });
    expect(fill(meter().firstElementChild)).toBe("0.5");
    const bars = [...container.querySelectorAll(".mbar-fill")].map(fill);
    expect(bars).toEqual(["0.4", "0.1"]); // a model that first appears in an event gets a bar; nothing else is drawn
    expect(container.textContent).toContain("$5.0000");
    expect(container.textContent).toContain("$4.0000");
  });
  it("no cap, no meter and no bars (nothing to measure against)", async () => {
    stubFetch(contractFetch({ run: { ...running, spend: { ...running.spend, cap_usd: 0 } } }));
    const { container } = renderApp(`/runs/${rid}`);
    await screen.findByText("Spend");
    expect(screen.queryByRole("meter")).toBeNull();
    expect(container.querySelectorAll(".mbar-fill")).toHaveLength(0);
  });
});
