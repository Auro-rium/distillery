import { act, cleanup, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { contract, contractFetch } from "../../testutil/contract";
import { FakeES, renderApp, stubFetch } from "../../testutil/render";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
const rid = contract.run.run_id;
const fill = (el: Element | null) => (el as HTMLElement).style.getPropertyValue("--f");

describe("spend numbers are exact in every frame (D1); only the bars move", () => {
  const notReduced = (q: string) => ({ matches: false, media: q, onchange: null, addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {}, dispatchEvent: () => false });
  beforeEach(() => { FakeES.all = []; vi.stubGlobal("EventSource", FakeES); });

  it("a spend event shows the payload value at once and at every animation frame after it", async () => {
    vi.stubGlobal("matchMedia", notReduced); // motion is allowed: a count-up would run here
    const frames: FrameRequestCallback[] = [];
    vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => frames.push(cb));
    vi.stubGlobal("cancelAnimationFrame", () => undefined);
    const start = { ...contract.run, status: "running", spend: { total_usd: 2.5, cap_usd: 10, finetune_usd_estimate: 1, by_model: { "m-1": { usd: 0.5, calls: 1, input_tokens: 100, output_tokens: 10 } } } };
    stubFetch(contractFetch({ run: start }));
    const { container } = renderApp(`/runs/${rid}`);
    await screen.findByRole("meter", { name: "Spend against cap" });
    await waitFor(() => expect(FakeES.all).toHaveLength(1));
    act(() => {
      FakeES.all[0].onopen!();
      FakeES.all[0].emit("spend", 1, { total_usd: 5, cap_usd: 10, finetune_usd_estimate: 3, by_model: { "m-1": { usd: 4, calls: 3, input_tokens: 300, output_tokens: 30 } } });
    });
    const seen = () => ["$5.0000", "$4.0000", "$3.0000"].filter((t) => container.textContent!.includes(t));
    expect(seen()).toEqual(["$5.0000", "$4.0000", "$3.0000"]); // exact final values before any frame runs
    for (const t of [0, 200, 400, 800, 1600]) {
      act(() => { frames.splice(0).forEach((cb) => cb(t)); });
      expect(seen()).toEqual(["$5.0000", "$4.0000", "$3.0000"]);
      // and no in-between number was ever put on screen: the old values are gone, nothing new appeared
      expect(container.textContent).not.toMatch(/\$(2\.5000|0\.5000|1\.0000)/);
    }
  });
});

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

describe("spend labels", () => {
  it("labels the total an estimate and shows fine-tune billed and ceiling apart", async () => {
    const { SpendPanel } = await import("./SpendPanel");
    const { render } = await import("@testing-library/react");
    const sp = { total_usd: 1, cap_usd: 5, by_model: {}, finetune_usd_estimate: 5, finetune_billed_usd: 2, finetune_ceiling_usd: 3 };
    const { container, rerender } = render(<SpendPanel spend={sp} title="Spend" />);
    const t = () => container.textContent!;
    expect(t()).toContain("Total (estimate)");
    expect(t()).toContain("Fine-tune (measured x price)$2.0000");
    expect(t()).toContain("Fine-tune (ceiling)$3.0000");
    rerender(<SpendPanel spend={{ ...sp, basis: "billed-basis (measured)" }} title="Spend" />);
    expect(t()).not.toContain("Total (estimate)");
  });
});
