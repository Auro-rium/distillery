// Layout-independent checks of the Live screen: what is on it and what it says, not where it sits.
import { act, cleanup, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { NOT_MEASURED } from "../../api/format";
import { contract, contractFetch } from "../../testutil/contract";
import { unexplained, visibleText } from "../../testutil/provenance";
import { FakeES, renderApp, stubFetch } from "../../testutil/render";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.useRealTimers(); });
const rid = contract.run.run_id;
const region = (name: string | RegExp) => screen.findByRole("region", { name });

describe("Live screen structure", () => {
  it("sits inside the run shell: one h1 with the run id, the section tabs, the label banner from the flags", async () => {
    stubFetch(contractFetch());
    renderApp(`/runs/${rid}`);
    await region("Spend");
    const h1 = screen.getAllByRole("heading", { level: 1 });
    expect(h1).toHaveLength(1);
    expect(h1[0].textContent).toContain(rid);
    expect(screen.getByRole("tab", { name: "Run" }).getAttribute("aria-selected")).toBe("true");
    expect(screen.getAllByRole("status").map((e) => e.textContent)).toContain("DRY RUN — fake models, numbers are NOT results");
  });

  it("has a named region for every panel", async () => {
    stubFetch(contractFetch());
    renderApp(`/runs/${rid}`);
    for (const name of ["Stages", "Spend", "Sandbox", "Verifier", "Log"]) expect(await region(name)).toBeTruthy();
  });

  it("shows the real stages, spend by model and self-test exactly as the payload has them", async () => {
    stubFetch(contractFetch());
    renderApp(`/runs/${rid}`);
    const spend = await region("Spend");
    for (const t of ["fake-t", "$0.5000", "$2.5000", "$10.0000", "$2.0000"]) expect(spend.textContent).toContain(t);
    const stages = await region("Stages");
    for (const s of contract.run.stages) expect(stages.textContent).toContain(s.name);
    const v = await region("Verifier");
    const st = contract.run.verifier.selftest!;
    for (const t of [String(st.accepted_gold), String(st.rejected_corruptions), String(st.failures)]) expect(v.textContent).toContain(t);
  });

  it("renders null fields as 'not measured' (never 0 or blank): sandbox counters, verifier code", async () => {
    stubFetch(contractFetch());
    renderApp(`/runs/${rid}`);
    const sb = await region("Sandbox");
    expect(sb.textContent!.split(NOT_MEASURED).length - 1).toBe(2); // operations and peak concurrency
    expect(sb.textContent).not.toMatch(/\b0\b/);
    const v = await region("Verifier");
    expect(v.textContent).toMatch(/verifier code[^a-z]*not measured/i);
    expect(within(v).queryByRole("button", { name: /copy/i })).toBeNull(); // no code, nothing to copy
  });

  it("renders a missing self-result, fine-tune estimate and spend as 'not measured'", async () => {
    const r = {
      ...contract.run,
      spend: { ...contract.run.spend, finetune_usd_estimate: null },
      verifier: { ...contract.run.verifier, selftest: null },
    };
    stubFetch(contractFetch({ run: r }));
    renderApp(`/runs/${rid}`);
    expect((await region("Spend")).textContent).toContain(NOT_MEASURED);
    expect((await region("Verifier")).textContent).toContain(NOT_MEASURED);
  });

  it("shows verifier code in a code viewer when the server has it", async () => {
    const r = { ...contract.run, verifier: { ...contract.run.verifier, language: "python", code: "def verify(x):\n    return x" } };
    stubFetch(contractFetch({ run: r }));
    renderApp(`/runs/${rid}`);
    const v = await region("Verifier");
    expect(v.textContent).toContain("def verify(x):");
    expect(within(v).getByRole("button", { name: /copy/i })).toBeTruthy();
  });

  it("an empty log says why: a finished run's log is not streamed, so it is not called 'no lines yet'", async () => {
    stubFetch(contractFetch());
    renderApp(`/runs/${rid}`);
    expect((await region("Log")).textContent).toMatch(/only streamed while a run is active/);
  });

  it("a run that is not running has no cancel form and no live status strip", async () => {
    stubFetch(contractFetch());
    renderApp(`/runs/${rid}`);
    await region("Spend");
    expect(screen.queryByLabelText("Admin token")).toBeNull();
    expect(screen.queryByRole("group", { name: "Run status" })).toBeNull();
    expect(document.querySelector(".hb")).toBeNull();
  });
});

describe("Live screen while the run is running", () => {
  beforeEach(() => { FakeES.all = []; vi.stubGlobal("EventSource", FakeES); });
  const running = {
    ...contract.run, status: "running",
    stages: [
      { name: "schema", status: "done", started_at: null, ended_at: null },
      { name: "gold_crosscheck", status: "running", started_at: "2026-09-30T06:00:00+00:00", ended_at: null },
    ],
  };

  it("an empty log on an active run is just empty so far", async () => {
    stubFetch(contractFetch({ run: running }));
    renderApp(`/runs/${rid}`);
    expect((await region("Log")).textContent).toContain("No log lines yet.");
  });

  it("has a status strip with the connection, the running stage and the total spend, and a labelled cancel form", async () => {
    stubFetch(contractFetch({ run: running }));
    renderApp(`/runs/${rid}`);
    const strip = await screen.findByRole("group", { name: "Run status" });
    expect(strip.querySelector(".hb")).not.toBeNull();
    expect(strip.textContent).toContain("gold_crosscheck");
    expect(strip.textContent).toContain("$2.5000");
    const token = screen.getByLabelText("Admin token") as HTMLInputElement;
    expect(token.type).toBe("password");
    expect(screen.getByRole("button", { name: "Cancel run" })).toBeTruthy();
  });

  it("the strip follows real events: a new running stage replaces the old one", async () => {
    stubFetch(contractFetch({ run: running }));
    renderApp(`/runs/${rid}`);
    const strip = await screen.findByRole("group", { name: "Run status" });
    await waitFor(() => expect(FakeES.all).toHaveLength(1));
    act(() => {
      FakeES.all[0].onopen!();
      FakeES.all[0].emit("stage", 1, { at: null, name: "gold_crosscheck", status: "done" });
      FakeES.all[0].emit("stage", 2, { at: "2026-09-30T06:00:05+00:00", name: "verifier_selftest", status: "running" });
    });
    expect(strip.textContent).toContain("verifier_selftest");
    expect(strip.textContent).not.toContain("gold_crosscheck");
  });

  it("the header badge follows the done event (status comes from the stream, not from a guess)", async () => {
    stubFetch(contractFetch({ run: running }));
    const { container } = renderApp(`/runs/${rid}`);
    await screen.findByRole("group", { name: "Run status" });
    await waitFor(() => expect(FakeES.all).toHaveLength(1));
    expect(container.querySelector('[data-slot="status"]')!.textContent).toBe("running");
    act(() => { FakeES.all[0].onopen!(); FakeES.all[0].emit("done", 1, { status: "failed", decision: null }); });
    expect(container.querySelector('[data-slot="status"]')!.textContent).toBe("failed");
    expect(screen.queryByRole("group", { name: "Run status" })).toBeNull();
    expect(screen.queryByLabelText("Admin token")).toBeNull();
  });

  it("cancelling sends the typed admin token and reports the server's answer", async () => {
    const ok = contractFetch({ run: running });
    const calls: string[] = [];
    stubFetch((u, init) => {
      if (u.endsWith("/cancel")) { calls.push(`${init?.method} ${u}`); return new Response(JSON.stringify({ status: "cancelling" }), { status: 200 }); }
      return ok(u, init);
    });
    renderApp(`/runs/${rid}`);
    fireEvent.change(await screen.findByLabelText("Admin token"), { target: { value: "tok" } });
    fireEvent.click(screen.getByRole("button", { name: "Cancel run" }));
    await waitFor(() => expect(screen.getByText(/Cancel requested \(status: cancelling\)/)).toBeTruthy());
    expect(calls).toEqual([`POST /api/runs/${rid}/cancel`]);
  });

  it("long lists collapse finished groups while the run is active", async () => {
    const stages = [
      "schema", "questions", "gold_crosscheck", "verifier_selftest", "split", "headroom", "teacher_data",
      "finetune_r1", "dev_eval_r1", "analysis_r1", "targeted_r1", "sandbox_branch_r1", "finetune_r2",
    ].map((name, i, a) => ({ name, status: i === a.length - 1 ? "running" : "done", started_at: null, ended_at: null }));
    stubFetch(contractFetch({ run: { ...running, stages } }));
    renderApp(`/runs/${rid}`);
    const setup = await screen.findByRole("button", { name: /^Setup/ });
    expect(setup.getAttribute("aria-expanded")).toBe("false");
    expect(screen.getByRole("button", { name: /^r2/ }).getAttribute("aria-expanded")).toBe("true");
  });
});

describe("Live screen provenance with rounds", () => {
  it("every number on screen is in the payload; the round keys are read from the real stage names", async () => {
    const stages = ["schema", "finetune_r1", "dev_eval_r1", "finetune_r2", "final_eval"].map((name) => ({ name, status: "done", started_at: null, ended_at: "2026-09-30T06:47:09.145236+00:00" }));
    const run = {
      ...contract.run, stages,
      spend: { total_usd: 0.348524, cap_usd: 10, finetune_usd_estimate: 0, by_model: { "fake-a": { usd: 0.136806, calls: 119, input_tokens: 126142, output_tokens: 5332 }, "fake-b": { usd: 0.205925, calls: 181, input_tokens: 189187, output_tokens: 8369 } } },
    };
    stubFetch(contractFetch({ run }));
    const { container } = renderApp(`/runs/${rid}`);
    await region("Spend");
    const text = visibleText(container);
    // "r1" / "r2" group headers: the digits come from the stage names in the payload (finetune_r1 -> r1), not from a computation.
    const ROUND_KEY = { pattern: /\br\d+\b/g, why: "round key read from the real stage names (finetune_r1 -> r1)" };
    expect(unexplained(text, [run], [ROUND_KEY])).toEqual([]);
    expect(text).toContain("r2");
  });
});
