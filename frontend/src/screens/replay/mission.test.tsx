import { cleanup, fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { fmtInt, fmtPercent } from "../../api/format";
import { contract, contractFetch, type Overrides } from "../../testutil/contract";
import evidence from "../../testutil/contract/evidence.json";
import { unexplained, visibleText, type Allow } from "../../testutil/provenance";
import { json, renderApp, stubFetch } from "../../testutil/render";
import { fmtSig, MAX_PROOFS, perMillion } from "./mission";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

const recorded = { dry_run: false, recorded: true, recorded_at: "2026-10-04T15:51:34Z" };
const featuredRun = { ...contract.replay[0], ...recorded, run_id: "rec-featured", decision: "PROMOTE" };
const report = { ...contract.report, ...recorded, run_id: "rec-featured" };
const replay = [contract.replay[0], featuredRun];
const gate = contract.report.evaluation.gate;

/** Contract fixtures plus /api/evidence; `over.evidence` replaces it (a Response for an error). */
function serve(over: Overrides & { evidence?: unknown } = {}) {
  const { evidence: ev = evidence, ...rest } = over;
  const base = contractFetch({ replay, report, ...rest });
  return stubFetch((u, i) => {
    if (u.endsWith("/api/evidence")) return ev instanceof Response ? ev : json(ev);
    return base(u, i);
  });
}
const ready = () => waitFor(() => expect(document.querySelector(".hero .m-verdict")).not.toBeNull(), { timeout: 3000 });
const hero = () => document.querySelector<HTMLElement>(".hero")!;
const section = (name: string) => screen.getByRole("heading", { level: 2, name }).closest("section")!;

describe("Mission hero verdict (featured run)", () => {
  it("features the recorded PROMOTE run: chip, one sentence without digits, three numerals, and only the evidence button on the right", async () => {
    serve();
    renderApp("/mission");
    await ready();
    const h = hero();
    expect(within(h).getAllByRole("heading", { level: 1 })).toHaveLength(1);
    expect(h.querySelector(".m-chip")!.textContent).toBe("PROMOTE");
    expect(h.querySelector(".m-sentence")!.textContent).not.toMatch(/\d/);
    const nums = [...h.querySelectorAll(".m-nums .m-num")].map((n) => [n.querySelector(".l")!.textContent, n.querySelector(".v")!.textContent]);
    expect(nums).toEqual([
      ["Student", fmtPercent(gate.student_acc, 1)],
      ["Teacher", fmtPercent(gate.teacher_acc, 1)],
      ["Base", fmtPercent(gate.base_acc, 1)],
    ]);
    // the threshold chart (the gate-minimum line) and the facts under it are gone; the right column is the one button
    expect(h.querySelector("figure.ci-line")).toBeNull();
    expect(h.textContent).not.toContain("gate minimum");
    const right = h.querySelector(".m-right")!;
    expect([...right.children]).toHaveLength(1);
    const cta = within(h).getByRole("link", { name: /Open the evidence/ });
    expect(cta.className).toContain("m-cta");
    expect(cta.getAttribute("href")).toBe("/runs/rec-featured/report");
  });

  it("labels the hero from the featured run's own flags, and features nothing when only dry runs exist", async () => {
    serve({ replay: contract.replay });
    const { container } = renderApp("/mission");
    await waitFor(() => screen.getByText(/no recorded run to feature/));
    expect(container.querySelector(".m-verdict")).toBeNull();
    expect(container.querySelector(".hero .label-banner")).toBeNull();
    expect(section("Training").textContent).toContain("Shown once a recorded run is available to feature.");
    expect(container.querySelector("main .state:not(.error)")).toBeNull();
  });

  it("a failing report is an error in the hero, and the sections that need it say so instead of looking empty", async () => {
    const ok = contractFetch({ replay });
    stubFetch((u, i) => (u.endsWith("/report") ? json({ error: "report_gone", message: "cannot read report" }, 500) : u.endsWith("/api/evidence") ? json(evidence) : ok(u, i)));
    renderApp("/mission");
    await waitFor(() => within(hero()).getByRole("alert"));
    expect(hero().textContent).toContain("report_gone");
    expect(hero().textContent).toContain("cannot read report");
    expect(section("Training").textContent).toMatch(/Unavailable: the featured run.s report did not load/);
    expect(section("Limits & cost").textContent).toMatch(/Unavailable/);
  });
});

describe("Mission sections", () => {
  it("proof ladder: at most six cards, each with its verdict, headline and a link to its evidence file", async () => {
    serve();
    renderApp("/mission");
    await waitFor(() => screen.getByRole("list", { name: "Proof ladder" }));
    const cards = [...section("Proof ladder").querySelectorAll<HTMLElement>(".proof-card")];
    const proofs = evidence.proofs.slice(0, MAX_PROOFS);
    expect(cards).toHaveLength(proofs.length);
    cards.forEach((c, i) => {
      const p = proofs[i];
      expect(c.querySelector(".proof-title")!.textContent).toBe(p.title);
      expect(c.querySelector(".proof-value")!.textContent).toBe(fmtSig(p.headline.value));
      // Evidence opens on this page (a button), not another one (no link), and the internal id is not shown
      expect(c.querySelector("button.proof-link")!.textContent).toMatch(/^Evidence/);
      expect(c.querySelector("a.proof-link")).toBeNull();
      expect(c.querySelector(".proof-rung")).toBeNull();
      expect(c.textContent).not.toContain(p.id);
      expect(c.querySelector(".proof-head .badge")!.textContent).toBe(p.verdict);
      if (p.verdict === "PARTIAL") expect(c.classList.contains("is-partial")).toBe(true);
    });
    // the overfit series becomes a sparkline on its card
    const withSeries = proofs.findIndex((p) => (p.series ?? []).length > 0);
    expect(withSeries).toBeGreaterThanOrEqual(0);
    expect(cards[withSeries].querySelector(".spark svg")).not.toBeNull();
  });

  it("proof ladder: Evidence opens the record as JSON in a dialog on the same page, and closes again", async () => {
    serve();
    renderApp("/mission");
    await waitFor(() => screen.getByRole("list", { name: "Proof ladder" }));
    const p = evidence.proofs[0];
    const receipt = evidence.receipts.find((r) => r.id === p.id)!;
    fireEvent.click(section("Proof ladder").querySelector<HTMLButtonElement>(".proof-card button.proof-link")!);
    const dlg = await screen.findByRole("dialog");
    const view = within(dlg).getByLabelText("Evidence record (JSON)");
    // the text is exactly the payload's proof entry and receipt, pretty-printed
    expect(JSON.parse(view.textContent!)).toEqual({ proof: p, receipt });
    expect(within(dlg).getByRole("link", { name: new RegExp(p.evidence_path) }).getAttribute("href")).toBe(p.evidence_url);
    expect(window.location.pathname).toBe("/"); // the test router is in memory: nothing navigated away
    fireEvent.keyDown(dlg, { key: "Escape" });
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
  });

  it("proof ladder: an API error is an error, while the hero still shows the verdict", async () => {
    serve({ evidence: json({ error: "evidence_down", message: "no evidence file" }, 503) });
    renderApp("/mission");
    await ready();
    await waitFor(() => within(section("Proof ladder")).getByRole("alert"));
    expect(section("Proof ladder").textContent).toContain("evidence_down");
    expect(section("Proof ladder").textContent).not.toMatch(/lists no proofs/);
  });

  it("training: the loss chart and a green DATA menu with everything else; no interpretation paragraph", async () => {
    serve();
    renderApp("/mission");
    await ready();
    const s = section("Training");
    const ft = contract.report.finetune[0];
    expect(s.querySelector("figure.loss")).not.toBeNull();
    expect(s.querySelector(".m-interp")).toBeNull();
    expect(s.textContent).not.toContain("Interpretation");
    const btn = within(s).getByRole("button", { name: /^Data/ });
    expect(btn.className).toContain("ok"); // green, like PROMOTE
    expect(btn.getAttribute("aria-expanded")).toBe("false");
    expect(within(s).queryByRole("list", { name: "Fine-tune record" })).toBeNull(); // closed: not in the accessibility tree
    fireEvent.click(btn);
    expect(btn.getAttribute("aria-expanded")).toBe("true");
    const list = within(s).getByRole("list", { name: "Fine-tune record" });
    const text = list.textContent!;
    expect(text).toContain(`trained tokens ${fmtInt(ft.trained_tokens)}`);
    expect(text).toContain(`steps ${fmtInt(ft.trained_steps)} / ${fmtInt(ft.total_steps)}`);
    expect(text).toContain("learning rate 0.0001"); // small values never read as 0.000
    expect(text).toContain("dev accuracy");
    expect(text).toContain("sealed held-out accuracy");
    // the last checkpoint's own losses
    const last = ft.loss_curve[ft.loss_curve.length - 1];
    expect(text).toContain(`train loss ${fmtSig(last.train_loss)}`);
    expect(text).toContain(`validation loss ${fmtSig(last.valid_loss)}`);
  });

  it("training: clicking train loss or validation loss opens a short definition beside it, and clicking again closes it", async () => {
    serve();
    renderApp("/mission");
    await ready();
    const s = section("Training");
    fireEvent.click(within(s).getByRole("button", { name: /^Data/ }));
    const train = within(s).getByRole("button", { name: /train loss/ });
    expect(within(s).queryByRole("note")).toBeNull();
    fireEvent.click(train);
    const note = within(s).getByRole("note");
    expect(note.textContent).toMatch(/practice questions/);
    expect(note.className).toContain("right");
    expect(train.getAttribute("aria-expanded")).toBe("true");
    // only one at a time
    fireEvent.click(within(s).getByRole("button", { name: /validation loss/ }));
    expect(within(s).getAllByRole("note")).toHaveLength(1);
    expect(within(s).getByRole("note").textContent).toMatch(/kept out of training/);
    fireEvent.click(within(s).getByRole("button", { name: /validation loss/ }));
    expect(within(s).queryByRole("note")).toBeNull();
  });

  it("limits & cost: stress bars for unseen families, overlap rate, run total and fine-tune lines with a derived per-million price", async () => {
    serve();
    renderApp("/mission");
    await ready();
    const s = section("Limits & cost");
    expect(s.querySelector("figure.stress-bars")).not.toBeNull();
    for (const f of contract.report.evaluation.stress.families) expect(s.textContent).toContain(f);
    expect(s.querySelector(".m-overlap")!.textContent).toContain(fmtPercent(contract.report.data.heldout_skeleton_overlap_rate, 1));
    expect(s.querySelector(".m-total .v")!.textContent).toBe(`$${contract.report.cost.run_total_usd.toFixed(2)}`);
    const lines = within(s).getByRole("list", { name: "Fine-tune cost" }).querySelectorAll("li");
    expect(lines).toHaveLength(contract.report.cost.finetune_lines.length);
    expect(lines[0].textContent).toContain("per million tokens (derived)");
    expect(s.querySelector("details.m-basis")!.textContent).toContain(contract.report.cost.basis);
  });

  it("clutter budget: three content sections between the hero and the calls to action (How it works has its own page), two charts in all (the threshold chart is gone)", async () => {
    serve();
    const { container } = renderApp("/mission");
    await ready();
    const h2 = [...container.querySelectorAll("main h2")].map((h) => h.textContent);
    expect(h2.slice(0, 4)).toEqual(["Proof ladder", "Training", "Limits & cost", "Try it"]);
    expect(container.querySelectorAll("figure.chart")).toHaveLength(2);
    const ctas = section("Try it");
    expect(within(ctas).getByRole("button", { name: /Watch a live demo run/ })).toBeTruthy();
    expect(within(ctas).getByRole("link", { name: "Start a live run" }).getAttribute("href")).toBe("/new");
    expect(within(ctas).getByRole("link", { name: /GitHub/ }).getAttribute("href")).toMatch(/^https:\/\/github\.com\//);
  });
});

describe("Mission provenance: every number on the page comes from a payload", () => {
  const DERIVED: Allow = {
    pattern: /\$\d+\.\d+ per million tokens \(derived\)/g,
    why: "fine-tune price per million tokens, derived on screen as usd / units x 1e6 from one cost.finetune_lines entry and labelled 'derived'",
  };
  it("with a featured run, the hero, proof ladder, training and limits numbers are all explained", async () => {
    serve();
    const { container } = renderApp("/mission");
    await ready();
    await waitFor(() => screen.getByRole("list", { name: "Proof ladder" }));
    await new Promise((r) => setTimeout(r, 50));
    const text = visibleText(container);
    expect(unexplained(text, [replay, report, evidence, contract.runs], [DERIVED])).toEqual([]);
  });
  it("perMillion is not measured for absent or zero token counts", () => {
    expect(perMillion({ usd: 1, units: 0, basis: null, model: null, kind: null })).toBe("not measured");
    expect(perMillion({ usd: null, units: 5, basis: null, model: null, kind: null })).toBe("not measured");
    expect(perMillion({ usd: 2.337012, units: 5842530, basis: null, model: null, kind: null })).toBe("$0.40");
  });
  it("fmtSig keeps small values readable and passes strings through", () => {
    expect(fmtSig(0.000343)).toBe("0.000343");
    expect(fmtSig(0.0002)).toBe("0.0002");
    expect(fmtSig(320)).toBe("320");
    expect(fmtSig(0.5)).toBe("0.5");
    expect(fmtSig("64/64")).toBe("64/64");
    expect(fmtSig(null)).toBe("not measured");
  });
});
