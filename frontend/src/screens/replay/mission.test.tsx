import { cleanup, screen, waitFor, within } from "@testing-library/react";
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
  it("features the recorded PROMOTE run: chip, one sentence without digits, three numerals, CI line, n, McNemar, evidence link", async () => {
    serve();
    renderApp("/");
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
    expect(h.querySelector("figure.ci-line")).not.toBeNull();
    expect(h.querySelector("figure.ci-line")!.textContent).toContain(String(gate.thresholds.ratio_lower_bound_min));
    expect(h.textContent).toContain(`n = ${fmtInt(gate.n)}`);
    expect(h.textContent).toMatch(/McNemar vs base: p = 0\.021 \(α = 0\.05\)/);
    expect(within(h).getByRole("link", { name: /Open the evidence/ }).getAttribute("href")).toBe("/runs/rec-featured/report");
  });

  it("a McNemar p below what three decimals can show reads 'far below α', never 0.000", async () => {
    const tiny = { ...report, evaluation: { ...report.evaluation, gate: { ...gate, mcnemar_p: 1.6e-57 } } };
    serve({ report: tiny });
    renderApp("/");
    await ready();
    expect(hero().textContent).toContain("McNemar vs base: p far below α (α = 0.05)");
    expect(hero().textContent).not.toContain("0.000");
  });

  it("labels the hero from the featured run's own flags, and features nothing when only dry runs exist", async () => {
    serve({ replay: contract.replay });
    const { container } = renderApp("/");
    await waitFor(() => screen.getByText(/no recorded run to feature/));
    expect(container.querySelector(".m-verdict")).toBeNull();
    expect(container.querySelector(".hero .label-banner")).toBeNull();
    expect(section("Training").textContent).toContain("Shown once a recorded run is available to feature.");
    expect(container.querySelector("main .state:not(.error)")).toBeNull();
  });

  it("a failing report is an error in the hero, and the sections that need it say so instead of looking empty", async () => {
    const ok = contractFetch({ replay });
    stubFetch((u, i) => (u.endsWith("/report") ? json({ error: "report_gone", message: "cannot read report" }, 500) : u.endsWith("/api/evidence") ? json(evidence) : ok(u, i)));
    renderApp("/");
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
    renderApp("/");
    await waitFor(() => screen.getByRole("list", { name: "Proof ladder" }));
    const cards = [...section("Proof ladder").querySelectorAll<HTMLElement>(".proof-card")];
    const proofs = evidence.proofs.slice(0, MAX_PROOFS);
    expect(cards).toHaveLength(proofs.length);
    cards.forEach((c, i) => {
      const p = proofs[i];
      expect(c.querySelector(".proof-title")!.textContent).toBe(p.title);
      expect(c.querySelector(".proof-value")!.textContent).toBe(fmtSig(p.headline.value));
      expect(c.querySelector<HTMLAnchorElement>("a.proof-link")!.getAttribute("href")).toBe(p.evidence_url);
      expect(c.querySelector(".proof-head .badge")!.textContent).toBe(p.verdict);
      if (p.verdict === "PARTIAL") expect(c.classList.contains("is-partial")).toBe(true);
    });
    // the overfit series becomes a sparkline on its card
    const withSeries = proofs.findIndex((p) => (p.series ?? []).length > 0);
    expect(withSeries).toBeGreaterThanOrEqual(0);
    expect(cards[withSeries].querySelector(".spark svg")).not.toBeNull();
  });

  it("proof ladder: an API error is an error, while the hero still shows the verdict", async () => {
    serve({ evidence: json({ error: "evidence_down", message: "no evidence file" }, 503) });
    renderApp("/");
    await ready();
    await waitFor(() => within(section("Proof ladder")).getByRole("alert"));
    expect(section("Proof ladder").textContent).toContain("evidence_down");
    expect(section("Proof ladder").textContent).not.toMatch(/lists no proofs/);
  });

  it("training: loss chart, trained tokens and steps, config chips, and the interpretation caption has no number", async () => {
    serve();
    renderApp("/");
    await ready();
    const s = section("Training");
    const ft = contract.report.finetune[0];
    expect(s.querySelector("figure.loss")).not.toBeNull();
    const chips = within(s).getByRole("list", { name: "Fine-tune record" }).textContent!;
    expect(chips).toContain(`trained tokens ${fmtInt(ft.trained_tokens)}`);
    expect(chips).toContain(`steps ${fmtInt(ft.trained_steps)} / ${fmtInt(ft.total_steps)}`);
    expect(chips).toContain("learning rate 0.0001"); // small values never read as 0.000
    const interp = s.querySelector(".m-interp")!.textContent!;
    expect(interp).toMatch(/^Interpretation/);
    expect(interp).toContain("execution accuracy");
    expect(interp).not.toMatch(/\d/);
  });

  it("limits & cost: stress bars for unseen families, overlap rate, run total and fine-tune lines with a derived per-million price", async () => {
    serve();
    renderApp("/");
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

  it("clutter budget: four content sections between the hero and the calls to action, three charts in all", async () => {
    serve();
    const { container } = renderApp("/");
    await ready();
    const h2 = [...container.querySelectorAll("main h2")].map((h) => h.textContent);
    expect(h2.slice(0, 5)).toEqual(["Proof ladder", "Training", "How it works", "Limits & cost", "Try it"]);
    expect(container.querySelectorAll("figure.chart")).toHaveLength(3);
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
    const { container } = renderApp("/");
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
