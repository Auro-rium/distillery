import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import fixture from "../../../../docs/fixtures/sample-dry-run-report.json";
import type { Report } from "../../api/types";
import { contract, contractFetch } from "../../testutil/contract";
import { json, renderApp, stubFetch } from "../../testutil/render";
import { unexplained, visibleText, type Allow } from "../../testutil/provenance";
import { ReportView } from ".";

const base = { ...(fixture as unknown as Report), dry_run: true, recorded: false, recorded_at: null };
const view = (r: Report) => render(<MemoryRouter><ReportView r={r} /></MemoryRouter>);
beforeEach(() => stubFetch(() => json([])));
afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });

describe("summary", () => {
  it("shows the decision, the label, n and the thresholds the gate used, all from the payload", () => {
    view(base);
    const s = screen.getByRole("region", { name: "Gate decision" });
    expect(within(s).getByText(base.decision)).toBeTruthy();
    expect(within(s).getByText("dry run")).toBeTruthy();
    const g = base.evaluation.gate;
    expect(s.textContent).toContain(`n=${g.n}`);
    expect(s.textContent).toContain(g.ratio_lo.toFixed(3));
    expect(s.textContent).toContain(g.thresholds.ratio_lower_bound_min.toFixed(2));
    expect(s.textContent).toContain(g.thresholds.mcnemar_alpha.toFixed(2));
    expect(s.textContent).toContain(g.mcnemar_p.toFixed(4));
  });

  it("keeps the gate reasons collapsed until asked, then shows them verbatim", () => {
    view(base);
    for (const r of base.decision_reasons) expect(screen.queryByText(r)).toBeNull();
    const toggle = screen.getByRole("button", { name: "Gate reasons" });
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    fireEvent.click(toggle);
    expect(toggle.getAttribute("aria-expanded")).toBe("true");
    for (const r of base.decision_reasons) expect(screen.getByText(r)).toBeTruthy();
    expect(document.getElementById(toggle.getAttribute("aria-controls")!)).not.toBeNull();
    fireEvent.click(toggle);
    for (const r of base.decision_reasons) expect(screen.queryByText(r)).toBeNull();
  });

  it("says so when the gate reported no reasons, and offers no toggle", () => {
    view({ ...base, decision_reasons: [] });
    expect(screen.getByText("No reasons were reported by the gate.")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Gate reasons" })).toBeNull();
  });

  it("labels the numbers with the report's own flags: recorded, and unknown when the flag is missing", () => {
    const rec = view({ ...base, dry_run: false, recorded: true, recorded_at: "2026-01-02T03:04:05Z" });
    expect(screen.getByText("recorded")).toBeTruthy();
    rec.unmount();
    const bare = { ...base } as Partial<Report>;
    delete bare.dry_run; delete bare.recorded;
    view(bare as Report);
    expect(screen.getByText("label unknown")).toBeTruthy();
    expect(screen.queryByText("recorded")).toBeNull();
    expect(screen.queryByText("dry run")).toBeNull();
  });
});

describe("accuracy", () => {
  it("shows every model's exact value and the CI only where the report has one", () => {
    const { container } = view(base);
    const main = container.querySelector(".acc")!;
    const acc = base.evaluation.accuracy;
    for (const k of ["base", "student", "teacher"] as const)
      expect(main.querySelector(`.acc-row.${k}`)!.textContent).toContain(`${(acc[k] * 100).toFixed(1)}%`);
    expect(main.querySelector(".acc-row.student")!.textContent).toContain(`CI ${(base.evaluation.gate.student_ci![0] * 100).toFixed(1)}% to ${(base.evaluation.gate.student_ci![1] * 100).toFixed(1)}%`);
    expect(main.querySelectorAll(".acc-ci")).toHaveLength(1);
    expect(main.textContent).toContain("no CI in report");
    expect(screen.getAllByRole("list", { name: "Chart key" })).toHaveLength(base.evaluation.stress ? 3 : 2); // headline, class split, stress
  });

  it("splits by held-out class as small multiples with their own n, marking classes the student did not train on", () => {
    const { container } = view(base);
    const classes = Object.keys(base.evaluation.accuracy_by_class);
    expect(classes.length).toBeGreaterThan(0);
    for (const c of classes) {
      const panel = screen.getByRole("group", { name: c });
      expect(within(panel).getByText(`n=${base.evaluation.class_counts[c]}`)).toBeTruthy();
      expect(panel.querySelectorAll(".acc-row")).toHaveLength(3);
    }
    expect(screen.queryAllByText("not in training").length).toBe(classes.filter((c) => c.startsWith("unseen")).length);
    expect(container.querySelectorAll(".acc-row")).toHaveLength(3 + 3 * classes.length + (base.evaluation.stress ? 3 : 0));
    // whiskers exist only on the headline chart: the report has no per-class CI
    expect(container.querySelectorAll(".acc-ci")).toHaveLength(1);
  });

  it("offers a table twin whose cells repeat the drawn values", () => {
    view(base);
    const tables = screen.getAllByRole("table", { hidden: true });
    const text = tables.map((t) => t.textContent).join(" ");
    expect(text).toContain(`${(Object.values(base.evaluation.accuracy_by_class)[0].base * 100).toFixed(1)}%`);
    expect(text).toContain(`${(base.evaluation.gate.student_ci![0] * 100).toFixed(1)}% to ${(base.evaluation.gate.student_ci![1] * 100).toFixed(1)}%`);
    for (const c of Object.keys(base.evaluation.accuracy_by_class)) expect(text).toContain(c);
  });

  it("reports unparseable outputs per model", () => {
    view(base);
    const u = base.evaluation.unparseable;
    expect(screen.getByText(`Unparseable outputs: base ${u.base}, student ${u.student}, teacher ${u.teacher}`)).toBeTruthy();
  });
});

describe("cost and latency", () => {
  it("shows the backend's own unavailable text verbatim and marks latency not measured", () => {
    view(base);
    expect(screen.getByText(base.cost.cost_per_1k_tasks.student as string)).toBeTruthy();
    expect(screen.getByText("the report has no latency data")).toBeTruthy();
    expect(screen.getByText(`$${(base.cost.cost_per_1k_tasks.teacher as { usd_per_1k_tasks: number }).usd_per_1k_tasks.toFixed(5)}`)).toBeTruthy();
    expect(screen.getAllByText("not measured").length).toBeGreaterThan(1);
  });
  it("lists spend per model with the payload's calls, tokens and dollars", () => {
    view(base);
    const t = screen.getByRole("table", { name: "Spend by model" });
    for (const [model, m] of Object.entries(base.cost.llm_by_model)) {
      const row = within(t).getByRole("row", { name: new RegExp(model) });
      expect(row.textContent).toContain(String(m.calls));
      expect(row.textContent).toContain(`$${m.usd.toFixed(4)}`);
    }
  });
});

describe("counters", () => {
  it("groups drop and error counters by stage, and lists retries per purpose", () => {
    view(base);
    const gc = screen.getByRole("table", { name: "gold_crosscheck" });
    expect(within(gc).getByText("discarded")).toBeTruthy();
    const attempts = screen.getByRole("table", { name: "LLM attempts by purpose" });
    // The purpose is split into inline parts so it can wrap after "_"; jsdom joins parts of a name with a space, browsers do not.
    const row = within(attempts).getByRole("row", { name: /eval_\s?teacher/ });
    expect(row.textContent).toContain(String(base.llm_attempt_counters["eval_teacher:schema_retries"]));
    expect(screen.getByText(/LLM errors by purpose: none recorded/)).toBeTruthy();
  });
  it("says so when a report has no counters or clusters or class split", () => {
    view({ ...base, counters: {}, llm_attempt_counters: {}, rounds: [], evaluation: { ...base.evaluation, accuracy_by_class: {} } });
    expect(screen.getByText("No drop or error counters in this report")).toBeTruthy();
    expect(screen.getByText("No failure clusters in this report")).toBeTruthy();
    expect(screen.getByText("No class split in this report")).toBeTruthy();
  });
});

describe("failure clusters", () => {
  it("groups clusters by round with their families, and marks the selected candidate", () => {
    view(base);
    const round = base.rounds.find((r) => r.clusters.length > 0)!;
    const c = round.clusters[0];
    expect(screen.getByText(c.name)).toBeTruthy();
    expect(screen.getByText(c.description)).toBeTruthy();
    expect(screen.getAllByText(c.target_families[0]).length).toBeGreaterThan(0);
    expect(screen.getByText(base.rounds_stop_reason)).toBeTruthy();
  });
});

describe("ReportScreen inside the run shell", () => {
  const id = contract.run.run_id;
  it("wraps the report in the run shell on the Report tab", async () => {
    stubFetch(contractFetch());
    renderApp(`/runs/${id}/report`);
    await screen.findByRole("region", { name: "Gate decision" });
    const tab = screen.getByRole("tab", { name: "Report" });
    expect(tab.getAttribute("data-state")).toBe("active");
    expect(screen.getAllByText("DRY RUN — fake models, numbers are NOT results").length).toBe(1);
  });
  it("still labels the numbers when the run detail cannot be loaded", async () => {
    const ok = contractFetch();
    stubFetch((u, i) => (/\/runs\/[^/]+$/.test(u) ? json({ error: "gone", message: "no run" }, 404) : ok(u, i)));
    renderApp(`/runs/${id}/report`);
    await screen.findByRole("region", { name: "Gate decision" });
    await waitFor(() => expect(screen.getAllByText("DRY RUN — fake models, numbers are NOT results").length).toBe(1));
  });
  it("says the report is not ready, distinct from an error", async () => {
    stubFetch((u) => (u.endsWith("/report") ? json({ error: "report_not_ready", message: "still running" }, 404) : json(contract.run)));
    renderApp(`/runs/${id}/report`);
    await screen.findByText("Report not ready");
    expect(screen.getByRole("link", { name: "Watch it live" })).toBeTruthy();
    expect(screen.queryByRole("alert")).toBeNull();
  });
  it("puts the summary just below the sticky run header, using its measured height", async () => {
    stubFetch(contractFetch());
    const real = window.getComputedStyle.bind(window);
    vi.spyOn(window, "getComputedStyle").mockImplementation((el: Element, pseudo?: string | null) => {
      const cs = real(el, pseudo);
      return el.classList?.contains("run-sticky") ? new Proxy(cs, { get: (t, k) => (k === "position" ? "sticky" : Reflect.get(t, k)) }) : cs;
    });
    const { container } = renderApp(`/runs/${id}/report`);
    const summary = await screen.findByRole("region", { name: "Gate decision" });
    Object.defineProperty(container.querySelector(".run-sticky")!, "offsetHeight", { configurable: true, value: 123 });
    Object.defineProperty(summary, "offsetHeight", { configurable: true, value: 45 });
    window.dispatchEvent(new Event("resize"));
    const root = container.querySelector<HTMLElement>(".rp")!;
    await waitFor(() => expect(root.style.getPropertyValue("--rp-top")).toBe("123px"));
    expect(root.style.getPropertyValue("--rp-sum")).toBe("45px");
  });
});

describe("provenance with everything loaded", () => {
  const AXIS: Allow = { pattern: /(?<![\d.])(0|50|100)%(?=\s|$)/g, why: "fixed axis tick labels (0, 50, 100 percent)" };
  const CARD: Allow = { pattern: /\bcost \/ 1k tasks/g, why: "'1k' is part of the metric name 'cost per 1k tasks'" };
  const SHOWN: Allow = { pattern: /(capped list, \d+ shown|showing \d+ of \d+)/g, why: "count of items in a returned list" };
  it("every number on the page, examples tabs included, comes from the report or the examples response", async () => {
    stubFetch(contractFetch());
    const { container } = renderApp(`/runs/${contract.run.run_id}/report`);
    await screen.findByRole("tab", { name: "Fixed" }, { timeout: 4000 });
    fireEvent.mouseDown(screen.getByRole("tab", { name: "Still wrong" }), { button: 0 });
    fireEvent.click(screen.getByRole("button", { name: "Gate reasons" }));
    const text = visibleText(container);
    expect(text).toContain("showing");
    expect(unexplained(text, [contract.report, contract.examples, contract.examplesHeaders], [AXIS, CARD, SHOWN])).toEqual([]);
  });
});

describe("stress set and in-distribution caveat", () => {
  const withNew = contract.report as unknown as Report;
  const oldRun = (() => {
    const o = JSON.parse(JSON.stringify(base)) as Report;
    delete o.evaluation.stress;
    for (const k of ["heldout_skeleton_overlap_rate", "train_distinct_skeletons", "stress_tasks"] as const) delete o.data[k];
    // a recorded run from before the benchmark rebuild: family-holdout classes
    const acc = Object.values(o.evaluation.accuracy_by_class)[0];
    o.evaluation.accuracy_by_class = { seen: acc, unseen_family: acc };
    o.evaluation.class_counts = { seen: 10, unseen_family: 10 };
    return o;
  })();

  it("shows the stress section with captions, per-model and per-family accuracy", () => {
    view(withNew);
    const card = screen.getByText("Stress set").closest("section, div.card, article") as HTMLElement;
    expect(card.textContent).toContain("Reserved families, never in train/dev/gate, NOT a gate input");
    const s = withNew.evaluation.stress!;
    expect(card.textContent).toContain(`n=${s.n}`);
    for (const f of Object.keys(s.accuracy_by_family)) expect(card.textContent).toContain(f);
    expect(card.querySelectorAll("table tbody tr")).toHaveLength(1 + Object.keys(s.accuracy_by_family).length);
  });

  it("renders nothing for the stress set when it is null or missing", () => {
    view({ ...withNew, evaluation: { ...withNew.evaluation, stress: null } });
    expect(screen.queryByText("Stress set")).toBeNull();
    cleanup();
    view(oldRun);
    expect(screen.queryByText("Stress set")).toBeNull();
  });

  it("shows the caveat with the percentage and skeleton count from the payload", () => {
    view(withNew);
    const note = screen.getByRole("note");
    const d = withNew.data;
    expect(note.textContent).toContain("Gate set is in-distribution");
    expect(note.textContent).toContain(`${(d.heldout_skeleton_overlap_rate! * 100).toFixed(0)}% of gate questions`);
    expect(note.textContent).toContain(`${d.train_distinct_skeletons} distinct training skeletons`);
    expect(note.textContent).toContain("not novel query structure");
  });

  it("omits the caveat for an old run and still renders its unseen classes", () => {
    view(oldRun);
    expect(screen.queryByText(/in-distribution/)).toBeNull();
    expect(screen.getAllByText("not in training").length).toBeGreaterThan(0);
  });
});
