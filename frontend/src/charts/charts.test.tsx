// Every chart draws only the numbers it is given: rendered text is checked with the same provenance
// checker the screens use, against the props as the only payload. Plus a11y names and absent-data cases.
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { unexplained, visibleText } from "../testutil/provenance";
import {
  AccuracyBars, CiNumberLine, DiscordantMatrix, JobLedger, LossChart, ProofCard, Sparkline, StageRail, StressBars,
} from ".";

afterEach(cleanup);

/** Numbers on screen that the given props do not explain (empty = nothing invented). */
const invented = (container: HTMLElement, props: unknown) => unexplained(visibleText(container), [props]);

describe("CiNumberLine", () => {
  const p = { point: 1.027, lo: 0.993, hi: 1.061, threshold: 0.95, title: "Student / teacher ratio", caption: "Ratio of accuracies with its interval." };
  it("labels point, interval and threshold from props, and nothing else", () => {
    const { container } = render(<CiNumberLine {...p} />);
    const text = visibleText(container);
    for (const v of ["1.027", "0.993", "1.061", "0.950"]) expect(text).toContain(v);
    expect(invented(container, p)).toEqual([]);
    expect(screen.getByRole("figure", { name: p.title })).toBeTruthy();
    expect(screen.getByText(p.caption).tagName).toBe("FIGCAPTION");
  });
  it("colours pass when the lower bound clears the threshold and fail when it does not", () => {
    const { container, rerender } = render(<CiNumberLine {...p} />);
    expect(container.querySelector("figure")!.className).toContain("is-pass");
    rerender(<CiNumberLine {...p} lo={0.9} />);
    expect(container.querySelector("figure")!.className).toContain("is-fail");
    rerender(<CiNumberLine {...p} lo={0.9} pass />); // the API's verdict wins when given
    expect(container.querySelector("figure")!.className).toContain("is-pass");
  });
  it("the point is focusable and named with all four values", () => {
    render(<CiNumberLine {...p} />);
    const dot = screen.getByRole("img", { name: /point 1\.027, interval 0\.993 to 1\.061, threshold 0\.950/ });
    expect(dot.getAttribute("tabindex")).toBe("0");
  });
  it("absent values read 'not measured' and draw no interval", () => {
    const { container } = render(<CiNumberLine point={null} lo={null} hi={null} threshold={null} title="Ratio" />);
    expect(container.textContent).toContain("not measured");
    expect(container.querySelector(".ci-bar")).toBeNull();
  });
});

describe("LossChart", () => {
  const points = [
    { step: 10, train_loss: 0.094, valid_loss: 0.353 },
    { step: 20, train_loss: 0.061, valid_loss: 0.471 },
    { step: 30, train_loss: 0.042, valid_loss: 0.55 },
  ];
  const props = { points, title: "Fine-tune loss", caption: "Token-level loss per checkpoint." };
  it("handles three checkpoints: a marker per value, straight segments, labels only from props", () => {
    const { container } = render(<LossChart {...props} />);
    expect(container.querySelectorAll(".marker")).toHaveLength(6);
    // no smoothing: polylines (straight segments), no curved paths
    expect(container.querySelectorAll("polyline.line")).toHaveLength(2);
    expect(container.querySelector("path")).toBeNull();
    const text = visibleText(container);
    for (const s of ["10", "20", "30", "0.042", "0.550"]) expect(text).toContain(s);
    expect(invented(container, props)).toEqual([]);
  });
  it("every checkpoint is inspectable by keyboard with its exact value", () => {
    render(<LossChart {...props} />);
    const m = screen.getByRole("img", { name: "Validation loss at step 20: 0.471" });
    expect(m.getAttribute("tabindex")).toBe("0");
    expect(screen.getByRole("img", { name: "Train loss at step 30: 0.042" })).toBeTruthy();
  });
  it("shows the hover tooltip with the same text on focus", async () => {
    render(<LossChart {...props} />);
    fireEvent.focus(screen.getByRole("img", { name: "Train loss at step 10: 0.094" }));
    expect((await screen.findByRole("tooltip")).textContent).toBe("Train loss at step 10: 0.094");
  });
  it("breaks the line at a missing value instead of drawing it as zero", () => {
    const gap = [points[0], { step: 20, train_loss: null, valid_loss: 0.471 }, points[2]];
    const { container } = render(<LossChart points={gap} title="Loss" />);
    expect(container.querySelectorAll(".s-train polyline")).toHaveLength(2);
    expect(container.querySelectorAll(".s-train .marker")).toHaveLength(2);
    expect(container.querySelectorAll(".s-valid polyline")).toHaveLength(1);
  });
  it("has a legend naming both series and an empty state for no data", () => {
    const { container, rerender } = render(<LossChart {...props} />);
    expect(within(screen.getByRole("list", { name: "Chart key" })).getAllByRole("listitem").map((l) => l.textContent)).toEqual(["Train loss", "Validation loss"]);
    rerender(<LossChart points={[]} title="Loss" />);
    expect(container.textContent).toContain("not measured");
  });
});

describe("Sparkline", () => {
  it("is an image named with its first and last values", () => {
    const values = [2.31, 1.2, null, 0.4, 0.012];
    const { container } = render(<Sparkline values={values} label="Overfit train loss" showLast />);
    expect(screen.getByRole("img", { name: "Overfit train loss: 2.310 to 0.012" })).toBeTruthy();
    expect(container.querySelectorAll("polyline")).toHaveLength(2); // gap at the null
    expect(invented(container, { values })).toEqual([]);
  });
  it("with no values says not measured", () => {
    render(<Sparkline values={[null]} label="Loss" />);
    expect(screen.getByText("Loss: not measured")).toBeTruthy();
  });
});

describe("AccuracyBars", () => {
  const rows = [
    { key: "base" as const, label: "Base", value: 0.257, ci: null },
    { key: "student" as const, label: "Student", value: 0.923, ci: [0.9, 0.944] as [number, number] },
    { key: "teacher" as const, label: "Teacher", value: 0.9, ci: null },
  ];
  it("shows each value as text, CI in the bar's name, colour class per model, no invented ticks", () => {
    const { container } = render(<AccuracyBars rows={rows} title="Held-out accuracy" caption="Execution accuracy on the sealed set." legend />);
    const text = visibleText(container);
    for (const v of ["25.7%", "92.3%", "90.0%"]) expect(text).toContain(v);
    expect(screen.getByRole("img", { name: "Student: 92.3%, CI 90.0% to 94.4%" })).toBeTruthy();
    expect(container.querySelectorAll(".bar.m-student, .bar.m-teacher, .bar.m-base")).toHaveLength(3);
    expect(container.querySelectorAll(".whisker")).toHaveLength(1);
    expect(invented(container, { rows })).toEqual([]);
  });
  it("a missing value is 'not measured' with a zero-width bar", () => {
    const { container } = render(<AccuracyBars rows={[{ key: "teacher", label: "Teacher", value: null }]} title="Acc" />);
    expect(container.textContent).toContain("not measured");
    expect(container.querySelector(".bar-fill")!.getAttribute("width")).toBe("0");
  });
});

describe("StressBars", () => {
  it("one group per family with student and teacher bars, n from props", () => {
    const rows = [
      { name: "window functions", n: 40, values: { student: 0.625, teacher: 0.8 } },
      { name: "recursive CTE", n: 12, values: { student: 0.5, teacher: null } },
    ];
    const { container } = render(<StressBars rows={rows} title="Stress set" caption="Unseen families." />);
    expect(container.querySelectorAll(".bar")).toHaveLength(4);
    expect(screen.getByRole("img", { name: "recursive CTE, Teacher: not measured" })).toBeTruthy();
    expect(visibleText(container)).toContain("n 40");
    expect(invented(container, { rows })).toEqual([]);
  });
});

describe("DiscordantMatrix", () => {
  it("shows the two discordant counts with their meaning", () => {
    const p = { studentOnly: 202, baseOnly: 2, title: "Discordant pairs" };
    const { container } = render(<DiscordantMatrix {...p} />);
    const cells = container.querySelectorAll(".dm-cell");
    expect(cells).toHaveLength(2);
    expect(within(cells[0] as HTMLElement).getByText("Student right, Base wrong")).toBeTruthy();
    expect(within(cells[0] as HTMLElement).getByText("202")).toBeTruthy();
    expect(within(cells[1] as HTMLElement).getByText("2")).toBeTruthy();
    expect(invented(container, p)).toEqual([]);
  });
  it("becomes a 2x2 when concordant counts are given; absent is not measured", () => {
    const { container } = render(<DiscordantMatrix studentOnly={5} baseOnly={null} bothRight={80} bothWrong={3} title="M" />);
    expect(container.querySelectorAll(".dm-cell")).toHaveLength(4);
    expect(container.textContent).toContain("not measured");
  });
});

describe("ProofCard", () => {
  it("shows rung, status chip, the given metric and an evidence link", () => {
    const p = { rung: "W2", title: "Pilot gate", status: "pass" as const, metric: "142/150", metricLabel: "pilot items correct", href: "https://github.com/x/y/blob/abc/e.json" };
    const { container } = render(<ProofCard {...p} />);
    expect(screen.getByRole("heading", { name: "Pilot gate" })).toBeTruthy();
    expect(container.querySelector(".badge.ok")!.textContent).toBe("PASS");
    const a = screen.getByRole("link", { name: /Evidence/ });
    expect(a.getAttribute("href")).toBe(p.href);
    expect(a.getAttribute("target")).toBe("_blank");
    // The rung id ("W2") is a short label, not a measurement; screens allow-list it the same way.
    const allow = [{ pattern: /\bW\d\b/g, why: "proof-ladder rung id" }];
    expect(unexplained(visibleText(container), [{ ...p, n: [142, 150] }], allow)).toEqual([]);
  });
  it("fail and unknown states say so in text, not only colour", () => {
    const { rerender, container } = render(<ProofCard rung="W3" title="T" status="fail" />);
    expect(container.querySelector(".badge")!.textContent).toBe("FAIL");
    rerender(<ProofCard rung="W3" title="T" status="unknown" />);
    expect(container.querySelector(".badge")!.textContent).toBe("No verdict");
    expect(screen.queryByRole("link")).toBeNull();
  });
});

describe("StageRail", () => {
  it("is a labelled ordered list with status text and the caller's elapsed text", () => {
    render(<StageRail live stages={[
      { name: "generate", status: "done", elapsed: "4m 12s" },
      { name: "finetune", status: "running" },
      { name: "evaluate", status: "pending" },
    ]} />);
    const list = screen.getByRole("list", { name: "Pipeline stages" });
    const items = within(list).getAllByRole("listitem");
    expect(items.map((i) => i.querySelector(".rail-status")!.textContent)).toEqual(["done", "running", "pending"]);
    expect(items[1].getAttribute("aria-current")).toBe("step");
    expect(items[0].textContent).toContain("4m 12s");
  });
});

describe("JobLedger", () => {
  it("lists the rows given, toned by kind, with timestamps verbatim", () => {
    const rows = [
      { kind: "finetune_job_started", at: "2026-09-30T10:00:00Z", jobId: "ftjob-abc" },
      { kind: "finetune_job_adopted", at: "2026-09-30T10:20:00Z", jobId: "ftjob-abc", detail: "resumed after restart" },
      { kind: "finetune_job_closed", at: null },
    ];
    const { container } = render(<JobLedger rows={rows} />);
    const items = within(screen.getByRole("list", { name: "Job ledger" })).getAllByRole("listitem");
    expect(items).toHaveLength(3);
    expect(items.map((i) => i.className)).toEqual([
      "ledger-row tone-info", "ledger-row tone-warn", "ledger-row tone-ok",
    ]);
    expect(items[2].textContent).toContain("not measured");
    expect(invented(container, rows)).toEqual([]);
  });
  it("says there are no rows instead of drawing an empty ledger", () => {
    render(<JobLedger rows={[]} />);
    expect(screen.getByText("No job events recorded for this run.")).toBeTruthy();
  });
});
