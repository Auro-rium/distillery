import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Report } from "../../api/types";
import { setAdminToken } from "../../api/client";
import { contract } from "../../testutil/contract";
import { json, stubFetch } from "../../testutil/render";
import { ReportView } from "../report";
import Humanset from ".";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); setAdminToken(null); });

const report = contract.report as unknown as Report;

describe("Gate B section of the report", () => {
  it("shows the human decision, n, discard and rejection counts and the bias note from the payload", () => {
    stubFetch(() => json([]));
    render(<MemoryRouter><ReportView r={report} /></MemoryRouter>);
    const card = screen.getByText("Human held-out set (Gate B)").closest("section") as HTMLElement;
    const h = report.evaluation.human!;
    const d = report.data.human!;
    expect(card.textContent).toContain(String(report.decision_human));
    expect(card.textContent).toContain(`Human n${h.n}`);
    expect(card.textContent).toContain(`Rejected by the human${d.counts.rejected}`);
    expect(card.textContent).toContain(h.note!);
    for (const reason of h.gate.reasons) expect(card.textContent).toContain(reason);
  });

  it("renders nothing when the run has no human set", () => {
    stubFetch(() => json([]));
    const none = { ...report, decision_human: null, evaluation: { ...report.evaluation, human: null } };
    render(<MemoryRouter><ReportView r={none as Report} /></MemoryRouter>);
    expect(screen.queryByText("Human held-out set (Gate B)")).toBeNull();
  });
});

describe("admin confirm screen", () => {
  const drafts = {
    set: "ab12cd34", sets: ["ab12cd34"], teacher_model: "fake-teacher", n_questions: 3,
    discarded_by_reason: { llm_error: 0, no_sql: 0, exec_error: 0, empty: 1, disagree: 0 },
    tally: { confirmed: 0, rejected: 0, skipped: 0, undecided: 1 },
    items: [{
      task_id: "h-0-00000000", question: "How many accounts?", gold_sql: "SELECT COUNT(*) FROM accounts",
      requires_order: false, decision: null, preview: { columns: ["n"], rows: [[3]], row_count: 1 },
    }],
  };

  it("sends the admin token, lists the drafts and posts a decision", async () => {
    const calls: { url: string; init?: RequestInit }[] = [];
    stubFetch((url, init) => {
      calls.push({ url, init });
      return json(url.includes("/decide") ? { decision: "confirm", tally: drafts.tally } : drafts);
    });
    render(<MemoryRouter><Humanset /></MemoryRouter>);
    fireEvent.change(screen.getByLabelText("Admin token"), { target: { value: "tok" } });
    fireEvent.click(screen.getByRole("button", { name: "Load drafts" }));
    await screen.findByText("How many accounts?");
    expect(screen.getByText("SELECT COUNT(*) FROM accounts")).toBeTruthy();
    expect(new Headers(calls[0].init?.headers).get("X-Admin-Token")).toBe("tok");
    fireEvent.click(screen.getByRole("button", { name: "Correct" }));
    await waitFor(() => expect(calls.some((c) => c.url.endsWith("/api/humanset/decide"))).toBe(true));
    const post = calls.find((c) => c.url.endsWith("/api/humanset/decide"))!;
    expect(JSON.parse(String(post.init?.body))).toEqual({ set: "ab12cd34", task_id: "h-0-00000000", decision: "confirm" });
  });

  it("shows the server's refusal and no drafts when the token is wrong", async () => {
    stubFetch(() => json({ error: "forbidden", message: "invalid admin token" }, 403));
    render(<MemoryRouter><Humanset /></MemoryRouter>);
    fireEvent.change(screen.getByLabelText("Admin token"), { target: { value: "nope" } });
    fireEvent.click(screen.getByRole("button", { name: "Load drafts" }));
    await screen.findByText(/invalid admin token/);
    expect(screen.queryByText("How many accounts?")).toBeNull();
  });
});
