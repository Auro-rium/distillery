import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import Playground, { RowsPreview } from ".";
import { contract } from "../../testutil/contract";
import { unexplained, visibleText } from "../../testutil/provenance";
import { json, stubFetch } from "../../testutil/render";
import { normalizeRows } from "./rows";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

const A = contract.acmeAsk;
const ST = contract.gatewayStatus;
const SV = contract.gatewaySavings;
const DR = contract.gatewayDrift;

interface Routes {
  status?: () => Response;
  savings?: () => Response;
  drift?: () => Response;
  ask?: (body: { question: string }) => Response;
}
/** The four endpoints the screen reads; each one defaults to the generated contract fixture. */
function stub(r: Routes = {}) {
  return stubFetch((url, init) => {
    if (url.endsWith("/gateway/status")) return r.status ? r.status() : json(ST);
    if (url.endsWith("/gateway/savings")) return r.savings ? r.savings() : json(SV);
    if (url.endsWith("/gateway/drift")) return r.drift ? r.drift() : json(DR);
    if (url.endsWith("/acme/ask")) return r.ask ? r.ask(JSON.parse(String(init?.body))) : json(A);
    return json({ error: "not_found", message: url }, 404);
  });
}
const open = () => render(<MemoryRouter><Playground /></MemoryRouter>);
async function ask(text = "how many accounts?") {
  const view = open();
  await screen.findByText(/Savings so far/);
  fireEvent.change(screen.getByLabelText("Question"), { target: { value: text } });
  fireEvent.click(screen.getByRole("button", { name: "Ask" }));
  return view;
}
const asked = (f: ReturnType<typeof stub>) => f.mock.calls.filter(([u]) => String(u).endsWith("/acme/ask"));

describe("rows: the gateway sends {columns, rows, row_count}", () => {
  it("normalises the object form, the array-of-arrays form and the array-of-objects form", () => {
    expect(normalizeRows({ columns: ["n", "s"], rows: [[1, "a"], [2, null]], row_count: 7 })).toEqual({ columns: ["n", "s"], rows: [[1, "a"], [2, null]], total: 7 });
    expect(normalizeRows([[1, "x"]])).toEqual({ columns: ["col 1", "col 2"], rows: [[1, "x"]], total: null });
    expect(normalizeRows([{ a: 1 }, { a: null }])).toEqual({ columns: ["a"], rows: [[1], [null]], total: null });
    expect(normalizeRows([])).toEqual({ columns: [], rows: [], total: null });
  });
  it("returns null for a shape it does not know, instead of guessing or throwing", () => {
    expect(normalizeRows("nope")).toBeNull();
    expect(normalizeRows({ columns: "x", rows: 3 })).toBeNull();
    expect(normalizeRows({ rows: [[1]] })).toBeNull();
  });
  it("renders a table with the payload's columns and row_count; NULL is shown as NULL", () => {
    render(<RowsPreview rows={{ columns: ["user_id", "name"], rows: [[1, "ada"], [2, null]], row_count: 42 }} />);
    const t = screen.getByRole("table");
    expect(within(t).getByRole("columnheader", { name: "user_id" })).toBeTruthy();
    expect(within(t).getByText("ada")).toBeTruthy();
    expect(within(t).getByText("NULL")).toBeTruthy();
    expect(screen.getByText("showing 2 of 42 rows")).toBeTruthy();
  });
  it("says so in words when the query returned no rows, and when the shape is not recognised", () => {
    const a = render(<RowsPreview rows={{ columns: ["a"], rows: [], row_count: 0 }} />);
    expect(screen.getByText("Query returned no rows.")).toBeTruthy();
    a.unmount();
    render(<RowsPreview rows={"weird"} />);
    expect(screen.getByText(/could not be shown/i)).toBeTruthy();
  });
});

describe("the question box", () => {
  it("has a real label, no placeholder, a counter, and Ask is disabled until there is a question", async () => {
    stub();
    open();
    const box = await screen.findByLabelText("Question");
    expect(box.tagName).toBe("TEXTAREA");
    expect(document.querySelector(`label[for="${box.id}"]`)!.textContent).toBe("Question");
    expect(box.getAttribute("placeholder")).toBeNull();
    const btn = screen.getByRole("button", { name: "Ask" }) as HTMLButtonElement;
    expect(btn.disabled).toBe(true);
    fireEvent.change(box, { target: { value: "abc" } });
    expect(document.body.textContent).toContain("3 / 500");
    expect(btn.disabled).toBe(false);
  });
  it("offers example questions, including odd phrasings and one outside the data, and a try-to-break-it hint", async () => {
    stub();
    open();
    const list = await screen.findByRole("list", { name: "Examples" });
    const items = within(list).getAllByRole("button");
    expect(items.length).toBeGreaterThanOrEqual(4);
    expect(items.length).toBeLessThanOrEqual(6);
    expect(items.some((b) => /rn|ppl|gimme/.test(b.textContent ?? ""))).toBe(true);
    expect(items.some((b) => /net revenue retention/.test(b.textContent ?? ""))).toBe(true);
    expect(screen.getByText(/Try to break it/)).toBeTruthy();
    fireEvent.click(items[1]);
    expect((screen.getByLabelText("Question") as HTMLTextAreaElement).value).toBe(items[1].textContent);
  });
  it("also offers questions in the style the specialist is certified on, which fill the box", async () => {
    stub();
    open();
    const list = await screen.findByRole("list", { name: "The specialist's home ground" });
    const items = within(list).getAllByRole("button");
    expect(items.length).toBe(3);
    fireEvent.click(items[0]);
    expect((screen.getByLabelText("Question") as HTMLTextAreaElement).value).toBe(items[0].textContent);
  });
  it("sends the trimmed question to POST /api/acme/ask; Ctrl+Enter also sends", async () => {
    const f = stub();
    open();
    const box = await screen.findByLabelText("Question");
    fireEvent.change(box, { target: { value: "  how many?  " } });
    fireEvent.keyDown(box, { key: "Enter", ctrlKey: true });
    await waitFor(() => expect(asked(f)).toHaveLength(1));
    expect(JSON.parse(String(asked(f)[0][1]?.body))).toEqual({ question: "how many?" });
    expect(asked(f)[0][1]?.method).toBe("POST");
  });
  it("says a request is running while it runs, and Ask is busy", async () => {
    let done!: (r: Response) => void;
    stubFetch((u) => (u.endsWith("/acme/ask") ? new Promise<Response>((r) => { done = r; }) as unknown as Response : u.endsWith("/status") ? json(ST) : u.endsWith("/drift") ? json(DR) : json(SV)));
    await ask();
    const btn = screen.getByRole("button", { name: "Ask" }) as HTMLButtonElement;
    await waitFor(() => expect(btn.disabled).toBe(true));
    expect(btn.getAttribute("aria-busy")).toBe("true");
    expect(screen.getByText("Asking the Acme analyst")).toBeTruthy();
    done(json(A));
    await screen.findByText("Student (certified)");
    expect(screen.queryByText("Asking the Acme analyst")).toBeNull();
  });
});

describe("an answer served by the certified student", () => {
  it("shows who served it, confidence against lambda, the reason, the SQL, the rows, the summary, the cost and the latency", async () => {
    stub();
    await ask();
    expect(await screen.findByText("Student (certified)")).toBeTruthy();
    expect(screen.queryByText("Teacher (deferred)")).toBeNull();
    const d = A.distillery;
    expect(screen.getByText(d.reason)).toBeTruthy();
    expect(screen.getByText(d.score.toFixed(3))).toBeTruthy();
    expect(screen.getAllByText(d.lambda.toFixed(3)).length).toBeGreaterThan(0);
    expect(document.querySelector("pre code")!.textContent).toBe(A.answer.sql);
    expect(screen.getByText("verified: ran read-only and returned rows")).toBeTruthy();
    const t = screen.getByRole("table");
    expect(within(t).getByText(String(A.answer.rows[0][0]))).toBeTruthy();
    expect(screen.getByText(A.summary)).toBeTruthy();
    expect(screen.getByText(`$${d.cost_usd.toFixed(6)}`)).toBeTruthy();
    expect(screen.getByText(`${d.latency_ms.toFixed(1)} ms`)).toBeTruthy();
    expect(screen.getByText(d.request_id)).toBeTruthy();
    // the basis of the cost figures is on the page as text, not only in a tooltip
    expect(screen.getByText(d.usage_basis)).toBeTruthy();
    expect(screen.getByText(d.counterfactual_basis)).toBeTruthy();
    expect(screen.queryByRole("alert")).toBeNull();
  });
  it("every number on the card is explained by the response (render provenance)", async () => {
    stub();
    const { container } = await ask();
    await screen.findByText("Student (certified)");
    const allow = [
      { pattern: /\d+ \/ 500/g, why: "character counter against the 500-character input limit" },
      { pattern: /(?<=\s)08(?=\s)/g, why: "tail of the response value $0.001408 left after the checker removed the server display string $0.0014 (a substring of it)" },
    ];
    expect(unexplained(visibleText(container), [A, ST, SV, DR], allow)).toEqual([]);
  });
});

describe("an answer deferred to the teacher", () => {
  const deferred = {
    ...A,
    distillery: { ...A.distillery, served_by: "teacher", score: 0.41, reason: "confidence 0.41 is below the threshold 0.80: handed to the teacher", cost_usd: 0.0014, teacher_cost_usd_counterfactual: 0.0014 },
  };
  it("is badged Teacher (deferred) and says why, with the confidence and lambda it was given", async () => {
    stub({ ask: () => json(deferred) });
    await ask();
    expect(await screen.findByText("Teacher (deferred)")).toBeTruthy();
    expect(screen.queryByText("Student (certified)")).toBeNull();
    expect(screen.getByText("confidence 0.41 is below the threshold 0.80: handed to the teacher")).toBeTruthy();
    expect(screen.getByText("0.410")).toBeTruthy();
    expect(screen.getAllByText("$0.001400").length).toBeGreaterThan(0);
  });
  it("shows an unknown served_by as it is, and absent score/lambda as not measured", async () => {
    stub({ ask: () => json({ ...A, distillery: { ...A.distillery, served_by: "mystery", score: null, lambda: null } }) });
    await ask();
    expect(await screen.findByText("mystery")).toBeTruthy();
    expect(screen.getAllByText("not measured").length).toBeGreaterThanOrEqual(2);
  });
  it("shows a SQL error as given and no table; a failed summary says why it is missing", async () => {
    stub({ ask: () => json({ ...A, summary: null, summary_error: { code: "rate_limited", message: "limit reached", status: 429, retry_after: "30" }, answer: { ...A.answer, verified: false, error: "query failed: no such column: nope", rows: [], row_count: 0 } }) });
    await ask();
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("query failed: no such column: nope");
    expect(screen.queryByRole("table")).toBeNull();
    expect(screen.getByText(/not verified/)).toBeTruthy();
    expect(screen.getByText(/Summary unavailable/).textContent).toContain("limit reached");
    expect(screen.getByText(/Summary unavailable/).textContent).toContain("30");
  });
});

describe("the gateway is not certified", () => {
  const off = { ...ST, certified: false, reason: "not certified: all traffic goes to the teacher", lambda: null, coverage: null, selective_risk: null, alpha: null, delta: null, prereg_commit: null, model_sha256: null };
  it("says so plainly, with the server's reason, and does not invent a certificate", async () => {
    stub({ status: () => json(off) });
    open();
    expect(await screen.findByText("Not certified yet: all traffic goes to the teacher.")).toBeTruthy();
    expect(screen.getByText("not certified: all traffic goes to the teacher")).toBeTruthy();
    expect(screen.getByText("not certified", { selector: ".badge" })).toBeTruthy();
    expect(screen.queryByText("certified", { selector: ".badge" })).toBeNull();
    expect(screen.getAllByText("not measured").length).toBeGreaterThanOrEqual(4);
    expect(screen.getByText("not recorded")).toBeTruthy();
  });
  it("a certified gateway shows alpha, delta, coverage, lambda and the short commit", async () => {
    stub({ status: () => json({ ...ST, prereg_commit: "0123456789abcdef0123456789abcdef01234567" }) });
    open();
    expect(await screen.findByText("certified", { selector: ".badge" })).toBeTruthy();
    expect(screen.queryByText(/Not certified yet/)).toBeNull();
    const dl = document.querySelector(".pg-st-list")!;
    expect(dl.textContent).toContain("0.10");
    expect(dl.textContent).toContain("0.05");
    expect(dl.textContent).toContain("62.0%");
    expect(dl.textContent).toContain("0.800");
    const commit = screen.getByText("0123456");
    expect(commit.getAttribute("title")).toBe("0123456789abcdef0123456789abcdef01234567");
  });
  it("a status that cannot be loaded is an error with its code, not an empty or uncertified state", async () => {
    stub({ status: () => json({ error: "gw_down", message: "no gateway" }, 500) });
    open();
    const alert = await screen.findByText(/gw_down/);
    expect(alert.closest("[role=alert]")!.textContent).toContain("no gateway");
    expect(screen.queryByText(/Not certified yet/)).toBeNull();
  });
});

describe("failures are told apart, and none of them is an empty answer", () => {
  it("rate limited: its own heading, the wait from Retry-After, and 'later' when the API gave none", async () => {
    stub({ ask: () => json({ error: "rate_limited", message: "limit of the gateway reached", request_id: "req_x" }, 429, { "Retry-After": "42" }) });
    await ask();
    const a = await screen.findByRole("alert");
    expect(within(a).getByText("Rate limit reached")).toBeTruthy();
    expect(within(a).getByText("Try again in 42 seconds.")).toBeTruthy();
    expect(a.textContent).toContain("rate_limited");
    expect(screen.queryByText("Answer")).toBeNull();
    expect(screen.queryByText("Student (certified)")).toBeNull();
    cleanup();
    stub({ ask: () => json({ error: "rate_limited", message: "m" }, 429) });
    await ask();
    expect(await screen.findByText("Try again later.")).toBeTruthy();
  });
  it("daily budget used up: its own heading and the way to the replay", async () => {
    stub({ ask: () => json({ error: "daily_budget_exhausted", message: "resets at 00:00 UTC" }, 429, { "Retry-After": "100" }) });
    await ask();
    const a = await screen.findByRole("alert");
    expect(within(a).getByText("Demo budget exhausted")).toBeTruthy();
    expect(within(a).getByRole("link", { name: "See replay" }).getAttribute("href")).toBe("/");
  });
  it("any other error shows the code and message it gave; a network failure is an error too", async () => {
    stub({ ask: () => json({ error: "upstream_unavailable", message: "no upstream model provider is configured" }, 503) });
    await ask();
    const a = await screen.findByRole("alert");
    expect(a.textContent).toContain("upstream_unavailable");
    expect(a.textContent).toContain("no upstream model provider is configured");
    cleanup();
    vi.stubGlobal("fetch", vi.fn((u: string) => (u.endsWith("/acme/ask") ? Promise.reject(new TypeError("Failed to fetch")) : Promise.resolve(json(u.endsWith("/status") ? ST : u.endsWith("/drift") ? DR : SV)))));
    await ask();
    expect((await screen.findByRole("alert")).textContent).toContain("network");
  });
  it("an old answer is cleared when the next request fails", async () => {
    let n = 0;
    stub({ ask: () => (++n === 1 ? json(A) : json({ error: "boom", message: "later" }, 500)) });
    await ask();
    await screen.findByText("Student (certified)");
    fireEvent.click(screen.getByRole("button", { name: "Ask" }));
    await screen.findByText(/boom/);
    expect(screen.queryByText("Student (certified)")).toBeNull();
  });
});

describe("the savings counter", () => {
  it("shows the server's display strings and counts, and says the student cost basis is pending H1", async () => {
    stub();
    open();
    await screen.findByText(/Savings so far/);
    expect(screen.getAllByText(SV.saved_display).length).toBeGreaterThan(0);
    const sv = within(document.querySelector(".pg-savings") as HTMLElement);
    expect(sv.getByText(SV.actual_display)).toBeTruthy();
    expect(screen.getAllByText(SV.teacher_only_display).length).toBeGreaterThan(0);
    expect(screen.getByText(SV.saved_pct_display!)).toBeTruthy();
    expect(screen.getByText("student serving cost: pending H1")).toBeTruthy();
    expect(screen.getByRole("note").textContent).toContain("pending H1");
    // the basis of every figure is on the page
    const basis = document.querySelector(".pg-basis")!.textContent!;
    for (const b of [SV.basis.actual, SV.basis.teacher_only, SV.basis.saved]) expect(basis).toContain(b);
  });
  it("with a measured price there is no pending note and the basis is the server's", async () => {
    const measured = { ...SV, student_cost_basis: "measured: bench run (as of 2026-10-20)", student_usd_per_1k: 0.5, student_cost_usd: 0.0005, basis: { actual: "a", teacher_only: "t", saved: "s" } };
    stub({ savings: () => json(measured) });
    open();
    await screen.findByText(/Savings so far/);
    expect(screen.getByText("student serving cost: measured: bench run (as of 2026-10-20)")).toBeTruthy();
    expect(screen.queryByRole("note")).toBeNull();
    expect(document.body.textContent).not.toContain("pending H1");
  });
  it("without a percentage says it is not available instead of showing zero", async () => {
    stub({ savings: () => json({ ...SV, saved_pct: null, saved_pct_display: null, requests: 0 }) });
    open();
    expect(await screen.findByText("percentage not available yet")).toBeTruthy();
  });
  it("is re-read after a question, so the counter shows the server's new figures", async () => {
    let n = 0;
    stub({ savings: () => json(++n === 1 ? SV : { ...SV, saved_usd: 9.1234, saved_display: "$9.1234", requests: 5 }) });
    await ask();
    await screen.findByText("$9.1234");
  });
  it("a savings read that fails is an error with its code", async () => {
    stub({ savings: () => json({ error: "sv_down", message: "cannot read the log" }, 500) });
    open();
    expect(await screen.findByText(/sv_down/)).toBeTruthy();
    expect(screen.queryByText(/Savings so far/)).toBeNull();
  });
});

describe("the page renders no number the API did not send", () => {
  it("before any question", async () => {
    stub();
    const { container } = open();
    await screen.findByText(/Savings so far/);
    await screen.findByText("Gateway");
    const allow = [{ pattern: /\d+ \/ 500/g, why: "character counter against the 500-character input limit" }];
    expect(unexplained(visibleText(container), [ST, SV, DR], allow)).toEqual([]);
  });
});
