import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import Playground, { Pane, RowsPreview } from ".";
import { normalizeRows } from "./rows";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

const res = (b: unknown, status = 200, headers: Record<string, string> = {}) => new Response(JSON.stringify(b), { status, headers });
const cfg = (pg: Partial<{ enabled: boolean; per_ip_per_hour: number; daily_cap_usd: number; spent_today_usd: number }> = {}) =>
  ({ playground: { enabled: true, per_ip_per_hour: 10, daily_cap_usd: 1, spent_today_usd: 0.25, ...pg } });
const NA = (reason: string) => ({ available: false, reason, sql: null, verified: null, rows_preview: null, error: null });
const answer = (teacher: unknown, cost = 0.001) => ({ results: { teacher, base: NA("base is not served"), student: NA("student is not served") }, cost_usd: cost, note: "the note" });

/** GET /config answers with `config()` (a function so a later read can differ); every other call is the POST. */
function stub(post: () => Response, config: () => unknown = () => cfg()) {
  const f = vi.fn((url: string) => Promise.resolve(url.endsWith("/config") ? res(config()) : post()));
  vi.stubGlobal("fetch", f);
  return f;
}
const open = () => render(<MemoryRouter><Playground /></MemoryRouter>);
async function ask(text = "how many users?") {
  const view = open();
  await waitFor(() => screen.getByText(/daily budget spent/));
  fireEvent.change(screen.getByLabelText("Question"), { target: { value: text } });
  fireEvent.click(screen.getByRole("button", { name: "Ask" }));
  return view;
}

describe("rows preview: the real server sends {columns, rows, row_count}", () => {
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
  it("renders the object form as a table with the payload's column names and row_count, and does not crash", () => {
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

describe("the form", () => {
  it("has a real label wired to the textarea, no placeholder, and the character counter", async () => {
    stub(() => res(answer(NA("x"))));
    open();
    const box = await screen.findByLabelText("Question");
    expect(box.tagName).toBe("TEXTAREA");
    const label = document.querySelector(`label[for="${box.id}"]`)!;
    expect(label.textContent).toBe("Question");
    expect(box.getAttribute("placeholder")).toBeNull();
    expect(document.body.textContent).toContain("0 / 500");
    fireEvent.change(box, { target: { value: "abc" } });
    expect(document.body.textContent).toContain("3 / 500");
  });
  it("keeps Ask disabled for an empty question and while a request is running, and says a request is running", async () => {
    let done!: (r: Response) => void;
    const f = vi.fn((url: string) => (url.endsWith("/config") ? Promise.resolve(res(cfg())) : new Promise<Response>((r) => { done = r; })));
    vi.stubGlobal("fetch", f);
    open();
    await screen.findByText(/daily budget spent/);
    const ask = screen.getByRole("button", { name: "Ask" }) as HTMLButtonElement;
    expect(ask.disabled).toBe(true);
    fireEvent.change(screen.getByLabelText("Question"), { target: { value: "q" } });
    expect(ask.disabled).toBe(false);
    fireEvent.click(ask);
    await waitFor(() => expect(ask.disabled).toBe(true));
    expect(ask.getAttribute("aria-busy")).toBe("true");
    expect(screen.getByText("Asking the models")).toBeTruthy();
    done(res(answer(NA("x"))));
    await waitFor(() => expect(ask.disabled).toBe(false));
    expect(screen.queryByText("Asking the models")).toBeNull();
  });
  it("submits with Ctrl+Enter from the textarea", async () => {
    const f = stub(() => res(answer(NA("x"))));
    open();
    const box = await screen.findByLabelText("Question");
    fireEvent.change(box, { target: { value: "q" } });
    fireEvent.keyDown(box, { key: "Enter", ctrlKey: true });
    await waitFor(() => expect(f.mock.calls.some(([u]) => String(u).endsWith("/playground"))).toBe(true));
  });
});

describe("caps and allowance are shown exactly as the API returned them, and nothing is worked out", () => {
  it("shows the per-IP limit and 'spent of cap', with a meter, and no invented 'remaining'", async () => {
    stub(() => res(answer(NA("x"))), () => cfg({ per_ip_per_hour: 12, spent_today_usd: 0.5, daily_cap_usd: 2 }));
    open();
    await screen.findByText("daily budget spent $0.50 of $2.00");
    expect(screen.getByText("Limit 12 requests per hour per IP")).toBeTruthy();
    const meter = screen.getByRole("meter");
    expect(meter.getAttribute("aria-valuenow")).toBe("0.5");
    expect(meter.getAttribute("aria-valuemax")).toBe("2");
    expect(document.body.textContent).not.toMatch(/remaining|left today/i);
  });
  it("re-reads the allowance after a request, so the spent figure is the server's new one", async () => {
    let spent = 0.25;
    stub(() => { spent = 0.26; return res(answer(NA("x"), 0.01)); }, () => cfg({ spent_today_usd: spent }));
    await ask();
    await screen.findByText("daily budget spent $0.26 of $1.00");
  });
  it("says the allowance could not be loaded (an error, not zero), and still lets the form be used", async () => {
    vi.stubGlobal("fetch", vi.fn((url: string) => Promise.resolve(url.endsWith("/config") ? res({ error: "cfg_down", message: "no config" }, 500) : res(answer(NA("x"))))));
    open();
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("cfg_down");
    expect(alert.textContent).toContain("no config");
    expect(screen.queryByText(/daily budget spent/)).toBeNull();
    expect(screen.queryByRole("meter")).toBeNull();
    expect((screen.getByRole("button", { name: "Ask" }) as HTMLButtonElement).disabled).toBe(true); // still empty
    fireEvent.change(screen.getByLabelText("Question"), { target: { value: "q" } });
    expect((screen.getByRole("button", { name: "Ask" }) as HTMLButtonElement).disabled).toBe(false);
  });
});

describe("disabled is its own state", () => {
  it("says the server reports the playground as disabled, disables the form, links to the replay, and invents no reason", async () => {
    stub(() => res(answer(NA("x"))), () => cfg({ enabled: false }));
    open();
    const status = await screen.findByText("playground disabled");
    expect(status.closest("[role=status]")).not.toBeNull();
    expect((screen.getByRole("button", { name: "Ask" }) as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByLabelText("Question") as HTMLTextAreaElement).disabled).toBe(true);
    expect(screen.getByRole("link", { name: "See replay" }).getAttribute("href")).toBe("/");
    expect(screen.getByText(/daily budget spent/)).toBeTruthy(); // the caps are still the API's
    expect(screen.queryByRole("alert")).toBeNull();
    expect(document.body.textContent).not.toMatch(/replay-only|not configured/i);
  });
});

describe("failures are told apart, and none of them is an empty result", () => {
  it("rate limited: its own heading, the wait from Retry-After, and 'later' when the API gave none", async () => {
    stub(() => res({ error: "rate_limited", message: "playground rate limit reached" }, 429, { "Retry-After": "42" }));
    await ask();
    const a = await screen.findByRole("alert");
    expect(within(a).getByText("Rate limit reached")).toBeTruthy();
    expect(within(a).getByText("Try again in 42 seconds.")).toBeTruthy();
    expect(a.textContent).toContain("rate_limited");
    expect(screen.queryByText("Cost of this request", { exact: false })).toBeNull();
    cleanup();
    stub(() => res({ error: "rate_limited", message: "m" }, 429));
    await ask();
    expect(await screen.findByText("Try again later.")).toBeTruthy();
  });
  it("budget exhausted: its own heading and the way to the replay", async () => {
    stub(() => res({ error: "demo_budget_exhausted", message: "demo budget exhausted, see replay" }, 503));
    await ask();
    const a = await screen.findByRole("alert");
    expect(within(a).getByText("Demo budget exhausted")).toBeTruthy();
    expect(within(a).getByRole("link", { name: "See replay" }).getAttribute("href")).toBe("/");
  });
  it("any other API error shows the code and message it gave, and no answer cards", async () => {
    stub(() => res({ error: "bad_request", message: "question too long" }, 422));
    await ask();
    const a = await screen.findByRole("alert");
    expect(a.textContent).toContain("bad_request");
    expect(a.textContent).toContain("question too long");
    expect(screen.queryByRole("region", { name: "Answers" })).toBeNull();
  });
  it("a network failure is an error too", async () => {
    vi.stubGlobal("fetch", vi.fn((url: string) => (url.endsWith("/config") ? Promise.resolve(res(cfg())) : Promise.reject(new TypeError("Failed to fetch")))));
    await ask();
    const a = await screen.findByRole("alert");
    expect(a.textContent).toContain("network");
  });
  it("an old answer is cleared when the next request fails", async () => {
    let n = 0;
    stub(() => (++n === 1 ? res(answer({ available: true, reason: null, sql: "SELECT 1", verified: null, rows_preview: null, error: null })) : res({ error: "boom", message: "later" }, 500)));
    await ask();
    await screen.findByText("SELECT 1");
    fireEvent.click(screen.getByRole("button", { name: "Ask" }));
    await screen.findByRole("alert");
    expect(screen.queryByText("SELECT 1")).toBeNull();
  });
});

describe("answers: what came back, and nothing else", () => {
  it("when no model could be served, says so, lists each model's own reason, and is not an error", async () => {
    stub(() => res({ results: { teacher: NA("teacher is not configured"), base: NA("base is not served"), student: NA("student is not served") }, cost_usd: 0, note: "none served" }));
    await ask();
    await screen.findByText("teacher is not configured");
    expect(screen.getByText("base is not served")).toBeTruthy();
    expect(screen.getByText("student is not served")).toBeTruthy();
    expect(screen.getByText("None of the models could answer this request.")).toBeTruthy();
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.getAllByText("not available")).toHaveLength(3);
    expect(screen.getByText(/none served/)).toBeTruthy();
  });
  it("does not print the 'none could answer' line when one model answered", async () => {
    stub(() => res(answer({ available: true, reason: null, sql: "SELECT 1", verified: true, rows_preview: null, error: null })));
    await ask();
    await screen.findByText("SELECT 1");
    expect(screen.queryByText("None of the models could answer this request.")).toBeNull();
  });
  it("shows the real rows_preview object, its row_count, and the API's cost with its note", async () => {
    stub(() => res(answer({ available: true, reason: null, sql: "SELECT id FROM t", verified: false, rows_preview: { columns: ["id"], rows: [[7], [8]], row_count: 9 }, error: null }, 0.000123)));
    await ask();
    await screen.findByText("SELECT id FROM t");
    expect(screen.getByText("verified: wrong result")).toBeTruthy();
    expect(screen.getByText("showing 2 of 9 rows")).toBeTruthy();
    expect(screen.getByText("$0.000123", { exact: false }).textContent).toContain("the note");
  });
  it("a model that ran but returned no SQL and no error says exactly that; an error string is shown as given", () => {
    const a = render(<Pane name="Teacher" r={{ available: true, reason: null, sql: null, verified: null, rows_preview: null, error: null }} />);
    expect(screen.getByText("No SQL returned.")).toBeTruthy();
    a.unmount();
    render(<Pane name="Teacher" r={{ available: true, reason: null, sql: "SELECT nope", verified: null, rows_preview: null, error: "no such column: nope" }} />);
    expect(screen.getByRole("alert").textContent).toContain("no such column: nope");
    expect(screen.queryByText("Query returned no rows.")).toBeNull();
  });
  it("puts each SQL string in a code block with a copy control outside the code, and never prints a verdict it was not given", () => {
    const { container } = render(<Pane name="Teacher" r={{ available: true, reason: null, sql: "SELECT 1", verified: null, rows_preview: null, error: null }} />);
    expect(container.querySelector("pre code")!.textContent).toBe("SELECT 1");
    expect(container.querySelector("pre")!.querySelector("button")).toBeNull();
    expect(screen.getByText(/not verified/)).toBeTruthy();
    expect(screen.queryByText(/verified: matches/)).toBeNull();
  });
});
