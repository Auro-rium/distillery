import { cleanup, fireEvent, screen, waitFor, within } from "@testing-library/react";
import fs from "node:fs";
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Receipt as ReceiptRow } from "../../api/types";
import { contract, contractFetch } from "../../testutil/contract";
import { unexplained, visibleText, type Allow } from "../../testutil/provenance";
import { json, renderApp, stubFetch } from "../../testutil/render";
import { SHORT_COMMIT, SHORT_SHA } from "./Receipt";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); document.title = ""; });

const EV = contract.evidence;
const ROWS = EV.receipts as unknown as ReceiptRow[];
const row = (id: string): ReceiptRow => ROWS.find((r) => r.id === id)!;

async function open(evidence: unknown = EV) {
  stubFetch(contractFetch({ evidence }));
  const view = renderApp("/receipts");
  await screen.findByRole("list", { name: "Receipts" });
  return view;
}
const card = (id: string) => within(document.getElementById(id) as HTMLElement);

describe("receipts: one row per claim from /api/evidence", () => {
  it("lists every receipt in order, each with its claim as the heading", async () => {
    await open();
    const items = Array.from(screen.getByRole("list", { name: "Receipts" }).children) as HTMLElement[];
    expect(items).toHaveLength(ROWS.length);
    ROWS.forEach((r, i) => {
      expect(items[i].querySelector("article")!.id).toBe(r.id);
      expect(within(items[i]).getByRole("heading", { level: 3 }).textContent).toBe(r.claim);
    });
  });
  it("is a page of its own: title, heading, nav entry, and where the evidence was built from", async () => {
    await open();
    expect(screen.getByRole("heading", { level: 1, name: "Receipts" })).toBeTruthy();
    await waitFor(() => expect(document.title).toBe("Receipts · Distillery"));
    expect(within(screen.getByRole("navigation", { name: "Main" })).getByRole("link", { name: "Receipts" }).getAttribute("aria-current")).toBe("page");
    expect(screen.getByText(EV.commit).getAttribute("class")).toContain("rc-hash");
    expect(document.body.textContent).toContain(EV.generated_at);
  });
});

describe("a verified row", () => {
  const r = row("W4");
  it("is badged VERIFIED and shows the value, the pinned artifact link, the adapter and the sealed hashes", async () => {
    await open();
    const c = card("W4");
    expect(c.getByText("VERIFIED")).toBeTruthy();
    expect(c.getByText(r.value_display!)).toBeTruthy();
    const link = c.getByRole("link", { name: new RegExp(r.artifact.replace(/[.]/g, "\\.")) });
    expect(link.getAttribute("href")).toBe(r.permalink);
    expect(link.getAttribute("target")).toBe("_blank");
    expect(link.getAttribute("rel")).toContain("noreferrer");
    const adapter = c.getByText(r.adapter_sha256!.slice(0, SHORT_SHA));
    expect(adapter.getAttribute("title")).toBe(r.adapter_sha256);
    for (const [name, sha] of Object.entries(r.sealed_sha256)) {
      expect(c.getByText(name)).toBeTruthy();
      expect(c.getByText(sha.slice(0, SHORT_SHA)).getAttribute("title")).toBe(sha);
    }
    expect(c.getByText("$4.2582")).toBeTruthy(); // ledger_total_usd, formatted
    expect(c.getByText("no pre-registration commit recorded in this evidence")).toBeTruthy();
  });
  it("keeps the gate inputs in a details block, values as the evidence has them", async () => {
    await open();
    const d = card("W4").getByText("Gate inputs").closest("details")!;
    expect(d.textContent).toContain("ratio_lower_bound_min");
    expect(d.textContent).toContain("0.85");
    expect(d.textContent).toContain(String(r.gate_inputs!.n));
  });
});

describe("a pending row", () => {
  it("says Pending: pre-registered, not yet scored; no number, no link, only where the result will be written", async () => {
    await open();
    const r = row("W-C1");
    const c = card("W-C1");
    expect(c.getByText("PENDING")).toBeTruthy();
    expect(c.getByText("Pending: pre-registered, not yet scored")).toBeTruthy();
    expect(c.queryAllByRole("link")).toEqual([]);
    expect(c.getByText("will be written to")).toBeTruthy();
    expect(c.getByText(r.artifact)).toBeTruthy();
    const text = document.getElementById("W-C1")!.textContent!.replace("W-C1", "").replace(r.artifact, "");
    expect(text).not.toMatch(/\d/);
    expect(text).not.toContain("not measured");
    expect(document.getElementById("W-C1")!.querySelector("details")).toBeNull();
  });
  it("every pending receipt in the payload renders as pending", async () => {
    await open();
    for (const r of ROWS.filter((x) => x.status === "pending")) {
      expect(card(r.id).getByText("Pending: pre-registered, not yet scored")).toBeTruthy();
      expect(card(r.id).queryByText("VERIFIED")).toBeNull();
    }
  });
});

describe("a failed row", () => {
  it("N1 is NOT MET, in the evidence's own words, with the pre-registration commit short and in full in the title", async () => {
    await open();
    const r = row("N1");
    const c = card("N1");
    expect(c.getByText("NOT MET")).toBeTruthy();
    expect(c.queryByText("VERIFIED")).toBeNull();
    expect(c.getByText(r.value_display!)).toBeTruthy();
    expect(r.value_display).toMatch(/^REJECT: ratio lower bound/);
    expect(c.getByText(/This claim was not met/)).toBeTruthy();
    const pre = c.getByText(r.prereg!.commit.slice(0, SHORT_COMMIT));
    expect(pre.getAttribute("title")).toBe(r.prereg!.commit);
    expect(c.getByText(r.prereg!.committed_at)).toBeTruthy();
  });
  it("a PARTIAL proof says partly met, not a bare failure", async () => {
    await open();
    expect(row("W2").verdict).toBe("PARTIAL");
    expect(card("W2").getByText("PARTLY MET")).toBeTruthy();
    expect(card("W2").queryByText("NOT MET")).toBeNull();
  });
  it("W-G, a test, is verified with its value and a link to the test file", async () => {
    await open();
    const c = card("W-G");
    expect(c.getByText("VERIFIED")).toBeTruthy();
    expect(c.getByText("same agent code, base URL swapped; test passes")).toBeTruthy();
    expect(c.getByRole("link").getAttribute("href")).toBe(row("W-G").permalink);
  });
});

describe("a receipt that has been scored since (pending -> verified)", () => {
  it("shows the value and a live link once the payload says verified", async () => {
    const scored = { ...row("W-C2"), status: "verified", verdict: "PASS", value_display: "9 of 10 wrong answers deferred", prereg: { commit: "0123456789abcdef0123456789abcdef01234567", committed_at: "2026-10-20T10:00:00+05:30" }, sealed_sha256: { test_split: "a".repeat(64) } };
    await open({ ...EV, receipts: EV.receipts.map((r) => (r.id === "W-C2" ? scored : r)) });
    const c = card("W-C2");
    expect(c.getByText("VERIFIED")).toBeTruthy();
    expect(c.getByText("9 of 10 wrong answers deferred")).toBeTruthy();
    expect(c.getByRole("link").getAttribute("href")).toBe(scored.permalink);
    expect(c.queryByText(/Pending/)).toBeNull();
  });
});

describe("states", () => {
  it("a failed request is an error with its code and a retry, not an empty page", async () => {
    stubFetch(() => json({ error: "evidence_unavailable", message: "evidence file not found or unreadable" }, 404));
    renderApp("/receipts");
    const alert = await screen.findByText(/evidence_unavailable/);
    expect(alert.closest("[role=alert]")!.textContent).toContain("Could not load the receipts");
    expect(screen.queryByRole("list", { name: "Receipts" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /retry/i }));
  });
  it("an evidence file from before receipts says so", async () => {
    const { receipts: _gone, ...old } = EV;
    stubFetch(contractFetch({ evidence: old }));
    renderApp("/receipts");
    expect(await screen.findByText("No receipts in the evidence file")).toBeTruthy();
    expect(screen.queryByRole("list", { name: "Receipts" })).toBeNull();
  });
  it("an empty list says so", async () => {
    stubFetch(contractFetch({ evidence: { ...EV, receipts: [] } }));
    renderApp("/receipts");
    expect(await screen.findByText("The evidence file lists no receipts.")).toBeTruthy();
  });
});

describe("render provenance and phone layout", () => {
  it("every number on the page is explained by the evidence payload", async () => {
    const { container } = await open();
    const shorts = [
      ...ROWS.flatMap((r) => [r.prereg?.commit.slice(0, SHORT_COMMIT), r.adapter_sha256?.slice(0, SHORT_SHA), ...Object.values(r.sealed_sha256).map((s) => s.slice(0, SHORT_SHA))]),
    ].filter((s): s is string => !!s);
    const allow: Allow[] = [
      { pattern: new RegExp(`(?<![0-9a-f])(?:${shorts.join("|")})(?![0-9a-f])`, "g"), why: "commit and sha256 values shortened for display; the whole value is in the title attribute and in the payload" },
      { pattern: /\b[A-Z]\d\b/g, why: "two-character proof ids from the payload (W2, A5, N1 ...): strings shorter than three characters are not matched verbatim by the checker" },
    ];
    expect(unexplained(visibleText(container), [EV], allow)).toEqual([]);
  });
  it("the allow-list for shortened hashes is exact: an invented digit run is still caught", () => {
    expect(unexplained("hash 0123456 and 7777", [{ s: "0123456789abcdef" }], [{ pattern: /(?<![0-9a-f])0123456(?![0-9a-f])/g, why: "test" }])).toEqual(["7777"]);
  });
  it("single column on a phone, and long values wrap instead of scrolling sideways", () => {
    const css = fs.readFileSync(path.resolve(__dirname, "receipts.css"), "utf8");
    expect(css).toMatch(/\.rc-list \{[^}]*grid-template-columns: minmax\(0, 1fr\)/);
    expect(css).toMatch(/@media \(min-width: 900px\) \{ \.rc-list/);
    for (const sel of [".rc .rc-path", ".rc .rc-hash", ".rc .rc-claim", ".rc .rc-value"]) {
      expect(css).toMatch(new RegExp(`${sel.replace(/\./g, "\\.")} \\{[^}]*overflow-wrap: anywhere`));
    }
    expect(css).toContain("min-width: 0");
  });
});
