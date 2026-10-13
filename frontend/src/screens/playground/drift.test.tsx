import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import Playground from ".";
import type { GatewayDrift } from "../../api/types";
import { contract } from "../../testutil/contract";
import { unexplained, visibleText } from "../../testutil/provenance";
import { json, stubFetch } from "../../testutil/render";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

/** The generated contract fixture: a guard that rolled back (state TEACHER) with the rollback's three events. */
const TEACHER: GatewayDrift = contract.gatewayDrift as GatewayDrift;
const CERTIFIED: GatewayDrift = {
  ...TEACHER,
  state: "CERTIFIED",
  allows_student: true,
  reason: null,
  detector: null,
  rolled_back_cert_sha256: [],
  window: { requests_seen: 9, window_requests: 4, window_deferred: 1, deferral_share: 0.25, audits_in_window: 2, audit_disagreements: 1, armed: true },
  audits: { audits: 2, inconclusive: 1, skipped_cap: 3, skipped_error: 0, spent_usd: 0.0123 },
  events: [],
};
const ROLLBACK: GatewayDrift = { ...TEACHER, state: "ROLLBACK" };
const DISABLED: GatewayDrift = {
  ...TEACHER,
  enabled: false, state: null, allows_student: null, reason: null, detector: null, params: null, window: null,
  rolled_back_cert_sha256: [], events: [],
  note: "drift guard not configured: set DISTILLERY_DRIFT_WINDOW and DISTILLERY_DRIFT_TAU",
};

function stub(drift: () => Response, ask?: () => Response) {
  return stubFetch((url) => {
    if (url.endsWith("/gateway/status")) return json(contract.gatewayStatus);
    if (url.endsWith("/gateway/savings")) return json(contract.gatewaySavings);
    if (url.endsWith("/gateway/drift")) return drift();
    if (url.endsWith("/acme/ask") && ask) return ask();
    return json({ error: "not_found", message: url }, 404);
  });
}
const open = () => render(<MemoryRouter><Playground /></MemoryRouter>);
const panel = async () => within((await screen.findByText("Drift guard")).closest("section") as HTMLElement);

describe("drift panel: the guard is certified", () => {
  it("shows the CERTIFIED badge, the parameters as configured, the window share as the server reports it and the audit counters", async () => {
    stub(() => json(CERTIFIED));
    open();
    const p = await panel();
    expect(p.getByText("CERTIFIED", { selector: ".badge" })).toBeTruthy();
    expect(p.queryByText(/Rolled back/)).toBeNull();
    const params = p.getByLabelText("Detector parameters").textContent!;
    for (const v of Object.values(CERTIFIED.params!)) expect(params).toContain(String(v));
    expect(p.getByText("deferral threshold tau").nextElementSibling!.textContent).toBe("0.75");
    expect(p.getByText("audit rate").nextElementSibling!.textContent).toBe("0");
    const w = p.getByLabelText("Window").textContent!;
    expect(p.getByText("deferral share").nextElementSibling!.textContent).toBe("0.25");
    expect(w).toContain("1 of 4");
    expect(w).toContain("yes");
    const a = p.getByLabelText("Audits");
    expect(a.textContent).toContain("$0.0123");
    expect(within(a).getByText("skipped (daily cap)").nextElementSibling!.textContent).toBe("3");
    expect(p.getByText("No drift events recorded.")).toBeTruthy();
    expect(p.getByText(CERTIFIED.cert_sha256!)).toBeTruthy();
  });
  it("is read-only: no button or form in the panel", async () => {
    stub(() => json(CERTIFIED));
    open();
    const p = await panel();
    expect(p.queryAllByRole("button")).toEqual([]);
    expect(p.queryAllByRole("textbox")).toEqual([]);
  });
  it("every number on the page is explained by the payloads", async () => {
    stub(() => json(CERTIFIED));
    const { container } = open();
    await panel();
    const allow = [{ pattern: /\d+ \/ 500/g, why: "character counter against the 500-character input limit" }];
    expect(unexplained(visibleText(container), [contract.gatewayStatus, contract.gatewaySavings, CERTIFIED], allow)).toEqual([]);
  });
});

describe("drift panel: rolled back to the teacher", () => {
  it("TEACHER is a warning: badge, the server's reason, the detector and an empty window shown as not measured", async () => {
    stub(() => json(TEACHER));
    open();
    const p = await panel();
    expect(p.getByText("TEACHER", { selector: ".badge" })).toBeTruthy();
    expect(p.getByText(/Rolled back: every skill request goes to the teacher/)).toBeTruthy();
    expect(p.getByText(TEACHER.reason!)).toBeTruthy();
    expect(p.getByText("deferral share").nextElementSibling!.textContent).toBe("not measured");
    expect(p.getByText("detector armed").nextElementSibling!.textContent).toBe("no");
    expect(p.queryByText("CERTIFIED", { selector: ".badge" })).toBeNull();
  });
  it("ROLLBACK is a warning too", async () => {
    stub(() => json(ROLLBACK));
    open();
    const p = await panel();
    expect(p.getByText("ROLLBACK", { selector: ".badge" })).toBeTruthy();
    expect(p.getByText(/Rolled back/)).toBeTruthy();
  });
  it("lists the events newest first with time, kind and detector; a queued re-distillation is not run", async () => {
    stub(() => json(TEACHER));
    open();
    const p = await panel();
    const items = p.getAllByRole("listitem");
    expect(items).toHaveLength(TEACHER.events.length);
    expect(items[0].textContent).toContain(TEACHER.events[0].created_at);
    expect(items[0].textContent).toContain("re-distill queued (not run)");
    expect(items[0].textContent).toContain("detector: deferral_rate");
    expect(items[1].textContent).toContain("teacher_only");
    expect(items[2].textContent).toContain("rollback");
    expect(items[2].textContent).toContain("CERTIFIED to ROLLBACK");
    expect(p.queryByText("redistill_queued")).toBeNull();
  });
  it("every number on the page is explained by the payloads", async () => {
    stub(() => json(TEACHER));
    const { container } = open();
    await panel();
    const allow = [{ pattern: /\d+ \/ 500/g, why: "character counter against the 500-character input limit" }];
    expect(unexplained(visibleText(container), [contract.gatewayStatus, contract.gatewaySavings, TEACHER], allow)).toEqual([]);
  });
});

describe("drift panel: not configured, and errors", () => {
  it("says Drift guard not configured, shows no state badge, parameters or window", async () => {
    stub(() => json(DISABLED));
    open();
    const p = await panel();
    expect(p.getByText("Drift guard not configured")).toBeTruthy();
    expect(p.getByText("not configured", { selector: ".badge" })).toBeTruthy();
    expect(p.getByText(DISABLED.note)).toBeTruthy();
    expect(p.queryByLabelText("Detector parameters")).toBeNull();
    expect(p.queryByLabelText("Window")).toBeNull();
    expect(p.queryByText("CERTIFIED")).toBeNull();
    expect(p.queryByText(/Events, newest first/)).toBeNull();
  });
  it("a drift read that fails is an error with its code, not an empty or certified panel", async () => {
    stub(() => json({ error: "drift_down", message: "cannot read the drift log" }, 500));
    open();
    const alert = await screen.findByText(/drift_down/);
    expect(alert.closest("[role=alert]")!.textContent).toContain("cannot read the drift log");
    expect(screen.queryByText("Drift guard")).toBeNull();
    expect(screen.queryByText("CERTIFIED")).toBeNull();
  });
  it("the rest of the page still loads when only the drift read fails", async () => {
    stub(() => json({ error: "drift_down", message: "x" }, 500));
    open();
    expect(await screen.findByText(/Savings so far/)).toBeTruthy();
    expect(await screen.findByText("Gateway")).toBeTruthy();
  });
});

describe("drift panel: re-read after a question", () => {
  it("shows a rollback that happened on the question just asked", async () => {
    let n = 0;
    stub(() => json(++n === 1 ? CERTIFIED : TEACHER), () => json(contract.acmeAsk));
    open();
    const p = await panel();
    expect(p.getByText("CERTIFIED", { selector: ".badge" })).toBeTruthy();
    fireEvent.change(await screen.findByLabelText("Question"), { target: { value: "how many accounts?" } });
    fireEvent.click(screen.getByRole("button", { name: "Ask" }));
    await waitFor(() => expect(screen.getByText("TEACHER", { selector: ".badge" })).toBeTruthy());
  });
});
