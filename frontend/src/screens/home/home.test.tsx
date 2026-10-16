import { cleanup, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { contract, contractFetch } from "../../testutil/contract";
import { renderApp, stubFetch } from "../../testutil/render";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

describe("overview", () => {
  it("leads with the product and links the analyst, the receipts and the distillation story", async () => {
    stubFetch(contractFetch());
    renderApp("/");
    expect(screen.getByRole("heading", { level: 1 }).textContent).toMatch(/Change one base URL/);
    expect(screen.getByRole("link", { name: "Ask the Acme analyst" }).getAttribute("href")).toBe("/playground");
    expect(screen.getByRole("link", { name: "Check the receipts" }).getAttribute("href")).toBe("/receipts");
    expect(screen.getByRole("link", { name: "the distillation run" }).getAttribute("href")).toBe("/mission");
    expect(await screen.findByText("This server, right now")).toBeTruthy();
  });

  it("shows the serving claims in order, and a pending claim with no number and no link", async () => {
    stubFetch(contractFetch());
    renderApp("/");
    const ids = (await screen.findAllByText(/^W-(C1|C2|D|E2E|G|\$)$/)).map((e) => e.textContent);
    expect(ids).toEqual(["W-C1", "W-C2", "W-D", "W-E2E", "W-G", "W-$"]);
    const pending = contract.evidence.receipts.find((r: { id: string }) => r.id === "W-E2E");
    const card = screen.getByText("W-E2E").closest(".h-claim") as HTMLElement;
    expect(within(card).getByText("Pending")).toBeTruthy();
    expect(within(card).getByText("Pre-registered, not yet scored.")).toBeTruthy();
    expect(within(card).queryByRole("link")).toBeNull();
    expect(pending?.status).toBe("pending");
  });

  it("explains a rolled-back gateway instead of calling it certified", async () => {
    stubFetch(contractFetch({ gatewayDrift: { ...contract.gatewayDrift, enabled: true, state: "TEACHER" } }));
    renderApp("/");
    expect(await screen.findByText(/the teacher answers everything until the router is re-certified/)).toBeTruthy();
  });

  it("says when the served checkpoint is not the certified one", async () => {
    stubFetch(contractFetch({
      gatewayStatus: { ...contract.gatewayStatus, student_adapter_sha256: "b".repeat(64), certified_adapter_sha256: "a".repeat(64) },
    }));
    renderApp("/");
    expect(await screen.findByText(/is not the certified checkpoint, so the teacher answers/)).toBeTruthy();
  });
});
