import { cleanup, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { contractFetch } from "../../testutil/contract";
import { renderApp, stubFetch } from "../../testutil/render";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

describe("overview", () => {
  it("leads with the product and links the mission and the distillation story", async () => {
    stubFetch(contractFetch());
    renderApp("/");
    expect(screen.getByRole("heading", { level: 1, name: /^An autonomous agent that distils a big model’s narrow skill/ })).toBeTruthy();
    expect(screen.getByText("Write → Test in Sandboxes → Fine-tune on Token Factory → Gate → Certify the router → Serve → Guard", { selector: ".glow-text" })).toBeTruthy();
    expect(screen.queryByText(/Change one base URL/)).toBeNull();
    expect(screen.queryByRole("link", { name: /Acme analyst/ })).toBeNull();
    expect(screen.getByRole("link", { name: "Go to mission" }).getAttribute("href")).toBe("/mission");
    expect(screen.queryByRole("link", { name: "Check the receipts" })).toBeNull();
    expect(screen.getByRole("link", { name: "the distillation run" }).getAttribute("href")).toBe("/mission");
    expect(screen.queryByText("This server, right now")).toBeNull();
    expect(screen.queryByText(/What has been shown/)).toBeNull();
  });
});
