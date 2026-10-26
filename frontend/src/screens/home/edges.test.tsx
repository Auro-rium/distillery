import { cleanup, fireEvent, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { contractFetch } from "../../testutil/contract";
import { renderApp, stubFetch } from "../../testutil/render";

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

describe("landing page side glow", () => {
  it("is on the landing page only, hidden from assistive technology, and goes when you leave", async () => {
    stubFetch(contractFetch());
    renderApp("/");
    const glow = document.querySelector(".h-edges")!;
    expect(glow.getAttribute("aria-hidden")).toBe("true");
    expect(glow.querySelectorAll(".h-edge")).toHaveLength(2);
    fireEvent.click(screen.getByRole("link", { name: "Receipts" }));
    await screen.findByRole("heading", { name: "Receipts" });
    expect(document.querySelector(".h-edges")).toBeNull();
  });
  it("is not on other pages", () => {
    stubFetch(contractFetch());
    renderApp("/new");
    expect(document.querySelector(".h-edges")).toBeNull();
  });
});
