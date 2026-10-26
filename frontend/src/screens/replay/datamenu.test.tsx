import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { DataMenu, type DataRow } from "./DataMenu";

afterEach(() => cleanup());

const rows: DataRow[] = [
  { key: "a", label: "steps", value: "10" },
  { key: "t", label: "train loss", value: "0.5", help: "Train loss help." },
  { key: "v", label: "validation loss", value: "0.7", help: "Validation loss help." },
];
const open = () => fireEvent.click(screen.getByRole("button", { name: /^Data/ }));

describe("DataMenu", () => {
  it("starts closed, opens downward with a green button, and closes on Esc or a click elsewhere", () => {
    render(<DataMenu rows={rows} />);
    const btn = screen.getByRole("button", { name: /^Data/ });
    expect(btn.className).toContain("ok");
    expect(screen.queryByRole("list")).toBeNull();
    open();
    expect(screen.getByRole("list", { name: "Fine-tune record" })).toBeTruthy();
    fireEvent.keyDown(document, { key: "Escape" });
    expect(btn.getAttribute("aria-expanded")).toBe("false");
    open();
    fireEvent.pointerDown(document.body);
    expect(btn.getAttribute("aria-expanded")).toBe("false");
  });

  it("only rows with a definition are buttons; plain rows are not", () => {
    render(<DataMenu rows={rows} />);
    open();
    expect(screen.getByRole("button", { name: /train loss/ })).toBeTruthy();
    expect(screen.queryByRole("button", { name: /steps/ })).toBeNull();
  });

  it("puts the definition to the right when it fits, below the row when it does not", () => {
    const real = HTMLElement.prototype.getBoundingClientRect;
    render(<DataMenu rows={rows} />);
    open();
    fireEvent.click(screen.getByRole("button", { name: /train loss/ }));
    expect(screen.getByRole("note").className).toContain("right");
    fireEvent.click(screen.getByRole("button", { name: /train loss/ })); // close
    HTMLElement.prototype.getBoundingClientRect = () => ({ right: window.innerWidth - 20 }) as DOMRect; // no room on the right
    try {
      fireEvent.click(screen.getByRole("button", { name: /train loss/ }));
      expect(screen.getByRole("note").className).toContain("below");
    } finally {
      HTMLElement.prototype.getBoundingClientRect = real;
    }
  });

  it("Esc closes the definition first and the list second", () => {
    render(<DataMenu rows={rows} />);
    open();
    fireEvent.click(screen.getByRole("button", { name: /validation loss/ }));
    expect(screen.getByRole("note")).toBeTruthy();
    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByRole("note")).toBeNull();
    expect(screen.getByRole("list")).toBeTruthy();
    fireEvent.keyDown(document, { key: "Escape" });
    expect(screen.queryByRole("list")).toBeNull();
  });
});
