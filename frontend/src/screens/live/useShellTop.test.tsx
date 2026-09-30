import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useShellTop } from "./useShellTop";

afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); });

function Probe() {
  const ref = useShellTop<HTMLDivElement>();
  return <div className="run-shell"><header className="run-sticky" style={{ position: "sticky" }} /><div ref={ref} className="live" /></div>;
}

describe("useShellTop", () => {
  it("publishes the run header's height as --live-shell while it is sticky, and zero when it is not", () => {
    vi.spyOn(HTMLElement.prototype, "offsetHeight", "get").mockReturnValue(173);
    const { container } = render(<Probe />);
    expect((container.querySelector(".live") as HTMLElement).style.getPropertyValue("--live-shell")).toBe("173px");
    cleanup();
    const { container: c2 } = render(<div className="run-shell"><header className="run-sticky" style={{ position: "static" }} /><Inner /></div>);
    expect((c2.querySelector(".live") as HTMLElement).style.getPropertyValue("--live-shell")).toBe("0px");
  });

  it("measures again when the header changes size", () => {
    let h = 173;
    vi.spyOn(HTMLElement.prototype, "offsetHeight", "get").mockImplementation(() => h);
    const cbs: Array<() => void> = [];
    vi.stubGlobal("ResizeObserver", class { constructor(cb: () => void) { cbs.push(cb); } observe() {} disconnect() {} });
    const { container } = render(<Probe />);
    const live = container.querySelector(".live") as HTMLElement;
    expect(live.style.getPropertyValue("--live-shell")).toBe("173px");
    h = 221;
    cbs.forEach((cb) => cb());
    expect(live.style.getPropertyValue("--live-shell")).toBe("221px");
  });

  it("does nothing outside a run shell", () => {
    function Bare() { const ref = useShellTop<HTMLDivElement>(); return <div ref={ref} className="live" />; }
    const { container } = render(<Bare />);
    expect((container.querySelector(".live") as HTMLElement).style.getPropertyValue("--live-shell")).toBe("");
  });
});

function Inner() {
  const ref = useShellTop<HTMLDivElement>();
  return <div ref={ref} className="live" />;
}

describe("useShellTop with a late element", () => {
  it("measures once the element appears (the screen shows a loading state first)", () => {
    vi.spyOn(HTMLElement.prototype, "offsetHeight", "get").mockReturnValue(150);
    function Late({ ready }: { ready: boolean }) {
      const ref = useShellTop<HTMLDivElement>();
      return <div className="run-shell"><header className="run-sticky" style={{ position: "sticky" }} />{ready ? <div ref={ref} className="live" /> : <p>loading</p>}</div>;
    }
    const { container, rerender } = render(<Late ready={false} />);
    expect(container.querySelector(".live")).toBeNull();
    rerender(<Late ready />);
    expect((container.querySelector(".live") as HTMLElement).style.getPropertyValue("--live-shell")).toBe("150px");
  });
});
