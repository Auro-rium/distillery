import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  Badge, Button, Card, Dialog, Field, IconButton, Input, Select, Skeleton, Spinner, Stat, Switch, Textarea, Toolbar, Tooltip,
} from ".";

afterEach(cleanup);

describe("Field wiring", () => {
  it("ties label, help text and control together by id", () => {
    render(
      <Field label="Budget" help="Help words">
        <Input defaultValue="x" />
      </Field>,
    );
    const input = screen.getByLabelText("Budget");
    expect(input.tagName).toBe("INPUT");
    const help = document.getElementById(input.getAttribute("aria-describedby")!.split(" ")[0]);
    expect(help?.textContent).toBe("Help words");
    expect(input.getAttribute("aria-invalid")).toBeNull();
  });

  it("marks the control invalid and describes it with the error text", () => {
    render(
      <Field label="Token" help="Held in memory" error="Missing token">
        <Input />
      </Field>,
    );
    const input = screen.getByLabelText("Token");
    expect(input.getAttribute("aria-invalid")).toBe("true");
    const ids = input.getAttribute("aria-describedby")!.split(" ");
    expect(ids).toHaveLength(2);
    expect(ids.map((i) => document.getElementById(i)?.textContent)).toEqual(["Held in memory", "Missing token"]);
  });

  it("invalid without a message only sets aria-invalid (the message lives elsewhere, e.g. a page alert)", () => {
    render(<Field label="Budget" invalid><Input /></Field>);
    const input = screen.getByLabelText("Budget");
    expect(input.getAttribute("aria-invalid")).toBe("true");
    expect(input.getAttribute("aria-describedby")).toBeNull();
  });

  it("works for Select and Textarea, and keeps a label note inside the label", () => {
    render(
      <>
        <Field label="Pack" note="(fixed)"><Select defaultValue="a"><option value="a">a</option></Select></Field>
        <Field label="Notes"><Textarea /></Field>
      </>,
    );
    expect(screen.getByLabelText("Pack (fixed)").tagName).toBe("SELECT");
    expect(screen.getByLabelText("Notes").tagName).toBe("TEXTAREA");
  });

  it("gives every control on a page a distinct id", () => {
    render(<><Field label="One"><Input /></Field><Field label="Two"><Input /></Field></>);
    expect(screen.getByLabelText("One").id).not.toBe(screen.getByLabelText("Two").id);
  });

  it("shows an adornment (unit) next to an input without changing its name", () => {
    render(<Field label="Budget cap (USD)"><Input suffix="USD" /></Field>);
    expect(screen.getByLabelText("Budget cap (USD)")).toBeTruthy();
    expect(screen.getByText("USD").getAttribute("aria-hidden")).toBe("true");
  });
});

describe("Switch", () => {
  function Demo() {
    const [on, setOn] = useState(false);
    return (
      <Field layout="inline" label="Dry run" help="Fake models">
        <Switch checked={on} onCheckedChange={setOn} />
      </Field>
    );
  }
  it("is a switch with the label's name, toggled by clicking the label or the control", () => {
    render(<Demo />);
    const sw = screen.getByRole("switch", { name: "Dry run" }) as HTMLInputElement;
    expect(sw.checked).toBe(false);
    fireEvent.click(screen.getByText("Dry run"));
    expect(sw.checked).toBe(true);
    fireEvent.click(sw);
    expect(sw.checked).toBe(false);
    expect(document.getElementById(sw.getAttribute("aria-describedby")!)?.textContent).toBe("Fake models");
  });
});

describe("Button and IconButton", () => {
  it("defaults to type=button and maps variants to classes", () => {
    render(<><Button>Plain</Button><Button variant="primary" type="submit">Go</Button></>);
    expect(screen.getByRole("button", { name: "Plain" }).getAttribute("type")).toBe("button");
    const go = screen.getByRole("button", { name: "Go" });
    expect(go.getAttribute("type")).toBe("submit");
    expect(go.className).toBe("btn primary");
  });
  it("loading disables the button and says it is busy", () => {
    render(<Button loading>Saving</Button>);
    const b = screen.getByRole("button", { name: "Saving" }) as HTMLButtonElement;
    expect(b.disabled).toBe(true);
    expect(b.getAttribute("aria-busy")).toBe("true");
  });
  it("IconButton takes its accessible name from label", () => {
    const onClick = vi.fn();
    render(<IconButton label="Keyboard shortcuts" onClick={onClick}>?</IconButton>);
    fireEvent.click(screen.getByRole("button", { name: "Keyboard shortcuts" }));
    expect(onClick).toHaveBeenCalledTimes(1);
  });
});

describe("Dialog", () => {
  it("is a named modal dialog that asks to close on Escape", () => {
    const onOpenChange = vi.fn();
    render(<Dialog open onOpenChange={onOpenChange} title="Keyboard shortcuts" description="Keys"><p>body</p></Dialog>);
    const d = screen.getByRole("dialog", { name: "Keyboard shortcuts" });
    expect(d.getAttribute("aria-describedby")).toBeTruthy();
    fireEvent.keyDown(d, { key: "Escape" });
    expect(onOpenChange).toHaveBeenCalledWith(false);
  });
  it("has a Close button and no dialog when closed", () => {
    const onOpenChange = vi.fn();
    const { rerender } = render(<Dialog open onOpenChange={onOpenChange} title="T"><p>b</p></Dialog>);
    fireEvent.click(screen.getByRole("button", { name: "Close" }));
    expect(onOpenChange).toHaveBeenCalledWith(false);
    rerender(<Dialog open={false} onOpenChange={onOpenChange} title="T"><p>b</p></Dialog>);
    expect(screen.queryByRole("dialog")).toBeNull();
  });
  it("returns focus to the element that had it when the dialog opened", async () => {
    function Host() {
      const [open, setOpen] = useState(false);
      return (
        <>
          <button onClick={() => setOpen(true)}>opener</button>
          <Dialog open={open} onOpenChange={setOpen} title="T"><p>b</p></Dialog>
        </>
      );
    }
    render(<Host />);
    const opener = screen.getByRole("button", { name: "opener" });
    opener.focus();
    fireEvent.click(opener);
    const d = screen.getByRole("dialog");
    fireEvent.keyDown(d, { key: "Escape" });
    expect(screen.queryByRole("dialog")).toBeNull();
    await waitFor(() => expect(document.activeElement).toBe(opener)); // Radix restores focus on the next tick
  });
});

describe("Tooltip", () => {
  it("shows its content on keyboard focus of the trigger and keeps the trigger's own name", async () => {
    render(<Tooltip content="Explains it"><button>Trigger</button></Tooltip>);
    fireEvent.focus(screen.getByRole("button", { name: "Trigger" }));
    expect((await screen.findByRole("tooltip")).textContent).toBe("Explains it");
  });
});

describe("Card, Stat, Badge, Skeleton, Toolbar, Spinner keep the DOM screens rely on", () => {
  it("renders the same markup the screens and tests already use", () => {
    const { container } = render(
      <>
        <Card title="T" className="lift">body</Card>
        <Stat label="L" value="V" hint="H" />
        <Badge tone="ok">ok</Badge>
        <Badge>plain</Badge>
      </>,
    );
    expect(container.querySelector("section.card.lift > h3")!.textContent).toBe("T");
    expect(container.querySelector(".stat .v")!.textContent).toBe("V");
    expect(container.querySelector(".stat .l.eyebrow")!.textContent).toBe("L");
    expect(container.querySelector(".stat .h")!.textContent).toBe("H");
    expect(container.querySelector(".badge.ok")!.textContent).toBe("ok");
    expect(container.querySelector(".badge:not(.ok)")!.className).toBe("badge");
  });
  it("Spinner text is exactly its label, and it is a busy status with skeleton lines", () => {
    const { container } = render(<Spinner label="Loading report" />);
    const s = screen.getByRole("status");
    expect(s.textContent).toBe("Loading report");
    expect(s.getAttribute("aria-busy")).toBe("true");
    expect(container.querySelectorAll(".sk-line").length).toBe(3);
  });
  it("Skeleton is decorative and Toolbar is a labelled group", () => {
    const { container } = render(<><Skeleton /><Toolbar label="Zoom"><button>a</button></Toolbar></>);
    expect(container.querySelector(".sk-line")!.getAttribute("aria-hidden")).toBe("true");
    expect(screen.getByRole("group", { name: "Zoom" })).toBeTruthy();
  });
});
