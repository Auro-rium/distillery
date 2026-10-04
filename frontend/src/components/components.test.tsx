import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { LabelBanner, DRY_RUN_TEXT } from ".";

describe("LabelBanner", () => {
  it("shows the exact dry-run text", () => {
    render(<LabelBanner dry_run recorded={false} />);
    expect(screen.getByRole("status").textContent).toBe("DRY RUN — fake models, numbers are NOT results");
    expect(DRY_RUN_TEXT).toBe("DRY RUN — fake models, numbers are NOT results");
  });
  it("shows recorded label", () => {
    render(<LabelBanner dry_run={false} recorded recorded_at="2026-01-01" />);
    expect(screen.getByRole("status").textContent).toBe("Recorded run · 2026-01-01 · real Token Factory jobs");
  });
  it("labels an explicit live run", () => {
    render(<LabelBanner dry_run={false} recorded={false} />);
    expect(screen.getByRole("status").textContent).toBe("Live run, not a recorded replay");
  });
  it("labels a completed non-dry local run as completed, and keeps Live run only while running", () => {
    const { unmount } = render(<LabelBanner dry_run={false} recorded={false} status="complete" />);
    expect(screen.getByRole("status").textContent).toBe("Completed run (real Token Factory jobs)");
    unmount();
    for (const status of ["running", "pending"]) {
      const r = render(<LabelBanner dry_run={false} recorded={false} status={status} />);
      expect(screen.getByRole("status").textContent).toBe("Live run, not a recorded replay");
      r.unmount();
    }
  });
  it("never renders missing flags as real", () => {
    for (const props of [{}, { dry_run: null }, { dry_run: undefined, recorded: false }, { dry_run: false }]) {
      const { unmount } = render(<LabelBanner {...props} />);
      expect(screen.getByRole("status").textContent).toMatch(/^Label unknown/);
      unmount();
    }
  });
});
