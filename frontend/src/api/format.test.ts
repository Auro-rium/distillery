import { describe, expect, it } from "vitest";
import { fmtInt, fmtNumber, fmtP, fmtPercent, fmtText, fmtUsd } from "./format";

const ABSENT = [null, undefined, NaN, Infinity];
describe("format helpers", () => {
  it.each([fmtNumber, fmtPercent, fmtUsd, fmtInt])("absent values read 'not measured'", (f) => {
    for (const v of ABSENT) expect(f(v)).toBe("not measured");
  });
  it("never renders zero-ish placeholders for absent data", () => {
    for (const f of [fmtNumber, fmtPercent, fmtUsd, fmtInt, fmtText])
      for (const v of [null, undefined]) expect(f(v as never)).not.toMatch(/^(0|0%|\$0\.0*|-|)$/);
  });
  it("real zero is shown as zero", () => {
    expect(fmtPercent(0)).toBe("0.0%");
    expect(fmtUsd(0)).toBe("$0.0000");
    expect(fmtInt(0)).toBe("0");
  });
  it("formats", () => {
    expect(fmtNumber(0.85123)).toBe("0.851");
    expect(fmtPercent(0.7)).toBe("70.0%");
    expect(fmtUsd(1.11195, 5)).toBe("$1.11195");
    expect(fmtInt(20325)).toBe("20325");
  });
  it("passes backend strings through verbatim", () => {
    const s = "unavailable: serving path undecided (spike S4)";
    for (const f of [fmtNumber, fmtPercent, fmtUsd, fmtInt]) expect(f(s)).toBe(s);
  });
});

describe("fmtP", () => {
  it("keeps fixed decimals for ordinary p-values and never rounds a real tiny p to zero", () => {
    expect(fmtP(0.0215)).toBe("0.0215");
    expect(fmtP(1.6e-57)).toBe("1.6e-57");
    expect(fmtP(0)).toBe("0.0000");
    expect(fmtP(null)).toBe(fmtNumber(null));
  });
});
