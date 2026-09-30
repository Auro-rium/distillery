import { describe, expect, it } from "vitest";
import { diffAgainstGold } from "./diff";

const join = (s: { text: string }[]) => s.map((x) => x.text).join("");
const changedText = (s: { text: string; changed: boolean }[]) => s.filter((x) => x.changed).map((x) => x.text.trim());

describe("diffAgainstGold", () => {
  it("joining each side's segments gives back exactly the input string", () => {
    const gold = "SELECT name,\n  COUNT(*) AS n\nFROM t  WHERE x = 1";
    const other = "SELECT name, COUNT(*) FROM t WHERE x = 2 ORDER BY n";
    const d = diffAgainstGold(gold, other)!;
    expect(join(d.gold)).toBe(gold);
    expect(join(d.other)).toBe(other);
  });
  it("identical strings produce no marks", () => {
    const d = diffAgainstGold("SELECT 1", "SELECT 1")!;
    expect(d.identical).toBe(true);
    expect([...d.gold, ...d.other].some((s) => s.changed)).toBe(false);
  });
  it("whitespace-only differences are not marked", () => {
    const d = diffAgainstGold("SELECT  a\nFROM t", "SELECT a FROM t")!;
    expect(d.identical).toBe(true);
  });
  it("marks only tokens outside the common sequence, on the side they belong to", () => {
    const d = diffAgainstGold("SELECT a FROM t WHERE x = 1", "SELECT a FROM t WHERE x = 2 LIMIT 3")!;
    expect(changedText(d.gold)).toEqual(["1"]);
    expect(changedText(d.other)).toEqual(["2 LIMIT 3"]);
    expect(d.identical).toBe(false);
  });
  it("every marked token exists in the string it came from (nothing invented)", () => {
    const gold = "SELECT c FROM orders JOIN users ON a=b";
    const other = "SELECT c FROM users";
    const d = diffAgainstGold(gold, other)!;
    for (const t of changedText(d.gold).flatMap((x) => x.split(/\s+/))) expect(gold.split(/\s+/)).toContain(t);
    for (const t of changedText(d.other).flatMap((x) => x.split(/\s+/))) expect(other.split(/\s+/)).toContain(t);
    expect(changedText(d.gold).join(" ")).toContain("orders");
  });
  it("handles empty strings and gives up (null) on huge inputs instead of freezing", () => {
    expect(join(diffAgainstGold("", "SELECT 1")!.other)).toBe("SELECT 1");
    const big = Array.from({ length: 3000 }, (_, i) => `w${i}`).join(" ");
    expect(diffAgainstGold(big, big + " x")).toBeNull();
  });
});
