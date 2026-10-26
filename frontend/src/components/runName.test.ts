import { describe, expect, it } from "vitest";
import { packLabel, runSentence, runTitle } from "./runName";

describe("plain-words run names", () => {
  it("names the task from a known pack at the start of the id, with or without the dry prefix", () => {
    expect(packLabel("sql-gated-20261004T155134Z-66ff347c")).toBe("SQL");
    expect(packLabel("dry-sql-tiny-24932bcc")).toBe("SQL");
    expect(runTitle("sql-gated-20261004T155134Z-66ff347c")).toEqual({ pack: "SQL", rest: " run" });
  });
  it("never guesses: an id that does not start with a known pack gets the neutral wording", () => {
    for (const id of ["rec-1", "demo-9", "", "gated-sql"]) {
      expect(packLabel(id), id).toBeNull();
      expect(runTitle(id)).toEqual({ pack: null, rest: "Run" });
      expect(runSentence(id, false)).toBe("One test of the small model.");
    }
  });
  it("says in one sentence what the page is, newest or not", () => {
    expect(runSentence("sql-gated-1", true)).toBe("The newest recorded test of the small model on SQL questions.");
    expect(runSentence("sql-gated-1", false)).toBe("One test of the small model on SQL questions.");
  });
  it("the sentences carry no figures", () => {
    for (const s of [runSentence("sql-x", true), runSentence("sql-x", false), runSentence("zzz", true)]) expect(s).not.toMatch(/\d/);
  });
});
