import { describe, expect, it } from "vitest";
import { answerLanguage } from "./packs";

describe("answerLanguage", () => {
  it("maps each pack to its code-block language", () => {
    expect(answerLanguage("sql")).toBe("sql");
    expect(answerLanguage("toolcall")).toBe("json");
    expect(answerLanguage(undefined)).toBe("sql");
    expect(answerLanguage("mystery")).toBe("text");
  });
});
