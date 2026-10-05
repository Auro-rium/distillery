/** Code language of a pack's answers, for the code block. Unknown packs fall back to plain text. */
const LANGUAGE: Record<string, string> = { sql: "sql", toolcall: "json" };

export const answerLanguage = (pack: string | undefined): string => LANGUAGE[pack ?? "sql"] ?? "text";
