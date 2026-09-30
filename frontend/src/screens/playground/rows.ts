// The playground's rows_preview arrives in more than one shape. The server today sends an object,
// {columns, rows, row_count}; the API contract only says "rows_preview|null", and api/types.ts describes
// arrays. Both are accepted here. A shape that is none of these is reported as such (null), never guessed at.

export type Cell = unknown;
export interface Rows {
  columns: string[];
  rows: Cell[][];
  /** row_count as the server gave it (the number of rows the query returned, of which `rows` is a preview); null when not given. */
  total: number | null;
}

const isRecord = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);

export function normalizeRows(v: unknown): Rows | null {
  if (Array.isArray(v)) {
    if (v.length === 0) return { columns: [], rows: [], total: null };
    if (v.every(Array.isArray)) {
      const width = Math.max(...(v as unknown[][]).map((r) => r.length));
      return { columns: Array.from({ length: width }, (_, i) => `col ${i + 1}`), rows: v as unknown[][], total: null };
    }
    if (v.every(isRecord)) {
      const columns = Object.keys(v[0] as Record<string, unknown>);
      return { columns, rows: (v as Record<string, unknown>[]).map((r) => columns.map((c) => r[c])), total: null };
    }
    return null;
  }
  if (isRecord(v) && Array.isArray(v.columns) && v.columns.every((c) => typeof c === "string") && Array.isArray(v.rows) && v.rows.every(Array.isArray)) {
    const total = typeof v.row_count === "number" && Number.isFinite(v.row_count) ? v.row_count : null;
    return { columns: v.columns as string[], rows: v.rows as unknown[][], total };
  }
  return null;
}

/** One cell as text. SQL NULL is `null`; anything that is not a plain value is shown as JSON, never as "[object Object]". */
export function cellText(v: Cell): { text: string; isNull: boolean } {
  if (v === null || v === undefined) return { text: "NULL", isNull: true };
  if (typeof v === "object") return { text: JSON.stringify(v), isNull: false };
  return { text: String(v), isNull: false };
}
