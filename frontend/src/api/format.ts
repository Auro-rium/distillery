// Display helpers. They format numbers the backend sent; they never invent one.
// Absent data (null / undefined / NaN / Infinity) is the literal text NOT_MEASURED, never 0, 0%,
// $0.00, blank or "-". Backend strings (e.g. "unavailable: serving path undecided") pass through
// verbatim.
export const NOT_MEASURED = "not measured";

type Val = number | string | null | undefined;

const num = (v: number): boolean => Number.isFinite(v);

/** Fixed decimals (display rounding only). */
export function fmtNumber(v: Val, digits = 3): string {
  if (typeof v === "string") return v;
  if (v === null || v === undefined || !num(v)) return NOT_MEASURED;
  return v.toFixed(digits);
}

/** `v` is a fraction (0.7); shown as a percentage. The x100 is display scaling only. */
export function fmtPercent(v: Val, digits = 1): string {
  if (typeof v === "string") return v;
  if (v === null || v === undefined || !num(v)) return NOT_MEASURED;
  return `${(v * 100).toFixed(digits)}%`;
}

export function fmtUsd(v: Val, digits = 4): string {
  if (typeof v === "string") return v;
  if (v === null || v === undefined || !num(v)) return NOT_MEASURED;
  return `$${v.toFixed(digits)}`;
}

export function fmtInt(v: Val): string {
  if (typeof v === "string") return v;
  if (v === null || v === undefined || !num(v)) return NOT_MEASURED;
  return String(Math.round(v));
}

/** Plain text that may be absent (timestamps, ids, free text). */
export function fmtText(v: string | null | undefined): string {
  return v === null || v === undefined || v === "" ? NOT_MEASURED : v;
}
