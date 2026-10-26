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

/** A p-value: fixed decimals, or scientific notation when that would round a real value to 0. */
export function fmtP(v: Val, digits = 4): string {
  if (typeof v === "number" && num(v) && v > 0 && v < 10 ** -digits) return v.toExponential(1);
  return fmtNumber(v, digits);
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

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** An ISO timestamp as "4 Oct 2026, 18:54:32 (UTC)". Absent or unreadable is "not measured", never a made-up time. */
export function fmtDateTimeUtc(iso: string | null | undefined): string {
  if (!iso) return NOT_MEASURED;
  const t = new Date(iso);
  if (Number.isNaN(t.getTime())) return NOT_MEASURED;
  const p = (n: number) => String(n).padStart(2, "0");
  return `${t.getUTCDate()} ${MONTHS[t.getUTCMonth()]} ${t.getUTCFullYear()}, ${p(t.getUTCHours())}:${p(t.getUTCMinutes())}:${p(t.getUTCSeconds())} (UTC)`;
}

/** Plain text that may be absent (timestamps, ids, free text). */
export function fmtText(v: string | null | undefined): string {
  return v === null || v === undefined || v === "" ? NOT_MEASURED : v;
}
