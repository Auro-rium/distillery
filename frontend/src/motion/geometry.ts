/**
 * Bar geometry only. Turns a real value into a 0..1 scale factor for a CSS transform. The
 * result is never displayed as text; every label still shows the payload value itself.
 * Returns null when either number is absent or the reference is not positive, so no bar is drawn.
 */
export function frac(value: number | null | undefined, of: number | null | undefined): number | null {
  if (typeof value !== "number" || typeof of !== "number") return null;
  if (!Number.isFinite(value) || !Number.isFinite(of) || of <= 0) return null;
  return Math.min(1, Math.max(0, value / of));
}
