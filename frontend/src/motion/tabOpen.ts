// "Opens out of the button": when a run tab is clicked, the page that replaces the old one is revealed through a window
// that starts as a strip under that tab and widens to the whole page, like an app opening from its icon. The click
// leaves the tab's box here; the new page reads it once on mount. Position only, never shown. Skipped under reduced
// motion, and ignored if the click was not just now.

export interface TabBox { x: number; y: number; left: number; right: number; top: number; bottom: number }

let origin: (TabBox & { at: number }) | null = null;

const MAX_AGE_MS = 1500;

/** Remember where a tab was clicked (its box and centre, in viewport coordinates). */
export function rememberTabOrigin(el: Element): void {
  const r = el.getBoundingClientRect();
  origin = { x: r.left + r.width / 2, y: r.top + r.height / 2, left: r.left, right: r.right, top: r.top, bottom: r.bottom, at: Date.now() };
}

/** The remembered click box, once; null if there is none or it is stale. */
export function takeTabOrigin(): TabBox | null {
  const o = origin;
  origin = null;
  if (!o || Date.now() - o.at > MAX_AGE_MS) return null;
  const { at: _at, ...box } = o;
  return box;
}
