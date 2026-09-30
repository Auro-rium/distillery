import { useCallback, useEffect, useRef } from "react";
import { PAN_STEP, ZOOM_MAX, ZOOM_MIN, ZOOM_STEP } from "../../motion/timing";

const DRAG_PX = 4;
const PAD = 16;
const clampK = (k: number) => Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, k));

/**
 * Pan and zoom of an SVG layer. The transform lives in a ref and is written straight to the DOM
 * (no React state, no per-frame renders). Wheel zooms only when the tree is larger than its
 * viewport (or with ctrl/cmd, which is also what a trackpad pinch sends), so page scrolling is
 * never hijacked for small trees. Drag pans; buttons and keys zoom, pan and fit.
 */
export function usePanZoom(content: { w: number; h: number }, resetKey: unknown) {
  const box = useRef<HTMLDivElement>(null);
  const layer = useRef<SVGGElement>(null);
  const v = useRef({ x: 0, y: 0, k: 1 });
  const moved = useRef(false);

  const apply = useCallback((smooth: boolean) => {
    const el = layer.current;
    if (!el) return;
    const { x, y, k } = v.current;
    el.classList.toggle("smooth", smooth);
    el.style.transform = `translate(${x}px, ${y}px) scale(${k})`;
  }, []);

  const zoomAt = useCallback((factor: number, cx: number, cy: number, smooth = true) => {
    const cur = v.current;
    const k = clampK(cur.k * factor);
    v.current = { k, x: cx - ((cx - cur.x) * k) / cur.k, y: cy - ((cy - cur.y) * k) / cur.k };
    apply(smooth);
  }, [apply]);

  const center = () => {
    const b = box.current;
    return { cx: (b?.clientWidth ?? 0) / 2, cy: (b?.clientHeight ?? 0) / 2 };
  };
  const zoomBy = useCallback((factor: number) => { const c = center(); zoomAt(factor, c.cx, c.cy); }, [zoomAt]);
  const panBy = useCallback((dx: number, dy: number) => { v.current = { ...v.current, x: v.current.x + dx, y: v.current.y + dy }; apply(true); }, [apply]);
  const fit = useCallback(() => {
    const b = box.current;
    const cw = b?.clientWidth ?? 0, ch = b?.clientHeight ?? 0;
    if (cw > 0 && ch > 0 && content.w > 0 && content.h > 0) {
      const k = clampK(Math.min(1, (cw - 2 * PAD) / content.w, (ch - 2 * PAD) / content.h));
      v.current = { k, x: PAD, y: PAD };
    } else v.current = { x: 0, y: 0, k: 1 };
    apply(true);
  }, [apply, content.w, content.h]);

  /** Pan just enough that a node at (x, y, w, h) in content coordinates is inside the viewport. */
  const reveal = useCallback((x: number, y: number, w: number, h: number) => {
    const b = box.current;
    if (!b || b.clientWidth === 0) return;
    const { x: ox, y: oy, k } = v.current;
    let dx = 0, dy = 0;
    if (x * k + ox < PAD) dx = PAD - (x * k + ox);
    else if ((x + w) * k + ox > b.clientWidth - PAD) dx = b.clientWidth - PAD - ((x + w) * k + ox);
    if (y * k + oy < PAD) dy = PAD - (y * k + oy);
    else if ((y + h) * k + oy > b.clientHeight - PAD) dy = b.clientHeight - PAD - ((y + h) * k + oy);
    if (dx || dy) panBy(dx, dy);
  }, [panBy]);

  // Start fitted when the tree is wider than the viewport, otherwise at natural size.
  useEffect(() => {
    const b = box.current;
    if (b && b.clientWidth > 0 && (content.w > b.clientWidth || content.h > b.clientHeight)) fit();
    else { v.current = { x: 0, y: 0, k: 1 }; apply(false); }
  }, [resetKey, content.w, fit, apply]);

  useEffect(() => {
    const el = box.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      const larger = content.w * v.current.k > el.clientWidth || content.h * v.current.k > el.clientHeight;
      if (!larger && !e.ctrlKey && !e.metaKey) return; // let the page scroll
      e.preventDefault();
      const r = el.getBoundingClientRect();
      zoomAt(e.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP, e.clientX - r.left, e.clientY - r.top, false);
    };
    let start: { x: number; y: number; id: number; ox: number; oy: number } | null = null;
    const onDown = (e: PointerEvent) => {
      if (e.button !== 0) return;
      moved.current = false;
      start = { x: e.clientX, y: e.clientY, id: e.pointerId, ox: v.current.x, oy: v.current.y };
    };
    const onMove = (e: PointerEvent) => {
      if (!start || e.pointerId !== start.id) return;
      const dx = e.clientX - start.x, dy = e.clientY - start.y;
      if (!moved.current && Math.hypot(dx, dy) < DRAG_PX) return;
      if (!moved.current) { moved.current = true; el.setPointerCapture?.(e.pointerId); el.classList.add("dragging"); }
      v.current = { ...v.current, x: start.ox + dx, y: start.oy + dy };
      apply(false);
    };
    const onUp = () => { start = null; el.classList.remove("dragging"); };
    // a drag must not count as a click on the node under the pointer
    const onClick = (e: MouseEvent) => { if (moved.current) { e.stopPropagation(); e.preventDefault(); moved.current = false; } };
    el.addEventListener("wheel", onWheel, { passive: false });
    el.addEventListener("pointerdown", onDown);
    el.addEventListener("pointermove", onMove);
    el.addEventListener("pointerup", onUp);
    el.addEventListener("pointercancel", onUp);
    el.addEventListener("click", onClick, true);
    return () => {
      el.removeEventListener("wheel", onWheel);
      el.removeEventListener("pointerdown", onDown);
      el.removeEventListener("pointermove", onMove);
      el.removeEventListener("pointerup", onUp);
      el.removeEventListener("pointercancel", onUp);
      el.removeEventListener("click", onClick, true);
    };
  }, [content.w, content.h, zoomAt, apply]);

  /** Keys: + - zoom, 0 fits (anywhere in the viewport); arrows pan only when the viewport itself has focus. */
  const onKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === "+" || e.key === "=") { e.preventDefault(); zoomBy(ZOOM_STEP); }
    else if (e.key === "-" || e.key === "_") { e.preventDefault(); zoomBy(1 / ZOOM_STEP); }
    else if (e.key === "0") { e.preventDefault(); fit(); }
    else if (e.target === e.currentTarget) {
      const d: Record<string, [number, number]> = { ArrowLeft: [PAN_STEP, 0], ArrowRight: [-PAN_STEP, 0], ArrowUp: [0, PAN_STEP], ArrowDown: [0, -PAN_STEP] };
      if (d[e.key]) { e.preventDefault(); panBy(...d[e.key]); }
    }
  }, [zoomBy, fit, panBy]);

  return { box, layer, zoomBy, fit, reveal, onKeyDown };
}
