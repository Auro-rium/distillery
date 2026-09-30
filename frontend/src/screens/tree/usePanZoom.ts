import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { PAN_STEP, ZOOM_MAX, ZOOM_STEP } from "../../motion/timing";
import { FIT_PAD, READABLE_MIN, ZOOM_FLOOR } from "./constants";

/** A rectangle in content coordinates. */
export interface Rect { x: number; y: number; w: number; h: number }
/** What a fit shows: everything, the path to the selected node, or (automatic) everything when it is readable and the path otherwise. */
export type FitMode = "all" | "path" | "auto";

const DRAG_PX = 4;
const clampK = (k: number) => Math.min(ZOOM_MAX, Math.max(ZOOM_FLOOR, k));

/** Size of an element, kept up to date by a ResizeObserver (0 x 0 until it is mounted or where there is no layout). */
export function useElementSize(el: HTMLElement | null): { w: number; h: number } {
  const [size, setSize] = useState({ w: 0, h: 0 });
  useLayoutEffect(() => {
    if (!el) return;
    const read = () => setSize((s) => (s.w === el.clientWidth && s.h === el.clientHeight ? s : { w: el.clientWidth, h: el.clientHeight }));
    read();
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(read);
    ro.observe(el);
    return () => ro.disconnect();
  }, [el]);
  return size;
}

/**
 * Pan and zoom of a layer inside a viewport element. The transform lives in a ref and is written straight to
 * the DOM (no React state, no per-frame renders).
 *
 * Auto-fit: on load, when the tree or its orientation changes, and whenever the viewport is resized (until the
 * user pans or zooms), the whole tree is fitted into the viewport, centred, never above natural size. If that would
 * shrink it below READABLE_MIN the view is instead framed on `focus` (the boxes of the path from the root to the
 * selected node), at a readable zoom, ending at the selected node. The Fit button and the 0 key always show everything.
 *
 * Wheel zooms only when the tree is larger than its viewport (or with ctrl/cmd, which is also what a trackpad
 * pinch sends), so page scrolling is never hijacked for small trees. Drag pans; buttons and keys zoom, pan and fit.
 */
export function usePanZoom(el: HTMLDivElement | null, content: { w: number; h: number }, resetKey: unknown, focus: Rect[] = []) {
  const layer = useRef<HTMLDivElement>(null);
  const v = useRef({ x: 0, y: 0, k: 1 });
  const moved = useRef(false);
  const touched = useRef(false); // the user has taken control since the last fit
  const focusRef = useRef(focus); // read when a fit runs; a change of selection alone must not move the view
  focusRef.current = focus;

  const apply = useCallback((smooth: boolean) => {
    const node = layer.current;
    if (!node) return;
    const { x, y, k } = v.current;
    node.classList.toggle("smooth", smooth);
    node.style.transform = `translate(${x}px, ${y}px) scale(${k})`;
  }, []);

  const zoomAt = useCallback((factor: number, cx: number, cy: number, smooth = true) => {
    touched.current = true;
    const cur = v.current;
    const k = clampK(cur.k * factor);
    v.current = { k, x: cx - ((cx - cur.x) * k) / cur.k, y: cy - ((cy - cur.y) * k) / cur.k };
    apply(smooth);
  }, [apply]);

  const zoomBy = useCallback((factor: number) => {
    zoomAt(factor, (el?.clientWidth ?? 0) / 2, (el?.clientHeight ?? 0) / 2);
  }, [zoomAt, el]);
  const panBy = useCallback((dx: number, dy: number) => {
    touched.current = true;
    v.current = { ...v.current, x: v.current.x + dx, y: v.current.y + dy };
    apply(true);
  }, [apply]);

  const fit = useCallback((smooth = true, mode: FitMode = "all") => {
    touched.current = false;
    const cw = el?.clientWidth ?? 0, ch = el?.clientHeight ?? 0;
    const room = (extent: number) => extent - 2 * FIT_PAD;
    const kFit = (w: number, h: number) => Math.min(1, room(cw) / w, room(ch) / h);
    if (cw > 0 && ch > 0 && content.w > 0 && content.h > 0) {
      const kAll = clampK(kFit(content.w, content.h));
      const boxes = focusRef.current;
      if (boxes.length > 0 && (mode === "path" || (mode === "auto" && kAll < READABLE_MIN))) {
        const x0 = Math.min(...boxes.map((b) => b.x)), y0 = Math.min(...boxes.map((b) => b.y));
        const x1 = Math.max(...boxes.map((b) => b.x + b.w)), y1 = Math.max(...boxes.map((b) => b.y + b.h));
        const last = boxes[boxes.length - 1];
        const k = clampK(Math.max(READABLE_MIN, kFit(x1 - x0, y1 - y0)));
        // On an axis where the path fits it is centred. Where it does not, the selected node is kept in view at the edge
        // the rest of the path leads away from (the far edge when the path comes from before it, the near edge otherwise).
        const place = (start: number, end: number, extent: number, lastStart: number, lastEnd: number) => {
          if ((end - start) * k <= room(extent)) return (extent - (end - start) * k) / 2 - start * k;
          return (lastStart + lastEnd) / 2 >= (start + end) / 2 ? extent - FIT_PAD - lastEnd * k : FIT_PAD - lastStart * k;
        };
        v.current = { k, x: place(x0, x1, cw, last.x, last.x + last.w), y: place(y0, y1, ch, last.y, last.y + last.h) };
      } else v.current = { k: kAll, x: (cw - content.w * kAll) / 2, y: (ch - content.h * kAll) / 2 };
    } else v.current = { x: 0, y: 0, k: 1 }; // no layout to fit into (not laid out yet)
    apply(smooth);
  }, [apply, el, content.w, content.h]);

  /** Pan just enough that a node at (x, y, w, h) in content coordinates is inside the viewport. */
  const reveal = useCallback((x: number, y: number, w: number, h: number) => {
    if (!el || el.clientWidth === 0) return;
    const { x: ox, y: oy, k } = v.current;
    let dx = 0, dy = 0;
    if (x * k + ox < FIT_PAD) dx = FIT_PAD - (x * k + ox);
    else if ((x + w) * k + ox > el.clientWidth - FIT_PAD) dx = el.clientWidth - FIT_PAD - ((x + w) * k + ox);
    if (y * k + oy < FIT_PAD) dy = FIT_PAD - (y * k + oy);
    else if ((y + h) * k + oy > el.clientHeight - FIT_PAD) dy = el.clientHeight - FIT_PAD - ((y + h) * k + oy);
    if (dx || dy) panBy(dx, dy);
  }, [panBy, el]);

  // Fit on load, and when the tree (or its orientation, which changes its size) changes.
  useLayoutEffect(() => { fit(false, "auto"); }, [resetKey, fit]);
  // Re-fit when the viewport is resized, unless the user has taken control.
  useEffect(() => {
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => { if (!touched.current) fit(false, "auto"); });
    ro.observe(el);
    return () => ro.disconnect();
  }, [el, fit]);

  useEffect(() => {
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
      if (!moved.current) { moved.current = true; touched.current = true; el.setPointerCapture?.(e.pointerId); el.classList.add("dragging"); }
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
  }, [el, content.w, content.h, zoomAt, apply]);

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

  const showPath = useCallback(() => fit(true, "path"), [fit]);
  return { layer, zoomBy, fit, showPath, reveal, onKeyDown };
}
