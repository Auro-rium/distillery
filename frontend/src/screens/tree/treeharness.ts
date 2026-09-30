// Test helpers for the tree screens (not shipped: only *.test files import this).
import { render } from "@testing-library/react";
import { createElement } from "react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { act } from "@testing-library/react";
import { vi } from "vitest";
import type { RunDetail, TreeNode } from "../../api/types";
import Tree from ".";

export const n = (id: string, parent: string | null, extra: Partial<TreeNode> = {}): TreeNode => ({
  id, parent_id: parent, label: id, round: null, hypothesis: null, data_delta: null,
  dev_score: null, cost_usd: null, sandbox_image: null, selected: false, ...extra,
});

export const RUN = { run_id: "r", dry_run: true, recorded: false, recorded_at: null, status: "complete" } as unknown as RunDetail;

export function show(nodes: TreeNode[], run: unknown = RUN, opts: { runFails?: boolean } = {}) {
  vi.stubGlobal("fetch", vi.fn((u: string) => {
    if (u.endsWith("/tree")) return Promise.resolve(new Response(JSON.stringify({ nodes }), { status: 200 }));
    if (u.endsWith("/report")) return Promise.resolve(new Response(JSON.stringify({ decision: "REJECT" }), { status: 200 }));
    if (opts.runFails) return Promise.resolve(new Response(JSON.stringify({ error: "run_down", message: "run unavailable" }), { status: 500 }));
    return Promise.resolve(new Response(JSON.stringify(run), { status: 200 }));
  }));
  return render(createElement(MemoryRouter, { initialEntries: ["/runs/r/tree"] },
    createElement(Routes, null, createElement(Route, { path: "/runs/:id/tree", element: createElement(Tree) }))));
}

/** Make the tree viewport report a size (jsdom has no layout) and let tests fire a resize. */
export function fakeViewport(initial: { w: number; h: number }) {
  const size = { ...initial };
  const desc = (p: string) => Object.getOwnPropertyDescriptor(HTMLElement.prototype, p) ?? Object.getOwnPropertyDescriptor(Element.prototype, p);
  const saved = { w: desc("clientWidth"), h: desc("clientHeight") };
  Object.defineProperty(HTMLElement.prototype, "clientWidth", { configurable: true, get(this: HTMLElement) { return this.classList.contains("pz-box") ? size.w : 0; } });
  Object.defineProperty(HTMLElement.prototype, "clientHeight", { configurable: true, get(this: HTMLElement) { return this.classList.contains("pz-box") ? size.h : 0; } });
  const observers: Array<{ cb: ResizeObserverCallback; el: Element | null }> = [];
  class RO {
    el: Element | null = null;
    constructor(public cb: ResizeObserverCallback) { observers.push(this); }
    observe(el: Element) { this.el = el; }
    unobserve() {}
    disconnect() { const i = observers.indexOf(this); if (i >= 0) observers.splice(i, 1); }
  }
  vi.stubGlobal("ResizeObserver", RO);
  return {
    resize(w: number, h: number) {
      size.w = w; size.h = h;
      act(() => { for (const o of [...observers]) o.cb([], o as unknown as ResizeObserver); });
    },
    restore() {
      if (saved.w) Object.defineProperty(HTMLElement.prototype, "clientWidth", saved.w); else delete (HTMLElement.prototype as unknown as Record<string, unknown>).clientWidth;
      if (saved.h) Object.defineProperty(HTMLElement.prototype, "clientHeight", saved.h); else delete (HTMLElement.prototype as unknown as Record<string, unknown>).clientHeight;
    },
  };
}

/** The layer's current transform as numbers. */
export function view(layer: Element) {
  const m = /translate\((-?[\d.]+)px,\s*(-?[\d.]+)px\)\s*scale\((-?[\d.]+)\)/.exec((layer as HTMLElement).style.transform);
  if (!m) throw new Error(`no transform: ${(layer as HTMLElement).style.transform}`);
  return { x: Number(m[1]), y: Number(m[2]), k: Number(m[3]) };
}

/** Left/top of a node button in content pixels. */
export const pos = (el: HTMLElement) => ({ x: parseFloat(el.style.left), y: parseFloat(el.style.top) });
