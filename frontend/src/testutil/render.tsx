import { render } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { vi } from "vitest";
import { App } from "../App";

/** Render the whole app at `path` (real router, real screens). */
export function renderApp(path: string) {
  return render(<MemoryRouter initialEntries={[path]}><App /></MemoryRouter>);
}

export const json = (b: unknown, status = 200, h: Record<string, string> = {}) =>
  new Response(JSON.stringify(b), { status, headers: h });

export function stubFetch(f: (url: string, init?: RequestInit) => Promise<Response> | Response) {
  const fn = vi.fn((url: string, init?: RequestInit) => Promise.resolve(f(String(url), init)));
  vi.stubGlobal("fetch", fn);
  return fn;
}

/** A minimal EventSource stand-in that records instances. */
export class FakeES {
  static all: FakeES[] = [];
  static CLOSED = 2;
  readyState = 1;
  onopen: (() => void) | null = null;
  onerror: (() => void) | null = null;
  ls = new Map<string, (e: MessageEvent) => void>();
  constructor(public url: string) { FakeES.all.push(this); }
  addEventListener(t: string, f: (e: MessageEvent) => void) { this.ls.set(t, f); }
  close() { this.readyState = 2; }
  emit(type: string, id: number, data: object) {
    this.ls.get(type)!(new MessageEvent(type, { data: JSON.stringify({ observed_at: "seen", ...data }), lastEventId: String(id) }));
  }
}
