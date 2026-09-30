import type {
  Config, ExampleKind, ExamplesResult, Health, NewRunBody, PlaygroundResponse, Report, RunDetail,
  RunSummary, Tree,
} from "./types";

export class ApiError extends Error {
  constructor(
    public status: number,
    public code: string,
    message: string,
    public retryAfter: number | null = null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

// API origin. Empty (default) = same origin. Set VITE_API_BASE at build time when the frontend is
// hosted apart from the API (Vercel + Nebius). It is public: never put a token in it.
const API_BASE = ((import.meta.env.VITE_API_BASE as string | undefined) ?? "").trim().replace(/\/+$/, "");

/** Absolute-or-relative URL for an API path that already starts with "/api", e.g. "/api/health".
 * Use it for every fetch and EventSource so they all follow VITE_API_BASE. */
export const apiUrl = (path: string): string => `${API_BASE}${path}`;

// Admin token lives in memory only. Never persisted.
let adminToken: string | null = null;
export const setAdminToken = (t: string | null): void => {
  adminToken = t && t.length > 0 ? t : null;
};
export const hasAdminToken = (): boolean => adminToken !== null;

function examplesMeta(h: Headers): Omit<ExamplesResult, "items"> {
  const avail = h.get("X-Examples-Available");
  const cap = Number(h.get("X-Examples-Cap-Per-Kind"));
  let totals: ExamplesResult["totals"] = null;
  try {
    const t = JSON.parse(h.get("X-Examples-Totals") ?? "null") as Record<string, unknown> | null;
    const n = (v: unknown) => (typeof v === "number" && Number.isFinite(v) ? v : null);
    if (t) totals = { fixed: n(t.fixed), still_wrong: n(t.still_wrong), regressed: n(t.regressed) };
  } catch {
    /* malformed header: totals stay absent */
  }
  return {
    available: avail === null ? null : avail === "true",
    capPerKind: h.get("X-Examples-Cap-Per-Kind") !== null && Number.isFinite(cap) ? cap : null,
    totals,
  };
}

async function request<T>(path: string, init: RequestInit = {}, admin = false): Promise<T> {
  return (await requestRaw<T>(path, init, admin)).data;
}

async function requestRaw<T>(path: string, init: RequestInit = {}, admin = false): Promise<{ data: T; headers: Headers }> {
  const headers = new Headers(init.headers);
  if (init.body !== undefined) headers.set("Content-Type", "application/json");
  if (admin && adminToken) headers.set("X-Admin-Token", adminToken);
  let res: Response;
  try {
    res = await fetch(apiUrl(`/api${path}`), { ...init, headers });
  } catch {
    throw new ApiError(0, "network", "Cannot reach the API server.");
  }
  if (!res.ok) {
    let code = `http_${res.status}`;
    let message = res.statusText || "Request failed";
    try {
      const b = (await res.json()) as Partial<{ error: string; message: string }>;
      if (b.error) code = b.error;
      if (b.message) message = b.message;
    } catch {
      /* body was not JSON */
    }
    const ra = Number(res.headers.get("Retry-After"));
    throw new ApiError(res.status, code, message, Number.isFinite(ra) && ra > 0 ? ra : null);
  }
  return { data: (await res.json()) as T, headers: res.headers };
}

const post = (body: unknown): RequestInit => ({ method: "POST", body: JSON.stringify(body) });

export const api = {
  health: () => request<Health>("/health"),
  config: () => request<Config>("/config"),
  runs: () => request<RunSummary[]>("/runs"),
  replay: () => request<RunSummary[]>("/replay"),
  run: (id: string) => request<RunDetail>(`/runs/${encodeURIComponent(id)}`),
  report: (id: string) => request<Report>(`/runs/${encodeURIComponent(id)}/report`),
  tree: (id: string) => request<Tree>(`/runs/${encodeURIComponent(id)}/tree`),
  examples: async (id: string, kind: ExampleKind = "all", limit = 20): Promise<ExamplesResult> => {
    const { data, headers } = await requestRaw<ExamplesResult["items"]>(
      `/runs/${encodeURIComponent(id)}/examples?kind=${kind}&limit=${limit}`,
    );
    return { items: data, ...examplesMeta(headers) };
  },
  createRun: (body: NewRunBody) => request<{ run_id: string }>("/runs", post(body), true),
  cancelRun: (id: string) =>
    request<{ status: string }>(`/runs/${encodeURIComponent(id)}/cancel`, { method: "POST" }, true),
  playground: (question: string) =>
    request<PlaygroundResponse>("/playground", post({ question })),
};
