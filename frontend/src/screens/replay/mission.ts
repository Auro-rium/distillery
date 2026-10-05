// Pure mapping from API payloads to what the Mission page shows. Every value read here comes from a
// payload; nothing is computed except where a comment says "derived" (and the page labels it so).
// Payloads are treated as untrusted shapes: a wrong shape yields null, never a throw.
import { fmtInt, fmtNumber, fmtUsd } from "../../api/format";
import type { Evidence, FinetuneRecord, Gate, ProofItem, Report, StressEval } from "../../api/types";
import type { ProofStatus } from "../../charts";

const obj = (v: unknown): v is Record<string, unknown> => !!v && typeof v === "object" && !Array.isArray(v);
const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

/** Below this, a McNemar p-value is shown as "far below alpha" rather than as a rounded 0.000. */
export const P_SHOW_MIN = 1e-3;
/** Tokens per "per million tokens" price (unit conversion for the derived fine-tune price). */
export const PER_MILLION = 1e6;
/** At most this many proof cards (clutter budget). */
export const MAX_PROOFS = 6;
/** Significant decimals kept when a small payload value would otherwise round to zero. */
const SIG = 3;
const MIN_DIGITS = 3;

export function gateOf(r: unknown): Gate | null {
  if (!obj(r) || !obj(r.evaluation)) return null;
  const g = r.evaluation.gate;
  return obj(g) ? (g as unknown as Gate) : null;
}

export function finetuneOf(r: unknown): FinetuneRecord | null {
  if (!obj(r) || !Array.isArray(r.finetune)) return null;
  const f = r.finetune[0];
  return obj(f) ? (f as unknown as FinetuneRecord) : null;
}

export function stressOf(r: unknown): StressEval | null {
  if (!obj(r) || !obj(r.evaluation)) return null;
  const s = r.evaluation.stress;
  return obj(s) && obj(s.accuracy_by_family) ? (s as unknown as StressEval) : null;
}

export function reportData(r: unknown): Partial<Report["data"]> | null {
  return obj(r) && obj(r.data) ? (r.data as Partial<Report["data"]>) : null;
}

export function costOf(r: unknown): Partial<Report["cost"]> | null {
  return obj(r) && obj(r.cost) ? (r.cost as Partial<Report["cost"]>) : null;
}

/** Dev accuracy of the round the gate evaluated (candidate_round), if the report recorded it. */
export function candidateDevAcc(r: unknown): number | null {
  if (!obj(r) || !Array.isArray(r.rounds)) return null;
  const round = r.candidate_round;
  const hit = r.rounds.find((x): x is Record<string, unknown> => obj(x) && x.round === round);
  return hit && isNum(hit.dev_acc) ? hit.dev_acc : null;
}

/** One fine-tune cost line as the report wrote it (`cost.finetune_lines`, not in the Report type). */
export interface FinetuneLine { usd: number | null; units: number | null; basis: string | null; model: string | null; kind: string | null }

export function finetuneLines(r: unknown): FinetuneLine[] {
  const c = costOf(r) as Record<string, unknown> | null;
  if (!c || !Array.isArray(c.finetune_lines)) return [];
  return c.finetune_lines.filter(obj).map((l) => ({
    usd: isNum(l.usd) ? l.usd : null,
    units: isNum(l.units) ? l.units : null,
    basis: typeof l.basis === "string" ? l.basis : null,
    model: typeof l.model === "string" ? l.model : null,
    kind: typeof l.kind === "string" ? l.kind : null,
  }));
}

/** Derived: usd / units x one million. Absent or zero units give "not measured", never a number. */
export function perMillion(l: FinetuneLine): string {
  if (l.usd === null || l.units === null || l.units <= 0) return fmtUsd(null);
  return fmtUsd((l.usd / l.units) * PER_MILLION, 2);
}

/**
 * A payload number shown with enough decimals that a small value never reads as 0.000; trailing zeros
 * are dropped. Integers print as integers; strings pass through verbatim.
 */
export function fmtSig(v: unknown): string {
  if (typeof v === "string") return v;
  if (!isNum(v)) return fmtNumber(null);
  if (Number.isInteger(v)) return fmtInt(v);
  const lead = v === 0 ? 0 : -Math.floor(Math.log10(Math.abs(v)));
  const digits = Math.max(MIN_DIGITS, lead + SIG - 1);
  const s = fmtNumber(v, digits);
  return s.includes(".") ? s.replace(/0+$/, "").replace(/\.$/, "") : s;
}

/** Evidence verdict to card state. PARTIAL has no ProofCard state of its own (see the page). */
export function proofStatus(v: unknown): ProofStatus {
  return v === "PASS" ? "pass" : v === "PARTIAL" ? "partial" : v === "FAIL" ? "fail" : "unknown";
}

export function proofsOf(e: unknown): ProofItem[] | null {
  if (!obj(e) || !Array.isArray(e.proofs)) return null;
  return (e as unknown as Evidence).proofs.filter((p): p is ProofItem => obj(p) && typeof p.id === "string").slice(0, MAX_PROOFS);
}

/** Shortest form of a proof id for the card corner: "W1-P1.11" stays as is (it is the payload's id). */
export const rungOf = (p: ProofItem): string => p.id;

/** Hyperparameter chips worth showing, in this order, when present. */
export const HP_KEYS = ["lora_r", "lora_alpha", "learning_rate", "n_epochs", "batch_size"] as const;
export const HP_LABEL: Record<(typeof HP_KEYS)[number], string> = {
  lora_r: "LoRA rank",
  lora_alpha: "LoRA alpha",
  learning_rate: "learning rate",
  n_epochs: "epochs",
  batch_size: "batch size",
};

export function hpChips(f: FinetuneRecord | null): { key: string; label: string; value: string }[] {
  const hp = f && obj(f.hyperparameters) ? f.hyperparameters : null;
  if (!hp) return [];
  return HP_KEYS.filter((k) => hp[k] !== undefined && hp[k] !== null).map((k) => ({ key: k, label: HP_LABEL[k], value: fmtSig(hp[k]) }));
}
