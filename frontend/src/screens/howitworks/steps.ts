// The tour's words. Stage order and facts are the pipeline's own (src/distillery/orchestrator.py, docs/ARCHITECTURE.md,
// README): this file carries no figures and no run data, only what each stage does and why it can be trusted.

export type Actor = "pack" | "sandbox" | "teacher" | "factory" | "planner" | "code" | "router" | "gateway";

export const ACTOR: Record<Actor, string> = {
  pack: "Task pack",
  sandbox: "Nebius sandbox",
  teacher: "Teacher model",
  factory: "Token Factory",
  planner: "Planner model",
  code: "Plain code, no AI",
  router: "Certified router",
  gateway: "Gateway",
};

export interface Step {
  id: string;
  title: string;
  /** What happens, in one plain sentence. */
  what: string;
  /** Why it can be trusted. */
  trust: string;
  who: Actor[];
  /** This stop sends the run back to an earlier one (the retraining loop). */
  loopsTo?: string;
}

export const BUILD: Step[] = [
  { id: "questions", title: "Write the questions", who: ["pack"], what: "A task pack brings a database schema and many questions about it.", trust: "The same fixed database every time, so every answer can be checked." },
  { id: "gold", title: "Check the answers", who: ["sandbox"], what: "Each reference answer is real SQL, run in a sandbox. It is kept only if it runs and returns rows.", trust: "Answers are run, never trusted on sight." },
  { id: "seal", title: "Seal the exam", who: ["code"], what: "An exam set and a harder stress set are locked away before any training.", trust: "Only the evaluator may open them, and a test checks that nothing else does." },
  { id: "teach", title: "Teacher writes lessons", who: ["teacher", "sandbox"], what: "The big model writes SQL for the training questions. Each one is run; only correct ones become lessons.", trust: "A wrong lesson never reaches the student." },
  { id: "train", title: "Train the student", who: ["factory"], what: "The small model is fine-tuned on the verified lessons.", trust: "The training job and its cost are recorded, and a paid job is always closed." },
  { id: "practice", title: "Find the mistakes", who: ["planner"], what: "A practice test shows where the student fails, and the failures are grouped by kind.", trust: "Only the practice set is used here; the sealed exam stays shut." },
  { id: "target", title: "Targeted lessons", who: ["teacher", "sandbox"], what: "New lessons aimed at the weak spots are made on a branch of the sandbox, and the student trains again.", trust: "Every round is a branch, so you can see exactly what changed.", loopsTo: "train" },
  { id: "exam", title: "Final exam, once", who: ["code"], what: "The best round sits the sealed exam once, next to the teacher and the untrained base model.", trust: "One sitting, no retakes, no peeking." },
  { id: "gate", title: "The gate decides", who: ["code"], what: "A fixed rule says PROMOTE or REJECT: close enough to the teacher, and clearly better than the base.", trust: "Plain code with thresholds written down beforehand. Nobody can talk it into a yes." },
];

export const SERVE: Step[] = [
  { id: "certify", title: "Certify the router", who: ["router"], what: "A small scorer learns when the student is likely right, and its cut-off is certified on sealed data.", trust: "The error bound is proven on questions it never saw." },
  { id: "answer", title: "Answer a request", who: ["gateway"], what: "Your app changes one base URL. Confident and certified: the student answers. Otherwise the teacher does.", trust: "Every answer says who gave it, and why." },
  { id: "guard", title: "Watch for drift", who: ["gateway", "teacher"], what: "If questions stop looking like the certified ones, or audits disagree, everything goes back to the teacher.", trust: "Rollback is automatic. Coming back needs a new certificate." },
  { id: "receipts", title: "Show the receipts", who: ["code"], what: "Every claim is listed with the file that proves it.", trust: "Claims not yet measured say pending, with no number." },
];
