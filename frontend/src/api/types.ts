// Types per docs/API_CONTRACT.md. Report types follow the report.json the pipeline writes.

export type RunStatus = "pending" | "running" | "complete" | "failed";
export type Decision = "PROMOTE" | "REJECT";
export type StageStatus = "pending" | "running" | "done" | "failed";

export interface ApiErrorBody {
  error: string;
  message: string;
}

export interface Health {
  ok: boolean;
  mode: "live" | "replay-only";
  version: string;
}

export interface Config {
  // null = no model id configured on this server
  models: { planner: string | null; teacher: string | null; triage: string | null; student: string | null };
  thresholds: { ratio_lower_bound_min: number; mcnemar_alpha: number; bootstrap_resamples: number };
  run_cap_usd: number;
  playground: {
    enabled: boolean;
    per_ip_per_hour: number;
    daily_cap_usd: number;
    spent_today_usd: number;
  };
}

export interface RunSummary {
  run_id: string;
  dry_run: boolean;
  recorded: boolean;
  recorded_at: string | null;
  status: RunStatus;
  decision: Decision | null;
  created_at: string | null; // the bundled replay reports null
}

export interface Stage {
  name: string;
  status: StageStatus;
  started_at: string | null;
  ended_at: string | null;
}

export interface ModelSpend {
  usd: number;
  calls: number;
  input_tokens: number;
  output_tokens: number;
}

export interface Spend {
  total_usd: number;
  cap_usd: number;
  by_model: Record<string, ModelSpend>;
  finetune_usd_estimate: number | null;
}

export interface RunDetail {
  run_id: string;
  dry_run: boolean;
  recorded: boolean;
  recorded_at: string | null;
  status: RunStatus;
  error: string | null;
  stages: Stage[];
  spend: Spend;
  sandbox: { operations: number | null; concurrency_peak: number | null };
  verifier: {
    language: "python" | "sql";
    code: string | null;
    selftest: { accepted_gold: number; rejected_corruptions: number; failures: number } | null;
  };
}

// SSE payloads. `observed_at` is when the server noticed the change while polling, NOT when it
// happened. Only `stage.at` is an event time (null if unknown).
export interface StageEvent { observed_at: string; at: string | null; name: string; status: StageStatus }
export interface SpendEvent extends Spend { observed_at: string }
export interface LogEvent { observed_at: string; level: string; message: string }
export interface DoneEvent { observed_at: string; status: RunStatus; decision: Decision | null }
export type RunEvent =
  | { id: number; type: "stage"; data: StageEvent }
  | { id: number; type: "spend"; data: SpendEvent }
  | { id: number; type: "log"; data: LogEvent }
  | { id: number; type: "done"; data: DoneEvent };

// ---- Report (the report.json the pipeline writes) ----
export type AccTriple = { base: number; student: number; teacher: number };

export interface GateThresholds {
  bootstrap_resamples: number;
  mcnemar_alpha: number;
  ratio_lower_bound_min: number;
  seed: number;
}

export interface Gate {
  base_acc: number;
  base_only_vs_student: number;
  bootstrap_resamples_used: number;
  bootstrap_skipped: number;
  decision: Decision;
  mcnemar_p: number;
  n: number;
  ratio_hi: number;
  ratio_lo: number;
  ratio_point: number;
  reasons: string[];
  student_acc: number;
  student_ci: [number, number];
  student_only_vs_base: number;
  teacher_acc: number;
  thresholds: GateThresholds;
}

export interface Cluster {
  description: string;
  name: string;
  target_families: string[];
}

export interface SandboxBranch {
  label: string;
  parent: string;
  uuid: string;
}

export interface FinetuneRecord {
  round: number;
  job_id: string;
  base_model: string | null;
  hyperparameters: Record<string, number | boolean | string | null> | null;
  trained_tokens: number | null;
  trained_steps: number | null;
  total_steps: number | null;
  loss_curve: { step: number | null; train_loss: number | null; valid_loss: number | null }[];
  events: { created_at: number | null; level: string | null; message: string }[];
  diagnostics_error?: string;
}

export interface ReportRound {
  adapter_sha256: string;
  clusters: Cluster[];
  dev_acc: number;
  dev_failures: number;
  finetune_stage: string;
  job_id: string;
  round: number;
  train_rows: number;
  sandbox_branch?: SandboxBranch;
  targeted_new_rows?: number;
}

export interface TeacherCostPer1k {
  basis: string;
  held_out_tasks: number;
  input_tokens: number;
  output_tokens: number;
  usd: number;
  usd_per_1k_tasks: number;
}

export interface StressEval {
  accuracy: AccTriple;
  accuracy_by_family: Record<string, AccTriple>;
  families: string[];
  n: number;
  note?: string;
  sha256: string;
  unparseable?: Partial<Record<"base" | "student" | "teacher", number>>;
}

/** Gate B: the human held-out set, scored with the same thresholds as the gate. */
export interface HumanEval {
  accuracy: AccTriple;
  gate: Gate;
  n: number;
  note?: string;
  sha256: string;
  unparseable?: Partial<Record<"base" | "student" | "teacher", number>>;
}

export interface HumanCounts {
  confirmed: number;
  discarded: number;
  discarded_by_reason: Record<string, number>;
  duplicates_dropped: number;
  kept: number;
  questions: number;
  rejected: number;
  skipped: number;
  undecided: number;
}

export interface HumanData {
  counts: HumanCounts;
  dropped_exact_overlap: { dev: number; gate: number; total: number; train: number };
  n: number;
  note: string;
  sha256: string;
  skeleton_in_train: number;
  teacher_model?: string;
}

export interface HumansetItem {
  decision: "confirm" | "reject" | "skip" | null;
  gold_sql: string;
  preview: { columns: string[]; row_count: number; rows: (string | number | null)[][] };
  question: string;
  requires_order: boolean;
  task_id: string;
}

export interface HumansetDrafts {
  discarded_by_reason: Record<string, number>;
  items: HumansetItem[];
  n_questions: number;
  set: string;
  sets: string[];
  tally: { confirmed: number; rejected: number; skipped: number; undecided: number };
  teacher_model: string;
}

export interface Report {
  candidate_round: number;
  candidate_selection: string;
  config: {
    gate_thresholds: GateThresholds;
    model_ids: { planner: string; student: string; teacher: string; triage: string };
    pipeline: Record<string, unknown> & {
      dry_run: boolean;
      pack: string;
      scale: { dev: number; heldout: number; name: string; train: number };
    };
    prices: Record<
      string,
      { date: string; input_per_mtok: number; output_per_mtok: number; source: string }
    >;
    seeds: { bootstrap_seed: number; db_seed: number; task_seed: number };
  };
  cost: {
    // student is a string ("unavailable: ...") when the serving path is unknown
    cost_per_1k_tasks: { student: string | TeacherCostPer1k; teacher: string | TeacherCostPer1k };
    finetune_usd: string | number;
    llm_by_model: Record<string, ModelSpend>;
    run_cap_usd: number;
    run_total_usd: number;
  };
  counters: Record<string, Record<string, number>>;
  data: {
    dev_tasks: number;
    heldout_class_counts: Record<string, number>;
    heldout_families: string[];
    heldout_sealed_sha256: string;
    heldout_tasks: number;
    // Added with the stress set / in-distribution gate; absent in older recorded runs.
    heldout_kind?: string;
    heldout_skeleton_overlap_rate?: number;
    stress_families?: string[];
    stress_sealed_sha256?: string;
    stress_tasks?: number;
    // null when no human set was sealed; absent in older runs
    human?: HumanData | null;
    train_distinct_skeletons?: number;
    spot_check_file?: string;
    teacher_verified_rows_round1: number;
    train_tasks: number;
  };
  decision: Decision;
  decision_reasons: string[];
  // Gate B decision; null (or absent in older runs) when no human set was sealed
  decision_human?: Decision | null;
  dry_run: boolean;
  evaluation: {
    accuracy: AccTriple;
    accuracy_by_class: Record<string, AccTriple>;
    artifact: { adapter_sha256: string; checkpoint_id: string; job_id: string };
    class_counts: Record<string, number>;
    gate: Gate;
    heldout_sha256: string;
    n: number;
    // null (or absent in older runs) when no stress set was evaluated
    stress?: StressEval | null;
    // Gate B; null (or absent in older runs) when no human set was sealed
    human?: HumanEval | null;
    unparseable: { base: number; student: number; teacher: number };
  };
  headroom: { base_dev_acc: number; max_allowed: number };
  label?: string;
  llm_attempt_counters: Record<string, number>;
  llm_errors_by_purpose: Record<string, number>;
  pack: string;
  rounds: ReportRound[];
  rounds_stop_reason: string;
  run_id: string;
  sandbox_lineage: SandboxBranch[];
  stages: { output_sha256: string; stage: string }[];
  finetune?: FinetuneRecord[]; // per round; absent in reports recorded before this key existed
  // added by the API
  recorded?: boolean;
  recorded_at?: string | null;
}

export interface TreeNode {
  id: string;
  parent_id: string | null;
  label: string;
  round: number | null;
  hypothesis: string | null;
  data_delta: { added_rows: number; families: string[] } | null;
  dev_score: number | null;
  cost_usd: number | null;
  sandbox_image: string | null;
  selected: boolean;
}
export interface Tree { nodes: TreeNode[] }

export type ExampleKind = "fixed" | "still_wrong" | "regressed" | "all";
export interface Example {
  kind: "fixed" | "still_wrong" | "regressed";
  task_id: string;
  family: string;
  heldout_class: string;
  question: string;
  gold_sql: string;
  base_sql: string;
  student_sql: string;
  teacher_sql: string;
  base_ok: boolean;
  student_ok: boolean;
  teacher_ok: boolean;
}

export interface ExamplesResult {
  items: Example[];
  /** X-Examples-Available; null when the header is missing. */
  available: boolean | null;
  capPerKind: number | null;
  /** Whole-set counts (X-Examples-Totals); a count is null when the report did not record it. */
  totals: { fixed: number | null; still_wrong: number | null; regressed: number | null } | null;
}

export interface NewRunBody {
  pack: "sql";
  scale: "tiny" | "small" | "full";
  dry_run: boolean;
  budget_usd?: number;
  run_id?: string;
  approve_spend?: boolean;
}

export interface PlaygroundResult {
  available: boolean;
  reason: string | null;
  sql: string | null;
  verified: boolean | null;
  rows_preview: unknown[][] | Record<string, unknown>[] | null;
  error: string | null;
}
export interface PlaygroundResponse {
  results: { teacher: PlaygroundResult; base: PlaygroundResult; student: PlaygroundResult };
  cost_usd: number;
  note: string;
}
