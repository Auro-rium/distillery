// Props-driven chart and card components ("mission control"). No API calls, no payload types: callers
// map API data to these plain props and every label is formatted via api/format.ts.
export { AccuracyBars, MODEL_ORDER, type AccuracyRow } from "./AccuracyBars";
export { CiNumberLine, type CiNumberLineProps } from "./CiNumberLine";
export { ChartLegend, MODEL_LABEL, type Formatter, type ModelKey } from "./core";
export { DiscordantMatrix, type DiscordantMatrixProps } from "./DiscordantMatrix";
export { JobLedger, type LedgerRow } from "./JobLedger";
export { LossChart, type LossChartProps, type LossPoint } from "./LossChart";
export { ProofCard, type ProofCardProps, type ProofStatus } from "./ProofCard";
export { Sparkline, type SparklineProps } from "./Sparkline";
export { StageRail, type RailStage, type RailStatus } from "./StageRail";
export { StressBars, type StressRow } from "./StressBars";
