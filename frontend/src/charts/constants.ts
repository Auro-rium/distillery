// Geometry and timing for the charts. Kept out of .tsx so no literal can be mistaken for a displayed value.
// None of these is ever shown as text: they are pixel sizes, paddings and fractions of an axis.

/** Width used before the container has been measured (and in jsdom, which has no layout). */
export const DEFAULT_WIDTH = 640;
export const MIN_WIDTH = 240;

/** Padding around plot areas, in px. */
export const PAD = { top: 18, right: 20, bottom: 34, left: 52 } as const;
export const PAD_COMPACT = { top: 10, right: 16, bottom: 26, left: 16 } as const;

/** Number line. */
export const NL_HEIGHT = 112;
export const NL_AXIS_Y = 64;
/** Fraction of the data span added on each side so the interval never touches the frame. */
export const NL_DOMAIN_PAD = 0.25;
/** Smallest span (in data units) used when interval and threshold coincide. */
export const NL_MIN_SPAN = 0.01;

/** Loss chart. */
export const LOSS_HEIGHT = 240;
export const LOSS_Y_PAD = 0.12;
export const MARKER_R = 4.5;
export const MARKER_HIT_R = 11;

/** Sparkline. */
export const SPARK_W = 120;
export const SPARK_H = 32;
export const SPARK_PAD = 3;

/** Bars. */
export const BAR_H = 18;
export const BAR_GAP = 12;
export const BAR_LABEL_W = 96;
export const BAR_VALUE_W = 72;
export const GROUP_GAP = 18;
export const WHISKER_CAP = 5;

/** Axis 0..1 for accuracy-type charts: gridlines only, never labelled with invented numbers. */
export const UNIT_GRID = [0, 0.25, 0.5, 0.75, 1] as const;
