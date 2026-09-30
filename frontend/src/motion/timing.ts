// Animation constants. Kept out of JSX so no number literal can be mistaken for displayed data.
export const COUNT_MS = 700;
export const HEARTBEAT_QUIET_MS = 30_000;
export const COPY_FEEDBACK_MS = 1600;
export const ZOOM_MIN = 0.4;
export const ZOOM_MAX = 2.5;
export const ZOOM_STEP = 1.25;
export const PAN_STEP = 60;
export const easeOut = (p: number): number => 1 - (1 - p) ** 3;
