// Layout and interaction constants of the Tree screen. Kept out of .tsx so no literal here can be mistaken for a displayed value.

/** Top to bottom must show the tree this many times larger than left to right before it is chosen. */
export const PREFER_LR = 1.15;
/** Space kept free around the tree when it is fitted into the viewport (px). */
export const FIT_PAD = 16;
/** Lowest zoom. Lower than the app-wide floor so that a tree of dozens of nodes can be fitted whole. */
export const ZOOM_FLOOR = 0.08;
/** Automatic fitting never shrinks the tree below this zoom: node labels stay legible. A tree that would need less is framed on the path to the selected node instead (Fit still shows everything). */
export const READABLE_MIN = 0.6;
