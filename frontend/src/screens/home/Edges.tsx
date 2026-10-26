import { createPortal } from "react-dom";
import "./edges.css";

/**
 * A soft glow along the left and right edges of the landing page: violet, purple, light blue and green. Decoration
 * only. It is portalled to <body> so the route's enter animation cannot change how its fixed position resolves, and it
 * goes away with the page.
 */
export function Edges() {
  if (typeof document === "undefined") return null;
  return createPortal(
    <div className="h-edges" aria-hidden="true">
      <i className="h-edge h-edge-l" />
      <i className="h-edge h-edge-r" />
    </div>,
    document.body,
  );
}
