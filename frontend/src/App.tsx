import { useCallback, useEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";
import { Link, NavLink, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { currentTheme, setTheme, type Theme } from "./theme";
import { EmptyState } from "./components";
import { useReducedMotion } from "./motion/useReducedMotion";
import { useShortcuts } from "./motion/useShortcuts";
import Replay from "./screens/replay";
import LiveRun from "./screens/live";
import Tree from "./screens/tree";
import ReportScreen from "./screens/report";
import NewRun from "./screens/new";
import Playground from "./screens/playground";

function ThemeToggle({ theme, onToggle }: { theme: Theme; onToggle: () => void }) {
  const next: Theme = theme === "dark" ? "light" : "dark";
  return (
    <button className="btn theme-btn" aria-label={`Switch to ${next} theme`} onClick={onToggle}>
      <span className="theme-ico" data-t={theme} aria-hidden="true" />
      {theme === "dark" ? "Light" : "Dark"}
    </button>
  );
}

const HELP: [string, string][] = [
  ["?", "Show or hide this list"],
  ["t", "Switch theme"],
  ["g then r", "Go to Replay"],
  ["g then n", "Go to New run"],
  ["g then p", "Go to Playground"],
  ["+ and −", "Zoom the experiment tree"],
  ["Arrow keys", "Move between tree nodes, or pan the tree"],
  ["Esc", "Close"],
];

function ShortcutHelp({ onClose }: { onClose: () => void }) {
  const close = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    const prev = document.activeElement as HTMLElement | null;
    close.current?.focus();
    const esc = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", esc);
    return () => { window.removeEventListener("keydown", esc); prev?.focus?.(); };
  }, [onClose]);
  return (
    <div className="kbd-scrim" onClick={onClose}>
      <div className="kbd-panel" role="dialog" aria-modal="true" aria-label="Keyboard shortcuts" onClick={(e) => e.stopPropagation()}>
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h3 style={{ margin: 0 }}>Keyboard shortcuts</h3>
          <button ref={close} className="btn" onClick={onClose}>Close</button>
        </div>
        <dl className="kbd-list">
          {HELP.map(([k, d]) => (<div key={k}><dt><kbd>{k}</kbd></dt><dd>{d}</dd></div>))}
        </dl>
      </div>
    </div>
  );
}

type Doc = Document & { startViewTransition?: (cb: () => void) => unknown };

/**
 * Renders the routes. With the View Transitions API (and no reduced-motion preference) a route
 * change cross-fades the old page into the new one; the new route is committed inside the
 * transition callback. Everywhere else the new page simply plays a CSS fade/slide on mount.
 */
function RouteStage() {
  const location = useLocation();
  const reduced = useReducedMotion();
  const vt = !reduced && typeof (document as Doc).startViewTransition === "function";
  const [shown, setShown] = useState(location);
  useEffect(() => {
    if (!vt || shown === location) return;
    if (shown.pathname === location.pathname) { setShown(location); return; }
    try {
      (document as Doc).startViewTransition!(() => { flushSync(() => setShown(location)); });
    } catch {
      setShown(location);
    }
  }, [vt, location, shown]);
  const active = vt ? shown : location;
  return (
    <div key={active.pathname} className={vt ? "route" : "route route-enter"}>
      <Routes location={active}>
        <Route path="/" element={<Replay />} />
        <Route path="/runs/:id" element={<LiveRun />} />
        <Route path="/runs/:id/tree" element={<Tree />} />
        <Route path="/runs/:id/report" element={<ReportScreen />} />
        <Route path="/new" element={<NewRun />} />
        <Route path="/playground" element={<Playground />} />
        <Route path="*" element={<EmptyState title="Page not found"><Link to="/">Back to Replay</Link></EmptyState>} />
      </Routes>
    </div>
  );
}

export function App() {
  const [theme, setT] = useState<Theme>(currentTheme);
  const [help, setHelp] = useState(false);
  const nav = useNavigate();
  const closeHelp = useCallback(() => setHelp(false), []);
  const toggle = () => { const next: Theme = theme === "dark" ? "light" : "dark"; setTheme(next); setT(next); };
  useShortcuts({
    help: () => setHelp((h) => !h),
    theme: toggle,
    go: (t) => nav(t === "replay" ? "/" : t === "new" ? "/new" : "/playground"),
  });
  return (
    <>
      <div className="aurora" aria-hidden="true"><i /><i /><i /></div>
      <header className="site-header">
        <div className="container">
          <Link to="/" className="brand">Distillery</Link>
          <nav aria-label="Main">
            <NavLink to="/" end>Replay</NavLink>
            <NavLink to="/new">New run</NavLink>
            <NavLink to="/playground">Playground</NavLink>
          </nav>
          <button className="btn icon-btn" aria-label="Keyboard shortcuts" aria-haspopup="dialog" onClick={() => setHelp(true)}>?</button>
          <ThemeToggle theme={theme} onToggle={toggle} />
        </div>
      </header>
      {/* Screens render their own <LabelBanner> once they know the run's dry_run/recorded flags. */}
      <main className="container">
        <RouteStage />
      </main>
      {help && <ShortcutHelp onClose={closeHelp} />}
    </>
  );
}
