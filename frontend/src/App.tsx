import { useCallback, useEffect, useRef, useState } from "react";
import { flushSync } from "react-dom";
import { Link, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import { currentTheme, setTheme, type Theme } from "./theme";
import { EmptyState } from "./components";
import { ShortcutHelp, SiteFooter, SiteHeader, SkipLink, SITE_NAME, pageTitle } from "./components/Shell";
import { useReducedMotion } from "./motion/useReducedMotion";
import { useShortcuts } from "./motion/useShortcuts";
import Replay from "./screens/replay";
import LiveRun from "./screens/live";
import Tree from "./screens/tree";
import ReportScreen from "./screens/report";
import NewRun from "./screens/new";
import Playground from "./screens/playground";
import Humanset from "./screens/humanset";

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
        <Route path="/humanset" element={<Humanset />} />
        <Route path="*" element={<EmptyState title="Page not found"><Link to="/">Back to Replay</Link></EmptyState>} />
      </Routes>
    </div>
  );
}

export function App() {
  const [theme, setT] = useState<Theme>(currentTheme);
  const [help, setHelp] = useState(false);
  const [announce, setAnnounce] = useState("");
  const nav = useNavigate();
  const { pathname } = useLocation();
  const main = useRef<HTMLElement>(null);
  const firstPath = useRef(true);
  const closeHelp = useCallback(() => setHelp(false), []);
  const toggle = () => { const next: Theme = theme === "dark" ? "light" : "dark"; setTheme(next); setT(next); };
  useShortcuts({
    help: () => setHelp((h) => !h),
    theme: toggle,
    go: (t) => nav(t === "replay" ? "/" : t === "new" ? "/new" : "/playground"),
  });
  // Every route gets its own document title. After a client-side navigation focus moves to <main> and
  // the new page is announced, because there is no page load to tell keyboard and screen reader users.
  useEffect(() => {
    const title = pageTitle(pathname);
    document.title = `${title} · ${SITE_NAME}`;
    if (firstPath.current) { firstPath.current = false; return; }
    setAnnounce(title);
    main.current?.focus({ preventScroll: true });
    document.scrollingElement?.scrollTo?.({ top: 0 });
  }, [pathname]);
  return (
    <>
      <SkipLink onSkip={() => main.current?.focus()} />
      <div className="aurora" aria-hidden="true"><i /><i /><i /></div>
      <SiteHeader theme={theme} onToggleTheme={toggle} onShowHelp={() => setHelp(true)} />
      {/* Screens render their own <LabelBanner> once they know the run's dry_run/recorded flags. */}
      <main id="main" ref={main} tabIndex={-1} className="container">
        <RouteStage />
      </main>
      <SiteFooter />
      <div className="sr-only" aria-live="polite" data-route-announcer>{announce}</div>
      <ShortcutHelp open={help} onClose={closeHelp} />
    </>
  );
}
