import { useState } from "react";
import { NavLink, Route, Routes, Link } from "react-router-dom";
import { currentTheme, setTheme, type Theme } from "./theme";
import { EmptyState } from "./components";
import Replay from "./screens/replay";
import LiveRun from "./screens/live";
import Tree from "./screens/tree";
import ReportScreen from "./screens/report";
import NewRun from "./screens/new";
import Playground from "./screens/playground";

function ThemeToggle() {
  const [t, setT] = useState<Theme>(currentTheme);
  const next: Theme = t === "dark" ? "light" : "dark";
  return (
    <button className="btn" aria-label={`Switch to ${next} theme`} onClick={() => { setTheme(next); setT(next); }}>
      {t === "dark" ? "Light" : "Dark"}
    </button>
  );
}

export function App() {
  return (
    <>
      <header className="site-header">
        <div className="container">
          <Link to="/" className="brand">Distillery</Link>
          <nav aria-label="Main">
            <NavLink to="/" end>Replay</NavLink>
            <NavLink to="/new">New run</NavLink>
            <NavLink to="/playground">Playground</NavLink>
          </nav>
          <ThemeToggle />
        </div>
      </header>
      {/* Screens render their own <LabelBanner> once they know the run's dry_run/recorded flags. */}
      <main className="container">
        <Routes>
          <Route path="/" element={<Replay />} />
          <Route path="/runs/:id" element={<LiveRun />} />
          <Route path="/runs/:id/tree" element={<Tree />} />
          <Route path="/runs/:id/report" element={<ReportScreen />} />
          <Route path="/new" element={<NewRun />} />
          <Route path="/playground" element={<Playground />} />
          <Route path="*" element={<EmptyState title="Page not found"><Link to="/">Back to Replay</Link></EmptyState>} />
        </Routes>
      </main>
    </>
  );
}
