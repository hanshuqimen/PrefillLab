import {
  Activity,
  ArrowUpRight,
  BookOpen,
  ChevronDown,
  Cpu,
  FlaskConical,
  GitCompareArrows,
  LayoutDashboard,
  Plus,
  RefreshCw,
  Terminal,
} from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { api } from "./api/client";
import { ErrorState, Loading } from "./components/common";
import { useResource } from "./hooks/useResource";
import { Compare } from "./pages/Compare";
import { Detail } from "./pages/Detail";
import { Experiments } from "./pages/Experiments";
import { NewBenchmark } from "./pages/NewBenchmark";
import { Overview } from "./pages/Overview";
import { System } from "./pages/System";

export default function App() {
  const [route, setRoute] = useState(
    window.location.hash.slice(1) || "/overview",
  );
  const [selected, setSelected] = useState<string[]>([]);
  const [offset, setOffset] = useState(0);
  const loadExperiments = useCallback(() => api.experiments(offset), [offset]);
  const experiments = useResource(loadExperiments);
  const system = useResource(api.system);
  useEffect(() => {
    const update = () => setRoute(window.location.hash.slice(1) || "/overview");
    window.addEventListener("hashchange", update);
    return () => window.removeEventListener("hashchange", update);
  }, []);
  useEffect(() => {
    document.title = `PrefillLab · ${route.startsWith("/experiments/") ? "Experiment" : route.slice(1)}`;
    document.getElementById("main")?.focus();
  }, [route]);
  const items = experiments.data?.items ?? [];
  const toggle = (id: string) =>
    setSelected((current) =>
      current.includes(id)
        ? current.filter((value) => value !== id)
        : current.length < 6
          ? [...current, id]
          : current,
    );
  const refresh = () => {
    experiments.refresh();
    system.refresh();
  };
  const links = [
    { path: "/overview", name: "Overview", icon: LayoutDashboard },
    { path: "/experiments", name: "Experiments", icon: FlaskConical },
    { path: "/compare", name: "Compare", icon: GitCompareArrows },
    { path: "/new", name: "New benchmark", icon: Plus },
    { path: "/system", name: "System", icon: Cpu },
  ];
  const active = links.find(
    (link) =>
      link.path === route ||
      (link.path === "/experiments" && route.startsWith("/experiments/")),
  );
  let content;
  if (route === "/new")
    content = <NewBenchmark system={system.data} changed={refresh} />;
  else if (route === "/system")
    content = system.loading ? (
      <Loading />
    ) : system.error ? (
      <ErrorState message={system.error} retry={system.refresh} />
    ) : (
      <System system={system.data} />
    );
  else if (route.startsWith("/experiments/"))
    content = <Detail key={route} id={route.split("/")[2]} changed={refresh} />;
  else if (experiments.loading) content = <Loading />;
  else if (experiments.error)
    content = (
      <ErrorState message={experiments.error} retry={experiments.refresh} />
    );
  else if (route === "/experiments")
    content = <Experiments items={items} selected={selected} toggle={toggle} />;
  else if (route === "/compare")
    content = <Compare items={items} selected={selected} toggle={toggle} />;
  else
    content = (
      <Overview key={items.length ? "populated" : "empty"} items={items} />
    );
  return (
    <div className="app-shell">
      <a
        href="#main"
        className="skip-link"
        onClick={(event) => {
          event.preventDefault();
          document.getElementById("main")?.focus();
        }}
      >
        Skip to content
      </a>
      <aside className="sidebar">
        <a className="brand" href="#/overview">
          <span className="brand-mark">
            <svg viewBox="0 0 28 28" aria-hidden="true">
              <path
                d="M4 21V7h5v14M12 21V4h5v17M20 21V11h5v10"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              />
            </svg>
          </span>
          <b>
            PrefillLab<span>0.1</span>
          </b>
        </a>
        <div className="workspace">
          <span className="workspace-icon">
            <Terminal size={17} aria-hidden="true" />
          </span>
          <div>
            Local workspace<small>Inference research</small>
          </div>
          <ChevronDown size={14} aria-hidden="true" />
        </div>
        <p className="nav-label">Workspace</p>
        <nav aria-label="Main navigation">
          {links.map(({ path, name, icon: Icon }) => (
            <a
              key={path}
              href={`#${path}`}
              className={active?.path === path ? "active" : ""}
              aria-current={active?.path === path ? "page" : undefined}
            >
              <Icon size={17} aria-hidden="true" />
              {name}
              {path === "/experiments" && items.length > 0 && (
                <span>{experiments.data?.total}</span>
              )}
            </a>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <a
            href="https://github.com/hanshuqimen/PrefillLab#readme"
            target="_blank"
            rel="noreferrer"
          >
            <BookOpen size={16} aria-hidden="true" /> Documentation{" "}
            <ArrowUpRight size={14} aria-hidden="true" />
          </a>
          <div className="local-status">
            <span
              className={`status-dot ${!experiments.error ? "ready" : ""}`}
            />
            <div>
              {experiments.error ? "API unavailable" : "Local API connected"}
              <small>
                {system.data?.environment.gpu
                  ? String(system.data.environment.gpu)
                  : "CPU / simulation ready"}
              </small>
            </div>
          </div>
          <p>Measure. Understand. Improve.</p>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div>
            <span>Workspace</span>
            <span className="slash">/</span>
            <b>{active?.name ?? "Overview"}</b>
          </div>
          <div>
            <span className="version">v0.1.0</span>
            <button
              className="icon-button"
              aria-label="Refresh workspace"
              onClick={refresh}
            >
              <RefreshCw size={15} aria-hidden="true" />
            </button>
            <a
              className="api-link"
              href="/api/docs"
              target="_blank"
              rel="noreferrer"
            >
              <Activity size={14} aria-hidden="true" /> API docs
            </a>
          </div>
        </header>
        <main id="main" tabIndex={-1}>
          {content}
          {experiments.data &&
            experiments.data.total > 200 &&
            !route.startsWith("/experiments/") &&
            route !== "/new" &&
            route !== "/system" && (
              <div className="pagination">
                <span>
                  History window {offset + 1}–{offset + items.length} of{" "}
                  {experiments.data.total}. Statistics and comparisons use this
                  window.
                </span>
                <button
                  className="button"
                  disabled={offset === 0}
                  onClick={() => setOffset((value) => Math.max(0, value - 200))}
                >
                  Previous
                </button>
                <button
                  className="button"
                  disabled={offset + items.length >= experiments.data.total}
                  onClick={() => setOffset((value) => value + 200)}
                >
                  Next
                </button>
              </div>
            )}
          <footer className="app-footer">
            <span>
              <span className="tiny-mark">▥</span> PrefillLab
            </span>
            <span>Local inference. Reproducible evidence.</span>
            <a
              href="https://github.com/hanshuqimen/PrefillLab"
              target="_blank"
              rel="noreferrer"
            >
              Open source <ArrowUpRight size={12} aria-hidden="true" />
            </a>
          </footer>
        </main>
      </div>
    </div>
  );
}
