import { Download, Search } from "lucide-react";
import { useState } from "react";
import { ExperimentTable } from "../components/ExperimentTable";
import { Empty, exportCSV, Panel } from "../components/common";
import type { Experiment } from "../types";

export function Experiments({
  items,
  selected,
  toggle,
}: {
  items: Experiment[];
  selected: string[];
  toggle: (id: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [source, setSource] = useState("all");
  const filtered = items.filter(
    (r) =>
      `${r.config.model} ${r.config.backend} ${r.id}`
        .toLowerCase()
        .includes(query.toLowerCase()) &&
      (source === "all" || r.simulated === (source === "simulated")),
  );
  return (
    <>
      <div className="page-title">
        <div>
          <h1>Experiments</h1>
          <p>Every configuration. Every sample. One history.</p>
        </div>
        <a className="button primary" href="#/new">
          New benchmark +
        </a>
      </div>
      <div className="toolbar">
        <label className="search">
          <Search size={16} aria-hidden="true" />
          <input
            aria-label="Search experiments"
            placeholder="Search model, backend, or ID…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </label>
        <div className="toolbar-right">
          <select
            aria-label="Experiment source"
            value={source}
            onChange={(e) => setSource(e.target.value)}
          >
            <option value="all">All sources</option>
            <option value="simulated">Simulated</option>
            <option value="measured">Measured</option>
          </select>
          <button
            className="button"
            disabled={!filtered.length}
            onClick={() => exportCSV(filtered)}
          >
            <Download size={15} aria-hidden="true" /> Export CSV
          </button>
          <a className="button" href="#/compare">
            Compare ({selected.length})
          </a>
        </div>
      </div>
      {filtered.length ? (
        <Panel
          title="Experiment history"
          subtitle={`${filtered.length} results · select up to six to compare`}
        >
          <ExperimentTable
            items={filtered}
            selected={selected}
            toggle={toggle}
          />
        </Panel>
      ) : (
        <Empty
          title={items.length ? "No matching experiments." : undefined}
          text={items.length ? "Try another search or data source." : undefined}
        />
      )}
    </>
  );
}
