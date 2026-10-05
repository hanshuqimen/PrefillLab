import { ArrowRight, Download, Info } from "lucide-react";
import { useState } from "react";
import { ScalingChart } from "../charts/ScalingChart";
import { ExperimentTable } from "../components/ExperimentTable";
import { Empty, exportCSV, format, Metric, Panel } from "../components/common";
import type { Experiment, MetricKey } from "../types";

export function Overview({ items }: { items: Experiment[] }) {
  const [source, setSource] = useState(
    items.some((r) => r.simulated) ? "simulated" : "measured",
  );
  const [model, setModel] = useState("all");
  const [batch, setBatch] = useState("1");
  const [metric, setMetric] = useState<MetricKey>("ttft_ms");
  const filtered = items.filter(
    (r) =>
      r.simulated === (source === "simulated") &&
      (model === "all" || r.config.model === model),
  );
  const chartItems = filtered.filter(
    (r) => String(r.config.batch_size) === batch,
  );
  const titles: Partial<Record<MetricKey, [string, string]>> = {
    ttft_ms: ["Time to first token", "ms"],
    prefill_throughput_tokens_per_sec: ["Prefill throughput", "tok/s"],
    peak_gpu_memory_mb: ["Peak GPU allocation", "MiB"],
  };
  const [, unit] = titles[metric]!;
  if (!items.length) return <Empty />;
  return (
    <>
      <div className="page-title">
        <div>
          <h1>Overview</h1>
          <p>A closer look at your prefill performance.</p>
        </div>
        <a className="button primary" href="#/new">
          New benchmark <span>+</span>
        </a>
      </div>
      <div className="toolbar">
        <div className="segmented" aria-label="Data source">
          {["simulated", "measured"].map((value) => (
            <button
              key={value}
              aria-pressed={source === value}
              className={source === value ? "active" : ""}
              onClick={() => setSource(value)}
            >
              {value === "simulated" ? "Simulation" : "Measured"}
            </button>
          ))}
        </div>
        <div className="toolbar-right">
          <label className="inline-label">
            Model
            <select
              aria-label="Filter model"
              value={model}
              onChange={(e) => setModel(e.target.value)}
            >
              <option value="all">All models</option>
              {[...new Set(items.map((r) => r.config.model))].map((name) => (
                <option key={name}>{name}</option>
              ))}
            </select>
          </label>
          <span className="muted">{filtered.length} experiments</span>
        </div>
      </div>
      {source === "simulated" && (
        <div className="notice simulation">
          <Info size={16} aria-hidden="true" />
          <span>
            <strong>Simulation workspace.</strong> Explore performance patterns
            with synthetic data. These values are not measured GPU results.
          </span>
          <a href="#/new">
            Run a real model <ArrowRight size={14} aria-hidden="true" />
          </a>
        </div>
      )}
      <div className="metrics-grid">
        <Metric
          label="Experiments"
          value={format(filtered.length, 0)}
          note={`${new Set(filtered.map((r) => r.config.model)).size} models in this selection`}
        />
        <Metric
          label="Benchmark runs"
          value={format(
            filtered.reduce((sum, r) => sum + r.samples.length, 0),
            0,
          )}
          note="Warmup runs excluded"
        />
        <Metric
          label="Average TTFT"
          value={format(
            filtered.length
              ? filtered.reduce((sum, r) => sum + r.metrics.ttft_ms, 0) /
                  filtered.length
              : null,
          )}
          unit="ms"
          note="Across selected configurations"
        />
        <Metric
          label="Best throughput"
          value={format(
            filtered.length
              ? Math.max(
                  ...filtered.map(
                    (r) => r.metrics.prefill_throughput_tokens_per_sec,
                  ),
                ) / 1000
              : null,
          )}
          unit="k tok/s"
          note="Prompt tokens / prefill seconds"
        />
      </div>
      <Panel
        title="Context scaling"
        subtitle="See how a longer prompt changes the work."
        action={
          <label className="inline-label">
            Batch
            <select
              aria-label="Chart batch size"
              value={batch}
              onChange={(e) => setBatch(e.target.value)}
            >
              {[...new Set(items.map((r) => r.config.batch_size))]
                .sort((a, b) => a - b)
                .map((b) => (
                  <option key={b}>{b}</option>
                ))}
            </select>
          </label>
        }
      >
        <div className="chart-tabs">
          {Object.entries(titles).map(([key, value]) => (
            <button
              key={key}
              className={metric === key ? "active" : ""}
              aria-pressed={metric === key}
              onClick={() => setMetric(key as MetricKey)}
            >
              {value[0]}
            </button>
          ))}
          <span>{unit}</span>
        </div>
        <ScalingChart items={chartItems} metric={metric} unit={unit} />
      </Panel>
      <div className="overview-bottom">
        <Panel
          title="Recent experiments"
          subtitle="Your latest runs, ready to inspect."
          action={
            <a className="text-link" href="#/experiments">
              View all <ArrowRight size={14} aria-hidden="true" />
            </a>
          }
        >
          <ExperimentTable items={filtered.slice(0, 5)} />
          <div className="panel-footer">
            <span>{filtered.length} experiments in selected history</span>
            <button className="text-link" onClick={() => exportCSV(filtered)}>
              <Download size={14} aria-hidden="true" /> Export CSV
            </button>
          </div>
        </Panel>
        <Panel
          title="Read the evidence"
          subtitle="From a result to a useful next experiment."
          className="insight-panel"
        >
          <div className="insight-symbol">
            <svg viewBox="0 0 120 50" aria-hidden="true">
              <path
                d="M4 42L26 35L48 37L70 20L92 23L116 4"
                stroke="currentColor"
                strokeWidth="3"
                fill="none"
              />
            </svg>
          </div>
          <h3>One number is only the beginning.</h3>
          <p>
            Inspect module time, sample variability, and the environment before
            drawing a bottleneck conclusion.
          </p>
          <a className="text-link" href="#/compare">
            Compare experiments <ArrowRight size={15} aria-hidden="true" />
          </a>
          <div className="insight-note">
            <Info size={15} aria-hidden="true" /> All recommendations are
            heuristic hypotheses.
          </div>
        </Panel>
      </div>
    </>
  );
}
