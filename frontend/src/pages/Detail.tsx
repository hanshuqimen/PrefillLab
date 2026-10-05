import { ArrowLeft, Download, Trash2 } from "lucide-react";
import { useCallback, useState } from "react";
import { api } from "../api/client";
import {
  Badge,
  download,
  ErrorState,
  format,
  Loading,
  Metric,
  Panel,
} from "../components/common";
import { useResource } from "../hooks/useResource";

export function Detail({ id, changed }: { id: string; changed: () => void }) {
  const loader = useCallback(() => api.experiment(id), [id]);
  const resource = useResource(loader);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confirm, setConfirm] = useState(false);
  if (resource.loading) return <Loading />;
  if (resource.error)
    return <ErrorState message={resource.error} retry={resource.refresh} />;
  const result = resource.data;
  if (!result) return null;
  const times = Object.entries(result.breakdown).filter(
    (pair): pair is [string, number] =>
      pair[0].endsWith("_ms") && typeof pair[1] === "number",
  );
  const total = times.reduce((sum, [, value]) => sum + value, 0);
  const remove = async () => {
    setDeleting(true);
    try {
      await api.remove(id);
      changed();
      window.location.hash = "/experiments";
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Delete failed.");
      setDeleting(false);
    }
  };
  return (
    <>
      <a className="back-link" href="#/experiments">
        <ArrowLeft size={14} aria-hidden="true" /> Experiments
      </a>
      <div className="page-title">
        <div>
          <div className="heading-row">
            <h1>{result.config.model}</h1>
            <Badge simulated={result.simulated} />
          </div>
          <p>
            {result.config.backend} / {result.device} / {result.config.dtype} /{" "}
            {new Date(result.timestamp).toLocaleString()}
          </p>
        </div>
        <button
          className="button"
          onClick={() =>
            download(`experiment_${id}.json`, JSON.stringify(result, null, 2))
          }
        >
          <Download size={15} aria-hidden="true" /> Export JSON
        </button>
      </div>
      {error && <ErrorState message={error} />}
      <div className="config-strip">
        <span>
          Context <b>{format(result.config.input_length, 0)}</b>
        </span>
        <span>
          Batch <b>{result.config.batch_size}</b>
        </span>
        <span>
          Output <b>{result.config.output_length}</b>
        </span>
        <span>
          Runs <b>{result.samples.length}</b>
        </span>
        <span>
          Seed <b>{result.config.seed}</b>
        </span>
        <span>
          ID <b>{id.slice(0, 8)}</b>
        </span>
      </div>
      <div className="metrics-grid detail-metrics">
        <Metric
          label="Time to first token"
          value={format(result.metrics.ttft_ms, 2)}
          unit="ms"
        />
        <Metric
          label="Prefill latency"
          value={format(result.metrics.prefill_latency_ms, 2)}
          unit="ms"
        />
        <Metric
          label="Throughput"
          value={format(result.metrics.prefill_throughput_tokens_per_sec, 0)}
          unit="tok/s"
        />
        <Metric
          label="Peak GPU allocation"
          value={format(result.metrics.peak_gpu_memory_mb, 0)}
          unit="MiB"
        />
        <Metric
          label="GPU utilization"
          value={format(result.metrics.gpu_utilization)}
          unit="%"
        />
      </div>
      <p className="methodology">{result.methodology}</p>
      <div className="two-column">
        <Panel
          title="Where the time goes"
          subtitle={`${result.breakdown.source} · independent profiling pass`}
        >
          {times.length ? (
            <div className="breakdown">
              {times.map(([key, value]) => (
                <div className="breakdown-row" key={key}>
                  <span>{key.replace("_time_ms", "").toUpperCase()}</span>
                  <div>
                    <i
                      style={{ width: `${total ? (value / total) * 100 : 0}%` }}
                    />
                  </div>
                  <b>
                    {format(value, 2)} <small>ms</small>
                  </b>
                  <span className="muted">
                    {format(total ? (value / total) * 100 : 0)}%
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <p className="panel-body muted">
              Module breakdown unavailable or profiling disabled.
            </p>
          )}
        </Panel>
        <Panel
          title="Bottleneck analysis"
          subtitle="Heuristic evidence, not a hardware diagnosis."
        >
          <div className="panel-body">
            <h3>{result.analysis?.primary_bottleneck ?? "Unavailable"}</h3>
            <p>
              Confidence {format((result.analysis?.confidence ?? 0) * 100, 0)}%
            </p>
            <ul className="evidence">
              {result.analysis?.evidence.map((text) => (
                <li key={text}>{text}</li>
              ))}
            </ul>
          </div>
        </Panel>
      </div>
      <Panel
        title="Suggested next experiments"
        subtitle="Each suggestion is a hypothesis to validate on your workload."
      >
        <div className="recommendation-grid">
          {result.recommendations.map((item) => (
            <article className="recommendation" key={item.title}>
              <span className="tag">{item.category}</span>
              <h3>{item.title}</h3>
              <p>{item.reason}</p>
              <small>Confidence {format(item.confidence * 100, 0)}%</small>
            </article>
          ))}
        </div>
      </Panel>
      <Panel
        title="Sample distribution"
        subtitle="All statistics exclude warmup runs."
      >
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Metric</th>
                {["mean", "median", "p50", "p90", "min", "max", "std"].map(
                  (key) => (
                    <th key={key}>{key}</th>
                  ),
                )}
              </tr>
            </thead>
            <tbody>
              {Object.entries(result.statistics).map(([key, values]) => (
                <tr key={key}>
                  <td>{key}</td>
                  {Object.entries(values).map(([stat, value]) => (
                    <td className="mono" key={stat}>
                      {format(value, 3)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
      <div className="two-column">
        <Panel title="Environment">
          <dl className="key-values">
            {Object.entries(result.environment).map(([key, value]) => (
              <div key={key}>
                <dt>{key.replaceAll("_", " ")}</dt>
                <dd>{value ?? "Unavailable"}</dd>
              </div>
            ))}
          </dl>
        </Panel>
        <Panel title="Measurement notes">
          <ul className="evidence panel-body">
            {result.warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
            <li>
              GPU memory denotes PyTorch allocation, not device-wide memory
              usage.
            </li>
            <li>Missing measurements remain null in JSON.</li>
          </ul>
        </Panel>
      </div>
      {result.layers.length > 0 && (
        <Panel
          title="Slowest modules"
          subtitle="Exclusive time; direct child module time is subtracted."
        >
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Module</th>
                  <th>Category</th>
                  <th>Exclusive ms</th>
                  <th>Calls</th>
                </tr>
              </thead>
              <tbody>
                {result.layers.slice(0, 12).map((layer) => (
                  <tr key={layer.name}>
                    <td className="mono">{layer.name}</td>
                    <td>{layer.category}</td>
                    <td>{format(layer.time_ms, 3)}</td>
                    <td>{layer.calls}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}
      <details className="raw-data">
        <summary>Raw experiment JSON</summary>
        <pre>{JSON.stringify(result, null, 2)}</pre>
      </details>
      <div className="danger-zone">
        {confirm ? (
          <>
            <span>
              Delete this experiment from local history? This cannot be undone.
            </span>
            <button
              className="button danger"
              disabled={deleting}
              onClick={remove}
            >
              {deleting ? "Deleting…" : "Delete experiment"}
            </button>
            <button className="button" onClick={() => setConfirm(false)}>
              Cancel
            </button>
          </>
        ) : (
          <button className="text-link danger" onClick={() => setConfirm(true)}>
            <Trash2 size={14} aria-hidden="true" /> Delete experiment
          </button>
        )}
      </div>
    </>
  );
}
