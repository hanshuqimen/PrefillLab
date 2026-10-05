import { ArrowUpRight } from "lucide-react";
import type { Experiment } from "../types";
import { Badge, format } from "./common";

export function ExperimentTable({
  items,
  selected = [],
  toggle,
}: {
  items: Experiment[];
  selected?: string[];
  toggle?: (id: string) => void;
}) {
  return (
    <div className="table-scroll">
      <table className="experiment-table">
        <thead>
          <tr>
            {toggle && (
              <th>
                <span className="sr-only">Select</span>
              </th>
            )}
            <th>Model / experiment</th>
            <th>Context</th>
            <th>Batch</th>
            <th>
              TTFT <span>ms</span>
            </th>
            <th>
              Throughput <span>tok/s</span>
            </th>
            <th>
              Peak VRAM <span>MiB</span>
            </th>
            <th>Source</th>
            <th>
              <span className="sr-only">Open</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {items.map((r) => (
            <tr key={r.id}>
              {toggle && (
                <td>
                  <input
                    type="checkbox"
                    aria-label={`Select ${r.config.model} context ${r.config.input_length} batch ${r.config.batch_size} ${r.id.slice(0, 6)}`}
                    checked={selected.includes(r.id)}
                    disabled={selected.length >= 6 && !selected.includes(r.id)}
                    onChange={() => toggle(r.id)}
                  />
                </td>
              )}
              <td>
                <a className="model-link" href={`#/experiments/${r.id}`}>
                  {r.config.model}
                </a>
                <small>
                  {r.config.backend} <span className="divider">/</span>{" "}
                  {r.id.slice(0, 8)}
                </small>
              </td>
              <td className="mono">{format(r.config.input_length, 0)}</td>
              <td className="mono">{r.config.batch_size}</td>
              <td className="mono">{format(r.metrics.ttft_ms, 2)}</td>
              <td className="mono">
                {format(r.metrics.prefill_throughput_tokens_per_sec, 0)}
              </td>
              <td className="mono">
                {format(r.metrics.peak_gpu_memory_mb, 0)}
              </td>
              <td>
                <Badge simulated={r.simulated} />
              </td>
              <td>
                <a
                  className="icon-button"
                  aria-label={`Open experiment ${r.id.slice(0, 8)}`}
                  href={`#/experiments/${r.id}`}
                >
                  <ArrowUpRight size={16} aria-hidden="true" />
                </a>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
