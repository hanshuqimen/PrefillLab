import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ExperimentTable } from "../components/ExperimentTable";
import { format, Panel } from "../components/common";
import type { Experiment, MetricKey } from "../types";

const comparisons: { key: MetricKey; title: string; unit: string }[] = [
  { key: "ttft_ms", title: "Time to first token", unit: "ms" },
  {
    key: "prefill_throughput_tokens_per_sec",
    title: "Prefill throughput",
    unit: "tok/s",
  },
  { key: "peak_gpu_memory_mb", title: "Peak GPU allocation", unit: "MiB" },
];
export function Compare({
  items,
  selected,
  toggle,
}: {
  items: Experiment[];
  selected: string[];
  toggle: (id: string) => void;
}) {
  const chosen = selected
    .map((id) => items.find((r) => r.id === id))
    .filter((r): r is Experiment => r !== undefined);
  const data = chosen.map((r) => ({
    label: `${r.config.model} / ${r.config.input_length} / b${r.config.batch_size} / ${r.id.slice(0, 4)}`,
    ...r.metrics,
  }));
  const baseline = chosen[0];
  return (
    <>
      <div className="page-title">
        <div>
          <h1>Compare experiments</h1>
          <p>Select two to six runs. The first selected row is the baseline.</p>
        </div>
        <span className="badge">{chosen.length} selected</span>
      </div>
      {new Set(chosen.map((r) => r.simulated)).size > 1 && (
        <div className="notice simulation">
          Mixed simulated and measured sources. Relative differences are not
          evidence of real hardware performance.
        </div>
      )}
      <Panel
        title="Choose experiments"
        subtitle="Configuration differences are preserved in the labels."
      >
        <ExperimentTable items={items} selected={selected} toggle={toggle} />
      </Panel>
      {chosen.length >= 2 ? (
        <>
          <div className="comparison-charts">
            {comparisons.map(({ key, title, unit }) => (
              <Panel key={key} title={title} subtitle={unit}>
                <div className="compare-chart" role="img" aria-label={title}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={data}
                      margin={{ left: 0, right: 14, top: 8, bottom: 30 }}
                    >
                      <CartesianGrid stroke="#292d38" vertical={false} />
                      <XAxis
                        dataKey="label"
                        tick={{ fill: "#a7b0c0", fontSize: 10 }}
                        interval={0}
                        angle={-15}
                        textAnchor="end"
                      />
                      <YAxis
                        width={55}
                        tick={{ fill: "#a7b0c0", fontSize: 11 }}
                        tickFormatter={(v) => format(Number(v), 0)}
                      />
                      <Tooltip
                        contentStyle={{
                          background: "#191d26",
                          border: "1px solid #3c4353",
                        }}
                      />
                      <Bar
                        dataKey={key}
                        fill="#82aaff"
                        radius={[3, 3, 0, 0]}
                        isAnimationActive={false}
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Panel>
            ))}
          </div>
          <Panel
            title="Relative differences"
            subtitle={`Baseline: ${baseline.config.model} · ${baseline.id.slice(0, 8)}`}
          >
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Metric</th>
                    {chosen.map((r) => (
                      <th key={r.id}>
                        {r.config.model}
                        <small>{r.id.slice(0, 8)}</small>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {comparisons.map(({ key, title, unit }) => (
                    <tr key={key}>
                      <td>{title}</td>
                      {chosen.map((r) => {
                        const value = r.metrics[key];
                        const base = baseline.metrics[key];
                        const delta =
                          value != null && base
                            ? (value / base - 1) * 100
                            : null;
                        return (
                          <td key={r.id} className="mono">
                            {format(value)} {unit}
                            <small>
                              {delta == null
                                ? "Unavailable"
                                : `${delta >= 0 ? "+" : ""}${format(delta)}%`}
                            </small>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                  {["attention", "mlp"].map((category) => (
                    <tr key={category}>
                      <td>{category.toUpperCase()} share</td>
                      {chosen.map((r) => {
                        const total = Object.entries(r.breakdown).reduce(
                          (sum, [key, value]) =>
                            sum +
                            (key.endsWith("_ms") && typeof value === "number"
                              ? value
                              : 0),
                          0,
                        );
                        const value = r.breakdown[`${category}_time_ms`];
                        return (
                          <td key={r.id} className="mono">
                            {typeof value === "number" && total
                              ? `${format((value / total) * 100)}%`
                              : "Unavailable"}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
        </>
      ) : (
        <div className="selection-hint">
          Select at least two experiments to reveal comparison charts.
        </div>
      )}
    </>
  );
}
