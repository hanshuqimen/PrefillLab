import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Experiment, MetricKey } from "../types";
import { format } from "../components/common";

const colors = [
  "#82aaff",
  "#d9a577",
  "#76c7bf",
  "#b99ae6",
  "#e68d9a",
  "#b3cb7a",
];
export function ScalingChart({
  items,
  metric,
  unit,
}: {
  items: Experiment[];
  metric: MetricKey;
  unit: string;
}) {
  const groups = new Map<string, Experiment[]>();
  for (const result of items) {
    const { input_length: _, ...fixed } = result.config;
    const key = JSON.stringify([fixed, result.device, result.simulated]);
    groups.set(key, [...(groups.get(key) ?? []), result]);
  }
  const series = [...groups.values()].slice(0, 6);
  const contexts = [...new Set(items.map((r) => r.config.input_length))].sort(
    (a, b) => a - b,
  );
  const data = contexts.map((context) => {
    const row: Record<string, number | null> = { context };
    series.forEach((group, i) => {
      const values = group
        .filter((r) => r.config.input_length === context)
        .map((r) => r.metrics[metric])
        .filter((v): v is number => v != null);
      row[`series${i}`] = values.length
        ? values.reduce((a, b) => a + b, 0) / values.length
        : null;
    });
    return row;
  });
  if (!series.length)
    return (
      <div className="chart-empty">
        No data for this selection. Try another model or source.
      </div>
    );
  return (
    <>
      <div
        className="chart"
        aria-label={`${metric} versus context length`}
        role="img"
      >
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={data}
            margin={{ top: 15, right: 18, bottom: 12, left: 0 }}
          >
            <CartesianGrid
              stroke="#292d38"
              vertical={false}
              strokeDasharray="3 5"
            />
            <XAxis
              dataKey="context"
              tickFormatter={(v) =>
                Number(v) >= 1024 ? `${Number(v) / 1024}k` : String(v)
              }
              stroke="#363b47"
              tick={{ fill: "#969fae", fontSize: 11 }}
              tickLine={false}
              axisLine={false}
              label={{
                value: "Context length · tokens",
                position: "insideBottom",
                offset: -10,
                fill: "#969fae",
                fontSize: 11,
              }}
            />
            <YAxis
              tickFormatter={(v) => format(Number(v), 0)}
              tick={{ fill: "#969fae", fontSize: 11 }}
              width={64}
              stroke="#363b47"
              tickLine={false}
              axisLine={false}
            />
            <Tooltip
              contentStyle={{
                background: "#191d26",
                border: "1px solid #3c4353",
                borderRadius: 8,
              }}
              labelFormatter={(label) => `${format(Number(label), 0)} tokens`}
              formatter={(value) => [`${format(Number(value), 2)} ${unit}`]}
            />
            <Legend
              verticalAlign="top"
              align="right"
              iconType="plainline"
              wrapperStyle={{ fontSize: 11, paddingBottom: 18 }}
            />
            {series.map((group, i) => (
              <Line
                key={i}
                type="linear"
                dataKey={`series${i}`}
                name={`${group[0].config.model} · b${group[0].config.batch_size} · ${group[0].config.dtype} · ${group[0].simulated ? "sim." : group[0].device}`}
                stroke={colors[i % colors.length]}
                strokeWidth={2.5}
                strokeDasharray={group[0].simulated ? "5 3" : undefined}
                dot={{ r: 4, fill: "#14171e", strokeWidth: 2 }}
                activeDot={{ r: 6 }}
                isAnimationActive={false}
                connectNulls={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="chart-footnote">
        Each line holds configuration constant.{" "}
        {groups.size > 6
          ? "Showing the first 6 configuration groups; refine filters."
          : "Repeated configurations are averaged."}
      </p>
    </>
  );
}
