import type { ReactNode } from "react";
import {
  AlertCircle,
  ArrowUpRight,
  FlaskConical,
  RefreshCw,
} from "lucide-react";
import type { Experiment } from "../types";

export const format = (value: number | null | undefined, digits = 1) =>
  value == null
    ? "—"
    : new Intl.NumberFormat("en-US", { maximumFractionDigits: digits }).format(
        value,
      );
export function Badge({ simulated }: { simulated: boolean }) {
  return (
    <span className={`badge ${simulated ? "simulated" : "measured"}`}>
      {simulated ? "Simulated" : "Measured"}
    </span>
  );
}
export function Panel({
  title,
  subtitle,
  action,
  children,
  className = "",
}: {
  title: string;
  subtitle?: string;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel ${className}`}>
      <div className="panel-heading">
        <div>
          <h2>{title}</h2>
          {subtitle && <p>{subtitle}</p>}
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}
export function Metric({
  label,
  value,
  unit,
  note,
}: {
  label: string;
  value: string;
  unit?: string;
  note?: string;
}) {
  return (
    <div className="metric">
      <span>{label}</span>
      <div className="metric-value">
        {value}
        <small>{unit}</small>
      </div>
      {note && <p>{note}</p>}
    </div>
  );
}
export function Empty({
  title = "Your next insight starts with a run.",
  text = "Create a benchmark to begin exploring prefill performance.",
}: {
  title?: string;
  text?: string;
}) {
  return (
    <div className="empty">
      <FlaskConical size={32} aria-hidden="true" />
      <h2>{title}</h2>
      <p>{text}</p>
      <a className="button primary" href="#/new">
        New benchmark <ArrowUpRight size={16} aria-hidden="true" />
      </a>
    </div>
  );
}
export function ErrorState({
  message,
  retry,
}: {
  message: string;
  retry?: () => void;
}) {
  return (
    <div className="notice error" role="alert">
      <AlertCircle size={18} aria-hidden="true" />
      <div>{message}</div>
      {retry && (
        <button className="button" onClick={retry}>
          <RefreshCw size={14} aria-hidden="true" /> Retry
        </button>
      )}
    </div>
  );
}
export function Loading() {
  return (
    <div className="loading" role="status">
      <RefreshCw className="spin" size={18} aria-hidden="true" /> Loading
      experiments…
    </div>
  );
}
export function download(
  name: string,
  text: string,
  type = "application/json",
) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export function exportCSV(items: Experiment[]) {
  const headers = [
    "id",
    "model",
    "backend",
    "device",
    "dtype",
    "simulated",
    "input_length",
    "batch_size",
    "ttft_ms",
    "prefill_latency_ms",
    "prefill_throughput_tokens_per_sec",
    "peak_gpu_memory_mb",
  ];
  const cell = (value: unknown) => {
    const text = String(value ?? "");
    return `"${/^[=+\-@\t\r]/.test(text) ? "'" : ""}${text.replaceAll('"', '""')}"`;
  };
  const rows = items.map((r) => [
    r.id,
    r.config.model,
    r.config.backend,
    r.device,
    r.config.dtype,
    r.simulated,
    r.config.input_length,
    r.config.batch_size,
    r.metrics.ttft_ms,
    r.metrics.prefill_latency_ms,
    r.metrics.prefill_throughput_tokens_per_sec,
    r.metrics.peak_gpu_memory_mb,
  ]);
  download(
    "prefilllab-experiments.csv",
    [
      headers.map(cell).join(","),
      ...rows.map((row) => row.map(cell).join(",")),
    ].join("\n"),
    "text/csv",
  );
}
