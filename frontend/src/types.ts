export interface Config {
  model: string;
  backend: string;
  device: string;
  dtype: string;
  batch_size: number;
  input_length: number;
  output_length: number;
  warmup_runs: number;
  benchmark_runs: number;
  seed: number;
  profile: boolean;
  mock_hidden_size: number;
  mock_num_layers: number;
}
export interface Metrics {
  ttft_ms: number;
  first_token_latency_ms: number;
  prefill_latency_ms: number;
  prefill_throughput_tokens_per_sec: number;
  peak_gpu_memory_mb: number | null;
  allocated_gpu_memory_mb: number | null;
  process_rss_mb: number | null;
  gpu_utilization: number | null;
  memory_utilization: number | null;
  memory_capacity_percent: number | null;
  decode_latency_ms: number;
  tokens: number;
}
export type MetricKey = keyof Metrics;
export interface Experiment {
  id: string;
  timestamp: string;
  schema_version: number;
  config: Config;
  simulated: boolean;
  device: string;
  metrics: Metrics;
  breakdown: Record<string, number | string | null>;
  statistics: Record<
    string,
    {
      mean: number;
      median: number;
      p50: number;
      p90: number;
      min: number;
      max: number;
      std: number;
    }
  >;
  samples: {
    ttft_ms: number;
    prefill_latency_ms: number;
    decode_latency_ms: number;
    gpu_elapsed_ms: number | null;
  }[];
  layers: { name: string; category: string; time_ms: number; calls: number }[];
  kernels: {
    name: string;
    cpu_time_ms: number;
    device_time_ms: number | null;
    calls: number;
  }[];
  environment: Record<string, string | number | null>;
  analysis: {
    primary_bottleneck: string;
    confidence: number;
    heuristic: boolean;
    evidence: string[];
  } | null;
  recommendations: {
    title: string;
    reason: string;
    confidence: number;
    category: string;
  }[];
  warnings: string[];
  methodology: string;
}
export interface ExperimentList {
  items: Experiment[];
  total: number;
  limit: number;
  offset: number;
}
export interface Backend {
  name: string;
  available: boolean;
  implemented: boolean;
  version: string | null;
}
export interface SystemInfo {
  environment: Record<string, string | number | null>;
  backends: Backend[];
  real_models_enabled: boolean;
}
