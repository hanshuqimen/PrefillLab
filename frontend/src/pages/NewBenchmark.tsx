import { Play, Info, RefreshCw } from "lucide-react";
import { useState, type FormEvent } from "react";
import { api } from "../api/client";
import { ErrorState, Panel } from "../components/common";
import type { Config, SystemInfo } from "../types";

const initial: Config = {
  model: "mock-7b",
  backend: "mock",
  device: "auto",
  dtype: "float32",
  batch_size: 1,
  input_length: 4096,
  output_length: 1,
  warmup_runs: 2,
  benchmark_runs: 5,
  seed: 42,
  profile: true,
  mock_hidden_size: 4096,
  mock_num_layers: 32,
};
export function NewBenchmark({
  system,
  changed,
}: {
  system: SystemInfo | null;
  changed: () => void;
}) {
  const [config, setConfig] = useState(initial);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const update = (key: keyof Config, value: string | number | boolean) =>
    setConfig((current) => ({ ...current, [key]: value }));
  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const result = await api.run(config);
      changed();
      window.location.hash = `/experiments/${result.id}`;
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Benchmark failed.");
      setBusy(false);
    }
  };
  const fields: {
    key: keyof Config;
    label: string;
    min: number;
    max: number;
    hint: string;
  }[] = [
    {
      key: "input_length",
      label: "Input length",
      min: 1,
      max: 131072,
      hint: "Synthetic prompt tokens per request",
    },
    {
      key: "batch_size",
      label: "Batch size",
      min: 1,
      max: 256,
      hint: "Requests in one forward pass",
    },
    {
      key: "output_length",
      label: "Output length",
      min: 1,
      max: 4096,
      hint: "Generated tokens including the first",
    },
    {
      key: "warmup_runs",
      label: "Warmup runs",
      min: 0,
      max: 100,
      hint: "Excluded from the sample statistics",
    },
    {
      key: "benchmark_runs",
      label: "Benchmark runs",
      min: 1,
      max: 1000,
      hint: "Independent measured repetitions",
    },
    {
      key: "seed",
      label: "Random seed",
      min: 0,
      max: 4294967295,
      hint: "Controls synthetic input and simulation",
    },
  ];
  return (
    <>
      <div className="page-title">
        <div>
          <h1>New benchmark</h1>
          <p>Define the workload. Measure the prefill.</p>
        </div>
      </div>
      <div className="benchmark-layout">
        <form onSubmit={submit}>
          <Panel
            title="Experiment configuration"
            subtitle="Use simulation to explore, or a real model to measure."
          >
            {error && (
              <div className="form-error">
                <ErrorState message={error} />
              </div>
            )}
            <fieldset disabled={busy}>
              <div className="form-grid">
                <label>
                  Backend
                  <select
                    aria-label="Backend"
                    value={config.backend}
                    onChange={(e) => {
                      const backend = e.target.value;
                      setConfig((current) => ({
                        ...current,
                        backend,
                        model: backend === "mock" ? "mock-7b" : "gpt2",
                      }));
                    }}
                  >
                    <option value="mock">Mock · simulation</option>
                    <option
                      value="transformers"
                      disabled={
                        !system?.real_models_enabled ||
                        !system.backends.find((b) => b.name === "transformers")
                          ?.available
                      }
                    >
                      Transformers ·{" "}
                      {system?.real_models_enabled
                        ? "real inference"
                        : "API opt-in required"}
                    </option>
                  </select>
                  <small>vLLM and SGLang adapters are on the roadmap.</small>
                </label>
                <label>
                  Model
                  <input
                    aria-label="Model"
                    required
                    maxLength={512}
                    value={config.model}
                    onChange={(e) => update("model", e.target.value)}
                  />
                  <small>Hugging Face ID or a local model path</small>
                </label>
                <label>
                  Device
                  <select
                    value={config.device}
                    onChange={(e) => update("device", e.target.value)}
                  >
                    <option value="auto">Auto</option>
                    <option value="cpu">CPU</option>
                    <option value="cuda">CUDA</option>
                  </select>
                </label>
                <label>
                  Precision
                  <select
                    value={config.dtype}
                    onChange={(e) => update("dtype", e.target.value)}
                  >
                    <option value="float32">float32</option>
                    <option value="float16">float16</option>
                    <option value="bfloat16">bfloat16</option>
                  </select>
                </label>
                {fields.map((field) => (
                  <label key={field.key}>
                    {field.label}
                    <input
                      aria-label={field.label}
                      aria-describedby={`hint-${field.key}`}
                      required
                      type="number"
                      min={field.min}
                      max={field.max}
                      step={1}
                      value={String(config[field.key])}
                      onChange={(e) =>
                        update(
                          field.key,
                          e.target.value === "" ? "" : Number(e.target.value),
                        )
                      }
                    />
                    <small id={`hint-${field.key}`}>{field.hint}</small>
                  </label>
                ))}
              </div>
              {config.backend === "mock" && (
                <div className="form-grid mock-dimensions">
                  <label>
                    Simulated hidden size
                    <input
                      type="number"
                      required
                      min={64}
                      max={32768}
                      value={config.mock_hidden_size}
                      onChange={(e) =>
                        update("mock_hidden_size", Number(e.target.value))
                      }
                    />
                  </label>
                  <label>
                    Simulated layers
                    <input
                      type="number"
                      required
                      min={1}
                      max={256}
                      value={config.mock_num_layers}
                      onChange={(e) =>
                        update("mock_num_layers", Number(e.target.value))
                      }
                    />
                  </label>
                </div>
              )}
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={config.profile}
                  onChange={(e) => update("profile", e.target.checked)}
                />{" "}
                Collect module and operator profiles
              </label>
              <div className="form-footer">
                <span>
                  {config.backend === "mock"
                    ? "Simulated results are labeled in every view."
                    : "The first run may download model weights."}
                </span>
                <button
                  className="button primary"
                  type="submit"
                  disabled={busy}
                >
                  {busy ? (
                    <RefreshCw size={16} className="spin" aria-hidden="true" />
                  ) : (
                    <Play size={16} aria-hidden="true" />
                  )}
                  {busy ? "Running benchmark…" : "Run benchmark"}
                </button>
              </div>
            </fieldset>
          </Panel>
        </form>
        <aside>
          <Panel title="A reproducible measurement">
            <div className="panel-body">
              <div className="workflow">
                <div>
                  <span>1</span>
                  <p>
                    <b>Prepare</b>Load the model and seeded synthetic tokens.
                  </p>
                </div>
                <div>
                  <span>2</span>
                  <p>
                    <b>Warm up</b>Let one-time initialization settle.
                  </p>
                </div>
                <div>
                  <span>3</span>
                  <p>
                    <b>Measure</b>Capture repeated prefill and first-token
                    timings.
                  </p>
                </div>
                <div>
                  <span>4</span>
                  <p>
                    <b>Inspect</b>Profile separately, then preserve the
                    evidence.
                  </p>
                </div>
              </div>
              <div className="small-notice">
                <Info size={17} aria-hidden="true" />
                <p>
                  TTFT here is local inference latency. Network, tokenization,
                  and queue time are excluded.
                </p>
              </div>
            </div>
          </Panel>
          {busy && (
            <div className="notice" role="status">
              Loading, warming up, and measuring. Keep this page open; model
              downloads can take several minutes.
            </div>
          )}
        </aside>
      </div>
    </>
  );
}
