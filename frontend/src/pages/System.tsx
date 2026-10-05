import { Check, Minus } from "lucide-react";
import { Panel } from "../components/common";
import type { SystemInfo } from "../types";

export function System({ system }: { system: SystemInfo | null }) {
  return (
    <>
      <div className="page-title">
        <div>
          <h1>System</h1>
          <p>Know the environment behind your measurements.</p>
        </div>
        <span className="badge measured">Local workspace</span>
      </div>
      <div className="two-column">
        <Panel
          title="Environment"
          subtitle="Missing optional dependencies do not block simulation."
        >
          <dl className="key-values">
            {Object.entries(system?.environment ?? {}).map(([key, value]) => (
              <div key={key}>
                <dt>{key.replaceAll("_", " ")}</dt>
                <dd>{value ?? <span className="muted">Unavailable</span>}</dd>
              </div>
            ))}
          </dl>
        </Panel>
        <Panel
          title="Inference backends"
          subtitle="Installed packages and implemented capabilities."
        >
          <div className="backend-list">
            {system?.backends.map((backend) => (
              <div key={backend.name}>
                <div className="backend-icon">
                  {backend.available ? (
                    <Check size={18} aria-hidden="true" />
                  ) : (
                    <Minus size={18} aria-hidden="true" />
                  )}
                </div>
                <div>
                  <b>{backend.name}</b>
                  <p>
                    {backend.implemented
                      ? backend.available
                        ? `Available · ${backend.version ?? "custom adapter"}`
                        : "Install optional dependencies"
                      : "Future adapter · not runnable in v0.1.0"}
                  </p>
                </div>
                <span
                  className={`status-dot ${backend.available ? "ready" : ""}`}
                />
              </div>
            ))}
          </div>
          <div className="panel-body">
            <h3>Real-model API access</h3>
            <p>
              {system?.real_models_enabled
                ? "Enabled for this trusted local server."
                : "Disabled by default. Run real models through the CLI, or set PREFILLLAB_ALLOW_REAL_MODELS=1 before starting the server."}
            </p>
            <pre>pip install -e '.[transformers]'</pre>
            <p>
              Use <code>prefilllab doctor</code> for optional CUDA, driver,
              NVML, and FlashAttention diagnostics.
            </p>
          </div>
        </Panel>
      </div>
    </>
  );
}
