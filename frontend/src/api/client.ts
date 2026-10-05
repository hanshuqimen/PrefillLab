import type { Config, Experiment, ExperimentList, SystemInfo } from "../types";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...options?.headers },
  });
  if (!response.ok) {
    let message = `Request failed (${response.status}).`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      message =
        typeof body.detail === "string"
          ? body.detail
          : JSON.stringify(body.detail ?? message);
    } catch {
      /* HTTP status is the fallback when a proxy returns non-JSON content. */
    }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}
export const api = {
  experiments: (offset = 0) =>
    request<ExperimentList>(`/experiments?limit=200&offset=${offset}`),
  experiment: (id: string) =>
    request<Experiment>(`/experiments/${encodeURIComponent(id)}`),
  run: (config: Config) =>
    request<Experiment>("/benchmarks/run", {
      method: "POST",
      body: JSON.stringify(config),
    }),
  remove: (id: string) =>
    request<{ deleted: boolean }>(`/experiments/${encodeURIComponent(id)}`, {
      method: "DELETE",
    }),
  system: () => request<SystemInfo>("/system"),
};
