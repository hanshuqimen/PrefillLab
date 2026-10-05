# Measurement methodology

## Timing boundaries

Model/tokenizer loading, synthetic token construction, and host-to-device transfer happen before warmups and timing. Each measured request synchronizes the selected CUDA device before starting a wall clock timer, runs one full prompt forward with `use_cache=True`, synchronizes, and records prefill latency. Greedy argmax produces the first token; another synchronization establishes TTFT. `first_token_latency_ms` is an explicit alias of this local TTFT. This is not serving TTFT: queueing, admission, tokenization, networking, and scheduler delays are excluded.

CUDA events separately report stream elapsed prefill time in each raw sample. Wall clock prefill includes host launch overhead and synchronization. Synchronizing between forward and token selection is deliberate for separating phases, and can slightly increase local TTFT versus a fully asynchronous production implementation. CUDA execution is single-device in this release.

Remaining output tokens are generated with the returned KV cache and a full-length attention mask. Decode latency records this tail separately and never changes TTFT. Generation ignores EOS to keep output length fixed. The default output length of one measures no decode iterations. Decode validates the model context limit. PyTorch runs in `eval()` and `inference_mode()`.

## Sampling and statistics

Two warmup requests and five measured requests are the defaults. Warmups exercise the complete requested output length. Statistics include mean, median, p50, p90, minimum, maximum, and population standard deviation (`ddof=0`), using NumPy percentiles. With five samples, tail percentiles are descriptive, not reliable service SLO estimates. Increase repetitions and repeat whole experiments to assess stability.

Prompt throughput = `batch_size × input_length / mean_prefill_seconds`. It is not output-token throughput or concurrent serving throughput. Each sample uses the same seeded synthetic prompt and a fresh cache, so these experiments do not measure prefix-cache reuse. Seeded inputs do not guarantee bit-identical GPU kernels or timing across devices, software versions, or runs.

## Module and operator profiling

Latency runs are uninstrumented. Two additional prefill-only passes collect (1) nested module hook timing, and (2) PyTorch profiler operator statistics. Those timings do not contribute to request statistics or peak memory. CPU hooks use a wall clock; CUDA hooks place events on the current stream and synchronize once after the pass. Direct child durations are subtracted from parent inclusive duration, preventing nested attention/QKV/MLP double counting. Unknown functional work is attributed to the exclusive time of its enclosing module.

The classifier uses module names and class names, with custom rules taking precedence. Fused kernels, shared modules, asynchronous execution, compilation, or unusual architectures can limit attribution. CPU hook overhead is visible in the instrumented pass. CUDA event instrumentation measures current-stream elapsed intervals; it is not a kernel trace and cannot reliably partition concurrent multi-stream execution. Breakdown totals need not equal uninstrumented mean latency. Results preserve the source, layer names, and raw times instead of scaling a profile to look exact.

`kernels` contains aggregated PyTorch **operator** summaries, not a complete CUDA kernel trace. Each entry records exclusive CPU time and, where available, exclusive device time. Names are retained verbatim. GPU integration and driver-specific NVML behavior require validation on target hardware; CPU integration exercises a real tiny local GPT-2 model with random weights without network access.

## Memory and utilization

`peak_gpu_memory_mb`: maximum PyTorch allocated bytes during the measured request series, divided by 2²⁰. Counters reset after warmups. The peak includes model, input, prompt cache, logits, and requested decode allocations. It excludes subsequent instrumented passes.

`allocated_gpu_memory_mb`: allocation remaining at the end of measured requests after request-local outputs have been released. `process_rss_mb`: process resident set size at collection time, not CPU peak memory. CPU experiments leave GPU allocation fields null.

`gpu_utilization` and `memory_utilization`: means of best-effort NVML device-wide samples, every 50 ms during measured requests. NVML memory utilization describes activity, not allocated capacity or HBM bandwidth. The driver's internal averaging period and short experiments may yield sparse or stale samples; other processes can affect values. CUDA device UUID mapping avoids assuming logical indices equal NVML indices. If the selected device cannot be mapped or sampling fails, fields remain null and the error appears in warnings.

`memory_capacity_percent`: peak allocated bytes / physical GPU capacity. This differs from device-wide memory use, reserved allocator memory, and NVML activity. MiB fields retain the historic `_mb` schema naming.

## Simulation

The mock adapter is an analytical educational model, not calibrated to real GPUs. Linear projection/MLP cost and quadratic attention cost depend on configured hidden size, layer count, context, and batch efficiency. Seeded bounded jitter produces repeatable sample variation. Memory estimates represent weights, KV cache, and a fixed workspace allowance. Mock model names are labels; use `mock_hidden_size` and `mock_num_layers` to change geometry. Demo presets explicitly configure different geometry for 7B-like and 13B-like workloads. Simulated metrics and conclusions are labeled throughout CLI, JSON, reports, and dashboard.

## Heuristics and scaling

An attention-dominated profile, low sampled GPU utilization, or high measured allocation capacity can motivate further experiments. They do not establish compute-bound versus memory-bandwidth-bound execution. Recommendations contain reasons and confidence but are not calibrated probabilities. Prefix caching and chunked prefill advice explicitly depends on real serving workloads that this tool does not simulate.

Scaling fits group fixed configurations, device, and data source. Three distinct axis values are required for a descriptive log-log exponent and doubling factor; fewer points produce `unknown`. The fit does not model uncertainty, thermal behavior, compilation thresholds, or OOM limits. Compare like-for-like dtype, device, model, output length, and backend before interpreting changes. Never use simulated/measured differences as hardware evidence.

## References

- [PyTorch CUDA events](https://docs.pytorch.org/docs/stable/generated/torch.cuda.Event.html)
- [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html)
- [Hugging Face cache explanation](https://huggingface.co/docs/transformers/cache_explanation)

## Reproducibility checklist

Keep full JSON alongside CSV; pin package versions; preserve the model revision and weights externally; choose representative prompts; control background GPU load; document power/clock settings; repeat trials and report spread; label simulation; preserve missing values. Environment collection records available Python, OS, CPU/RAM, GPU capacity, driver/CUDA, PyTorch/Transformers/backend versions, PrefillLab version, and installed source Git commit. An installed wheel without Git metadata records a null commit. Model revision hashes, power/clock state, and workload trace replay are not captured automatically in v0.1.0.
