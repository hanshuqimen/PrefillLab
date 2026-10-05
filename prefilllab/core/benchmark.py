"""Warmups, independent samples, separate instrumentation, and reliable cleanup."""

import logging
from typing import Any

from prefilllab.analyzers.bottleneck import BottleneckAnalyzer
from prefilllab.analyzers.recommendations import RecommendationEngine
from prefilllab.backends.base import create_backend
from prefilllab.backends.mock import MockBackend
from prefilllab.core.environment import collect_environment
from prefilllab.core.models import (
    Breakdown,
    ExperimentConfig,
    ExperimentResult,
    KernelProfile,
    LayerProfile,
    Metrics,
)
from prefilllab.errors import BenchmarkError, PrefillLabError, ProfilerError
from prefilllab.profilers.collector import ProfileCollector
from prefilllab.profilers.memory import MemoryProfiler
from prefilllab.profilers.nvml import NVMLProfiler
from prefilllab.profilers.timer import TimerProfiler, summarize
from prefilllab.profilers.torch_profiler import TorchProfiler

logger = logging.getLogger(__name__)


class Benchmark:
    """Run local inference experiments. Heavy dependencies are loaded only when requested."""

    def __init__(self, model: str = "mock-7b", backend: str = "mock", **options: Any) -> None:
        self.defaults = {"model": model, "backend": backend, **options}

    def run(self, **options: Any) -> ExperimentResult:
        config = ExperimentConfig(**{**self.defaults, **options})
        backend = create_backend(config.backend, config.seed)
        samples = []
        warnings = []
        breakdown = Breakdown()
        layers: list[LayerProfile] = []
        kernels: list[KernelProfile] = []
        memory: dict[str, Any] = {}
        utilization: dict[str, Any] = {}
        try:
            logger.info("Loading model %s (%s)...", config.model, config.backend)
            backend.load_model(config.model, config.dtype, config.device)
            inputs = backend.prepare_inputs(config.batch_size, config.input_length)
            for index in range(config.warmup_runs):
                logger.info("Warmup %d/%d", index + 1, config.warmup_runs)
                if isinstance(backend, MockBackend):
                    backend.simulate(config)
                else:
                    TimerProfiler().measure(backend, inputs, config.output_length)
            nvml = NVMLProfiler(backend.device)
            collector = ProfileCollector([MemoryProfiler(backend.device), nvml])
            collector.start()
            try:
                for index in range(config.benchmark_runs):
                    logger.info("Benchmark %d/%d", index + 1, config.benchmark_runs)
                    if isinstance(backend, MockBackend):
                        sample, breakdown, profile = backend.simulate(config)
                        memory = profile.model_dump()
                    else:
                        sample = TimerProfiler().measure(backend, inputs, config.output_length)
                    samples.append(sample)
            finally:
                captured = collector.stop()
            if not backend.simulated:
                memory = {key: value for key, value in captured.items() if key.endswith("_mb")}
                utilization = {
                    key: captured[key] for key in ("gpu_utilization", "memory_utilization")
                }
                if nvml.error:
                    warnings.append(nvml.error)
                if nvml.samples:
                    warnings.append(
                        f"NVML: {len(nvml.samples)} samples, device-wide utilization; includes other processes."
                    )
                if config.profile:
                    logger.info("Profiling separate prefill passes...")
                    try:
                        breakdown, layers, kernels = TorchProfiler().collect(backend, inputs)
                    except ProfilerError as exc:
                        warnings.append(str(exc))
                if backend.device == "cpu":
                    warnings.append("Device: CPU. GPU metrics unavailable.")
            else:
                utilization = {
                    "gpu_utilization": min(95.0, 30 + config.batch_size * 8),
                    "memory_utilization": None,
                }
                warnings.append(
                    "SIMULATED: analytical educational model, not measured or hardware-calibrated."
                )
                if not config.profile:
                    breakdown = Breakdown()
            environment = collect_environment(config.backend, gpu=backend.device.startswith("cuda"))
            logger.info("Analyzing results...")
            statistics = {
                name: summarize([getattr(sample, name) for sample in samples])
                for name in ("ttft_ms", "prefill_latency_ms", "decode_latency_ms")
            }
            prefill = statistics["prefill_latency_ms"].mean
            tokens = config.batch_size * config.input_length
            peak = memory.get("peak_gpu_memory_mb")
            capacity = (
                min(100.0, peak / environment.gpu_memory_mb * 100)
                if peak is not None and environment.gpu_memory_mb
                else None
            )
            result = ExperimentResult(
                config=config,
                device=backend.device,
                simulated=backend.simulated,
                metrics=Metrics(
                    ttft_ms=statistics["ttft_ms"].mean,
                    first_token_latency_ms=statistics["ttft_ms"].mean,
                    prefill_latency_ms=prefill,
                    prefill_throughput_tokens_per_sec=tokens / prefill * 1000,
                    decode_latency_ms=statistics["decode_latency_ms"].mean,
                    tokens=tokens,
                    memory_capacity_percent=capacity,
                    **memory,
                    **utilization,
                ),
                statistics=statistics,
                samples=samples,
                breakdown=breakdown,
                layers=layers,
                kernels=kernels,
                environment=environment,
                warnings=warnings,
            )
            result.analysis = BottleneckAnalyzer().analyze(result)
            result.recommendations = RecommendationEngine().recommend(result)
            return result
        except PrefillLabError:
            raise
        except (RuntimeError, ValueError, OSError) as exc:
            raise BenchmarkError(f"Benchmark failed: {exc}") from exc
        finally:
            backend.cleanup()
