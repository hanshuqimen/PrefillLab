from typing import Protocol

from prefilllab.core.models import ExperimentResult, Recommendation


class OptimizationAdvisor(Protocol):
    """Extension point for workload/hardware-aware advice without changing inference."""

    def recommend(self, result: ExperimentResult) -> list[Recommendation]: ...


class RecommendationEngine:
    def recommend(self, result: ExperimentResult) -> list[Recommendation]:
        config, metrics = result.config, result.metrics
        suggestions: list[Recommendation] = []

        def add(title: str, reason: str, confidence: float, category: str) -> None:
            if result.simulated:
                reason = "Simulation-based hypothesis. " + reason
            suggestions.append(
                Recommendation(title=title, reason=reason, confidence=confidence, category=category)
            )

        if config.batch_size == 1:
            add(
                "Benchmark a larger batch",
                "Batch size is one; a controlled sweep can test throughput gains and latency cost.",
                0.55,
                "batching",
            )
        if config.input_length >= 2048:
            add(
                "Compare an optimized attention implementation",
                "Long contexts increase attention work; compare SDPA or FlashAttention on supported hardware.",
                0.65,
                "attention",
            )
            add(
                "Evaluate chunked prefill in a serving engine",
                "Long prompts may affect scheduling fairness. This local benchmark cannot measure queue interference.",
                0.4,
                "scheduling",
            )
            add(
                "Evaluate prefix caching on representative requests",
                "Caching can help only when real requests share token prefixes; synthetic tokens do not measure cache hits.",
                0.3,
                "caching",
            )
        if metrics.memory_capacity_percent is not None and metrics.memory_capacity_percent > 85:
            add(
                "Reduce context or evaluate quantization",
                "Measured peak allocation is above 85% of GPU capacity. Validate accuracy and context requirements before changing precision.",
                0.75,
                "memory",
            )
        if not suggestions:
            add(
                "Collect a context and batch sweep",
                "One experiment cannot establish a scaling trend or an optimal serving configuration.",
                0.6,
                "measurement",
            )
        return suggestions
