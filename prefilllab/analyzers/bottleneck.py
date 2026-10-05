from prefilllab.core.models import BottleneckReport, ExperimentResult


class BottleneckAnalyzer:
    def analyze(self, result: ExperimentResult) -> BottleneckReport:
        metrics = result.metrics
        values = result.breakdown.model_dump()
        total = sum(
            value for key, value in values.items() if key.endswith("_ms") and value is not None
        )
        evidence = ["Heuristic analysis; module proportions are from a separate instrumented pass."]
        if result.simulated:
            evidence.append("Simulation only: these observations describe a hypothetical device.")
        if metrics.memory_capacity_percent is not None and metrics.memory_capacity_percent > 85:
            return BottleneckReport(
                primary_bottleneck="Memory capacity pressure",
                confidence=0.75,
                evidence=[
                    *evidence,
                    f"Allocated peak uses {metrics.memory_capacity_percent:.1f}% of device capacity.",
                ],
            )
        attention = result.breakdown.attention_time_ms
        if (
            total
            and attention is not None
            and attention / total > 0.4
            and result.config.input_length >= 2048
        ):
            return BottleneckReport(
                primary_bottleneck="Attention-dominated prefill",
                confidence=0.72,
                evidence=[
                    *evidence,
                    f"Attention accounts for {attention / total:.1%} at context {result.config.input_length}.",
                ],
            )
        if (
            metrics.gpu_utilization is not None
            and metrics.gpu_utilization < 40
            and result.config.batch_size <= 2
        ):
            return BottleneckReport(
                primary_bottleneck="Possible GPU underutilization",
                confidence=0.5,
                evidence=[
                    *evidence,
                    "Low sampled utilization with a small batch; short requests may undersample NVML.",
                ],
            )
        if (
            total
            and result.breakdown.mlp_time_ms is not None
            and result.breakdown.mlp_time_ms / total > 0.4
        ):
            return BottleneckReport(
                primary_bottleneck="MLP-dominated prefill",
                confidence=0.65,
                evidence=[*evidence, "MLP accounts for more than 40% of classified module time."],
            )
        return BottleneckReport(
            primary_bottleneck="Insufficient evidence",
            confidence=0.2,
            evidence=[
                *evidence,
                "No measured rule establishes a dominant bottleneck; compute vs bandwidth requires hardware counters.",
            ],
        )
