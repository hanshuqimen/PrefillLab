"""Versioned, finite, JSON-safe experiment schemas."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ExperimentConfig(Schema):
    model: str = Field(default="mock-7b", min_length=1, max_length=512)
    backend: str = Field(default="mock", min_length=1, max_length=64)
    device: str = Field(default="auto", pattern=r"^(auto|cpu|cuda(?::\d+)?)$")
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    batch_size: int = Field(default=1, ge=1, le=256)
    input_length: int = Field(default=4096, ge=1, le=131072)
    output_length: int = Field(default=1, ge=1, le=4096)
    warmup_runs: int = Field(default=2, ge=0, le=100)
    benchmark_runs: int = Field(default=5, ge=1, le=1000)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    profile: bool = True
    mock_hidden_size: int = Field(default=4096, ge=64, le=32768)
    mock_num_layers: int = Field(default=32, ge=1, le=256)

    @model_validator(mode="after")
    def budget(self) -> "ExperimentConfig":
        if self.batch_size * self.input_length > 2**22:
            raise ValueError("batch_size × input_length must not exceed 4,194,304 tokens")
        return self


class EnvironmentInfo(Schema):
    python: str
    os: str
    cpu: str | None = None
    ram_mb: float | None = None
    gpu: str | None = None
    gpu_memory_mb: float | None = None
    driver: str | None = None
    cuda: str | None = None
    pytorch: str | None = None
    transformers: str | None = None
    backend_version: str | None = None
    git_commit: str | None = None
    prefilllab_version: str = "0.1.0"


class Statistics(Schema):
    mean: float = Field(ge=0)
    median: float = Field(ge=0)
    p50: float = Field(ge=0)
    p90: float = Field(ge=0)
    min: float = Field(ge=0)
    max: float = Field(ge=0)
    std: float = Field(ge=0)


class RunSample(Schema):
    prefill_latency_ms: float = Field(gt=0)
    ttft_ms: float = Field(gt=0)
    decode_latency_ms: float = Field(default=0, ge=0)
    gpu_elapsed_ms: float | None = Field(default=None, ge=0)


class MemoryProfile(Schema):
    peak_gpu_memory_mb: float | None = Field(default=None, ge=0)
    allocated_gpu_memory_mb: float | None = Field(default=None, ge=0)
    process_rss_mb: float | None = Field(default=None, ge=0)


class Metrics(MemoryProfile):
    ttft_ms: float = Field(gt=0)
    first_token_latency_ms: float = Field(gt=0)
    prefill_latency_ms: float = Field(gt=0)
    prefill_throughput_tokens_per_sec: float = Field(gt=0)
    decode_latency_ms: float = Field(ge=0)
    gpu_utilization: float | None = Field(default=None, ge=0, le=100)
    memory_utilization: float | None = Field(default=None, ge=0, le=100)
    memory_capacity_percent: float | None = Field(default=None, ge=0, le=100)
    tokens: int = Field(ge=1)


class ModuleCategory(StrEnum):
    QKV = "qkv"
    ATTENTION = "attention"
    OUTPUT = "output"
    MLP = "mlp"
    NORM = "norm"
    EMBEDDING = "embedding"
    OTHER = "other"


class LayerProfile(Schema):
    name: str
    category: ModuleCategory
    time_ms: float = Field(ge=0)
    calls: int = Field(ge=1)


class KernelProfile(Schema):
    name: str
    cpu_time_ms: float = Field(ge=0)
    device_time_ms: float | None = Field(default=None, ge=0)
    calls: int = Field(ge=1)


class Breakdown(Schema):
    qkv_time_ms: float | None = Field(default=None, ge=0)
    attention_time_ms: float | None = Field(default=None, ge=0)
    output_time_ms: float | None = Field(default=None, ge=0)
    mlp_time_ms: float | None = Field(default=None, ge=0)
    norm_time_ms: float | None = Field(default=None, ge=0)
    embedding_time_ms: float | None = Field(default=None, ge=0)
    other_time_ms: float | None = Field(default=None, ge=0)
    source: str = "unavailable"


class BottleneckReport(Schema):
    primary_bottleneck: str
    confidence: float = Field(ge=0, le=1)
    evidence: list[str]
    heuristic: bool = True


class Recommendation(Schema):
    title: str
    reason: str
    confidence: float = Field(ge=0, le=1)
    category: str


class ExperimentResult(Schema):
    schema_version: Literal[1] = 1
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    config: ExperimentConfig
    device: str
    simulated: bool
    metrics: Metrics
    statistics: dict[str, Statistics]
    samples: list[RunSample]
    breakdown: Breakdown = Field(default_factory=Breakdown)
    layers: list[LayerProfile] = Field(default_factory=list)
    kernels: list[KernelProfile] = Field(default_factory=list)
    environment: EnvironmentInfo
    analysis: BottleneckReport | None = None
    recommendations: list[Recommendation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    methodology: str = (
        "Prepared synthetic tokens; local request TTFT; no network/queue/tokenization."
    )

    @model_validator(mode="after")
    def validate_provenance(self) -> "ExperimentResult":
        if self.config.backend == "mock" and not self.simulated:
            raise ValueError("Mock backend results must be marked simulated.")
        if len(self.samples) != self.config.benchmark_runs:
            raise ValueError("Sample count must equal benchmark_runs.")
        if self.metrics.tokens != self.config.batch_size * self.config.input_length:
            raise ValueError("Token count must equal batch_size × input_length.")
        if self.timestamp.tzinfo is None:
            raise ValueError("Experiment timestamp must include a timezone.")
        return self

    @property
    def ttft_ms(self) -> float:
        return self.metrics.ttft_ms
