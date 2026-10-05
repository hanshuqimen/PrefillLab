import math

import pytest
from pydantic import ValidationError

from prefilllab import Benchmark, ExperimentConfig, ExperimentResult, Sweep
from prefilllab.analyzers.scaling import ScalingAnalyzer
from prefilllab.backends.base import create_backend
from prefilllab.classification.modules import ModuleClassifier
from prefilllab.core.models import ModuleCategory
from prefilllab.errors import BackendUnavailableError


@pytest.mark.parametrize(
    "kwargs",
    [
        {"input_length": 0},
        {"batch_size": -1},
        {"benchmark_runs": 0},
        {"device": "bogus"},
        {"seed": -1},
        {"dtype": "int8"},
    ],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValidationError):
        ExperimentConfig(**kwargs)


def test_mock_determinism_and_statistics():
    first = Benchmark().run(input_length=1024)
    second = Benchmark().run(input_length=1024)
    assert first.simulated and first.device == "simulated"
    assert first.samples == second.samples
    assert len(first.samples) == 5
    assert first.statistics["ttft_ms"].mean == first.ttft_ms
    assert (
        first.statistics["ttft_ms"].min
        <= first.statistics["ttft_ms"].p90
        <= first.statistics["ttft_ms"].max
    )
    assert first.metrics.prefill_throughput_tokens_per_sec == pytest.approx(
        1024 / first.metrics.prefill_latency_ms * 1000
    )
    assert first.analysis.heuristic
    assert first.recommendations and all(item.reason for item in first.recommendations)
    assert ExperimentResult.model_validate_json(first.model_dump_json()) == first


def test_mock_output_and_dimensions():
    small = Benchmark().run(input_length=1024, output_length=1)
    large = Benchmark().run(input_length=8192, output_length=8)
    assert large.ttft_ms > small.ttft_ms
    assert large.metrics.decode_latency_ms > 0
    assert small.metrics.decode_latency_ms == 0
    assert large.metrics.peak_gpu_memory_mb > small.metrics.peak_gpu_memory_mb
    assert large.analysis.primary_bottleneck == "Attention-dominated prefill"


def test_scaling_groups_do_not_mix_batches_or_sources():
    results = Sweep(input_lengths=[512, 1024, 2048], batch_sizes=[1, 2]).run()
    groups = ScalingAnalyzer().analyze(results)
    assert len(groups) == 2
    assert all(group["exponent"] is not None and len(group["points"]) == 3 for group in groups)
    short = ScalingAnalyzer().analyze(results[:2])
    assert short[0]["trend"] == "unknown"


@pytest.mark.parametrize(
    "name,category",
    [
        ("layers.0.self_attn.q_proj", ModuleCategory.QKV),
        ("transformer.h.0.attn.c_proj", ModuleCategory.OUTPUT),
        ("layers.0.mlp.down_proj", ModuleCategory.MLP),
        ("model.norm", ModuleCategory.NORM),
    ],
)
def test_classifier(name, category):
    assert ModuleClassifier().classify(name) == category
    assert (
        ModuleClassifier([(r"q_proj$", ModuleCategory.OTHER)]).classify("q_proj")
        == ModuleCategory.OTHER
    )


def test_unknown_and_future_adapters_fail_explicitly():
    for name in ("sglang", "vllm", "unknown"):
        with pytest.raises(BackendUnavailableError):
            create_backend(name, 42)


def test_finite_schema():
    result = Benchmark().run()
    data = result.model_dump()
    data["metrics"]["ttft_ms"] = math.nan
    with pytest.raises(ValidationError):
        ExperimentResult.model_validate(data)
