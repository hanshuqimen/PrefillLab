"""Deterministic educational simulation. Never measured hardware performance."""

from typing import Any

import numpy as np

from prefilllab.core.models import Breakdown, ExperimentConfig, MemoryProfile, RunSample


class MockBackend:
    simulated = True
    device = "simulated"
    model: Any = None

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.name = "mock-7b"

    def load_model(self, model: str, dtype: str, device: str) -> None:
        self.name = model

    def prepare_inputs(self, batch_size: int, input_length: int) -> tuple[int, int]:
        return batch_size, input_length

    def prefill(self, inputs: tuple[int, int]) -> tuple[int, int]:
        return inputs

    def first_token(self, output: Any) -> int:
        return 1

    def decode(self, inputs: Any, output: Any, token: Any, max_new_tokens: int) -> list[int]:
        return [1] * max_new_tokens

    def generate(self, inputs: Any, max_new_tokens: int) -> list[int]:
        return self.decode(inputs, self.prefill(inputs), 1, max_new_tokens)

    def synchronize(self) -> None:
        return None

    def cleanup(self) -> None:
        return None

    def simulate(self, config: ExperimentConfig) -> tuple[RunSample, Breakdown, MemoryProfile]:
        # Model dimensions and constants describe a hypothetical device, not a calibrated GPU.
        length, batch = config.input_length, config.batch_size
        scale = (config.mock_hidden_size / 4096) ** 2 * config.mock_num_layers / 32
        efficiency = 1 + 0.48 * (batch - 1)
        linear = length / 1024 * batch / efficiency * scale
        attention = 3.6 * (length / 1024) ** 2 * batch / efficiency * scale
        jitter = float(self.rng.uniform(0.97, 1.03))
        breakdown = Breakdown(
            qkv_time_ms=4.4 * linear * jitter,
            attention_time_ms=attention * jitter,
            output_time_ms=1.5 * linear * jitter,
            mlp_time_ms=11.2 * linear * jitter,
            norm_time_ms=0.5 * linear * jitter,
            embedding_time_ms=0.3 * linear * jitter,
            other_time_ms=(1.2 + 0.5 * linear) * jitter,
            source="simulation",
        )
        prefill = sum(v for k, v in breakdown.model_dump().items() if k.endswith("_ms"))
        sample = RunSample(
            prefill_latency_ms=prefill,
            ttft_ms=prefill + 0.7 * jitter,
            decode_latency_ms=(config.output_length - 1) * 2.5 * scale * jitter,
        )
        bytes_per_element = 4 if config.dtype == "float32" else 2
        weights_mb = (
            12 * config.mock_num_layers * config.mock_hidden_size**2 * bytes_per_element / 2**20
        )
        kv_mb = (
            2
            * config.mock_num_layers
            * batch
            * length
            * config.mock_hidden_size
            * bytes_per_element
            / 2**20
        )
        return (
            sample,
            breakdown,
            MemoryProfile(
                peak_gpu_memory_mb=weights_mb + kv_mb + 256,
                allocated_gpu_memory_mb=weights_mb + kv_mb,
            ),
        )
