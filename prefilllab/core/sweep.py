from itertools import product
from typing import Any

from prefilllab.core.benchmark import Benchmark
from prefilllab.core.models import ExperimentResult


class Sweep:
    """Cartesian sweep; each experiment loads independently to preserve isolation."""

    def __init__(
        self,
        model: str = "mock-7b",
        backend: str = "mock",
        input_lengths: list[int] | None = None,
        batch_sizes: list[int] | None = None,
        **options: Any,
    ) -> None:
        self.benchmark = Benchmark(model=model, backend=backend, **options)
        self.input_lengths = input_lengths or [128, 256, 512, 1024]
        self.batch_sizes = batch_sizes or [1]

    def run(self) -> list[ExperimentResult]:
        return [
            self.benchmark.run(input_length=length, batch_size=batch)
            for batch, length in product(self.batch_sizes, self.input_lengths)
        ]
