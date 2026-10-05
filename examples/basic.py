"""Run a CPU-friendly simulation and export the complete experiment."""

from pathlib import Path

from prefilllab import Benchmark
from prefilllab.storage.repository import write_json

result = Benchmark(model="mock-7b", backend="mock").run(input_length=4096)
print(f"Simulated={result.simulated}; local TTFT={result.ttft_ms:.2f} ms")
write_json(result, Path("results/basic.json"))
