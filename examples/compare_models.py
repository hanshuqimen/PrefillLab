"""Compare simulated model geometry; these values are not GPU measurements."""

from prefilllab import Benchmark
from prefilllab.storage.repository import export_frame

small = Benchmark(model="mock-7b", mock_hidden_size=4096, mock_num_layers=32).run()
large = Benchmark(model="mock-13b", mock_hidden_size=5120, mock_num_layers=40).run()
print(
    export_frame([small, large])[
        ["model", "simulated", "ttft_ms", "prefill_throughput_tokens_per_sec"]
    ]
)
