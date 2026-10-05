from time import perf_counter
from typing import Any

import numpy as np

from prefilllab.backends.base import InferenceBackend
from prefilllab.core.models import RunSample, Statistics


def summarize(values: list[float]) -> Statistics:
    data = np.asarray(values, dtype=float)
    return Statistics(
        mean=float(data.mean()),
        median=float(np.median(data)),
        p50=float(np.percentile(data, 50)),
        p90=float(np.percentile(data, 90)),
        min=float(data.min()),
        max=float(data.max()),
        std=float(data.std(ddof=0)),
    )


class TimerProfiler:
    """Wall clock request timing with synchronization; CUDA events reported separately."""

    def measure(self, backend: InferenceBackend, inputs: Any, output_length: int) -> RunSample:
        cuda = backend.device.startswith("cuda")
        start_event = end_event = None
        if cuda:
            import torch

            start_event, end_event = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
        backend.synchronize()
        start = perf_counter()
        if start_event is not None:
            start_event.record()
        output = backend.prefill(inputs)
        if end_event is not None:
            end_event.record()
        backend.synchronize()
        prefill_end = perf_counter()
        token = backend.first_token(output)
        backend.synchronize()
        first_end = perf_counter()
        backend.decode(inputs, output, token, output_length)
        backend.synchronize()
        decode_end = perf_counter()
        return RunSample(
            prefill_latency_ms=(prefill_end - start) * 1000,
            ttft_ms=(first_end - start) * 1000,
            decode_latency_ms=(decode_end - first_end) * 1000 if output_length > 1 else 0,
            gpu_elapsed_ms=start_event.elapsed_time(end_event)
            if start_event is not None and end_event is not None
            else None,
        )
