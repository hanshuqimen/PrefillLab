import os
from typing import Any

import psutil

from prefilllab.core.models import MemoryProfile


class MemoryProfiler:
    def __init__(self, device: str) -> None:
        self.device = device

    def start(self) -> None:
        if self.device.startswith("cuda"):
            import torch

            torch.cuda.reset_peak_memory_stats(self.device)

    def stop(self) -> dict[str, Any]:
        profile = MemoryProfile(
            process_rss_mb=psutil.Process(os.getpid()).memory_info().rss / 2**20
        )
        if self.device.startswith("cuda"):
            import torch

            profile.peak_gpu_memory_mb = torch.cuda.max_memory_allocated(self.device) / 2**20
            profile.allocated_gpu_memory_mb = torch.cuda.memory_allocated(self.device) / 2**20
        return profile.model_dump()
