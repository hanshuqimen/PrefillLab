"""Best-effort sampled NVML utilization; bandwidth utilization differs from capacity."""

import logging
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class NVMLProfiler:
    def __init__(self, device: str, interval: float = 0.05) -> None:
        self.device, self.interval = device, interval
        self.samples: list[tuple[float, float]] = []
        self.error: str | None = None
        self._event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.device.startswith("cuda"):
            return
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def _sample(self) -> None:
        try:
            import pynvml
            import torch

            pynvml.nvmlInit()
            try:
                # UUID mapping respects CUDA_VISIBLE_DEVICES and selected logical device.
                properties = torch.cuda.get_device_properties(self.device)
                gpu_uuid = str(properties.uuid)
                handle = pynvml.nvmlDeviceGetHandleByUUID(gpu_uuid)
                while not self._event.is_set():
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.samples.append((float(util.gpu), float(util.memory)))
                    self._event.wait(self.interval)
            finally:
                pynvml.nvmlShutdown()
        except Exception as exc:
            # Optional vendor API may raise driver-specific exceptions; surface the failure.
            self.error = f"NVML unavailable: {type(exc).__name__}: {exc}"
            logger.debug(self.error)

    def stop(self) -> dict[str, Any]:
        self._event.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if not self.samples:
            return {"gpu_utilization": None, "memory_utilization": None}
        average = np.mean(self.samples, axis=0)
        return {"gpu_utilization": float(average[0]), "memory_utilization": float(average[1])}
