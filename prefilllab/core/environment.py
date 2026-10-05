"""Best-effort reproducibility metadata without importing GPU packages by default."""

import importlib.metadata
import logging
import platform
import subprocess
from pathlib import Path

import psutil

from prefilllab.core.models import EnvironmentInfo

logger = logging.getLogger(__name__)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_environment(backend: str = "mock", gpu: bool = False) -> EnvironmentInfo:
    """Collect available metadata; unavailable optional measurements remain null."""
    result = EnvironmentInfo(
        python=platform.python_version(),
        os=platform.platform(),
        cpu=platform.processor() or platform.machine(),
        ram_mb=psutil.virtual_memory().total / 2**20,
        pytorch=package_version("torch"),
        transformers=package_version("transformers"),
        backend_version="0.1.0" if backend == "mock" else package_version(backend),
    )
    try:
        process = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if process.returncode == 0:
            result.git_commit = process.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.debug("Git metadata unavailable: %s", exc)
    if gpu and result.pytorch:
        try:
            import torch

            result.cuda = torch.version.cuda
            if torch.cuda.is_available():
                properties = torch.cuda.get_device_properties(torch.cuda.current_device())
                result.gpu = properties.name
                result.gpu_memory_mb = properties.total_memory / 2**20
        except (ImportError, RuntimeError, OSError) as exc:
            logger.debug("CUDA metadata unavailable: %s", exc)
    if gpu:
        try:
            import pynvml

            pynvml.nvmlInit()
            try:
                driver = pynvml.nvmlSystemGetDriverVersion()
                result.driver = driver.decode() if isinstance(driver, bytes) else str(driver)
            finally:
                pynvml.nvmlShutdown()
        except (ImportError, OSError, RuntimeError) as exc:
            logger.debug("NVML metadata unavailable: %s", exc)
        except Exception as exc:
            # pynvml defines driver-specific exception classes only when installed.
            logger.debug("NVML driver error: %s", exc)
    return result
