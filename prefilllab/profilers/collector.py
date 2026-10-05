"""Coordinator for independent profiler lifecycles."""

from typing import Any

from prefilllab.profilers.base import Profiler


class ProfileCollector:
    def __init__(self, profilers: list[Profiler]) -> None:
        self.profilers = profilers

    def start(self) -> None:
        for profiler in self.profilers:
            profiler.start()

    def stop(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for profiler in reversed(self.profilers):
            result.update(profiler.stop())
        return result
