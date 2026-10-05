from typing import Any, Protocol


class Profiler(Protocol):
    """Minimal lifecycle for experiment-level profiler extensions."""

    def start(self) -> None: ...
    def stop(self) -> dict[str, Any]: ...
