"""Backend protocol and explicit capabilities registry."""

from collections.abc import Callable
from typing import Any, Protocol

from prefilllab.core.environment import package_version
from prefilllab.errors import BackendUnavailableError


class InferenceBackend(Protocol):
    device: str
    simulated: bool
    model: Any

    def load_model(self, model: str, dtype: str, device: str) -> None: ...
    def prepare_inputs(self, batch_size: int, input_length: int) -> Any: ...
    def prefill(self, inputs: Any) -> Any: ...
    def first_token(self, output: Any) -> Any: ...
    def decode(self, inputs: Any, output: Any, token: Any, max_new_tokens: int) -> Any: ...
    def generate(self, inputs: Any, max_new_tokens: int) -> Any: ...
    def synchronize(self) -> None: ...
    def cleanup(self) -> None: ...


_registry: dict[str, Callable[[int], InferenceBackend]] = {}


def register_backend(name: str, factory: Callable[[int], InferenceBackend]) -> None:
    """Register a custom adapter factory receiving the experiment seed."""
    _registry[name] = factory


def create_backend(name: str, seed: int) -> InferenceBackend:
    if name in _registry:
        return _registry[name](seed)
    if name == "mock":
        from prefilllab.backends.mock import MockBackend

        return MockBackend(seed)
    if name == "transformers":
        from prefilllab.backends.transformers import TransformersBackend

        return TransformersBackend(seed)
    if name in ("vllm", "sglang"):
        raise BackendUnavailableError(
            f"{name} is a future adapter; installation alone does not enable it in v0.1.0. "
            "Use transformers or mock, or register_backend() with a custom adapter."
        )
    raise BackendUnavailableError(f"Unknown backend '{name}'. Use mock or transformers.")


def backend_catalog() -> list[dict[str, Any]]:
    return [
        {"name": "mock", "available": True, "implemented": True, "version": "0.1.0"},
        {
            "name": "transformers",
            "available": bool(package_version("transformers") and package_version("torch")),
            "implemented": True,
            "version": package_version("transformers"),
        },
        *[
            {
                "name": name,
                "available": False,
                "implemented": False,
                "version": package_version(name),
            }
            for name in ("vllm", "sglang")
        ],
        *[
            {"name": name, "available": True, "implemented": True, "version": None}
            for name in _registry
        ],
    ]
