import pytest
from pydantic import ValidationError

from prefilllab import Benchmark, ExperimentResult
from prefilllab.backends.base import register_backend
from prefilllab.errors import BenchmarkError


def test_adapter_cleanup_after_failure():
    class FailingAdapter:
        cleaned = False

        def load_model(self, model, dtype, device):
            raise BenchmarkError("Deliberate load failure")

        def cleanup(self):
            self.cleaned = True

    adapter = FailingAdapter()
    register_backend("test-failure", lambda seed: adapter)
    with pytest.raises(BenchmarkError, match="Deliberate"):
        Benchmark(backend="test-failure").run()
    assert adapter.cleaned


@pytest.mark.parametrize("change", ["source", "samples", "tokens", "timestamp"])
def test_imported_result_invariants(change):
    data = Benchmark().run().model_dump(mode="json")
    if change == "source":
        data["simulated"] = False
    elif change == "samples":
        data["samples"] = []
    elif change == "tokens":
        data["metrics"]["tokens"] = 1
    else:
        data["timestamp"] = "2026-10-05T00:00:00"
    with pytest.raises(ValidationError):
        ExperimentResult.model_validate(data)
