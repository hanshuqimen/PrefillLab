from uuid import NAMESPACE_URL, uuid5

from prefilllab import Benchmark
from prefilllab.storage.repository import ExperimentRepository


def seed_demo(repository: ExperimentRepository) -> None:
    """Add stable demo IDs once, preserving all existing research history."""
    for name, hidden, layers in (("mock-7b", 4096, 32), ("mock-13b", 5120, 40)):
        for batch in (1, 2, 4):
            for length in (512, 1024, 2048, 4096, 8192):
                experiment_id = uuid5(
                    NAMESPACE_URL, f"prefilllab:v0.1.0:demo:{name}:{batch}:{length}"
                )
                if repository.get(str(experiment_id)) is not None:
                    continue
                result = Benchmark(model=name, mock_hidden_size=hidden, mock_num_layers=layers).run(
                    input_length=length,
                    batch_size=batch,
                )
                result.id = experiment_id
                repository.save(result)
