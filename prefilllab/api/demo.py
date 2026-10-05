from prefilllab import Benchmark
from prefilllab.storage.repository import ExperimentRepository


def seed_demo(repository: ExperimentRepository) -> None:
    """Seed an empty database only; existing research history is never overwritten."""
    if repository.count():
        return
    for name, hidden, layers in (("mock-7b", 4096, 32), ("mock-13b", 5120, 40)):
        for batch in (1, 2, 4):
            for length in (512, 1024, 2048, 4096, 8192):
                result = Benchmark(model=name, mock_hidden_size=hidden, mock_num_layers=layers).run(
                    input_length=length,
                    batch_size=batch,
                )
                repository.save(result)
