"""Example research workflow; all mock data is explicitly simulated."""

from pathlib import Path

from prefilllab import Sweep
from prefilllab.storage.repository import export_frame


def main() -> None:
    Path("results").mkdir(exist_ok=True)
    results = Sweep(
        input_lengths=[128, 256, 512, 1024, 2048, 4096, 8192], batch_sizes=[1, 2, 4]
    ).run()
    export_frame(results).to_csv("results/scaling-study.csv", index=False)


if __name__ == "__main__":
    main()
