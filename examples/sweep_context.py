"""Export context scaling in a form usable by Pandas or Jupyter."""

from pathlib import Path

from prefilllab import Sweep
from prefilllab.analyzers.scaling import ScalingAnalyzer
from prefilllab.storage.repository import export_frame

results = Sweep(input_lengths=[512, 1024, 2048, 4096], batch_sizes=[1, 2]).run()
Path("results").mkdir(exist_ok=True)
export_frame(results).to_csv("results/context.csv", index=False)
for trend in ScalingAnalyzer().analyze(results):
    print(trend)
