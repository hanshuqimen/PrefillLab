from collections import defaultdict
from typing import Any

import numpy as np

from prefilllab.core.models import ExperimentResult


class ScalingAnalyzer:
    def analyze(
        self, results: list[ExperimentResult], axis: str = "input_length", metric: str = "ttft_ms"
    ) -> list[dict[str, Any]]:
        if axis not in {"input_length", "batch_size"}:
            raise ValueError("axis must be input_length or batch_size")
        groups: dict[tuple[Any, ...], list[tuple[int, float]]] = defaultdict(list)
        for result in results:
            config = result.config
            fixed = config.model_dump(
                exclude={axis, "seed", "warmup_runs", "benchmark_runs", "profile"}
            )
            key = (tuple(sorted(fixed.items())), result.device, result.simulated)
            value = getattr(result.metrics, metric)
            if value is not None and value > 0:
                groups[key].append((getattr(config, axis), value))
        reports = []
        for key, points in groups.items():
            merged: dict[int, list[float]] = defaultdict(list)
            for x, y in points:
                merged[x].append(y)
            points = sorted((x, float(np.mean(ys))) for x, ys in merged.items())
            exponent = None
            trend = "unknown"
            if len(points) >= 3:
                context_values, metric_values = np.asarray(points, dtype=float).T
                exponent = float(np.polyfit(np.log(context_values), np.log(metric_values), 1)[0])
                trend = (
                    "linear"
                    if 0.85 <= exponent <= 1.15
                    else "superlinear"
                    if exponent > 1.15
                    else "sublinear"
                )
            reports.append(
                {
                    "axis": axis,
                    "metric": metric,
                    "fixed": dict(key[0]),
                    "device": key[1],
                    "simulated": key[2],
                    "points": points,
                    "trend": trend,
                    "exponent": exponent,
                    "doubling_factor": 2**exponent if exponent is not None else None,
                    "note": "Descriptive log-log fit; at least three distinct points required. Not a hardware bottleneck diagnosis.",
                }
            )
        return reports
