"""Separate instrumented pass: exclusive nested module timing and operator summaries."""

from collections import defaultdict
from time import perf_counter
from typing import Any

from prefilllab.backends.base import InferenceBackend
from prefilllab.classification.modules import ModuleClassifier
from prefilllab.core.models import Breakdown, KernelProfile, LayerProfile, ModuleCategory
from prefilllab.errors import ProfilerError


class TorchProfiler:
    def __init__(self, classifier: ModuleClassifier | None = None) -> None:
        self.classifier = classifier or ModuleClassifier()

    def collect(
        self, backend: InferenceBackend, inputs: Any
    ) -> tuple[Breakdown, list[LayerProfile], list[KernelProfile]]:
        import torch

        cuda = backend.device.startswith("cuda")
        records: list[dict[str, Any]] = []
        stack: list[dict[str, Any]] = []
        handles = []
        names = {
            id(module): (name or "model", type(module).__name__)
            for name, module in backend.model.named_modules()
        }

        def before(module: Any, args: Any) -> None:
            begin: Any = torch.cuda.Event(enable_timing=True) if cuda else perf_counter()
            if cuda:
                begin.record()
            entry = {"module": id(module), "start": begin, "children": [], "end": None}
            if stack:
                stack[-1]["children"].append(entry)
            stack.append(entry)
            records.append(entry)

        def after(module: Any, args: Any, output: Any) -> None:
            entry = stack.pop()
            if cuda:
                entry["end"] = torch.cuda.Event(enable_timing=True)
                entry["end"].record()
            else:
                entry["end"] = perf_counter()

        try:
            for module in backend.model.modules():
                handles.extend(
                    [module.register_forward_pre_hook(before), module.register_forward_hook(after)]
                )
            backend.synchronize()
            output = backend.prefill(inputs)
            backend.synchronize()
            del output
        except Exception as exc:
            raise ProfilerError(f"Module profiling failed: {exc}") from exc
        finally:
            for handle in handles:
                handle.remove()

        def inclusive(entry: dict[str, Any]) -> float:
            return (
                float(entry["start"].elapsed_time(entry["end"]))
                if cuda
                else (entry["end"] - entry["start"]) * 1000
            )

        grouped: dict[str, tuple[float, int, ModuleCategory]] = {}
        totals: dict[ModuleCategory, float] = defaultdict(float)
        for entry in records:
            name, class_name = names[entry["module"]]
            category = self.classifier.classify(name, class_name)
            exclusive = max(
                0.0, inclusive(entry) - sum(inclusive(child) for child in entry["children"])
            )
            previous, calls, _ = grouped.get(name, (0.0, 0, category))
            grouped[name] = previous + exclusive, calls + 1, category
            totals[category] += exclusive
        layers = [
            LayerProfile(name=name, category=category, time_ms=elapsed, calls=calls)
            for name, (elapsed, calls, category) in grouped.items()
        ]
        breakdown = Breakdown(
            **{f"{category.value}_time_ms": totals[category] for category in ModuleCategory},
            source="cuda_module_events_exclusive" if cuda else "cpu_module_hooks_exclusive",
        )
        activities = [torch.profiler.ProfilerActivity.CPU]
        if cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        try:
            with torch.profiler.profile(activities=activities, record_shapes=True) as profile:
                output = backend.prefill(inputs)
                backend.synchronize()
                del output
            kernels = [
                KernelProfile(
                    name=event.key,
                    cpu_time_ms=event.self_cpu_time_total / 1000,
                    device_time_ms=getattr(event, "self_device_time_total", 0) / 1000
                    if cuda
                    else None,
                    calls=event.count,
                )
                for event in profile.key_averages()
            ]
        except Exception as exc:
            raise ProfilerError(f"PyTorch operator profiling failed: {exc}") from exc
        kernels.sort(key=lambda item: item.device_time_ms or item.cpu_time_ms, reverse=True)
        return breakdown, sorted(layers, key=lambda item: item.time_ms, reverse=True), kernels
