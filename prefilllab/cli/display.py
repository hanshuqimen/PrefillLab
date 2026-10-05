from rich.console import Console
from rich.table import Table

from prefilllab.core.models import ExperimentResult

console = Console()


def result_table(result: ExperimentResult) -> None:
    console.print(
        "[bold]PrefillLab[/bold] · "
        + ("[yellow]SIMULATED[/yellow]" if result.simulated else "MEASURED")
    )
    table = Table(show_header=False)
    for name, value in {
        "Model": result.config.model,
        "Backend": result.config.backend,
        "Device": result.device,
        "Input length": result.config.input_length,
        "Batch size": result.config.batch_size,
        "TTFT": f"{result.ttft_ms:.3f} ms",
        "Prefill latency": f"{result.metrics.prefill_latency_ms:.3f} ms",
        "Throughput": f"{result.metrics.prefill_throughput_tokens_per_sec:,.0f} tok/s",
        "Peak GPU memory": f"{result.metrics.peak_gpu_memory_mb:.1f} MiB"
        if result.metrics.peak_gpu_memory_mb is not None
        else "Unavailable",
    }.items():
        table.add_row(name, str(value))
    console.print(table)
    console.print("[bold]Breakdown[/bold] · " + result.breakdown.source)
    times = {
        key: value
        for key, value in result.breakdown.model_dump().items()
        if key.endswith("_ms") and value is not None
    }
    total = sum(times.values())
    for name, value in times.items():
        console.print(
            f"  {name.removesuffix('_time_ms'):12} {value:9.3f} ms   {value / total if total else 0:6.1%}"
        )
    if result.analysis:
        console.print("[bold]Bottleneck[/bold] (heuristic): " + result.analysis.primary_bottleneck)
    console.print("[bold]Recommendations[/bold]")
    for item in result.recommendations:
        console.print(f"  • {item.title}: {item.reason}")
    for warning in result.warnings:
        console.print(warning, style="yellow", markup=False)
