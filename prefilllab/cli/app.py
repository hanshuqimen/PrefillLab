"""Installed CLI: stable JSON on stdout, progress and errors on stderr."""

import functools
import json
import logging
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Any

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from prefilllab.analyzers.scaling import ScalingAnalyzer
from prefilllab.backends.base import backend_catalog
from prefilllab.cli.display import console, result_table
from prefilllab.core.benchmark import Benchmark
from prefilllab.core.environment import collect_environment, package_version
from prefilllab.core.models import ExperimentResult
from prefilllab.core.sweep import Sweep
from prefilllab.errors import PrefillLabError
from prefilllab.reports.html import render_report
from prefilllab.storage.repository import ExperimentRepository, export_frame, write_json

app = typer.Typer(help="Understand where your LLM prefill time goes.", no_args_is_help=True)
stderr = Console(stderr=True)


class OutputFormat(StrEnum):
    table = "table"
    json = "json"


def friendly(function: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except (PrefillLabError, ValidationError, OSError, ValueError) as exc:
            if kwargs.get("format") == OutputFormat.json:
                typer.echo(json.dumps({"error": {"type": type(exc).__name__, "message": str(exc)}}))
            else:
                stderr.print(f"Error: {exc}", markup=False)
            raise typer.Exit(1) from exc

    return wrapped


@app.callback()
def configure(
    verbose: bool = typer.Option(False, "--verbose", help="Enable debug logging."),
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO, format="%(message)s", force=True
    )


@app.command()
@friendly
def run(
    model: str = typer.Option("mock-7b"),
    backend: str = typer.Option("mock"),
    device: str = typer.Option("auto"),
    dtype: str = typer.Option("float32"),
    input_length: int = typer.Option(4096),
    batch_size: int = typer.Option(1),
    output_length: int = typer.Option(1),
    warmup_runs: int = typer.Option(2),
    benchmark_runs: int = typer.Option(5),
    seed: int = typer.Option(42),
    profile: bool = typer.Option(True, "--profile/--no-profile"),
    output: Path | None = typer.Option(None),
    format: OutputFormat = typer.Option(OutputFormat.table),
) -> None:
    """Run and persist a repeatable local benchmark."""
    result = Benchmark(model, backend).run(
        device=device,
        dtype=dtype,
        input_length=input_length,
        batch_size=batch_size,
        output_length=output_length,
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
        seed=seed,
        profile=profile,
    )
    repository = ExperimentRepository()
    try:
        repository.save(result)
    finally:
        repository.close()
    if output:
        write_json(result, output)
    if format == OutputFormat.json:
        typer.echo(result.model_dump_json(indent=2))
    else:
        result_table(result)


def positive_list(value: str) -> list[int]:
    numbers = [int(item.strip()) for item in value.split(",")]
    if not numbers or any(number <= 0 for number in numbers):
        raise ValueError("Sweep dimensions must be comma-separated positive integers.")
    return list(dict.fromkeys(numbers))


@app.command()
@friendly
def sweep(
    model: str = typer.Option("mock-7b"),
    backend: str = typer.Option("mock"),
    input_lengths: str = typer.Option("512,1024,2048,4096"),
    batch_sizes: str = typer.Option("1"),
    device: str = typer.Option("auto"),
    dtype: str = typer.Option("float32"),
    output_length: int = typer.Option(1),
    warmup_runs: int = typer.Option(2),
    benchmark_runs: int = typer.Option(5),
    seed: int = typer.Option(42),
    profile: bool = typer.Option(True, "--profile/--no-profile"),
    output_dir: Path = typer.Option(Path("results")),
    format: OutputFormat = typer.Option(OutputFormat.table),
) -> None:
    """Sweep context and batch, saving JSON, CSV, and descriptive scaling fits."""
    lengths, batches = positive_list(input_lengths), positive_list(batch_sizes)
    if len(lengths) * len(batches) > 256:
        raise ValueError(
            "A CLI sweep is limited to 256 experiments. Split larger grids explicitly."
        )
    results = Sweep(
        model,
        backend,
        lengths,
        batches,
        device=device,
        dtype=dtype,
        output_length=output_length,
        warmup_runs=warmup_runs,
        benchmark_runs=benchmark_runs,
        seed=seed,
        profile=profile,
    ).run()
    output_dir.mkdir(parents=True, exist_ok=True)
    repository = ExperimentRepository()
    try:
        for result in results:
            name = output_dir / f"experiment_{result.id}"
            write_json(result, name.with_suffix(".json"))
            export_frame([result]).to_csv(name.with_suffix(".csv"), index=False)
            repository.save(result)
    finally:
        repository.close()
    export_frame(results).to_csv(output_dir / "sweep.csv", index=False)
    scaling = [
        report
        for axis in ("input_length", "batch_size")
        for metric in ("ttft_ms", "prefill_throughput_tokens_per_sec", "peak_gpu_memory_mb")
        for report in ScalingAnalyzer().analyze(results, axis, metric)
    ]
    (output_dir / "scaling.json").write_text(json.dumps(scaling, indent=2), encoding="utf-8")
    if format == OutputFormat.json:
        typer.echo(
            json.dumps(
                {"experiments": [r.model_dump(mode="json") for r in results], "scaling": scaling}
            )
        )
    else:
        table = Table("Model", "Context", "Batch", "TTFT (ms)", "Throughput (tok/s)", "Source")
        for result in results:
            table.add_row(
                result.config.model,
                str(result.config.input_length),
                str(result.config.batch_size),
                f"{result.ttft_ms:.2f}",
                f"{result.metrics.prefill_throughput_tokens_per_sec:,.0f}",
                "SIMULATED" if result.simulated else "MEASURED",
            )
        console.print(table)
        console.print(f"Saved {len(results)} experiments to {output_dir.resolve()}")


@app.command()
@friendly
def compare(
    files: list[Path] = typer.Argument(...), format: OutputFormat = typer.Option(OutputFormat.table)
) -> None:
    """Compare existing experiment JSON files relative to the first file."""
    if len(files) < 2:
        raise ValueError("Choose at least two experiment files.")
    results = [
        ExperimentResult.model_validate_json(path.read_text(encoding="utf-8")) for path in files
    ]
    rows: list[dict[str, Any]] = []
    for metric in ("ttft_ms", "prefill_throughput_tokens_per_sec", "peak_gpu_memory_mb"):
        baseline = getattr(results[0].metrics, metric)
        values = [getattr(result.metrics, metric) for result in results]
        deltas = [
            (value / baseline - 1) * 100 if value is not None and baseline else None
            for value in values
        ]
        rows.append({"metric": metric, "values": values, "delta_percent": deltas})
    mixed = len({result.simulated for result in results}) > 1
    if format == OutputFormat.json:
        typer.echo(
            json.dumps(
                {
                    "experiments": [str(r.id) for r in results],
                    "mixed_sources": mixed,
                    "metrics": rows,
                }
            )
        )
    else:
        if mixed:
            console.print(
                "Warning: simulated and measured sources differ; deltas are not hardware evidence."
            )
        table = Table(
            "Metric",
            *[f"{r.config.model} ({'simulated' if r.simulated else 'measured'})" for r in results],
        )
        for row in rows:
            table.add_row(
                row["metric"],
                *[
                    f"{value:,.2f} ({delta:+.1f}%)" if delta is not None else "Unavailable"
                    for value, delta in zip(row["values"], row["delta_percent"], strict=True)
                ],
            )
        console.print(table)


@app.command()
@friendly
def report(
    file: Path = typer.Argument(...), output: Path = typer.Option(Path("prefilllab-report.html"))
) -> None:
    """Generate a self-contained offline HTML report with charts."""
    result = ExperimentResult.model_validate_json(file.read_text(encoding="utf-8"))
    console.print(str(render_report(result, output).resolve()))


@app.command()
@friendly
def doctor(format: OutputFormat = typer.Option(OutputFormat.table)) -> None:
    """Inspect optional inference packages and device availability without failing on absence."""
    environment = collect_environment(gpu=True)
    checks: dict[str, Any] = {
        "Python": environment.python,
        "PyTorch": environment.pytorch,
        "CUDA": environment.cuda,
        "GPU": environment.gpu,
        "Driver": environment.driver,
        "NVML": package_version("nvidia-ml-py"),
        "Transformers": environment.transformers,
        "vLLM": package_version("vllm"),
        "SGLang": package_version("sglang"),
        "FlashAttention": package_version("flash-attn"),
    }
    if format == OutputFormat.json:
        typer.echo(
            json.dumps(
                {
                    "checks": checks,
                    "environment": environment.model_dump(),
                    "backends": backend_catalog(),
                    "offline_simulation_available": True,
                }
            )
        )
    else:
        table = Table("PrefillLab environment", "Value")
        for name, value in checks.items():
            table.add_row(name, str(value) if value is not None else "Unavailable (optional)")
        console.print(table)


@app.command()
@friendly
def serve(
    demo: bool = typer.Option(False),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000, min=1, max=65535),
) -> None:
    """Start FastAPI and the bundled React dashboard; --demo seeds simulated history."""
    import uvicorn

    from prefilllab.api.app import create_app

    uvicorn.run(create_app(demo=demo), host=host, port=port)


def main() -> None:
    app()
