import html
import json
from collections.abc import Mapping
from pathlib import Path

from prefilllab.core.models import ExperimentResult


def render_report(result: ExperimentResult, output: Path) -> Path:
    """Produce a script-free offline HTML report, escaping all imported/user content."""

    def safe(value: object) -> str:
        return html.escape(str(value))

    def table(values: Mapping[str, object]) -> str:
        return (
            "<table>"
            + "".join(
                f"<tr><th>{safe(key)}</th><td>{safe(value) if value is not None else 'Unavailable'}</td></tr>"
                for key, value in values.items()
            )
            + "</table>"
        )

    times = {
        key.removesuffix("_time_ms"): value
        for key, value in result.breakdown.model_dump().items()
        if key.endswith("_ms") and value is not None
    }
    total = sum(times.values())
    bars = (
        "".join(
            f'<div class="bar"><span>{safe(key)}</span><div><i style="width:{value / total * 100 if total else 0:.2f}%"></i></div><b>{value:.2f} ms</b></div>'
            for key, value in times.items()
        )
        or "<p>Module breakdown unavailable.</p>"
    )
    samples = result.samples
    maximum = max(sample.ttft_ms for sample in samples)
    points = " ".join(
        f"{30 + index / max(1, len(samples) - 1) * 720:.1f},{165 - sample.ttft_ms / maximum * 135:.1f}"
        for index, sample in enumerate(samples)
    )
    recommendations = "".join(
        f"<li><strong>{safe(item.title)}</strong><p>{safe(item.reason)} (confidence {item.confidence:.0%})</p></li>"
        for item in result.recommendations
    )
    analysis = result.analysis.model_dump() if result.analysis else {"status": "Unavailable"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PrefillLab · {safe(result.config.model)}</title><style>
    *{{box-sizing:border-box}}body{{background:#0d1118;color:#e7edf6;font:16px/1.6 system-ui;margin:0}}main{{max-width:1000px;margin:auto;padding:44px 24px}}h1{{font-size:36px;margin-bottom:0}}h2{{margin-top:36px}}.badge{{display:inline-block;background:#253047;padding:4px 12px;border-radius:6px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}}.card{{border:1px solid #344056;padding:18px;border-radius:8px}}.card b{{display:block;font:28px monospace;margin-top:8px}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:9px;border-bottom:1px solid #283449;overflow-wrap:anywhere}}th{{width:40%;color:#b0bdd0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#151c29;padding:18px}}.bar{{display:grid;grid-template-columns:110px 1fr 110px;gap:14px;margin:12px 0}}.bar div{{background:#253047;height:16px;margin:auto 0}}.bar i{{display:block;height:16px;background:#7ca5fa}}svg{{width:100%;height:auto}}p{{color:#b0bdd0}}.notice{{border-left:3px solid #e5b970;padding-left:16px}}
    </style></head><body><main><span class="badge">{"SIMULATED · educational data" if result.simulated else "MEASURED · local inference"}</span><h1>PrefillLab</h1><p>{safe(result.config.model)} · {safe(result.timestamp.isoformat())}</p><div class="grid"><div class="card">TTFT<b>{result.ttft_ms:.2f} ms</b></div><div class="card">Prefill latency<b>{result.metrics.prefill_latency_ms:.2f} ms</b></div><div class="card">Throughput<b>{result.metrics.prefill_throughput_tokens_per_sec:,.0f} tok/s</b></div><div class="card">Peak GPU allocation<b>{f"{result.metrics.peak_gpu_memory_mb:.1f} MiB" if result.metrics.peak_gpu_memory_mb is not None else "Unavailable"}</b></div></div>
    <p class="notice">{safe(result.methodology)}</p><h2>First-token samples</h2><svg viewBox="0 0 800 200" role="img" aria-label="TTFT across benchmark runs"><path d="M30 20V170H770" fill="none" stroke="#54627a"/><polyline points="{points}" fill="none" stroke="#7ca5fa" stroke-width="3"/><text x="30" y="195" fill="#b0bdd0">Run 1</text><text x="650" y="195" fill="#b0bdd0">Run {len(samples)}</text><text x="40" y="20" fill="#b0bdd0">Max {maximum:.2f} ms</text></svg><h2>Module breakdown</h2><p>{safe(result.breakdown.source)} · separate profiling pass</p>{bars}<h2>Heuristic bottleneck analysis</h2>{table(analysis)}<h2>Recommendations</h2><ul>{recommendations}</ul><h2>Configuration</h2>{table(result.config.model_dump())}<h2>Environment</h2>{table(result.environment.model_dump())}<h2>Metrics and memory</h2>{table(result.metrics.model_dump())}<h2>Sample statistics</h2><pre>{safe(json.dumps({k: v.model_dump() for k, v in result.statistics.items()}, indent=2))}</pre><h2>Warnings</h2><ul>{"".join(f"<li>{safe(warning)}</li>" for warning in result.warnings)}</ul><details><summary>Full experiment JSON</summary><pre>{safe(result.model_dump_json(indent=2))}</pre></details></main></body></html>''',
        encoding="utf-8",
    )
    return output
