import json

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from prefilllab import Benchmark
from prefilllab.api.app import create_app
from prefilllab.cli.app import app
from prefilllab.reports.html import render_report
from prefilllab.storage.repository import ExperimentRepository, export_frame


def test_storage_roundtrip_pagination_delete(tmp_path):
    repository = ExperimentRepository(tmp_path / "history.db")
    result = Benchmark().run()
    repository.save(result)
    repository.save(result)
    assert repository.count() == 1
    assert repository.get(str(result.id)) == result
    assert repository.list(limit=1) == [result]
    assert repository.list(offset=1) == []
    assert repository.delete(str(result.id))
    assert not repository.delete(str(result.id))
    repository.close()
    frame = export_frame([result])
    assert frame.iloc[0]["simulated"]
    assert "ttft_ms_p90" in frame.columns


def test_api_crud_and_validation(tmp_path):
    with TestClient(create_app(tmp_path / "api.db")) as client:
        assert client.get("/api/health").json()["status"] == "ok"
        assert client.get("/api/docs").status_code == 200
        assert client.get("/api/backends").json()[0]["available"]
        assert client.get("/api/models").status_code == 200
        assert client.get("/api/system").status_code == 200
        assert client.get("/api/experiments").json()["total"] == 0
        response = client.post("/api/benchmarks/run", json={"input_length": 512})
        assert response.status_code == 201, response.text
        result = response.json()
        assert result["simulated"]
        assert client.get(f"/api/experiments/{result['id']}").json() == result
        assert client.get("/api/experiments?limit=1").json()["total"] == 1
        assert client.post("/api/experiments", json=result).status_code == 201
        assert client.post("/api/benchmarks/run", json={"input_length": -1}).status_code == 422
        assert (
            client.post("/api/benchmarks/run", json={"backend": "transformers"}).status_code == 403
        )
        assert (
            client.post(
                "/api/benchmarks/run", json={}, headers={"Origin": "https://evil.example"}
            ).status_code
            == 403
        )
        assert client.delete(f"/api/experiments/{result['id']}").json()["deleted"]
        assert client.get(f"/api/experiments/{result['id']}").status_code == 404
        assert client.get("/api/experiments?limit=201").status_code == 422


def test_demo_seed_is_idempotent(tmp_path):
    for _ in range(2):
        with TestClient(create_app(tmp_path / "demo.db", demo=True)) as client:
            response = client.get("/api/experiments").json()
            assert response["total"] == 30
            assert all(result["simulated"] for result in response["items"])


def test_demo_preserves_existing_research(tmp_path):
    database = tmp_path / "research.db"
    repository = ExperimentRepository(database)
    existing = Benchmark(model="my-existing-experiment").run()
    repository.save(existing)
    repository.close()
    for _ in range(2):
        with TestClient(create_app(database, demo=True)) as client:
            assert client.get("/api/experiments").json()["total"] == 31
            assert (
                client.get(f"/api/experiments/{existing.id}").json()["config"]["model"]
                == existing.config.model
            )


def test_cli_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("PREFILLLAB_DB", str(tmp_path / "cli.db"))
    runner = CliRunner()
    assert runner.invoke(app, ["--help"]).exit_code == 0
    assert runner.invoke(app, ["doctor", "--format", "json"]).exit_code == 0
    first = tmp_path / "first.json"
    outcome = runner.invoke(
        app, ["run", "--input-length", "512", "--format", "json", "--output", str(first)]
    )
    assert outcome.exit_code == 0, outcome.output
    assert json.loads(outcome.stdout)["simulated"]
    second = tmp_path / "second.json"
    assert (
        runner.invoke(app, ["run", "--input-length", "1024", "--output", str(second)]).exit_code
        == 0
    )
    compare = runner.invoke(app, ["compare", str(first), str(second), "--format", "json"])
    assert compare.exit_code == 0
    assert len(json.loads(compare.stdout)["metrics"]) == 3
    html = tmp_path / "report.html"
    assert runner.invoke(app, ["report", str(first), "--output", str(html)]).exit_code == 0
    assert html.exists() and "<svg" in html.read_text(encoding="utf-8")
    sweep = runner.invoke(
        app,
        [
            "sweep",
            "--input-lengths",
            "128,256,512",
            "--batch-sizes",
            "1,2",
            "--output-dir",
            str(tmp_path / "sweep"),
        ],
    )
    assert sweep.exit_code == 0, sweep.output
    assert len(list((tmp_path / "sweep").glob("experiment_*.json"))) == 6
    assert (tmp_path / "sweep" / "scaling.json").exists()
    invalid = runner.invoke(app, ["run", "--input-length", "-1", "--format", "json"])
    assert invalid.exit_code == 1
    assert "error" in json.loads(invalid.stdout)


def test_report_escapes_untrusted_model(tmp_path):
    result = Benchmark(model='<script>alert("x")</script>').run()
    output = render_report(result, tmp_path / "report.html").read_text(encoding="utf-8")
    assert '<script>alert("x")</script>' not in output
    assert "&lt;script&gt;" in output
    assert "SIMULATED" in output
