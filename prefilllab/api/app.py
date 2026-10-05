"""Synchronous local benchmark API with explicit opt-in for model downloads."""

import logging
import os
import threading
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from prefilllab.api.demo import seed_demo
from prefilllab.api.schemas import (
    BackendInfo,
    DeleteResponse,
    ExperimentList,
    HealthResponse,
    ModelInfo,
    SystemResponse,
)
from prefilllab.backends.base import backend_catalog
from prefilllab.core.benchmark import Benchmark
from prefilllab.core.environment import collect_environment
from prefilllab.core.models import ExperimentConfig, ExperimentResult
from prefilllab.errors import BackendUnavailableError, PrefillLabError
from prefilllab.storage.repository import ExperimentRepository

logger = logging.getLogger(__name__)


def create_app(db: str | Path | None = None, demo: bool = False) -> FastAPI:
    """Create isolated application state, initializing SQLite only on startup."""
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        repository = ExperimentRepository(db)
        application.state.repository = repository
        try:
            if demo:
                seed_demo(repository)
            yield
        finally:
            repository.close()

    application = FastAPI(
        title="PrefillLab",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )

    @application.middleware("http")
    async def local_boundary(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        origin = request.headers.get("origin")
        if request.method in {"POST", "DELETE", "PUT", "PATCH"} and origin:
            parsed = urlsplit(origin)
            if parsed.netloc != request.headers.get("host") and origin not in {
                "http://localhost:5173",
                "http://127.0.0.1:5173",
            }:
                return JSONResponse({"detail": "Cross-origin write rejected."}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @application.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @application.get("/api/experiments", response_model=ExperimentList)
    def experiments(
        request: Request, limit: int = Query(100, ge=1, le=200), offset: int = Query(0, ge=0)
    ) -> ExperimentList:
        repository = request.app.state.repository
        return ExperimentList(
            items=repository.list(limit, offset),
            total=repository.count(),
            limit=limit,
            offset=offset,
        )

    @application.get("/api/experiments/{experiment_id}", response_model=ExperimentResult)
    def experiment(experiment_id: UUID, request: Request) -> ExperimentResult:
        result = request.app.state.repository.get(str(experiment_id))
        if result is None:
            raise HTTPException(404, "Experiment not found.")
        return result

    @application.post("/api/experiments", response_model=ExperimentResult, status_code=201)
    def import_experiment(result: ExperimentResult, request: Request) -> ExperimentResult:
        request.app.state.repository.save(result)
        return result

    @application.delete("/api/experiments/{experiment_id}", response_model=DeleteResponse)
    def delete_experiment(experiment_id: UUID, request: Request) -> DeleteResponse:
        if not request.app.state.repository.delete(str(experiment_id)):
            raise HTTPException(404, "Experiment not found.")
        return DeleteResponse(deleted=True)

    @application.post("/api/benchmarks/run", response_model=ExperimentResult, status_code=201)
    def run_benchmark(config: ExperimentConfig, request: Request) -> ExperimentResult:
        if config.backend != "mock" and os.environ.get("PREFILLLAB_ALLOW_REAL_MODELS") != "1":
            raise HTTPException(
                403,
                "Real models are disabled for the API. Set PREFILLLAB_ALLOW_REAL_MODELS=1 and restart, or run through the CLI.",
            )
        if not lock.acquire(blocking=False):
            raise HTTPException(409, "A benchmark is already running. Retry when it completes.")
        try:
            result = Benchmark(**config.model_dump()).run()
            request.app.state.repository.save(result)
            return result
        except BackendUnavailableError as exc:
            raise HTTPException(422, str(exc)) from exc
        except PrefillLabError as exc:
            logger.warning("Benchmark failed: %s", exc)
            raise HTTPException(400, str(exc)) from exc
        finally:
            lock.release()

    @application.get("/api/system", response_model=SystemResponse)
    def system() -> SystemResponse:
        return SystemResponse(
            environment=collect_environment(gpu=True),
            backends=[BackendInfo(**item) for item in backend_catalog()],
            real_models_enabled=os.environ.get("PREFILLLAB_ALLOW_REAL_MODELS") == "1",
        )

    @application.get("/api/backends", response_model=list[BackendInfo])
    def backends() -> list[BackendInfo]:
        return [BackendInfo(**item) for item in backend_catalog()]

    @application.get("/api/models", response_model=list[ModelInfo])
    def models(request: Request) -> list[ModelInfo]:
        pairs = {("mock-7b", "mock"), ("mock-13b", "mock")}
        pairs.update(
            (result.config.model, result.config.backend)
            for result in request.app.state.repository.list(200)
        )
        return [ModelInfo(name=name, backend=backend) for name, backend in sorted(pairs)]

    web = Path(__file__).resolve().parents[1] / "web"
    if (web / "assets").exists():
        application.mount("/assets", StaticFiles(directory=web / "assets"), name="assets")

    @application.get("/", response_model=None, include_in_schema=False)
    def dashboard() -> FileResponse:
        if not (web / "index.html").exists():
            raise HTTPException(
                503, "Dashboard bundle missing. Run npm install and npm run build inside frontend/."
            )
        return FileResponse(web / "index.html")

    return application


app = create_app()
