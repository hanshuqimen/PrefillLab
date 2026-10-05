from prefilllab.core.models import EnvironmentInfo, ExperimentResult, Schema


class HealthResponse(Schema):
    status: str = "ok"
    version: str = "0.1.0"


class ExperimentList(Schema):
    items: list[ExperimentResult]
    total: int
    limit: int
    offset: int


class BackendInfo(Schema):
    name: str
    available: bool
    implemented: bool
    version: str | None


class ModelInfo(Schema):
    name: str
    backend: str


class SystemResponse(Schema):
    environment: EnvironmentInfo
    backends: list[BackendInfo]
    real_models_enabled: bool


class DeleteResponse(Schema):
    deleted: bool
