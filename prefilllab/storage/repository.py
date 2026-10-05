import os
from pathlib import Path
from typing import Any, cast

import pandas as pd
from sqlalchemy import JSON, Column, String, create_engine, delete, func, select
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import DeclarativeBase, Session

from prefilllab.core.models import ExperimentResult


class Base(DeclarativeBase):
    pass


class ExperimentRecord(Base):
    __tablename__ = "experiments"
    id = Column(String, primary_key=True)
    timestamp = Column(String, index=True, nullable=False)
    payload = Column(JSON, nullable=False)


class ExperimentRepository:
    """Full schema-preserving JSON payloads in SQLite, with bounded ordered pagination."""

    def __init__(self, location: str | Path | None = None) -> None:
        raw = str(
            location
            or os.environ.get("PREFILLLAB_DB", Path.home() / ".prefilllab" / "prefilllab.db")
        )
        if raw.startswith("sqlite:"):
            url = raw
        else:
            path = Path(raw).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            url = "sqlite:///" + path.as_posix()
        self.engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 30})
        Base.metadata.create_all(self.engine)

    def save(self, result: ExperimentResult) -> None:
        with Session(self.engine) as session:
            session.merge(
                ExperimentRecord(
                    id=str(result.id),
                    timestamp=result.timestamp.isoformat(),
                    payload=result.model_dump(mode="json"),
                )
            )
            session.commit()

    def get(self, experiment_id: str) -> ExperimentResult | None:
        with Session(self.engine) as session:
            row = session.get(ExperimentRecord, experiment_id)
            return ExperimentResult.model_validate(row.payload) if row else None

    def list(self, limit: int = 100, offset: int = 0) -> list[ExperimentResult]:
        with Session(self.engine) as session:
            rows = session.scalars(
                select(ExperimentRecord)
                .order_by(ExperimentRecord.timestamp.desc(), ExperimentRecord.id)
                .limit(limit)
                .offset(offset)
            )
            return [ExperimentResult.model_validate(row.payload) for row in rows]

    def delete(self, experiment_id: str) -> bool:
        with Session(self.engine) as session:
            outcome = session.execute(
                delete(ExperimentRecord).where(ExperimentRecord.id == experiment_id)
            )
            session.commit()
            return bool(cast(CursorResult[Any], outcome).rowcount)

    def count(self) -> int:
        with Session(self.engine) as session:
            return int(session.scalar(select(func.count()).select_from(ExperimentRecord)) or 0)

    def close(self) -> None:
        self.engine.dispose()


def export_frame(results: list[ExperimentResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                "id": str(result.id),
                "timestamp": result.timestamp.isoformat(),
                "simulated": result.simulated,
                **result.config.model_dump(),
                "resolved_device": result.device,
                **result.metrics.model_dump(),
                **result.breakdown.model_dump(),
                **{
                    f"{metric}_{stat}": value
                    for metric, stats in result.statistics.items()
                    for stat, value in stats.model_dump().items()
                },
            }
        )
    return pd.DataFrame(rows)


def write_json(result: ExperimentResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
