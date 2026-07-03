"""
SQLAlchemy 2.0 ORM models for the `jobs` and `results` tables.

Mirrors the schema in docs/SPEC.md section 5. Written in the modern typed style
(`Mapped[...]` + `mapped_column(...)`). These models are shared between the API
and the Celery workers via package import (CLAUDE.md), and they are the single
source of truth Alembic autogenerates migrations from.
"""

import enum
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    DateTime,
    Enum as SAEnum,
    ForeignKey,
    Index,
    Integer,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.db.database import Base


# ─────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────
# Members are UPPERCASE (idiomatic Python) but their VALUES are the lowercase
# strings the API and DB speak. By default SQLAlchemy's Enum persists the member
# *name* ("QUEUED"); `values_callable` below makes it persist the *value*
# ("queued") instead, so the native Postgres ENUM matches SPEC section 5 exactly.
class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelType(str, enum.Enum):
    YOLO = "yolo"
    FACE = "face"
    FIRE = "fire"
    MULTI = "multi"


def _enum_values(enum_cls: type[enum.Enum]) -> list[str]:
    """Return the ``.value`` of each member, for Enum(values_callable=...)."""
    return [member.value for member in enum_cls]


# ─────────────────────────────────────────────────────────────
# Job
# ─────────────────────────────────────────────────────────────
class Job(Base):
    """A single inference job and its lifecycle state."""

    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    status: Mapped[JobStatus] = mapped_column(
        SAEnum(JobStatus, name="job_status", values_callable=_enum_values),
        nullable=False,
        default=JobStatus.QUEUED,
    )
    model_type: Mapped[ModelType] = mapped_column(
        SAEnum(ModelType, name="model_type", values_callable=_enum_values),
        nullable=False,
    )
    input_url: Mapped[str] = mapped_column(Text, nullable=False)
    input_size_kb: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    worker_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )

    # One-to-one: a job has at most one result. cascade delete-orphan mirrors the
    # ON DELETE CASCADE on the FK so removing a Job removes its Result too.
    result: Mapped[Optional["Result"]] = relationship(
        back_populates="job",
        uselist=False,
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (
        Index("idx_jobs_status", "status"),
        # SPEC indexes created_at DESC; a plain btree index serves ORDER BY ...
        # DESC equally well (Postgres scans it backward) and keeps autogenerate
        # output clean.
        Index("idx_jobs_created_at", "created_at"),
    )


# ─────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────
class Result(Base):
    """Inference output for a completed job. Detections are stored as JSONB."""

    __tablename__ = "results"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
    )
    # Array of {label, confidence, bbox: [x1, y1, x2, y2]} objects.
    detections: Mapped[list] = mapped_column(JSONB, nullable=False)
    annotated_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    inference_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    model_version: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    job: Mapped["Job"] = relationship(back_populates="result")

    __table_args__ = (Index("idx_results_job_id", "job_id"),)
