"""
Pydantic v2 request/response schemas for the jobs API (SPEC §6).

These define the wire contract for ``/api/v1/jobs`` — what clients send and what
they receive — and are deliberately separate from the SQLAlchemy ORM models in
api/db/models.py (persistence shape vs. API shape). Response models set
``from_attributes=True`` so FastAPI can build them directly from ORM instances.
"""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from api.db.models import JobStatus, ModelType


# ─────────────────────────────────────────────────────────────
# Requests
# ─────────────────────────────────────────────────────────────
class JobCreate(BaseModel):
    """Body for POST /api/v1/jobs."""

    # protected_namespaces=() silences pydantic's warning about the ``model_``
    # prefix on ``model_type`` — the name is part of our API contract (SPEC §6).
    model_config = ConfigDict(protected_namespaces=())

    input_url: HttpUrl
    model_type: ModelType
    # Free-form per-model knobs (e.g. confidence_threshold, save_annotated).
    # Validated here but not persisted yet — Day 5 forwards them to Celery.
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_confidence_threshold(self) -> "JobCreate":
        """If options.confidence_threshold is set, it must be a number in [0, 1]."""
        threshold = self.options.get("confidence_threshold")
        if threshold is None:
            return self
        # bool is a subclass of int; reject it so `true` isn't read as 1.0.
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            raise ValueError("options.confidence_threshold must be a number")
        if not 0 <= threshold <= 1:
            raise ValueError("options.confidence_threshold must be between 0 and 1")
        return self


# ─────────────────────────────────────────────────────────────
# Responses
# ─────────────────────────────────────────────────────────────
class ResultResponse(BaseModel):
    """Nested inference result on a JobResponse (built from a Result ORM row)."""

    # from_attributes: build from the ORM object. protected_namespaces=():
    # silence the ``model_`` warning on ``model_version`` (part of the contract).
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    detections: list[dict]
    annotated_url: Optional[str] = None
    inference_ms: int
    model_version: str
    created_at: datetime


class JobResponse(BaseModel):
    """Full job representation for GET /api/v1/jobs/{job_id}, result included."""

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        protected_namespaces=(),
    )

    # Serialized as ``job_id`` to match the SPEC contract and the POST response;
    # populated from the ORM's ``id`` attribute via from_attributes.
    id: UUID = Field(serialization_alias="job_id")
    status: JobStatus
    model_type: ModelType
    input_url: str
    input_size_kb: Optional[int] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    worker_id: Optional[str] = None
    retry_count: int
    result: Optional[ResultResponse] = None


class JobListResponse(BaseModel):
    """Paginated list wrapper for GET /api/v1/jobs."""

    jobs: list[JobResponse]
    total: int
    limit: int
    offset: int


class JobSubmittedResponse(BaseModel):
    """202 response for POST /api/v1/jobs — the accepted job's handle."""

    job_id: UUID
    status: JobStatus
    websocket_url: str
    estimated_wait_seconds: int
