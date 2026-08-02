"""
Jobs API — submit, fetch, and list inference jobs (SPEC §6).

Routes are thin: they translate HTTP <-> schemas and delegate all persistence to
JobRepository (api/db/repositories.py), which is injected per request via
``Depends(get_job_repository)`` on top of the request-scoped session from
``get_db``.

POST persists a QUEUED job and enqueues the ``inference.run`` Celery task
(api/core/celery_client.py); a worker then flips it PROCESSING → COMPLETED.
``options`` on the request are validated by the schema and forwarded to the
task, but not stored (there is no column for them yet).
"""

from typing import Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.celery_client import enqueue_inference
from api.db.database import get_db
from api.db.models import JobStatus
from api.db.repositories import JobRepository
from api.metrics.prometheus import JOBS_SUBMITTED
from api.schemas.job import (
    JobCreate,
    JobListResponse,
    JobResponse,
    JobSubmittedResponse,
)

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])

logger = structlog.get_logger(__name__)

# Placeholder; a real estimate would come from current queue depth.
ESTIMATED_WAIT_SECONDS = 5


def get_job_repository(db: AsyncSession = Depends(get_db)) -> JobRepository:
    """Build a JobRepository from the request-scoped session."""
    return JobRepository(db)


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=JobSubmittedResponse,
)
async def submit_job(
    payload: JobCreate,
    repo: JobRepository = Depends(get_job_repository),
) -> JobSubmittedResponse:
    """Accept an inference job: persist it QUEUED, then enqueue the worker task.

    If enqueueing fails, the job is marked FAILED and a 500 is returned so the
    client isn't handed a job id that will never be processed. ``options`` are
    validated by the schema and forwarded to the task, but not stored.
    """
    job = await repo.create(
        {
            "model_type": payload.model_type,
            # HttpUrl -> str: the input_url column is Text.
            "input_url": str(payload.input_url),
            "status": JobStatus.QUEUED,
        }
    )
    JOBS_SUBMITTED.labels(model_type=payload.model_type.value).inc()
    logger.info(
        "job_submitted", job_id=str(job.id), model_type=payload.model_type.value
    )

    try:
        enqueue_inference(
            job_id=job.id,
            model_type=payload.model_type.value,
            input_url=str(payload.input_url),
            options=payload.options,
        )
    except Exception as exc:
        logger.error("job_enqueue_failed", job_id=str(job.id), error=str(exc))
        await repo.update_status(
            job.id, JobStatus.FAILED, error=f"enqueue failed: {exc}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to enqueue inference job",
        )

    return JobSubmittedResponse(
        job_id=job.id,
        status=job.status,
        websocket_url=f"/ws/jobs/{job.id}",
        estimated_wait_seconds=ESTIMATED_WAIT_SECONDS,
    )


@router.get(
    "/{job_id}",
    response_model=JobResponse,
    responses={status.HTTP_404_NOT_FOUND: {"description": "Job not found"}},
)
async def get_job(
    job_id: UUID,
    repo: JobRepository = Depends(get_job_repository),
) -> JobResponse:
    """Return a single job with its result, or 404 if it does not exist."""
    job = await repo.get_by_id_with_result(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )
    return JobResponse.model_validate(job)


@router.get("", response_model=JobListResponse)
async def list_jobs(
    status: Optional[JobStatus] = None,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    repo: JobRepository = Depends(get_job_repository),
) -> JobListResponse:
    """List jobs newest-first with pagination, optionally filtered by status."""
    jobs = await repo.list(status=status, limit=limit, offset=offset)
    total = await repo.count(status=status)
    return JobListResponse(
        jobs=[JobResponse.model_validate(job) for job in jobs],
        total=total,
        limit=limit,
        offset=offset,
    )
