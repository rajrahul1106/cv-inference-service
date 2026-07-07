"""
Repository layer: the only place that issues queries against the ORM models.

Each repository wraps an :class:`AsyncSession` passed in at construction time.
Routes/services build a repository from the request-scoped session yielded by
``get_db`` and call these methods, keeping SQLAlchemy specifics out of the
handlers.

For Day 2 the mutating methods commit internally — simple, and each request owns
exactly one session. When the service layer (api/services/job_service.py) lands,
committing can move up to a unit-of-work there and these methods would only
``flush``.
"""

from typing import Optional
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy.sql import func

from api.db.models import Job, JobStatus, Result


class JobRepository:
    """Read/write access to the ``jobs`` table."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, job_data: dict) -> Job:
        """Insert a new job and return it fully populated (id, created_at, ...)."""
        job = Job(**job_data)
        self.session.add(job)
        await self.session.commit()
        await self.session.refresh(job)
        return job

    async def get_by_id(self, job_id: UUID) -> Optional[Job]:
        """Fetch a job by primary key, or ``None`` if it does not exist."""
        return await self.session.get(Job, job_id)

    async def get_by_id_with_result(self, job_id: UUID) -> Optional[Job]:
        """Fetch a job by primary key with its ``result`` eagerly loaded.

        ``get_by_id`` uses ``session.get`` and loads no relationships. Async
        SQLAlchemy forbids implicit lazy loads, so any handler that serializes
        ``Job.result`` (the GET-by-id and list endpoints) must eager-load it up
        front — that is what ``selectinload`` does here.
        """
        stmt = (
            select(Job)
            .options(selectinload(Job.result))
            .where(Job.id == job_id)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list(
        self,
        status: Optional[JobStatus] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Job]:
        """List jobs newest-first, optionally filtered by status, with paging."""
        stmt = (
            select(Job)
            .options(selectinload(Job.result))
            .order_by(Job.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        if status is not None:
            stmt = stmt.where(Job.status == status)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def count(self, status: Optional[JobStatus] = None) -> int:
        """Count jobs, optionally filtered by status (drives list pagination)."""
        stmt = select(func.count()).select_from(Job)
        if status is not None:
            stmt = stmt.where(Job.status == status)
        result = await self.session.execute(stmt)
        return result.scalar_one()

    async def update_status(
        self,
        job_id: UUID,
        status: JobStatus,
        error: Optional[str] = None,
    ) -> None:
        """Transition a job to ``status``, stamping lifecycle timestamps.

        ``started_at`` is set when the job enters PROCESSING; ``completed_at`` is
        set when it reaches COMPLETED or FAILED. ``error`` is recorded when
        provided (typically on failure).
        """
        values: dict = {"status": status}
        if status == JobStatus.PROCESSING:
            values["started_at"] = func.now()
        elif status in (JobStatus.COMPLETED, JobStatus.FAILED):
            values["completed_at"] = func.now()
        if error is not None:
            values["error_message"] = error

        await self.session.execute(
            update(Job).where(Job.id == job_id).values(**values)
        )
        await self.session.commit()


class ResultRepository:
    """Read/write access to the ``results`` table."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, job_id: UUID, result_data: dict) -> Result:
        """Insert the result for a job and return it fully populated."""
        result = Result(job_id=job_id, **result_data)
        self.session.add(result)
        await self.session.commit()
        await self.session.refresh(result)
        return result

    async def get_by_job_id(self, job_id: UUID) -> Optional[Result]:
        """Fetch the result for a job, or ``None`` if none exists yet."""
        stmt = select(Result).where(Result.job_id == job_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
