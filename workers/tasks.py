"""
Celery tasks: the inference base class and the (currently mock) inference task.

Workers are synchronous (Celery prefork), so DB access here uses a *sync*
SQLAlchemy engine/session — deliberately separate from the API's async engine in
api/db/database.py. (See the Day 4 plan for why sync-in-workers beats
async-everywhere.)

Day 4 scope: run_inference is a mock — it flips job status and writes a fake
Result after a short sleep. Real ML inference replaces the body on Day 5.
"""

import logging
import time
from typing import Optional
from uuid import UUID

from celery import Task
from sqlalchemy import create_engine, func
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

from api.config import settings
from api.db.models import Job, JobStatus, Result
from workers.celery_app import app

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Sync DB access for workers
# ─────────────────────────────────────────────────────────────
_engine: Optional[Engine] = None


def _sync_database_url() -> str:
    """Translate the async DB URL (asyncpg) into a sync one (psycopg2)."""
    return settings.database_url.replace("+asyncpg", "+psycopg2")


def get_sync_engine() -> Engine:
    """Return a process-local sync engine, created lazily on first use.

    Created lazily so the engine is built *inside* the forked worker child (not
    the parent), and uses NullPool so no pooled connection is ever shared across
    a fork — both guard against Celery prefork's connection-reuse hazards. Fine
    for low-throughput mock work; revisit pooling when real load arrives.
    """
    global _engine
    if _engine is None:
        _engine = create_engine(_sync_database_url(), poolclass=NullPool, future=True)
    return _engine


def _set_job_status(
    session: Session,
    job_id: UUID,
    status: JobStatus,
    **fields: object,
) -> None:
    """Load a job in ``session`` and set its status plus any extra columns.

    The caller owns the transaction (commit/rollback). Raises ``ValueError`` if
    the job row is missing.
    """
    job = session.get(Job, job_id)
    if job is None:
        raise ValueError(f"job {job_id} not found")
    job.status = status
    for key, value in fields.items():
        setattr(job, key, value)


# ─────────────────────────────────────────────────────────────
# Inference task
# ─────────────────────────────────────────────────────────────
class InferenceTask(Task):
    """Base task: retry transient errors with exponential backoff + jitter.

    Per-process model loading will hang off this class on Day 5+; for now it
    carries only the retry policy.
    """

    autoretry_for = (ConnectionError, TimeoutError)
    retry_kwargs = {"max_retries": 3, "countdown": 5}
    retry_backoff = True
    retry_backoff_max = 60
    retry_jitter = True


# Mock output until real inference lands (Day 5).
_MOCK_DETECTIONS = [{"label": "person", "confidence": 0.92, "bbox": [120, 80, 340, 520]}]
_MOCK_INFERENCE_MS = 2000


@app.task(base=InferenceTask, bind=True, name="inference.run")
def run_inference(
    self,
    job_id: str,
    model_type: str,
    input_url: str,
    options: dict,
) -> dict:
    """Process one inference job (mock): PROCESSING → sleep → Result → COMPLETED.

    On any error the job is marked FAILED and the exception re-raised so Celery
    records the failure. The mock raises none of ``autoretry_for``'s exceptions,
    so there is no FAILED-then-retried conflict today; once Day 5 introduces real
    retryable errors, failure-marking should move to an ``on_failure`` hook that
    fires only after retries are exhausted.
    """
    job_uuid = UUID(job_id)
    engine = get_sync_engine()

    try:
        # 1) Mark PROCESSING in its own transaction so the state is visible
        #    (e.g. to GET /jobs/{id}) while the work below runs.
        with Session(engine) as session:
            _set_job_status(
                session,
                job_uuid,
                JobStatus.PROCESSING,
                started_at=func.now(),
                worker_id=self.request.hostname,
            )
            session.commit()
        logger.info("job %s processing on %s", job_id, self.request.hostname)

        # 2) Simulate inference work (becomes a real model call on Day 5).
        time.sleep(2)

        # 3) Persist the result and mark COMPLETED in one transaction.
        with Session(engine) as session:
            session.add(
                Result(
                    job_id=job_uuid,
                    detections=_MOCK_DETECTIONS,
                    annotated_url=None,
                    inference_ms=_MOCK_INFERENCE_MS,
                    model_version="mock-1.0",
                )
            )
            _set_job_status(
                session,
                job_uuid,
                JobStatus.COMPLETED,
                completed_at=func.now(),
            )
            session.commit()
        logger.info("job %s completed", job_id)

        return {
            "job_id": job_id,
            "status": "completed",
            "inference_ms": _MOCK_INFERENCE_MS,
        }

    except Exception as exc:
        logger.error("job %s failed: %s", job_id, exc)
        # Mark FAILED defensively: a secondary error here (e.g. DB down) must be
        # logged, not raised over the original failure the client cares about.
        try:
            with Session(engine) as session:
                _set_job_status(
                    session,
                    job_uuid,
                    JobStatus.FAILED,
                    completed_at=func.now(),
                    error_message=str(exc),
                )
                session.commit()
        except Exception:
            logger.exception("could not mark job %s FAILED", job_id)
        raise
