"""
Celery tasks: the inference base class and the inference task.

Workers are synchronous (Celery prefork), so DB access here uses a *sync*
SQLAlchemy engine/session — deliberately separate from the API's async engine in
api/db/database.py. (See the Day 4 plan for why sync-in-workers beats
async-everywhere.)

run_inference downloads the input image, runs the per-process-cached model
(workers/model_registry.py) over it, and stores the detections. It runs real
YOLOv8 object detection as of Day 5.
"""

import logging
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
from workers.model_registry import get_model
from workers.storage import cleanup_image, download_image, ensure_dirs

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


@app.task(base=InferenceTask, bind=True, name="inference.run")
def run_inference(
    self,
    job_id: str,
    model_type: str,
    input_url: str,
    options: dict,
) -> dict:
    """Process one inference job: PROCESSING → download → predict → COMPLETED.

    On any error the job is marked FAILED — with a stage-prefixed message so
    download vs inference failures are distinguishable — and the *original*
    exception is re-raised, preserving its type so ``InferenceTask``'s
    ``autoretry_for=(ConnectionError, TimeoutError)`` still fires for transient
    errors. The downloaded image is always cleaned up in ``finally``.

    Note: a retryable error both marks FAILED and retries, so a retried job can
    briefly flap FAILED→PROCESSING before its terminal state. Deferring FAILED
    marking to an on_failure hook (fires only after retries exhaust) is the clean
    fix; it lives in the author-owned retry logic and is out of scope here.
    """
    job_uuid = UUID(job_id)
    engine = get_sync_engine()
    image_path: Optional[str] = None
    stage = "processing"

    try:
        # Mark PROCESSING in its own transaction so the state is visible
        # (e.g. to GET /jobs/{id}) while inference runs.
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

        # Download the input image (transient working file).
        ensure_dirs()
        stage = "download"
        image_path = download_image(input_url, settings.upload_dir)

        # Run inference with the per-process-cached model.
        stage = "inference"
        model = get_model(model_type)
        result = model.predict(
            image_path,
            confidence_threshold=options.get("confidence_threshold", 0.25),
        )

        # Persist the result and mark COMPLETED in one transaction.
        stage = "persist"
        with Session(engine) as session:
            session.add(
                Result(
                    job_id=job_uuid,
                    detections=result.detections,
                    annotated_url=None,
                    inference_ms=result.inference_ms,
                    model_version=result.model_version,
                )
            )
            _set_job_status(
                session,
                job_uuid,
                JobStatus.COMPLETED,
                completed_at=func.now(),
            )
            session.commit()
        logger.info(
            "job %s completed: %d detection(s) in %d ms",
            job_id,
            len(result.detections),
            result.inference_ms,
        )

        return {
            "job_id": job_id,
            "status": "completed",
            "inference_ms": result.inference_ms,
        }

    except Exception as exc:
        logger.error("job %s failed during %s: %s", job_id, stage, exc)
        # Mark FAILED defensively: a secondary error here (e.g. DB down) must be
        # logged, not raised over the original failure the client cares about.
        try:
            with Session(engine) as session:
                _set_job_status(
                    session,
                    job_uuid,
                    JobStatus.FAILED,
                    completed_at=func.now(),
                    error_message=f"{stage} error: {exc}",
                )
                session.commit()
        except Exception:
            logger.exception("could not mark job %s FAILED", job_id)
        raise
    finally:
        if image_path is not None:
            cleanup_image(image_path)
