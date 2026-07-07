"""
API-side client for enqueueing Celery tasks.

Deliberately imports only the Celery ``app`` (workers/celery_app.py), never the
task functions in workers/tasks.py — tasks are dispatched **by name** via
``send_task``. That keeps the API decoupled from worker/ML code and avoids an
api<->workers import cycle.
"""

import logging
from uuid import UUID

from workers.celery_app import app

logger = logging.getLogger(__name__)


def enqueue_inference(
    job_id: UUID,
    model_type: str,
    input_url: str,
    options: dict,
) -> str:
    """Enqueue the ``inference.run`` task and return its Celery task id."""
    async_result = app.send_task(
        "inference.run",
        args=[str(job_id), model_type, input_url, options],
    )
    logger.info(
        "enqueued inference.run task_id=%s job_id=%s", async_result.id, job_id
    )
    return async_result.id
