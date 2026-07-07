"""
Celery application instance for the CV inference service.

Run the worker with ``celery -A workers.celery_app worker``. Broker and result
backend come from api.config.settings (Redis). Task modules are autodiscovered
from the ``workers`` package, so workers/tasks.py registers itself on startup.

Importing this module (e.g. from the API's celery_client) pulls in only the
Celery app and settings — never the task implementations — so there is no
api<->workers import cycle.
"""

from celery import Celery

from api.config import settings

app = Celery(
    "cv_inference",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

app.conf.update(
    # JSON everywhere — no pickle (safer, language-agnostic payloads).
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Emit a STARTED state so callers can tell PENDING (queued) from STARTED.
    task_track_started=True,
    # Hard + soft time limits: SoftTimeLimitExceeded fires first (graceful),
    # then the hard limit kills the worker process.
    task_time_limit=300,
    task_soft_time_limit=240,
    # Pull one task at a time so a slow task doesn't hoard the queue.
    worker_prefetch_multiplier=1,
    # Ack after completion so a crashed worker's task is redelivered, not lost.
    task_acks_late=True,
)

# Lazily discover workers.tasks on worker start (does not import it now).
app.autodiscover_tasks(["workers"])
