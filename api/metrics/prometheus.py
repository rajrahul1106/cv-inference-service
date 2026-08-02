"""
Prometheus metrics for the CV inference service (SPEC §9).

Metrics are declared on the default registry. Because the API and the Celery
workers run in *separate processes*, we use prometheus-client's multiprocess
mode: every process writes mmap files to ``PROMETHEUS_MULTIPROC_DIR`` and the
API's ``/metrics`` endpoint aggregates them via ``MultiProcessCollector``. See
the Day 10 plan (decision A) for why one shared dir beats a scrape port per
worker.
"""

import os

from fastapi.responses import PlainTextResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    multiprocess,
)

# In multiprocess mode the dir must exist before any metric is created.
_MULTIPROC_DIR = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
if _MULTIPROC_DIR:
    os.makedirs(_MULTIPROC_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Metric definitions
# ─────────────────────────────────────────────────────────────
JOBS_SUBMITTED = Counter(
    "cv_jobs_submitted_total",
    "Total jobs submitted",
    ["model_type"],
)
JOBS_COMPLETED = Counter(
    "cv_jobs_completed_total",
    "Total jobs completed",
    ["model_type", "status"],
)
INFERENCE_DURATION = Histogram(
    "cv_inference_duration_seconds",
    "Inference duration in seconds",
    ["model_type"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)
# Defined per SPEC; no writer is wired yet (would need a periodic Redis LLEN),
# so it reads 0 until that lands.
QUEUE_DEPTH = Gauge(
    "cv_queue_depth",
    "Pending jobs in the Redis queue",
    multiprocess_mode="livesum",
)
# Single total gauge (no job_id label) — per-job visibility lives in the logs.
WEBSOCKET_CONNECTIONS = Gauge(
    "cv_websocket_connections",
    "Active WebSocket connections",
    multiprocess_mode="livesum",
)


def get_metrics_response() -> PlainTextResponse:
    """Render metrics in Prometheus exposition format.

    In multiprocess mode, aggregate every process's samples via a fresh registry
    + ``MultiProcessCollector``; otherwise render the default registry.
    """
    if _MULTIPROC_DIR:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        data = generate_latest(registry)
    else:
        data = generate_latest()
    return PlainTextResponse(content=data, media_type=CONTENT_TYPE_LATEST)
