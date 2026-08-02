# Distributed CV Inference Service

Production-grade distributed microservice for computer vision inference. Built with FastAPI, Celery, Redis, PostgreSQL, and Docker Compose.

**Status:** In active development (Days 1-11 complete out of 14).

## Architecture

- **FastAPI gateway** - accepts job submissions, exposes REST + WebSocket endpoints
- **Redis** - Celery broker (task queue) + pub/sub channel for real-time status updates
- **Celery workers** - pull jobs, run ML inference (YOLOv8 object detection and MediaPipe face detection live; fire detection is a flagged placeholder)
- **PostgreSQL 16** - persistent job and result storage with JSONB detections
- **Alembic** - reversible database migrations
- **Docker Compose** - multi-service orchestration

## Progress

- [x] Day 1: Project scaffolding, Docker Compose, health endpoints
- [x] Day 2: Async SQLAlchemy 2.0 data layer, Alembic migrations, readiness probes
- [x] Day 3: Jobs REST API (submit, get, list) with Pydantic v2 validation
- [x] Day 4: Celery worker wired for end-to-end job processing
- [x] Day 5: Real YOLOv8 inference — per-process model registry, image download/cleanup. Warm inference pipeline latency: ~186ms end-to-end (includes disk read, preprocessing, model forward pass, and postprocessing). Raw YOLO forward pass alone is ~48ms.
- [x] Day 6: All three models live behind the shared InferenceModel interface — YOLOv8 object detection, MediaPipe face detection, and a YOLOv8-based fire detection placeholder — routed by the model registry
- [x] Day 7: Model registry with multi-model routing — unified yolo/face/fire routing behind the InferenceModel interface (shipped in Days 5-6)
- [x] Day 8-9: WebSocket real-time job status via Redis pub/sub — ConnectionManager + lifespan-managed background subscriber bridge; worker publishes each status transition
- [x] Day 10-11: Prometheus metrics + structured logging — /metrics aggregating API + worker processes (multiprocess mode), structlog JSON logs with job_id correlation across both
- [ ] Day 12-13: React dashboard
- [ ] Day 14: Load testing, benchmarks, demo

## Local Development

```bash
docker compose -f infra/docker-compose.yml up -d
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
alembic upgrade head
uvicorn api.main:app --reload --port 8000
celery -A workers.celery_app worker --loglevel=info --pool=solo
```

> **macOS note:** the worker runs with `--pool=solo` — tasks execute in the main
> process with no forking. On macOS, Celery's default prefork pool crashes with
> SIGSEGV when MediaPipe initializes its native C++ threading inside a forked
> child. Solo pool avoids this by not forking, at the cost of single-task
> concurrency. On **Linux**, the default prefork pool works fine; use
> `--concurrency=N` there for real parallelism.

### Troubleshooting

**Running prefork on macOS (e.g. to test concurrency):** disable Apple's
Objective-C fork-safety guard before launching the worker:

```bash
OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES \
  celery -A workers.celery_app worker --loglevel=info --concurrency=2
```

This is a workaround, not a fix — it silences the check that would otherwise
abort the forked child. Prefer `--pool=solo` for everyday macOS development, and
use Linux (or the Docker worker image) for genuine prefork concurrency.

## Viewing Metrics

Prometheus metrics are exposed at `GET /metrics`:

```bash
curl http://localhost:8000/metrics
```

Key series: `cv_jobs_submitted_total{model_type}`, `cv_jobs_completed_total{model_type,status}`, `cv_inference_duration_seconds{model_type}` (histogram), and `cv_websocket_connections`.

The API and Celery workers are separate processes, so metrics use prometheus-client's multiprocess mode. To aggregate **both** behind the one `/metrics` endpoint, point them at the same `PROMETHEUS_MULTIPROC_DIR` (a shared path — a shared volume under Docker):

```bash
export PROMETHEUS_MULTIPROC_DIR=./prometheus_multiproc
# start the API and the worker with this env var set
```

Without it, `/metrics` reports only the API process's own metrics.

## Tech Stack

Python 3.13 · FastAPI 0.115 · SQLAlchemy 2.0 async · Alembic · Pydantic v2 · Celery 5.4 · Redis 7.4 · PostgreSQL 16 · ultralytics/YOLOv8 · MediaPipe · Prometheus · structlog · Docker Compose
