# Distributed CV Inference Service

Production-grade distributed microservice for computer vision inference. Built with FastAPI, Celery, Redis, PostgreSQL, and Docker Compose.

**Status:** In active development (Days 1-6 complete out of 14).

## Architecture

- **FastAPI gateway** - accepts job submissions, exposes REST + WebSocket endpoints
- **Redis broker** - task queue between API and workers
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
- [x] Day 6: MediaPipe face detection (Tasks API) + placeholder fire detector — both behind the shared InferenceModel interface, routed by the model registry
- [ ] Day 7: Model registry with multi-model routing
- [ ] Day 8-9: WebSocket real-time updates via Redis pub/sub
- [ ] Day 10-11: Prometheus metrics + structured logging
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

## Tech Stack

Python 3.13 · FastAPI 0.115 · SQLAlchemy 2.0 async · Alembic · Pydantic v2 · Celery 5.4 · Redis 7.4 · PostgreSQL 16 · ultralytics/YOLOv8 · MediaPipe · Docker Compose
