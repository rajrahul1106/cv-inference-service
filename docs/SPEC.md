# CV Inference Service: Technical Specification

A production-grade distributed computer vision microservice. Built to demonstrate distributed systems patterns alongside ML serving for hybrid SDE and Applied ML hiring.

**Estimated build time:** 14 days, ~3 hours/day
**Stack:** FastAPI + Celery + Redis + PostgreSQL + WebSockets + Docker + React

---

## 1. The Pitch (memorise this for interviews)

A horizontally-scalable computer vision inference service that accepts image and video jobs through a REST API, queues them through Redis, runs ML inference on Celery worker pools, streams real-time job status updates over WebSockets, and exposes Prometheus metrics for observability. Three models are served: YOLOv8 object detection, MediaPipe face detection, and a custom fire detection model. The entire stack runs through Docker Compose.

This single project demonstrates: async API design, message queue patterns, worker pools, production ML serving, observability, containerisation, and a real-time UI.

---

## 2. System Architecture

```
                                              ┌─────────────────────┐
                                              │  React Dashboard    │
                                              │  (upload + status)  │
                                              └──────────┬──────────┘
                                                         │ HTTP + WS
                                                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                      FastAPI Gateway                              │
│   ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐    │
│   │ /api/jobs   │  │ /ws/jobs/:id │  │ /metrics  /health   │    │
│   └──────┬──────┘  └──────┬───────┘  └─────────────────────┘    │
└──────────┼─────────────────┼──────────────────────────────────────┘
           │                 │
           │ enqueue         │ subscribe
           ▼                 │
   ┌───────────────┐         │
   │  Redis Queue  │◄────────┤ pub/sub channel: job.status
   │   (broker)    │         │
   └───────┬───────┘         │
           │ dequeue         │ publish
           ▼                 │
   ┌───────────────────────────────────┐
   │  Celery Workers (N replicas)      │
   │  ┌──────────┐ ┌──────────┐        │
   │  │ Worker 1 │ │ Worker 2 │  ...   │
   │  │ YOLO+FD  │ │ YOLO+FD  │        │
   │  └────┬─────┘ └────┬─────┘        │
   └───────┼────────────┼──────────────┘
           │            │
           ▼            ▼
   ┌─────────────────────────┐
   │   PostgreSQL            │
   │   (jobs + results)      │
   └─────────────────────────┘
```

---

## 3. Tech Stack with Versions

| Layer | Tech | Version | Why |
|---|---|---|---|
| API | FastAPI | 0.115+ | Async, fast, you know it |
| ORM | SQLAlchemy | 2.0 async | Same as RailYatra |
| DB | PostgreSQL | 16 | Production standard |
| Queue Broker | Redis | 7.4 | Lighter than RabbitMQ for v1 |
| Worker | Celery | 5.4 | Most common in industry |
| ML | ultralytics | latest | YOLOv8/v11 |
| ML | mediapipe | 0.10+ | Face detection |
| ML | torch | 2.4+ | Fire detection model |
| Real-time | WebSockets | (FastAPI native) | Standard |
| Metrics | prometheus-client | latest | Industry standard |
| Frontend | React + Vite | 18 | Same as RailYatra |
| Container | Docker Compose | v2 | Standard |
| Load test | Locust | latest | Python, easy to write |

Pin exact versions in requirements.txt once you start. Do not let dependencies float.

---

## 4. Folder Structure

```
cv-inference-service/
├── api/
│   ├── __init__.py
│   ├── main.py                      # FastAPI app entry
│   ├── config.py                    # env config (pydantic-settings)
│   ├── routes/
│   │   ├── jobs.py                  # POST/GET /jobs
│   │   ├── health.py                # /health + /ready
│   │   └── websocket.py             # /ws/jobs/{id}
│   ├── schemas/
│   │   ├── job.py                   # Pydantic request/response
│   │   └── results.py
│   ├── db/
│   │   ├── database.py              # session, engine
│   │   ├── models.py                # SQLAlchemy models
│   │   └── repositories.py          # query layer
│   ├── services/
│   │   └── job_service.py           # business logic
│   ├── websocket/
│   │   └── connection_manager.py    # active connection tracking
│   ├── metrics/
│   │   └── prometheus.py            # counters, histograms
│   └── core/
│       ├── celery_client.py         # publishes tasks
│       └── redis_client.py          # pub/sub
│
├── workers/
│   ├── __init__.py
│   ├── celery_app.py                # Celery instance
│   ├── tasks.py                     # @app.task definitions
│   ├── inference/
│   │   ├── base.py                  # InferenceModel ABC
│   │   ├── yolo_detector.py         # YOLOv8 wrapper
│   │   ├── face_detector.py         # MediaPipe wrapper
│   │   └── fire_detector.py         # custom model wrapper
│   ├── model_registry.py            # singleton model cache
│   └── storage.py                   # save outputs (annotated images)
│
├── frontend/
│   ├── src/
│   │   ├── pages/Dashboard.jsx
│   │   ├── components/UploadCard.jsx
│   │   ├── components/JobList.jsx
│   │   ├── components/JobDetail.jsx
│   │   ├── hooks/useJobSocket.js
│   │   └── api/client.js
│   ├── package.json
│   └── vite.config.js
│
├── infra/
│   ├── docker/
│   │   ├── Dockerfile.api
│   │   ├── Dockerfile.worker
│   │   └── Dockerfile.frontend
│   ├── docker-compose.yml
│   ├── docker-compose.dev.yml
│   └── prometheus.yml
│
├── scripts/
│   ├── load_test.py                 # Locust scenarios
│   ├── seed_models.py               # download model weights
│   └── benchmark.py                 # latency/throughput report
│
├── tests/
│   ├── test_jobs_api.py
│   ├── test_inference.py
│   └── test_workers.py
│
├── .env.example
├── .gitignore
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 5. Database Schema

Two tables. Keep it simple.

```sql
CREATE TYPE job_status AS ENUM ('queued', 'processing', 'completed', 'failed');
CREATE TYPE model_type AS ENUM ('yolo', 'face', 'fire', 'multi');

CREATE TABLE jobs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status          job_status NOT NULL DEFAULT 'queued',
    model_type      model_type NOT NULL,
    input_url       TEXT NOT NULL,
    input_size_kb   INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    error_message   TEXT,
    worker_id       TEXT,
    retry_count     INTEGER DEFAULT 0
);

CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_created_at ON jobs(created_at DESC);

CREATE TABLE results (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id          UUID NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    detections      JSONB NOT NULL,          -- array of {label, conf, bbox}
    annotated_url   TEXT,                    -- path to annotated image
    inference_ms    INTEGER NOT NULL,
    model_version   TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_results_job_id ON results(job_id);
```

Use Alembic for migrations from day 1. Do not skip this.

---

## 6. API Contract

### POST /api/v1/jobs

Submit an inference job.

**Request:**
```json
{
  "input_url": "https://example.com/image.jpg",
  "model_type": "yolo",
  "options": {
    "confidence_threshold": 0.5,
    "save_annotated": true
  }
}
```

Or `multipart/form-data` with an image file upload.

**Response (202 Accepted):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "websocket_url": "/ws/jobs/550e8400-e29b-41d4-a716-446655440000",
  "estimated_wait_seconds": 3
}
```

### GET /api/v1/jobs/{job_id}

**Response (200 OK):**
```json
{
  "job_id": "550e...",
  "status": "completed",
  "created_at": "2026-06-25T10:30:00Z",
  "completed_at": "2026-06-25T10:30:04Z",
  "inference_ms": 234,
  "model_type": "yolo",
  "result": {
    "detections": [
      {"label": "person", "confidence": 0.92, "bbox": [120, 80, 340, 520]},
      {"label": "car", "confidence": 0.87, "bbox": [400, 200, 600, 380]}
    ],
    "annotated_url": "/api/v1/jobs/550e.../annotated"
  }
}
```

### GET /api/v1/jobs?status=queued&limit=20

List jobs with cursor pagination.

### WS /ws/jobs/{job_id}

WebSocket pushing status events:
```json
{"event": "status_change", "status": "processing", "worker_id": "worker-2"}
{"event": "status_change", "status": "completed", "inference_ms": 234}
```

### GET /metrics

Prometheus exposition format. Counters and histograms (see section 9).

### GET /health and GET /ready

Standard k8s-style health probes. /ready checks DB and Redis connectivity.

---

## 7. Celery Task Design

```python
# workers/tasks.py
from celery import Task
from workers.celery_app import app
from workers.model_registry import get_model
from workers.inference.base import InferenceResult

class InferenceTask(Task):
    """Base class so models are loaded once per worker process."""
    autoretry_for = (ConnectionError, TimeoutError)
    retry_kwargs = {'max_retries': 3, 'countdown': 5}
    retry_backoff = True
    retry_backoff_max = 60
    retry_jitter = True

@app.task(base=InferenceTask, bind=True, name='inference.run')
def run_inference(self, job_id: str, model_type: str, input_url: str, options: dict):
    publish_status(job_id, 'processing', worker_id=self.request.hostname)
    
    try:
        model = get_model(model_type)              # cached per process
        result: InferenceResult = model.predict(input_url, **options)
        save_result_to_db(job_id, result)
        publish_status(job_id, 'completed', inference_ms=result.duration_ms)
        return {'job_id': job_id, 'status': 'completed'}
    except Exception as e:
        mark_job_failed(job_id, str(e))
        publish_status(job_id, 'failed', error=str(e))
        raise
```

**Key design choices to defend in interviews:**

- Model is loaded once per worker process (in `get_model`, a module-level dict). This avoids the 2-3 second cold start per task.
- Retries use exponential backoff with jitter to avoid thundering herd.
- Status updates go through Redis pub/sub, not direct DB polling.
- Worker concurrency is set via `--concurrency=N` flag based on CPU count.

---

## 8. WebSocket Protocol

```python
# api/websocket/connection_manager.py
class ConnectionManager:
    def __init__(self):
        self.active: dict[str, set[WebSocket]] = defaultdict(set)
    
    async def connect(self, job_id: str, ws: WebSocket):
        await ws.accept()
        self.active[job_id].add(ws)
    
    async def disconnect(self, job_id: str, ws: WebSocket):
        self.active[job_id].discard(ws)
    
    async def broadcast(self, job_id: str, message: dict):
        dead = set()
        for ws in self.active[job_id]:
            try:
                await ws.send_json(message)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.active[job_id].discard(ws)
```

A background task in the API process subscribes to Redis pub/sub and calls `manager.broadcast()` whenever a worker publishes a status change. This decouples workers from WebSocket clients entirely.

---

## 9. Observability Plan

### Prometheus metrics to track

```python
from prometheus_client import Counter, Histogram, Gauge

JOBS_SUBMITTED = Counter('cv_jobs_submitted_total', 'Jobs submitted', ['model_type'])
JOBS_COMPLETED = Counter('cv_jobs_completed_total', 'Jobs completed', ['model_type', 'status'])
INFERENCE_DURATION = Histogram('cv_inference_duration_seconds', 'Inference time', ['model_type'],
                                buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])
QUEUE_DEPTH = Gauge('cv_queue_depth', 'Pending jobs in queue')
ACTIVE_WORKERS = Gauge('cv_active_workers', 'Workers processing jobs')
```

### Structured logging

JSON logs with these fields on every log line: `timestamp`, `level`, `service`, `job_id`, `worker_id`, `event`, `duration_ms`. Use `structlog` or stdlib logging with a JSON formatter.

### What to capture for your resume bullets

Run a benchmark on day 14 and record:
- p50, p95, p99 inference latency per model
- Max sustained throughput (jobs/second)
- Memory footprint per worker
- Cold start vs warm start latency delta

These numbers go directly into your CV.

---

## 10. 14-Day Implementation Plan

**Days 1-2: Scaffolding**
- Repo init, Docker Compose with Postgres + Redis
- FastAPI app with /health endpoint
- Alembic setup and first migration (jobs + results tables)
- POST /jobs endpoint that just writes to DB (no queue yet)
- Commit checkpoint: `feat: scaffold api with job creation`

**Days 3-4: Queue and worker basics**
- Celery worker that runs a fake task (sleep + return)
- Wire POST /jobs to enqueue Celery task
- Worker updates job status in DB on completion
- GET /jobs/{id} returns latest status
- Commit checkpoint: `feat: end-to-end job submission with mock worker`

**Days 5-6: First real model**
- Add ultralytics dependency
- Write `YOLODetector` class implementing the `InferenceModel` interface
- Model loaded once per worker (singleton pattern)
- Save detections as JSONB to results table
- Test with 5 sample images
- Commit checkpoint: `feat: yolo inference pipeline`

**Day 7: Second and third models**
- MediaPipe face detector (similar interface)
- Fire detection model (use a pretrained YOLOv8 fine-tuned on fire dataset, or train one quickly on a public fire dataset)
- Model registry routing based on `model_type` field
- Commit checkpoint: `feat: multi-model support`

**Days 8-9: Real-time updates**
- ConnectionManager class
- WS /ws/jobs/{id} endpoint
- Redis pub/sub channel for job status
- Worker publishes status changes
- Background task in API subscribes and broadcasts
- Commit checkpoint: `feat: realtime websocket status updates`

**Days 10-11: Observability**
- prometheus-client integration
- /metrics endpoint
- Counters, histograms, gauges as listed in section 9
- Structured JSON logging across api + workers
- Retry logic with exponential backoff
- Commit checkpoint: `feat: prometheus metrics and structured logging`

**Days 12-13: Frontend**
- React + Vite scaffold (reuse patterns from RailYatra)
- Upload component (drag-and-drop)
- Job list with live status updates over WebSocket
- Job detail view with annotated image and detection list
- Commit checkpoint: `feat: react dashboard with realtime updates`

**Day 14: Polish + benchmarks**
- Locust load test (50 concurrent users, ramp 1 minute)
- Record p50/p95/p99 latency per model
- Write README with architecture diagram and screenshots
- Record a 60-second demo gif using `peek` or `screentogif`
- Final commit: `docs: readme + benchmarks + demo`

If you fall behind by 2+ days, skip the frontend and use FastAPI's built-in /docs page as the UI. The backend is what matters for hiring. The frontend is bonus.

---

## 11. Resume Bullets to Earn

Write these as you go, fill in the real numbers on day 14:

- Built a distributed CV inference microservice handling **X** concurrent jobs/second with Redis-backed Celery worker pools, achieving p95 latency of **X**ms on a 4-core machine
- Designed async job lifecycle (queued → processing → completed/failed) with WebSocket real-time status updates over Redis pub/sub, decoupling **X** active clients from worker pools
- Served three production CV models (YOLOv8, MediaPipe face detection, custom fire detector) through a unified inference interface with per-process model caching, eliminating **X**s cold start per request
- Instrumented production-grade observability: Prometheus counters and histograms, structured JSON logging, exponential-backoff retry logic with jitter
- Containerised 4-service architecture (FastAPI + Celery workers + Redis + Postgres) with Docker Compose; Locust load testing harness validating **X** RPS sustained throughput

---

## 12. Interview Talking Points (rehearse these)

**Q: Why Celery over RabbitMQ or Kafka?**
Celery sits on top of a broker (Redis or RabbitMQ) and gives you task primitives, retries, scheduling, and worker management for free. Kafka is overkill for a job queue. RabbitMQ as the broker would also work, you picked Redis for v1 because the operational complexity is lower and the throughput meets your needs (under 1000 jobs/second).

**Q: What happens when a worker crashes mid-job?**
Celery acknowledges tasks late (`acks_late=True`) so an unacked task gets redelivered to another worker. The job stays in 'processing' state with a heartbeat timestamp. A reaper task marks stale 'processing' jobs back to 'queued' after a timeout.

**Q: How would this scale to 10x traffic?**
First, add more worker replicas (`docker compose up --scale worker=10`). Second, partition workers by model type so one slow model does not block others. Third, move from Redis broker to RabbitMQ if you need durable queues and dead-letter exchanges. Fourth, batch inference where applicable (process 8 images per GPU forward pass instead of 1).

**Q: How do you handle model versioning?**
Each result row stores `model_version`. Models are loaded by `{type}:{version}` key. Rolling a new model out means deploying workers that load the new version and routing traffic gradually.

**Q: What is the cost per inference?**
On a CPU-only setup: roughly 200ms inference time × Y$/hour for the box = Z paise per inference. On GPU this drops 10x but the box is more expensive. The break-even depends on your traffic.

**Q: Why WebSockets and not polling?**
Polling at 1s intervals creates 60 unnecessary requests per minute per client. WebSockets push exactly when state changes. For 100 concurrent clients waiting on jobs that take 5 seconds, polling generates 6000 requests/minute. WebSockets generate maybe 200.

**Q: What is in the JSONB detections column?**
An array of objects: `{label, confidence, bbox: [x1, y1, x2, y2]}`. Storing as JSONB instead of a separate detections table is a deliberate trade-off: faster reads, no joins, harder to query individual detections. For this workload, reads dominate writes and queries are always "give me all detections for a job."

---

## 13. README Template Outline

Your README must have:

1. Hero image or demo gif (60 seconds, end-to-end flow)
2. One-paragraph pitch (the section 1 text)
3. Architecture diagram (use the section 2 ASCII or recreate in excalidraw)
4. Tech stack table
5. Quick start (3 commands max: `git clone`, `docker compose up`, open localhost)
6. API documentation link (/docs)
7. Benchmark results table (latency, throughput, cold start)
8. Configuration table (env vars and what they do)
9. Architecture decision record (3-5 key choices and why)
10. License

Recruiters who open your GitHub spend 30 seconds. The hero gif and benchmark table are what make them keep scrolling.

---

## 14. Quick Start Commands

```bash
# clone and configure
git clone https://github.com/rajrahul1106/cv-inference-service.git
cd cv-inference-service
cp .env.example .env

# spin up everything
docker compose -f infra/docker-compose.yml up -d

# verify
curl http://localhost:8000/health
curl http://localhost:8000/metrics | head -20

# scale workers
docker compose -f infra/docker-compose.yml up -d --scale worker=4

# run load test
python scripts/load_test.py --users 50 --duration 60s

# follow logs
docker compose logs -f worker
```

---

## 15. Common Pitfalls (read before starting each phase)

**Phase 1 trap:** Skipping Alembic and using `create_all()`. Migrations matter on day 10 when you want to add a column.

**Phase 2 trap:** Loading the YOLO model inside the task function. Each task will take 3 seconds of model loading. Use a module-level singleton or Celery worker init signals.

**Phase 3 trap:** Forgetting to set `result_backend` for Celery. Without it, `task.get()` will hang forever.

**Phase 4 trap:** Holding DB sessions across WebSocket connection lifetimes. Sessions belong to a single request. Open and close per message.

**Phase 5 trap:** Logging `print(...)` statements that do not appear in Docker logs because of buffering. Set `PYTHONUNBUFFERED=1` in the Dockerfile.

**Phase 6 trap:** Putting React state for live job updates in component state. When the parent re-renders, the WebSocket reconnects. Lift the state to a context or use a custom hook with `useRef` for the socket.

---

## 16. What "Done" Looks Like

You can demo this in 90 seconds:
1. Open dashboard, upload a sample image with people and a car
2. Job appears in list with "queued" status
3. Status flips to "processing" then "completed" within 2 seconds
4. Annotated image renders with bounding boxes
5. Open /metrics in another tab, show the histogram values
6. Open /docs, show the OpenAPI spec
7. Run `docker compose ps`, show 4 services running
8. Show the README with benchmark table

If you can do this without the demo breaking, the project is ready for the resume.

---

Start with day 1 today. Create the repo, get docker-compose up with just postgres and redis running, and write the /health endpoint. That is your first commit.
