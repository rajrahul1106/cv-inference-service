"""
Locust load test for the jobs API (SPEC §9/§10 Day 14).

Simulates clients submitting inference jobs with a realistic model mix
(60% yolo / 30% face / 10% fire) and polling job status. Each simulated user
submits ~1 job/second (constant pacing).

Run headless (or use scripts/run_benchmarks.sh for the standardized run):

    locust -f scripts/load_test.py --headless -u 50 -r 5 -t 120s \
        --csv benchmarks/bench --host http://localhost:8000

NOTE: percentiles measured here are API-layer latencies (submit + status read).
With a single solo-pool worker the queue intentionally backs up; worker
inference latency is measured separately via the Prometheus histogram.
"""

import logging
import random
from collections import deque
from typing import Any

from locust import HttpUser, constant_pacing, events, task

logger = logging.getLogger(__name__)

# Real, stable test images (also used throughout the project's verification).
YOLO_IMAGE = "https://ultralytics.com/images/bus.jpg"
FACE_IMAGE = "https://ultralytics.com/images/zidane.jpg"
FIRE_IMAGE = "https://ultralytics.com/images/bus.jpg"

# Job ids created by this user, for follow-up status GETs. Bounded so a long
# run doesn't grow memory; oldest ids fall off as new ones arrive.
_RECENT_JOBS_MAX = 50


class InferenceUser(HttpUser):
    """A client that submits inference jobs and checks on their status."""

    host = "http://localhost:8000"
    wait_time = constant_pacing(1)  # each user fires ~1 task per second

    def on_start(self) -> None:
        self.recent_jobs: deque[str] = deque(maxlen=_RECENT_JOBS_MAX)
        self.submitted = 0
        self.completed_seen = 0

    def _submit(self, model_type: str, input_url: str) -> None:
        """POST a job and remember its id for later status reads."""
        with self.client.post(
            "/api/v1/jobs",
            json={"input_url": input_url, "model_type": model_type, "options": {}},
            name=f"POST /api/v1/jobs [{model_type}]",
            catch_response=True,
        ) as response:
            if response.status_code == 202:
                self.recent_jobs.append(response.json()["job_id"])
                self.submitted += 1
                response.success()
            else:
                response.failure(f"expected 202, got {response.status_code}")

    @task(6)
    def submit_yolo(self) -> None:
        self._submit("yolo", YOLO_IMAGE)

    @task(3)
    def submit_face(self) -> None:
        self._submit("face", FACE_IMAGE)

    @task(1)
    def submit_fire(self) -> None:
        self._submit("fire", FIRE_IMAGE)

    @task(3)
    def check_job_status(self) -> None:
        """GET a previously submitted job — the status-read path clients hammer."""
        if not self.recent_jobs:
            return
        job_id = random.choice(self.recent_jobs)
        with self.client.get(
            f"/api/v1/jobs/{job_id}",
            name="GET /api/v1/jobs/{job_id}",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                if response.json().get("status") == "completed":
                    self.completed_seen += 1
                response.success()
            else:
                response.failure(f"expected 200, got {response.status_code}")

    def on_stop(self) -> None:
        logger.info(
            "user done: submitted=%d completed_seen=%d",
            self.submitted,
            self.completed_seen,
        )


@events.test_start.add_listener
def on_test_start(environment: Any, **kwargs: Any) -> None:
    logger.info("load test starting against %s", environment.host)


@events.test_stop.add_listener
def on_test_stop(environment: Any, **kwargs: Any) -> None:
    stats = environment.stats.total
    logger.info(
        "load test finished: requests=%d failures=%d",
        stats.num_requests,
        stats.num_failures,
    )
