"""
FastAPI application entry point.

Creates the app, mounts routers, and runs a lifespan-managed background task that
subscribes to Redis pub/sub (``job.status.*``) and fans status events out to
WebSocket clients via the shared ConnectionManager. That background task is the
bridge decoupling workers from WebSocket clients entirely (SPEC §8).
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from redis.exceptions import RedisError

from api.core.redis_client import CHANNEL_PATTERN, get_redis
from api.routes import health, jobs, websocket
from api.websocket import connection_manager

logger = logging.getLogger(__name__)

# Backoff before the subscriber retries after a Redis error.
_SUBSCRIBER_RETRY_SECONDS = 2


async def _status_subscriber(redis_client) -> None:
    """Fan Redis status events out to WebSocket clients until cancelled.

    Pattern-subscribes to all job channels; for each message it parses the
    job_id from the channel name and broadcasts the event to that job's local
    WebSocket connections. Reconnects on Redis errors; exits on cancellation.
    """
    while True:
        pubsub = redis_client.pubsub()
        try:
            await pubsub.psubscribe(CHANNEL_PATTERN)
            logger.info("status subscriber listening on %s", CHANNEL_PATTERN)
            async for message in pubsub.listen():
                if message.get("type") != "pmessage":
                    continue  # skip (p)subscribe confirmations
                channel = message["channel"]
                try:
                    job_id = UUID(channel.rsplit(".", 1)[-1])
                    event = json.loads(message["data"])
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    logger.warning("dropping bad status message on %s: %s", channel, exc)
                    continue
                await connection_manager.broadcast(job_id, event)
        except asyncio.CancelledError:
            logger.info("status subscriber cancelled")
            raise
        except RedisError as exc:
            logger.warning(
                "status subscriber Redis error (%s); retrying in %ds",
                exc,
                _SUBSCRIBER_RETRY_SECONDS,
            )
            await asyncio.sleep(_SUBSCRIBER_RETRY_SECONDS)
        finally:
            try:
                await pubsub.aclose()
            except Exception:
                pass


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start/stop the Redis→WebSocket status subscriber alongside the app."""
    redis_client = get_redis()
    subscriber = asyncio.create_task(_status_subscriber(redis_client))
    logger.info("status subscriber started")
    try:
        yield
    finally:
        subscriber.cancel()
        try:
            await subscriber
        except asyncio.CancelledError:
            pass
        await redis_client.aclose()
        logger.info("status subscriber stopped")


# ─────────────────────────────────────────────────────
# App instance
# ─────────────────────────────────────────────────────
app = FastAPI(
    title="CV Inference Service",
    description="Distributed computer vision inference microservice",
    version="0.1.0",
    docs_url="/docs",          # Swagger UI at /docs
    redoc_url="/redoc",        # ReDoc UI at /redoc
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────────────
# CORS middleware (so the React frontend can call this API)
# ─────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────
# Routers: system health, jobs API, and the WebSocket endpoint
# ─────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(jobs.router)
app.include_router(websocket.router)


# ─────────────────────────────────────────────────────
# Root redirect to docs (nice DX for browser visits)
# ─────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "CV Inference Service. See /docs for API documentation."}
