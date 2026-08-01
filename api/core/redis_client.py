"""
Async Redis client + pub/sub helpers for real-time job status.

Workers publish status events to a per-job channel (``job.status.{job_id}``); the
API process subscribes and fans them out to WebSocket clients. This module owns
the async (API-side) client and the channel-name convention. Workers publish
with a *sync* client (see workers/tasks.py) using the same ``channel_for`` name.
"""

import json
import logging
from collections.abc import AsyncIterator
from uuid import UUID

import redis.asyncio as redis_async

from api.config import settings

logger = logging.getLogger(__name__)

# One channel per job. The API's background subscriber listens to all of them at
# once via the pattern ``job.status.*`` (psubscribe).
_CHANNEL_PREFIX = "job.status"
# Pattern the API's background subscriber uses to receive every job's events.
CHANNEL_PATTERN = f"{_CHANNEL_PREFIX}.*"


def channel_for(job_id: UUID) -> str:
    """Return the pub/sub channel name for a job's status events."""
    return f"{_CHANNEL_PREFIX}.{job_id}"


def get_redis() -> redis_async.Redis:
    """Create an async Redis client from ``settings.redis_url``.

    ``decode_responses=True`` so pub/sub payloads arrive as ``str`` (not bytes).
    ``from_url`` is lazy — an unreachable Redis surfaces on the first command,
    not here.
    """
    try:
        return redis_async.from_url(settings.redis_url, decode_responses=True)
    except Exception as exc:
        raise ConnectionError(
            f"could not create Redis client for {settings.redis_url}: {exc}"
        ) from exc


async def publish_status(
    redis_client: redis_async.Redis, job_id: UUID, event: dict
) -> None:
    """Publish a status ``event`` (as JSON) to the job's channel."""
    await redis_client.publish(channel_for(job_id), json.dumps(event))


async def subscribe_to_job_status(
    redis_client: redis_async.Redis, job_id: UUID
) -> AsyncIterator[dict]:
    """Yield decoded status events for a single job until the caller stops.

    Standalone per-job subscriber (handy for tests / a per-connection design).
    The wired bridge in api/main.py instead uses one process-wide ``psubscribe``.
    """
    channel = channel_for(job_id)
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(channel)
    try:
        async for message in pubsub.listen():
            if message.get("type") != "message":
                continue  # skip subscribe/unsubscribe confirmations
            try:
                yield json.loads(message["data"])
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("dropping malformed event on %s: %s", channel, exc)
    finally:
        await pubsub.unsubscribe(channel)
        await pubsub.aclose()
