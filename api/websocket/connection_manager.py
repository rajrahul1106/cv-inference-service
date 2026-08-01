"""
ConnectionManager: in-process registry of WebSocket clients, keyed by job_id.

The WS route registers/unregisters connections here; the background Redis
subscriber (api/main.py) calls ``broadcast`` to push status events to the
clients watching a given job. State is per-process/per-event-loop — with
multiple API replicas each keeps its own set and Redis pub/sub fans events to
all of them (see the Day 8 plan).

Authored per the Day 8 task spec; this is on CLAUDE.md's "write by hand" list.
"""

import logging
from collections import defaultdict
from uuid import UUID

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Tracks active WebSocket connections per job and broadcasts to them."""

    def __init__(self) -> None:
        self._active: dict[UUID, set[WebSocket]] = defaultdict(set)

    async def connect(self, job_id: UUID, ws: WebSocket) -> None:
        """Accept the socket and register it under ``job_id``."""
        await ws.accept()
        self._active[job_id].add(ws)
        logger.info(
            "ws connect job=%s connections=%d", job_id, len(self._active[job_id])
        )

    async def disconnect(self, job_id: UUID, ws: WebSocket) -> None:
        """Unregister a socket, dropping the job's entry when it empties."""
        # .get (not self._active[job_id]) so we never auto-create an empty set.
        connections = self._active.get(job_id)
        if connections is not None:
            connections.discard(ws)
            if not connections:
                del self._active[job_id]
        remaining = len(self._active.get(job_id) or ())
        logger.info("ws disconnect job=%s connections=%d", job_id, remaining)

    async def broadcast(self, job_id: UUID, message: dict) -> None:
        """Send ``message`` as JSON to every socket watching ``job_id``.

        Uses ``.get`` deliberately: the process-wide subscriber calls this for
        *every* job's events, so indexing a defaultdict here would leak an empty
        set per job_id. Sockets that fail to send are treated as dead and dropped.
        """
        connections = self._active.get(job_id)
        if not connections:
            return
        dead: set[WebSocket] = set()
        for ws in connections:
            try:
                await ws.send_json(message)
            except Exception:
                dead.add(ws)
        for ws in dead:
            connections.discard(ws)
        if dead:
            logger.info(
                "ws broadcast job=%s dropped %d dead connection(s)", job_id, len(dead)
            )
        if not connections:
            self._active.pop(job_id, None)
