"""
WebSocket endpoint for real-time job status: /ws/jobs/{job_id}.

The handler validates the job id, registers the socket with the shared
ConnectionManager, then keeps the connection open. It does NOT talk to Redis —
status events are pushed by the background subscriber (api/main.py) via the
ConnectionManager, which keeps workers fully decoupled from clients (SPEC §8).
"""

from uuid import UUID

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.metrics.prometheus import WEBSOCKET_CONNECTIONS
from api.websocket import connection_manager

logger = structlog.get_logger(__name__)

router = APIRouter()

# WebSocket close code for a policy violation (RFC 6455).
_WS_POLICY_VIOLATION = 1008


@router.websocket("/ws/jobs/{job_id}")
async def job_status_ws(websocket: WebSocket, job_id: str) -> None:
    """Stream status events for one job to a connected WebSocket client."""
    try:
        job_uuid = UUID(job_id)
    except ValueError:
        # Accept then close so the client gets a clean 1008 close frame rather
        # than a bare handshake rejection.
        await websocket.accept()
        await websocket.close(code=_WS_POLICY_VIOLATION, reason="invalid job_id")
        logger.info("ws_rejected_invalid_job_id", job_id=job_id)
        return

    await connection_manager.connect(job_uuid, websocket)
    WEBSOCKET_CONNECTIONS.inc()
    try:
        # Keep the socket open. We don't act on inbound frames, but receiving
        # lets us detect client disconnects and tolerates client-side pings.
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass  # normal client close
    except Exception as exc:
        logger.error("ws_error", job_id=str(job_uuid), error=str(exc))
    finally:
        WEBSOCKET_CONNECTIONS.dec()
        await connection_manager.disconnect(job_uuid, websocket)
