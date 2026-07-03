"""
Health and readiness probes (k8s-style).

- ``/health`` is a *liveness* probe: is the API process up? It touches no
  downstream dependencies.
- ``/ready`` is a *readiness* probe: can the API actually serve traffic? For now
  that means Postgres is reachable (a ``SELECT 1``). A Redis check will be added
  here once api/core/redis_client.py exists (SPEC section 6).
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.database import get_db

router = APIRouter(tags=["system"])

# Kept in sync with the app version in api/main.py.
SERVICE_NAME = "cv-inference-api"
SERVICE_VERSION = "0.1.0"


@router.get("/health")
async def health() -> dict:
    """Liveness probe. 200 as long as the process is serving requests."""
    return {"status": "ok", "service": SERVICE_NAME, "version": SERVICE_VERSION}


@router.get("/ready")
async def ready(db: AsyncSession = Depends(get_db)) -> dict:
    """Readiness probe. 200 if Postgres answers ``SELECT 1``, else 503."""
    try:
        await db.execute(text("SELECT 1"))
    except Exception as exc:  # broad: any DB error means "not ready"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "not ready",
                "postgres": "unreachable",
                "error": str(exc),
            },
        )
    return {"status": "ready", "postgres": "ok"}
