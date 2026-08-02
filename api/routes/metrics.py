"""
Prometheus scrape endpoint: GET /metrics.

Open access by design (no auth) — scrapers hit this from within the cluster.
Delegates to get_metrics_response(), which aggregates across processes when
multiprocess mode is active.
"""

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from api.metrics.prometheus import get_metrics_response

router = APIRouter()


@router.get("/metrics", include_in_schema=False)
async def metrics() -> PlainTextResponse:
    """Return metrics in Prometheus exposition format."""
    return get_metrics_response()
