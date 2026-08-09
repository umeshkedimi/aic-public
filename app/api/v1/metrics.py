"""Prometheus scrape endpoint.

Served from the application's own registry rather than the prometheus_client
global default, so the exported set is exactly what this process declared. The
global default silently accumulates collectors from any imported library, which
makes the exported surface a function of your import graph.
"""

from __future__ import annotations

from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.observability.metrics import get_registry

router = APIRouter(tags=["observability"])


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    response_class=Response,
    # Excluded from OpenAPI: the body is Prometheus text format, not JSON, and
    # documenting it as a JSON schema would be actively misleading.
    include_in_schema=False,
)
async def metrics() -> Response:
    """Render the registry in Prometheus text exposition format.

    ``generate_latest`` is synchronous and CPU-bound over the registry. That is
    acceptable at this registry size (tens of series) but is exactly why
    cardinality is controlled so carefully: a registry with 100k series would
    block the event loop on every scrape, degrading the service it is meant to
    be observing.
    """
    return Response(content=generate_latest(get_registry()), media_type=CONTENT_TYPE_LATEST)
