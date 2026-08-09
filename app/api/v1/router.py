"""API v1 router assembly.

Two routers, separated on purpose:

``operational_router``
    Health probes and metrics. **Unversioned and unprefixed** — mounted at
    ``/health/*`` and ``/metrics``. Kubernetes probes and Prometheus scrape
    configs are infrastructure, not API consumers; versioning them would mean a
    v2 release requires editing every manifest and scrape job.

``api_router``
    Business endpoints, mounted under ``/api/v1``. Versioned because these are
    a contract with external callers.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import health, metrics

operational_router = APIRouter()
operational_router.include_router(health.router)
operational_router.include_router(metrics.router)

api_router = APIRouter(prefix="/api/v1")
# Business routes (PRF jobs, approvals, events) mount here in Phase 2.
