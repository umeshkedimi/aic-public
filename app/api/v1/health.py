"""Health and readiness probes.

**Liveness and readiness must check different things.** Conflating them is one
of the most common and most damaging Kubernetes mistakes:

``/health/live`` — *is this process broken beyond recovery?*
    Checks **nothing external**. Answers "is the event loop still turning".
    Failing it gets the container **killed and restarted**.

    If liveness checked Postgres, a 30-second database blip would fail the
    probe on *every replica simultaneously*, and Kubernetes would restart the
    entire fleet. The database is still down, the restarted pods still cannot
    reach it, and now there is a thundering herd of cold starts and connection
    storms on a database that was already struggling. A brief dependency
    hiccup becomes a full outage — caused entirely by the health check.

``/health/ready`` — *can this process serve traffic right now?*
    Checks dependencies. Failing it removes the pod from Service endpoints but
    **leaves it running**, so it recovers on its own the moment the dependency
    returns. That is exactly the behaviour a dependency blip should produce.

``/health/startup`` — *has this process finished booting?*
    Gives slow startup a longer grace period without loosening the liveness
    threshold for steady-state operation.
"""

from __future__ import annotations

import asyncio
import time
from typing import Literal

from fastapi import APIRouter, Response
from pydantic import BaseModel, Field

from app.api.deps import AppStateDep
from app.observability.logging import get_logger
from app.state import AppState

logger = get_logger(__name__)

router = APIRouter(tags=["health"])

#: Total budget for all readiness checks combined. Must stay below the probe's
#: own `timeoutSeconds` in the Kubernetes manifest, or kubelet gives up first
#: and the response never arrives.
READINESS_BUDGET_SECONDS = 3.0


class DependencyStatus(BaseModel):
    name: str
    healthy: bool
    latency_ms: float = Field(description="Probe round-trip. A rising trend precedes failure.")
    error: str | None = None
    required: bool = Field(
        description="Whether an unhealthy result should fail readiness. Optional "
        "dependencies degrade the service without removing it from rotation."
    )


class LivenessResponse(BaseModel):
    status: Literal["alive"]
    service: str
    version: str


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    service: str
    version: str
    dependencies: list[DependencyStatus]
    checked_in_ms: float


@router.get(
    "/health/live",
    response_model=LivenessResponse,
    summary="Liveness probe — restart the container if this fails",
)
async def liveness(
    state: AppStateDep,
) -> LivenessResponse:
    """Confirm the process is running and the event loop is responsive.

    Intentionally does no I/O. Returning at all *is* the signal: if the event
    loop were blocked or deadlocked, this coroutine would never be scheduled
    and the probe would time out.
    """
    settings = state.settings
    return LivenessResponse(
        status="alive",
        service=settings.service_name,
        version=settings.service_version,
    )


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe — remove from load balancer if this fails",
)
async def readiness(
    response: Response,
    state: AppStateDep,
) -> ReadinessResponse:
    """Check every dependency needed to serve traffic.

    Checks run **concurrently** under one shared budget. Run serially, a slow
    Postgres plus a slow Redis would sum, and the probe would time out reporting
    nothing useful about either.

    Returns 503 when a *required* dependency is unhealthy. Optional dependencies
    are reported but do not fail the probe — losing Redis costs us the rate-limit
    fast path, not the ability to serve.
    """
    settings = state.settings
    started = time.perf_counter()

    try:
        async with asyncio.timeout(READINESS_BUDGET_SECONDS):
            results = await asyncio.gather(
                _check_postgres(state),
                _check_redis(state),
            )
    except TimeoutError:
        # Budget blown. Report not-ready rather than hanging — a probe that
        # never answers is treated as a failure anyway, but with no diagnostics.
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.warning("readiness_check_timed_out", budget_s=READINESS_BUDGET_SECONDS)
        response.status_code = 503
        return ReadinessResponse(
            status="not_ready",
            service=settings.service_name,
            version=settings.service_version,
            dependencies=[
                DependencyStatus(
                    name="all",
                    healthy=False,
                    latency_ms=round(elapsed_ms, 2),
                    error=f"readiness budget of {READINESS_BUDGET_SECONDS}s exceeded",
                    required=True,
                )
            ],
            checked_in_ms=round(elapsed_ms, 2),
        )

    elapsed_ms = (time.perf_counter() - started) * 1000
    ready = all(d.healthy for d in results if d.required)

    if not ready:
        failing = [d.name for d in results if d.required and not d.healthy]
        logger.warning("readiness_failed", failing_dependencies=failing)
        response.status_code = 503

    return ReadinessResponse(
        status="ready" if ready else "not_ready",
        service=settings.service_name,
        version=settings.service_version,
        dependencies=list(results),
        checked_in_ms=round(elapsed_ms, 2),
    )


@router.get(
    "/health/startup",
    response_model=ReadinessResponse,
    summary="Startup probe — allows a longer boot grace period",
)
async def startup(
    response: Response,
    state: AppStateDep,
) -> ReadinessResponse:
    """Same checks as readiness, probed with a longer failure threshold.

    Separated so the liveness threshold can stay tight for steady state while
    a cold start (migrations, pool warm-up) still gets room to finish.
    """
    return await readiness(response, state)


async def _check_postgres(state: AppState) -> DependencyStatus:
    """Postgres is required: it is the system of record and the job queue."""
    database = state.database
    try:
        healthy, elapsed = await database.health_check()
        return DependencyStatus(
            name="postgres",
            healthy=healthy,
            latency_ms=round(elapsed * 1000, 2),
            error=None if healthy else "health check returned unhealthy",
            required=True,
        )
    except Exception as exc:
        return DependencyStatus(
            name="postgres",
            healthy=False,
            latency_ms=0.0,
            error=type(exc).__name__,
            required=True,
        )


async def _check_redis(state: AppState) -> DependencyStatus:
    """Redis is optional by design.

    It holds only reconstructible state, and every consumer has a documented
    fallback (see :mod:`app.infrastructure.redis`). Marking it required would
    take the whole API out of rotation over a cache outage the service is
    explicitly built to survive.
    """
    redis_client = state.redis
    try:
        healthy, elapsed = await redis_client.health_check()
        return DependencyStatus(
            name="redis",
            healthy=healthy,
            latency_ms=round(elapsed * 1000, 2),
            error=None if healthy else "ping failed",
            required=False,
        )
    except Exception as exc:
        return DependencyStatus(
            name="redis",
            healthy=False,
            latency_ms=0.0,
            error=type(exc).__name__,
            required=False,
        )
