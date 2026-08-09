"""FastAPI application factory and process lifecycle.

Structured as a factory rather than a module-level ``app = FastAPI()`` so tests
can build an isolated instance with injected configuration. A module-level app
runs its lifespan against the ambient environment, which makes hermetic testing
essentially impossible.

Graceful shutdown (§41) is the subtle part and is documented inline below.
"""

from __future__ import annotations

import asyncio
import signal
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.errors import register_exception_handlers
from app.api.middleware import (
    AccessLogMiddleware,
    CorrelationMiddleware,
    MetricsMiddleware,
    RequestSizeLimitMiddleware,
)
from app.api.v1.router import api_router, operational_router
from app.config.settings import Settings, get_settings
from app.domain.errors import ConfigurationError
from app.observability.logging import configure_logging, get_logger
from app.observability.metrics import get_metrics
from app.state import AppState

logger = get_logger(__name__)

#: Seconds to keep serving after SIGTERM before closing connections.
#:
#: Kubernetes removes a pod from Service endpoints and sends SIGTERM
#: concurrently, and endpoint propagation to every kube-proxy is *eventually*
#: consistent. Exiting immediately means traffic still arrives at a socket that
#: is already closed — connection-refused errors during an otherwise healthy
#: deploy. This delay lets readiness flip to not-ready and propagate first.
DRAIN_DELAY_SECONDS = 5.0


def _validate_configuration(settings: Settings) -> None:
    """Fail fast on invalid configuration.

    Every problem is logged before raising, so one restart surfaces the full
    list rather than one error per deploy cycle.
    """
    problems = settings.validate_for_environment()
    if not problems:
        return
    for problem in problems:
        logger.error("configuration_invalid", problem=problem)
    raise ConfigurationError(
        f"{len(problems)} configuration problem(s) detected; refusing to start",
        component="config",
        details={"problems": problems},
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Own resource setup and teardown for the process."""
    settings: Settings = app.state.settings
    _validate_configuration(settings)

    state = AppState.create(settings)
    app.state.app_state = state

    get_metrics().set_build_info(
        version=settings.service_version,
        environment=settings.environment.value,
        service=settings.service_name,
    )

    await state.startup()
    _install_signal_handlers(state)

    logger.info(
        "api_started",
        environment=settings.environment.value,
        version=settings.service_version,
        docs_enabled=not settings.is_production,
    )

    try:
        yield
    finally:
        # Reached on both clean shutdown and startup failure. Draining here
        # rather than only in the signal handler covers the non-signal paths
        # (uvicorn reload, test teardown, an exception during startup).
        if state.accepting_traffic:
            state.begin_shutdown()
        await state.shutdown()


def _install_signal_handlers(state: AppState) -> None:
    """Flip to draining on SIGTERM/SIGINT, ahead of uvicorn's own teardown.

    We deliberately do **not** cancel in-flight work here. The handler only
    marks the process not-ready; uvicorn's shutdown then waits for active
    requests to finish. Killing in-flight requests to shut down faster is how a
    deploy turns into a burst of 502s.

    Registration is best-effort: ``add_signal_handler`` is unavailable on
    Windows and inside some embedded loops, and neither is a reason to refuse
    to boot.
    """
    loop = asyncio.get_running_loop()

    def _on_signal(signame: str) -> None:
        logger.info("shutdown_signal_received", signal=signame, drain_delay_s=DRAIN_DELAY_SECONDS)
        state.begin_shutdown()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _on_signal, sig.name)
        except (NotImplementedError, RuntimeError):
            logger.debug("signal_handler_unavailable", signal=sig.name)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application.

    :param settings: Injected configuration. Defaults to the process settings.
    """
    settings = settings or get_settings()
    configure_logging(settings)

    app = FastAPI(
        title="PRF Agentic AI — Production Engineering Lab",
        description=(
            "Queue-backed multi-agent PRF pipeline built for failure, load, and "
            "observability. See ARCHITECTURE.md and FAILURE_MODES.md."
        ),
        version=settings.service_version,
        lifespan=lifespan,
        root_path=settings.api.root_path,
        # Interactive docs expose the full schema surface. Fine locally, not in
        # production, where it is free reconnaissance.
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None if settings.is_production else "/redoc",
        openapi_url=None if settings.is_production else "/openapi.json",
    )
    app.state.settings = settings

    # --- Middleware ---------------------------------------------------------
    # Starlette applies middleware bottom-up: the LAST added is the OUTERMOST.
    # Correlation is added last so every inner layer — metrics, access logging,
    # and the exception handlers — already has request_id bound.
    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=settings.api.max_request_bytes)
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(CorrelationMiddleware)

    if settings.api.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.api.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID", "X-Correlation-ID"],
        )

    register_exception_handlers(app)

    app.include_router(operational_router)
    app.include_router(api_router)

    return app


# Uvicorn entry point: `uvicorn app.main:app`.
app = create_app()
