"""Application state container.

Holds the long-lived resources a process owns: settings, the database engine,
the Redis client. Constructed once during lifespan startup and torn down in
reverse order at shutdown.

Why a container rather than module-level globals: globals make it impossible to
run two independently-configured instances in one interpreter, which is exactly
what the test suite needs. Passing state explicitly also makes each component's
dependencies visible in its signature instead of hidden in an import.

The same container is used by the API and the worker. They are different
processes with different lifecycles, but they need identical wiring — building
it twice would guarantee they drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config.settings import Settings
from app.infrastructure.database import Database
from app.infrastructure.redis import RedisClient
from app.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class AppState:
    """Long-lived resources for one process."""

    settings: Settings
    database: Database
    redis: RedisClient
    #: Flipped false when SIGTERM arrives, so readiness starts failing *before*
    #: connections are torn down. See :meth:`begin_shutdown`.
    accepting_traffic: bool = field(default=True)

    @classmethod
    def create(cls, settings: Settings) -> AppState:
        """Build the container without opening any connections.

        Construction is deliberately side-effect free; :meth:`startup` performs
        the I/O. That split lets tests build state and swap a component before
        anything connects.
        """
        return cls(
            settings=settings,
            database=Database(settings.database),
            redis=RedisClient(settings.redis),
        )

    async def startup(self) -> None:
        """Open connections in dependency order.

        Postgres first: it is required, so failing fast on it avoids standing up
        a Redis pool we would immediately discard.
        """
        await self.database.connect()
        await self.redis.connect()
        logger.info(
            "app_state_started",
            environment=self.settings.environment.value,
            version=self.settings.service_version,
        )

    async def shutdown(self) -> None:
        """Close connections in reverse order, tolerating individual failures.

        Each teardown is guarded: a failure closing Redis must not prevent the
        database pool from being disposed. A half-finished shutdown leaks
        server-side connections, which during a rolling deploy can exhaust
        ``max_connections`` on a database nobody is actually using.
        """
        for name, closer in (("redis", self.redis.disconnect), ("database", self.database.disconnect)):
            try:
                await closer()
            except Exception as exc:
                logger.warning("shutdown_component_failed", component=name, error=str(exc))
        logger.info("app_state_stopped")

    def begin_shutdown(self) -> None:
        """Mark the process as draining.

        Called on SIGTERM, before connections close. Readiness then reports
        not-ready while the process is still serving, so the load balancer stops
        sending new work *before* in-flight work is disrupted. Skipping this
        step is what causes dropped requests during an otherwise clean deploy
        (§41).
        """
        self.accepting_traffic = False
        logger.info("draining_started")
