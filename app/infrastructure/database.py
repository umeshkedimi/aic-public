"""PostgreSQL engine, session management, and pool instrumentation.

Postgres is the system of record for jobs, workflow state, events, and the
audit trail. It is also the durable job queue (ADR-004). That concentration is
deliberate — it removes the dual-write problem between "job persisted" and
"job enqueued" — but it makes the connection pool the single most important
saturation point in the system, so it is instrumented from day one.

Two timeouts operate at different layers and are both required:

``pool_timeout``
    Client-side. How long a coroutine waits for a free connection. Bounded so
    that pool pressure surfaces as a fast 503 instead of unbounded latency.
``statement_timeout``
    Server-side, set per connection. Postgres itself kills a query that runs
    too long. Without it, a pathological query holds its connection
    indefinitely and starves the pool no matter what the client-side timeout says.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import Pool

from app.config.settings import DatabaseSettings
from app.domain.errors import DependencyDownError, TimeoutError_, TransientError
from app.observability.logging import get_logger
from app.observability.metrics import get_metrics

logger = get_logger(__name__)

_COMPONENT = "postgres"


def _install_pool_instrumentation(engine: AsyncEngine) -> None:
    """Emit pool checkout timing and exhaustion counters.

    Hooks the *sync* engine underneath the async wrapper, which is where
    SQLAlchemy's pool events actually fire.

    The checkout duration histogram is the leading indicator of pool
    exhaustion: it climbs well before requests start failing, which makes it
    the signal to alert on rather than the errors it eventually causes.
    """
    metrics = get_metrics()
    sync_engine = engine.sync_engine

    @event.listens_for(sync_engine, "checkout")
    def _on_checkout(dbapi_conn: Any, conn_record: Any, conn_proxy: Any) -> None:
        conn_record.info["checkout_at"] = time.perf_counter()
        _refresh_pool_gauges(sync_engine.pool, metrics)

    @event.listens_for(sync_engine, "checkin")
    def _on_checkin(dbapi_conn: Any, conn_record: Any) -> None:
        started = conn_record.info.pop("checkout_at", None)
        if started is not None:
            metrics.db_pool_checkout_duration_seconds.observe(time.perf_counter() - started)
        _refresh_pool_gauges(sync_engine.pool, metrics)


def _refresh_pool_gauges(pool: Pool, metrics: Any) -> None:
    """Update pool gauges. Guarded — gauge upkeep must never break a query."""
    try:
        metrics.db_pool_connections.labels(state="in_use").set(pool.checkedout())  # type: ignore[attr-defined]
        metrics.db_pool_connections.labels(state="available").set(pool.checkedin())  # type: ignore[attr-defined]
        metrics.db_pool_connections.labels(state="overflow").set(pool.overflow())  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        # NullPool and friends don't expose these. Not worth failing over.
        pass


def create_engine(settings: DatabaseSettings) -> AsyncEngine:
    """Build the async engine with explicit pool and timeout configuration."""
    engine = create_async_engine(
        settings.async_dsn,
        echo=settings.echo_sql,
        pool_size=settings.pool_size,
        max_overflow=settings.max_overflow,
        pool_timeout=settings.pool_timeout_seconds,
        pool_recycle=settings.pool_recycle_seconds,
        pool_pre_ping=settings.pool_pre_ping,
        connect_args={
            "timeout": settings.connect_timeout_seconds,
            # asyncpg applies server settings per connection. statement_timeout
            # is in milliseconds and must be a string.
            "server_settings": {
                "statement_timeout": str(int(settings.statement_timeout_seconds * 1000)),
                "application_name": "prf-agentic-lab",
            },
            # asyncpg caches prepared statements per connection. Harmless
            # normally, but it breaks against transaction-mode PgBouncer, which
            # is the likeliest production topology — so disable it up front
            # rather than discovering it during a pooler migration.
            "statement_cache_size": 0,
        },
    )
    _install_pool_instrumentation(engine)
    logger.info(
        "database_engine_created",
        pool_size=settings.pool_size,
        max_overflow=settings.max_overflow,
        pool_timeout_s=settings.pool_timeout_seconds,
        statement_timeout_s=settings.statement_timeout_seconds,
    )
    return engine


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Build the session factory.

    ``expire_on_commit=False`` so ORM objects stay usable after commit. With the
    default, touching an attribute post-commit triggers a lazy refresh — an
    implicit database round-trip, and in async code a confusing
    ``MissingGreenlet`` rather than an obvious error.
    """
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


class Database:
    """Owns the engine and session factory for a process.

    Constructed once at startup and disposed at shutdown. Holding it on an
    object rather than in module globals keeps tests able to stand up an
    isolated instance.
    """

    def __init__(self, settings: DatabaseSettings) -> None:
        self._settings = settings
        self._engine: AsyncEngine | None = None
        self._session_factory: async_sessionmaker[AsyncSession] | None = None

    async def connect(self) -> None:
        if self._engine is not None:
            return
        self._engine = create_engine(self._settings)
        self._session_factory = create_session_factory(self._engine)

    async def disconnect(self) -> None:
        """Dispose the pool, closing every connection.

        Called during graceful shutdown (§41). Skipping it leaves connections
        open on the server until they time out, which during a rolling deploy
        can exhaust ``max_connections`` with connections nobody is using.
        """
        if self._engine is not None:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info("database_engine_disposed")

    @property
    def engine(self) -> AsyncEngine:
        if self._engine is None:
            raise RuntimeError("Database.connect() has not been called")
        return self._engine

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield a session inside a transaction, committing on clean exit.

        Rolls back on any exception, then re-raises classified. Callers get a
        typed error instead of a driver-specific one, which is what makes retry
        decisions possible upstream.
        """
        if self._session_factory is None:
            raise RuntimeError("Database.connect() has not been called")

        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception as exc:
            await session.rollback()
            raise _classify_db_error(exc) from exc
        finally:
            await session.close()

    async def health_check(self, timeout_seconds: float = 2.0) -> tuple[bool, float]:
        """Probe connectivity. Returns ``(healthy, elapsed_seconds)``.

        Bounded by its own timeout, shorter than the readiness probe's budget:
        a health check that can hang is worse than none, because it takes the
        probe endpoint down with the dependency it was meant to report on.
        """
        metrics = get_metrics()
        started = time.perf_counter()
        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            elapsed = time.perf_counter() - started
            metrics.dependency_up.labels(dependency=_COMPONENT).set(1)
            metrics.dependency_check_duration_seconds.labels(dependency=_COMPONENT).observe(elapsed)
            return True, elapsed
        except Exception as exc:
            elapsed = time.perf_counter() - started
            metrics.dependency_up.labels(dependency=_COMPONENT).set(0)
            metrics.dependency_check_duration_seconds.labels(dependency=_COMPONENT).observe(elapsed)
            logger.warning("database_health_check_failed", error=str(exc), elapsed_s=elapsed)
            return False, elapsed


def _classify_db_error(exc: Exception) -> Exception:
    """Map driver exceptions onto the error taxonomy.

    Done by inspecting the exception chain's type names rather than importing
    asyncpg's exception hierarchy directly, so this module does not hard-couple
    to one driver. Swapping asyncpg for psycopg should not require rewriting
    classification.
    """
    from sqlalchemy.exc import DBAPIError, OperationalError, TimeoutError as SATimeoutError

    if isinstance(exc, SATimeoutError):
        # SQLAlchemy raises this when pool checkout exceeds pool_timeout.
        get_metrics().db_pool_exhausted_total.inc()
        return DependencyDownError(
            "database connection pool exhausted",
            component=_COMPONENT,
            details={"hint": "raise pool_size, lower concurrency, or shed load"},
            cause=exc,
        )

    if isinstance(exc, OperationalError):
        return TransientError(
            f"database operational error: {type(exc).__name__}",
            component=_COMPONENT,
            idempotent_operation=True,
            cause=exc,
        )

    if isinstance(exc, DBAPIError):
        chain = " ".join(type(e).__name__ for e in _exception_chain(exc))
        if "QueryCanceled" in chain or "statement timeout" in str(exc).lower():
            return TimeoutError_(
                "query exceeded statement_timeout",
                component=_COMPONENT,
                idempotent_operation=True,
                cause=exc,
            )
        if exc.connection_invalidated:
            return TransientError(
                "database connection invalidated mid-query",
                component=_COMPONENT,
                idempotent_operation=True,
                cause=exc,
            )

    return exc


def _exception_chain(exc: BaseException, limit: int = 10) -> list[BaseException]:
    """Walk ``__cause__``/``__context__``, bounded against cyclic chains."""
    seen: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and len(seen) < limit:
        seen.append(current)
        current = current.__cause__ or current.__context__
        if current in seen:
            break
    return seen
