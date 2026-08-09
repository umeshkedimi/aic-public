"""Redis client for ephemeral, reconstructible state.

**Redis is not a system of record here.** It holds rate-limit counters,
circuit-breaker state, the idempotency fast path, and worker wakeup
notifications — all of which can be rebuilt or safely defaulted. Jobs and
workflow state live in Postgres (ADR-003, ADR-004).

That boundary is what makes "Redis is down" a *degradation* rather than an
outage, and it is the single most useful property of this design:

- Rate limiting → fail open (serve traffic, lose enforcement) or fail closed,
  per call site. Both are defensible; the choice is explicit at each use.
- Circuit breaker → falls back to per-process in-memory state. Less coordinated
  across replicas, still functional.
- Idempotency → falls back to the Postgres uniqueness constraint, which is the
  actual correctness guarantee. Redis is only the fast path.
- Worker wakeup → falls back to polling. Higher latency, no lost work.

Every method here is bounded by ``socket_timeout``. An unbounded Redis call
inside a request handler converts a Redis brownout into API-wide latency.
"""

from __future__ import annotations

import time
from typing import Any, Final

import redis.asyncio as aioredis
from redis.asyncio.connection import ConnectionPool
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError
from redis.exceptions import TimeoutError as RedisTimeoutError

from app.config.settings import RedisSettings
from app.domain.errors import DependencyDownError, TimeoutError_
from app.observability.logging import get_logger
from app.observability.metrics import get_metrics

logger = get_logger(__name__)

_COMPONENT: Final = "redis"


class RedisClient:
    """Thin instrumented wrapper over ``redis.asyncio``.

    Wrapping rather than exposing the raw client buys three things the raw
    client does not give us: uniform metrics, uniform error classification, and
    a single place to enforce that no call is unbounded.
    """

    def __init__(self, settings: RedisSettings) -> None:
        self._settings = settings
        self._pool: ConnectionPool | None = None
        self._client: aioredis.Redis | None = None

    async def connect(self) -> None:
        if self._client is not None:
            return

        self._pool = ConnectionPool.from_url(
            str(self._settings.dsn),
            max_connections=self._settings.max_connections,
            socket_timeout=self._settings.socket_timeout_seconds,
            socket_connect_timeout=self._settings.socket_connect_timeout_seconds,
            health_check_interval=self._settings.health_check_interval_seconds,
            decode_responses=True,
        )
        self._client = aioredis.Redis(connection_pool=self._pool)
        logger.info(
            "redis_client_created",
            max_connections=self._settings.max_connections,
            socket_timeout_s=self._settings.socket_timeout_seconds,
        )

    async def disconnect(self) -> None:
        """Close the client and pool during graceful shutdown."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._pool is not None:
            await self._pool.aclose()
            self._pool = None
            logger.info("redis_client_closed")

    @property
    def client(self) -> aioredis.Redis:
        if self._client is None:
            raise RuntimeError("RedisClient.connect() has not been called")
        return self._client

    async def execute(self, operation: str, *args: Any, **kwargs: Any) -> Any:
        """Run a Redis command with timing, outcome metrics, and classification.

        ``operation`` labels the metric. It is the *command name* — a bounded
        set — never a key, which would be unbounded cardinality.
        """
        metrics = get_metrics()
        started = time.perf_counter()
        try:
            method = getattr(self.client, operation)
            result = await method(*args, **kwargs)
        except RedisTimeoutError as exc:
            self._record(metrics, operation, "timeout", started)
            raise TimeoutError_(
                f"redis {operation} timed out",
                component=_COMPONENT,
                idempotent_operation=True,
                cause=exc,
            ) from exc
        except RedisConnectionError as exc:
            self._record(metrics, operation, "unavailable", started)
            raise DependencyDownError(
                f"redis unavailable during {operation}",
                component=_COMPONENT,
                idempotent_operation=True,
                cause=exc,
            ) from exc
        except RedisError as exc:
            self._record(metrics, operation, "error", started)
            raise DependencyDownError(
                f"redis {operation} failed: {type(exc).__name__}",
                component=_COMPONENT,
                cause=exc,
            ) from exc
        else:
            self._record(metrics, operation, "success", started)
            return result

    @staticmethod
    def _record(metrics: Any, operation: str, outcome: str, started: float) -> None:
        elapsed = time.perf_counter() - started
        metrics.redis_operations_total.labels(operation=operation, outcome=outcome).inc()
        metrics.redis_operation_duration_seconds.labels(operation=operation).observe(elapsed)

    async def health_check(self) -> tuple[bool, float]:
        """Probe with PING. Returns ``(healthy, elapsed_seconds)``."""
        metrics = get_metrics()
        started = time.perf_counter()
        try:
            await self.client.ping()
            elapsed = time.perf_counter() - started
            metrics.dependency_up.labels(dependency=_COMPONENT).set(1)
            metrics.dependency_check_duration_seconds.labels(dependency=_COMPONENT).observe(elapsed)
            return True, elapsed
        except Exception as exc:
            elapsed = time.perf_counter() - started
            metrics.dependency_up.labels(dependency=_COMPONENT).set(0)
            metrics.dependency_check_duration_seconds.labels(dependency=_COMPONENT).observe(elapsed)
            logger.warning("redis_health_check_failed", error=str(exc), elapsed_s=elapsed)
            return False, elapsed
