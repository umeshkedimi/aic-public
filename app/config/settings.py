"""Typed application configuration.

Every tunable in this system lives here. That is deliberate: the spec's
reliability requirements (timeouts, retry budgets, concurrency ceilings,
backpressure thresholds) are only credible if they are *configurable and
inspectable*, not scattered as magic numbers across call sites.

Config is loaded from the environment with nested delimiter ``__``::

    DATABASE__POOL_SIZE=20
    RESILIENCE__LLM_TIMEOUT_SECONDS=30

Secrets are never defaulted to a real value. Where a secret is required in
production, the default is empty and :meth:`Settings.validate_for_environment`
fails fast at startup rather than letting the service run half-configured.
"""

from __future__ import annotations

import os
from enum import StrEnum
from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    LOCAL = "local"
    CI = "ci"
    STAGING = "staging"
    PRODUCTION = "production"


class DatabaseSettings(BaseSettings):
    """PostgreSQL connection and pool configuration.

    Pool sizing is a load-bearing reliability parameter, not a detail. The pool
    is the first thing to saturate under concurrency, so ``pool_size`` and
    ``max_overflow`` are explicit and measured (see PERFORMANCE.md) rather than
    left at SQLAlchemy's defaults.
    """

    model_config = SettingsConfigDict(env_prefix="DATABASE__", extra="ignore")

    dsn: PostgresDsn = Field(
        default=PostgresDsn("postgresql://prf:prf@localhost:5432/prf"),
        description="Base DSN. Driver-specific URLs are derived from this.",
    )

    # --- Pool ---
    pool_size: int = Field(default=20, ge=1, le=200)
    max_overflow: int = Field(default=10, ge=0, le=200)
    pool_timeout_seconds: float = Field(
        default=5.0,
        gt=0,
        description=(
            "How long a caller waits for a free connection before failing. Kept "
            "deliberately short: blocking indefinitely on pool checkout converts "
            "database pressure into unbounded request latency, which is the "
            "failure mode we most want to avoid."
        ),
    )
    pool_recycle_seconds: int = Field(
        default=1800,
        description="Recycle connections before infra idle-timeouts kill them mid-query.",
    )
    pool_pre_ping: bool = Field(
        default=True,
        description="Cheap liveness check on checkout; converts stale-connection errors into retries.",
    )

    # --- Statement-level limits ---
    statement_timeout_seconds: float = Field(
        default=10.0,
        gt=0,
        description="Server-side statement_timeout. A query that outlives this is killed by Postgres.",
    )
    connect_timeout_seconds: float = Field(default=5.0, gt=0)

    echo_sql: bool = False

    @computed_field  # type: ignore[prop-decorator]
    @property
    def async_dsn(self) -> str:
        """DSN for the asyncpg driver used by the application at runtime."""
        return str(self.dsn).replace("postgresql://", "postgresql+asyncpg://", 1)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sync_dsn(self) -> str:
        """DSN for the psycopg driver used by Alembic.

        Migrations run synchronously on purpose — an async engine inside
        Alembic's ``run_migrations_online`` buys nothing and complicates
        failure handling during deploys.
        """
        return str(self.dsn).replace("postgresql://", "postgresql+psycopg://", 1)


class RedisSettings(BaseSettings):
    """Redis configuration.

    Redis holds *ephemeral* state only — rate-limit counters, circuit-breaker
    state, idempotency response cache, worker wakeup notifications. It is
    explicitly **not** the system of record for jobs or workflow state, so
    losing Redis degrades the system without losing work. See ADR-003.
    """

    model_config = SettingsConfigDict(env_prefix="REDIS__", extra="ignore")

    dsn: RedisDsn = Field(default=RedisDsn("redis://localhost:6379/0"))

    max_connections: int = Field(default=50, ge=1)
    socket_timeout_seconds: float = Field(default=2.0, gt=0)
    socket_connect_timeout_seconds: float = Field(default=2.0, gt=0)
    health_check_interval_seconds: int = Field(default=30, ge=0)


class ResilienceSettings(BaseSettings):
    """Timeouts, retries, circuit breakers, and concurrency ceilings.

    Defaults here encode the timeout budget documented in ARCHITECTURE.md. The
    invariant: an inner timeout must always be shorter than the outer deadline
    that contains it, or the outer deadline fires first and the inner retry
    never happens.
    """

    model_config = SettingsConfigDict(env_prefix="RESILIENCE__", extra="ignore")

    # --- Timeouts (seconds) ---
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    embedding_timeout_seconds: float = Field(default=10.0, gt=0)
    tool_timeout_seconds: float = Field(default=10.0, gt=0)
    vector_search_timeout_seconds: float = Field(default=3.0, gt=0)
    http_timeout_seconds: float = Field(default=15.0, gt=0)

    # --- Retry policy ---
    max_retry_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Total attempts, not retries-after-first. 3 == initial try + 2 retries.",
    )
    retry_initial_backoff_seconds: float = Field(default=0.25, gt=0)
    retry_max_backoff_seconds: float = Field(default=8.0, gt=0)
    retry_backoff_multiplier: float = Field(default=2.0, gt=1)
    retry_jitter: bool = Field(
        default=True,
        description="Full jitter. Without it, synchronised clients retry in lockstep and self-DDoS.",
    )
    retry_budget_ratio: float = Field(
        default=0.2,
        gt=0,
        le=1.0,
        description=(
            "Ceiling on retries as a fraction of total calls. Once exceeded, "
            "retries are shed even if individually eligible — this is what stops "
            "a dependency brownout from becoming a retry storm."
        ),
    )

    # --- Circuit breaker ---
    breaker_failure_threshold: int = Field(default=5, ge=1)
    breaker_success_threshold: int = Field(
        default=2, ge=1, description="Consecutive successes in HALF_OPEN before closing."
    )
    breaker_reset_timeout_seconds: float = Field(
        default=30.0, gt=0, description="How long OPEN waits before probing with HALF_OPEN."
    )
    breaker_half_open_max_calls: int = Field(
        default=1, ge=1, description="Concurrent probes allowed while HALF_OPEN."
    )

    # --- Concurrency ceilings (per process) ---
    llm_max_concurrency: int = Field(default=50, ge=1)
    vector_max_concurrency: int = Field(default=30, ge=1)
    tool_max_concurrency: int = Field(default=20, ge=1)


class QueueSettings(BaseSettings):
    """Durable job queue and backpressure configuration.

    The queue is Postgres-backed (``SELECT ... FOR UPDATE SKIP LOCKED``); see
    ADR-004 for why, and for the throughput ceiling that choice implies.
    """

    model_config = SettingsConfigDict(env_prefix="QUEUE__", extra="ignore")

    # --- Lease / visibility ---
    lease_duration_seconds: int = Field(
        default=120,
        ge=5,
        description=(
            "How long a worker owns a claimed job. If the worker dies, the lease "
            "expires and the job becomes claimable again — this is the crash "
            "recovery mechanism, so it must exceed the longest expected step."
        ),
    )
    lease_heartbeat_seconds: int = Field(
        default=30, ge=1, description="Worker extends its lease at this interval while working."
    )

    # --- Retry / dead-letter ---
    max_attempts: int = Field(default=3, ge=1, description="Attempts before a job is dead-lettered.")
    retry_backoff_seconds: float = Field(default=5.0, gt=0)
    retry_backoff_multiplier: float = Field(default=3.0, gt=1)

    # --- Worker ---
    worker_concurrency: int = Field(
        default=8, ge=1, description="Concurrent workflow executions per worker process."
    )
    poll_interval_seconds: float = Field(default=1.0, gt=0)
    batch_size: int = Field(default=10, ge=1, description="Jobs claimed per poll.")
    shutdown_grace_seconds: float = Field(
        default=30.0, gt=0, description="SIGTERM → drain window before forced exit."
    )

    # --- Backpressure ---
    max_queue_depth: int = Field(
        default=1000,
        ge=1,
        description=(
            "Pending-job ceiling. Beyond this the API sheds load with 503 rather "
            "than accepting work it cannot deliver on. Admission control is what "
            "keeps latency bounded under overload."
        ),
    )
    queue_depth_warn_ratio: float = Field(default=0.7, gt=0, le=1.0)


class APISettings(BaseSettings):
    """HTTP surface configuration: limits, rate limiting, versioning."""

    model_config = SettingsConfigDict(env_prefix="API__", extra="ignore")

    host: str = "0.0.0.0"  # noqa: S104 — binding all interfaces is correct inside a container
    port: int = 8000
    root_path: str = ""

    max_request_bytes: int = Field(
        default=1_048_576, ge=1024, description="1 MiB. Rejects oversized bodies before parsing."
    )
    request_timeout_seconds: float = Field(default=30.0, gt=0)

    # --- Rate limiting (incoming). Distinct from provider-side limits; see ADR-005. ---
    rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = Field(default=600, ge=1)
    rate_limit_burst: int = Field(default=100, ge=1)

    # --- Idempotency ---
    idempotency_ttl_seconds: int = Field(
        default=86_400, ge=60, description="How long an Idempotency-Key result is replayable."
    )

    cors_origins: list[str] = Field(default_factory=list)


class ObservabilitySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OBSERVABILITY__", extra="ignore")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "console"] = "json"

    metrics_enabled: bool = True
    metrics_path: str = "/metrics"

    tracing_enabled: bool = False
    otlp_endpoint: str = Field(default="", description="e.g. http://otel-collector:4317")
    trace_sample_ratio: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="1.0 locally. Lower under load — tracing every request at 1000 RPM is its own load problem.",
    )


class Settings(BaseSettings):
    """Root settings object. Access via :func:`get_settings`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    service_name: str = "prf-agentic-lab"
    service_version: str = "0.1.0"
    environment: Environment = Environment.LOCAL
    debug: bool = False

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    resilience: ResilienceSettings = Field(default_factory=ResilienceSettings)
    queue: QueueSettings = Field(default_factory=QueueSettings)
    api: APISettings = Field(default_factory=APISettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    @field_validator("environment", mode="before")
    @classmethod
    def _normalise_environment(cls, v: object) -> object:
        return v.lower() if isinstance(v, str) else v

    @property
    def is_production(self) -> bool:
        return self.environment is Environment.PRODUCTION

    def validate_for_environment(self) -> list[str]:
        """Return configuration problems that should block startup.

        Returns a list rather than raising so the caller can log *all* problems
        at once. Booting with a bad config and discovering it one error per
        restart is a miserable way to run a deploy.
        """
        problems: list[str] = []

        if self.is_production:
            if self.debug:
                problems.append("debug=True is not permitted in production")
            if self.observability.log_format != "json":
                problems.append("production logs must be json for ingestion")
            if "localhost" in str(self.database.dsn):
                problems.append("database DSN still points at localhost in production")
            if self.observability.trace_sample_ratio == 1.0:
                problems.append(
                    "trace_sample_ratio=1.0 in production will emit a span per request; "
                    "set a sampled ratio"
                )

        # An inner timeout longer than the outer request deadline means the outer
        # deadline always wins and the inner retry policy is dead code.
        if self.resilience.llm_timeout_seconds >= self.api.request_timeout_seconds:
            problems.append(
                f"llm_timeout ({self.resilience.llm_timeout_seconds}s) >= api request_timeout "
                f"({self.api.request_timeout_seconds}s): inner timeout can never fire"
            )

        if self.queue.lease_heartbeat_seconds >= self.queue.lease_duration_seconds:
            problems.append(
                f"lease_heartbeat ({self.queue.lease_heartbeat_seconds}s) >= lease_duration "
                f"({self.queue.lease_duration_seconds}s): leases will expire under a live worker"
            )

        total_pool = self.database.pool_size + self.database.max_overflow
        if self.queue.worker_concurrency > total_pool:
            problems.append(
                f"worker_concurrency ({self.queue.worker_concurrency}) exceeds max db "
                f"connections ({total_pool}): workers will block on pool checkout"
            )

        return problems


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton.

    Cached because constructing this re-reads the environment and .env file.
    Tests that need different config should call ``get_settings.cache_clear()``.
    """
    return Settings()


def reset_settings_cache() -> None:
    """Drop the cached settings. Used by tests that mutate the environment."""
    get_settings.cache_clear()


def is_running_under_pytest() -> bool:
    return "PYTEST_CURRENT_TEST" in os.environ
