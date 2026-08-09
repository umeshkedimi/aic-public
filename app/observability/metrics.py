"""Prometheus metrics.

**Cardinality is the design constraint.** A Prometheus time series is created
for every unique label-value combination, and each one costs memory in the
scraper *and* the local registry — forever, until it ages out. The classic way
to take down a monitoring stack is a label like ``job_id`` or ``donor_id``:
unbounded distinct values, one series each.

Rules this module follows:

- Label values must come from a **bounded, enumerable set** — HTTP method, a
  route *template* (not the resolved path), an agent name, an error class.
- High-cardinality identifiers (``job_id``, ``request_id``, ``trace_id``) belong
  in **logs and traces**, never in labels. That's the division of labour: metrics
  answer "how often / how slow", traces answer "why was *this one* slow".
- Histogram buckets are chosen per-metric against the SLO being measured.
  Default buckets are tuned for fast RPCs and are useless for a 30s workflow.

Metrics are registered against an explicit registry rather than the global
default so tests can build a clean one and assert on it without cross-test
pollution.
"""

from __future__ import annotations

from typing import Final

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, Info

# ---------------------------------------------------------------------------
# Bucket definitions
# ---------------------------------------------------------------------------

# HTTP: the API is an enqueue-and-return surface, so anything past ~1s means
# something is wrong. Fine granularity below 250ms is where the SLO lives.
HTTP_BUCKETS: Final[tuple[float, ...]] = (
    0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0,
)

# Dependency calls (Postgres, Redis): sub-millisecond to a few seconds.
DEPENDENCY_BUCKETS: Final[tuple[float, ...]] = (
    0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
)


class Metrics:
    """Container for the process's metric instruments.

    Instantiated once and reached through :func:`get_metrics`. Grouping them on
    an object (rather than module-level globals) is what makes a fresh registry
    per test possible.
    """

    def __init__(self, registry: CollectorRegistry) -> None:
        self.registry = registry

        # --- Build / identity -------------------------------------------------
        self.build_info = Info(
            "prf_build",
            "Service build identity. Constant labels — one series, by design.",
            registry=registry,
        )

        # --- HTTP -------------------------------------------------------------
        # `endpoint` is the route TEMPLATE ("/api/v1/prf/jobs/{job_id}"), never
        # the resolved path. Using the raw path would mint a series per job id.
        self.http_requests_total = Counter(
            "http_requests_total",
            "HTTP requests by method, route template, and status code.",
            labelnames=("method", "endpoint", "status"),
            registry=registry,
        )
        self.http_request_duration_seconds = Histogram(
            "http_request_duration_seconds",
            "HTTP request latency.",
            labelnames=("method", "endpoint"),
            buckets=HTTP_BUCKETS,
            registry=registry,
        )
        self.http_requests_in_flight = Gauge(
            "http_requests_in_flight",
            "Requests currently being served. Rising with flat throughput means saturation.",
            registry=registry,
        )
        self.http_request_size_bytes = Histogram(
            "http_request_size_bytes",
            "Inbound request body size.",
            buckets=(256, 1024, 4096, 16_384, 65_536, 262_144, 1_048_576),
            registry=registry,
        )

        # --- Errors -----------------------------------------------------------
        # `error_type` is drawn from the ErrorCode enum — a closed set, so bounded.
        self.errors_total = Counter(
            "errors_total",
            "Errors by classification and originating component.",
            labelnames=("error_type", "component"),
            registry=registry,
        )

        # --- Dependencies -----------------------------------------------------
        self.dependency_up = Gauge(
            "dependency_up",
            "Last observed health of a dependency (1 healthy, 0 unhealthy).",
            labelnames=("dependency",),
            registry=registry,
        )
        self.dependency_check_duration_seconds = Histogram(
            "dependency_check_duration_seconds",
            "Health-check probe latency by dependency.",
            labelnames=("dependency",),
            buckets=DEPENDENCY_BUCKETS,
            registry=registry,
        )

        # --- Database pool ----------------------------------------------------
        # Pool exhaustion is the bottleneck the spec asks us to demonstrate
        # (§57), so the pool is instrumented before it ever hurts.
        self.db_pool_connections = Gauge(
            "db_pool_connections",
            "Connections by state: in_use, available, overflow.",
            labelnames=("state",),
            registry=registry,
        )
        self.db_pool_checkout_duration_seconds = Histogram(
            "db_pool_checkout_duration_seconds",
            "Time spent waiting for a pooled connection. Non-zero p95 means the pool is too small.",
            buckets=DEPENDENCY_BUCKETS,
            registry=registry,
        )
        self.db_pool_exhausted_total = Counter(
            "db_pool_exhausted_total",
            "Checkout attempts that timed out with no connection available.",
            registry=registry,
        )
        self.db_query_duration_seconds = Histogram(
            "db_query_duration_seconds",
            "Query latency by logical operation (not by SQL text — that would be unbounded).",
            labelnames=("operation",),
            buckets=DEPENDENCY_BUCKETS,
            registry=registry,
        )

        # --- Redis ------------------------------------------------------------
        self.redis_operation_duration_seconds = Histogram(
            "redis_operation_duration_seconds",
            "Redis command latency by command name.",
            labelnames=("operation",),
            buckets=DEPENDENCY_BUCKETS,
            registry=registry,
        )
        self.redis_operations_total = Counter(
            "redis_operations_total",
            "Redis commands by name and outcome.",
            labelnames=("operation", "outcome"),
            registry=registry,
        )

    def set_build_info(self, *, version: str, environment: str, service: str) -> None:
        self.build_info.info(
            {"version": version, "environment": environment, "service": service}
        )


# ---------------------------------------------------------------------------
# Process-wide accessor
# ---------------------------------------------------------------------------

_registry: CollectorRegistry | None = None
_metrics: Metrics | None = None


def get_registry() -> CollectorRegistry:
    """The process's metric registry, created on first use."""
    global _registry
    if _registry is None:
        _registry = CollectorRegistry()
    return _registry


def get_metrics() -> Metrics:
    """The process's metric instruments, created on first use."""
    global _metrics
    if _metrics is None:
        _metrics = Metrics(get_registry())
    return _metrics


def reset_metrics() -> None:
    """Drop the registry and instruments.

    Tests call this between cases: Prometheus counters are monotonic by
    contract, so assertions written against a dirty registry are order-dependent
    and flaky.
    """
    global _registry, _metrics
    _registry = None
    _metrics = None
