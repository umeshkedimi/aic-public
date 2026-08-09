"""Structured JSON logging with correlation enrichment and secret redaction.

Two non-negotiables from the spec drive this module:

1. Every log line carries the correlation fields (§26), so a single request can
   be reconstructed across API, queue, and worker.
2. Secrets and donor PII never reach the log store (§26, §34).

Redaction is enforced by a processor in the pipeline rather than by asking
developers to remember. A convention that depends on discipline fails on the
day someone logs an exception payload containing an API key — so the pipeline
scrubs by key name and by value shape regardless of the call site.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import MutableMapping
from typing import Any, Final

import structlog
from structlog.types import EventDict, Processor

from app.config.settings import Settings
from app.observability.context import current_context

REDACTED: Final = "[REDACTED]"

# Key names whose values are never safe to emit. Matched case-insensitively as
# substrings, so `openai_api_key`, `X-API-Key`, and `db_password` all hit.
_SENSITIVE_KEY_PATTERNS: Final[tuple[str, ...]] = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "auth_header",
    "credential",
    "private_key",
    "session_id",
    "cookie",
)

# Donor PII. The determinism/data boundary says donor records stay in Postgres
# and are referenced by ID; a donor's name or address appearing in a log line
# means something bypassed that boundary, so we scrub and keep the shape.
_PII_KEY_PATTERNS: Final[tuple[str, ...]] = (
    "email",
    "phone",
    "ssn",
    "tax_id",
    "address",
    "street",
    "donor_name",
    "full_name",
    "first_name",
    "last_name",
    "date_of_birth",
    "dob",
)

# Value-shaped detection, for secrets that arrive under an innocuous key —
# typically inside a stringified exception or an echoed request body.
_VALUE_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),  # OpenAI-style
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{16,}"),  # Anthropic-style
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{16,}", re.IGNORECASE),
    re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),  # JWT
    re.compile(r"postgres(?:ql)?://[^:]+:[^@]+@"),  # DSN with inline password
    re.compile(r"redis://[^:]*:[^@]+@"),
)

# Depth cap for recursive scrubbing. A cyclic or pathologically nested payload
# must not turn a log call into a stack overflow.
_MAX_SCRUB_DEPTH: Final = 6


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(p in lowered for p in _SENSITIVE_KEY_PATTERNS) or any(
        p in lowered for p in _PII_KEY_PATTERNS
    )


def _scrub_value(value: Any, depth: int = 0) -> Any:
    """Recursively redact secret-shaped content inside a value."""
    if depth >= _MAX_SCRUB_DEPTH:
        return "[TRUNCATED:depth]"

    if isinstance(value, str):
        scrubbed = value
        for pattern in _VALUE_PATTERNS:
            scrubbed = pattern.sub(REDACTED, scrubbed)
        return scrubbed

    if isinstance(value, MutableMapping):
        return {
            k: REDACTED if _is_sensitive_key(str(k)) else _scrub_value(v, depth + 1)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        scrubbed_items = [_scrub_value(v, depth + 1) for v in value]
        return type(value)(scrubbed_items) if isinstance(value, (list, tuple)) else scrubbed_items

    return value


def redact_processor(_logger: Any, _method: str, event_dict: EventDict) -> EventDict:
    """Strip secrets and PII from every event before it is rendered."""
    for key in list(event_dict.keys()):
        if _is_sensitive_key(str(key)):
            event_dict[key] = REDACTED
        else:
            event_dict[key] = _scrub_value(event_dict[key])
    return event_dict


def correlation_processor(_logger: Any, _method: str, event_dict: EventDict) -> EventDict:
    """Attach request/job/workflow IDs from the ambient context.

    Explicit values on the call win over ambient context — a caller naming a
    different job_id is describing that job, not the one it happens to run under.
    """
    for key, value in current_context().items():
        event_dict.setdefault(key, value)
    return event_dict


def _service_processor(settings: Settings) -> Processor:
    """Stamp every line with service identity, so a shared log store can filter."""

    def processor(_logger: Any, _method: str, event_dict: EventDict) -> EventDict:
        event_dict.setdefault("service", settings.service_name)
        event_dict.setdefault("version", settings.service_version)
        event_dict.setdefault("environment", settings.environment.value)
        return event_dict

    return processor


def _trace_processor(_logger: Any, _method: str, event_dict: EventDict) -> EventDict:
    """Attach the active OpenTelemetry trace/span IDs when tracing is live.

    This is the join key between logs and traces: it lets an engineer pivot
    from a slow span straight to the log lines emitted inside it.
    """
    try:
        from opentelemetry import trace
    except ImportError:  # pragma: no cover — otel is a hard dependency
        return event_dict

    span = trace.get_current_span()
    ctx = span.get_span_context()
    if ctx.is_valid:
        event_dict.setdefault("trace_id", format(ctx.trace_id, "032x"))
        event_dict.setdefault("span_id", format(ctx.span_id, "016x"))
    return event_dict


def configure_logging(settings: Settings) -> None:
    """Install the logging pipeline. Idempotent — safe to call per-process.

    Stdlib ``logging`` is routed through the same processors so that third-party
    libraries (uvicorn, sqlalchemy) emit the same JSON shape. Mixed formats in
    one stream break log ingestion.
    """
    level = getattr(logging, settings.observability.log_level)

    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        _service_processor(settings),
        correlation_processor,
        _trace_processor,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        # Redaction runs last among the enrichers so it also sees fields added
        # by everything above it.
        redact_processor,
    ]

    renderer: Processor = (
        structlog.processors.JSONRenderer()
        if settings.observability.log_format == "json"
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[
            *shared,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Route stdlib logging through structlog's formatter for a uniform stream.
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared,
            processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
        )
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    # uvicorn installs its own handlers; clear them so lines aren't emitted twice.
    for noisy in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(noisy)
        lg.handlers = []
        lg.propagate = True

    # SQLAlchemy's INFO tier is a firehose of SQL. Raise its floor unless echo
    # was explicitly requested.
    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.INFO if settings.database.echo_sql else logging.WARNING
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound logger. Module-level ``__name__`` is the usual argument."""
    return structlog.stdlib.get_logger(name)  # type: ignore[no-any-return]
