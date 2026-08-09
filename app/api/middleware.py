"""HTTP middleware: correlation, metrics, and request-size limiting.

Ordering is significant. Starlette runs middleware in reverse registration
order on the way in, so the *last* registered wraps outermost. Correlation is
registered last so it is outermost — every other layer, including the metrics
recorder and the error handlers, then runs with correlation IDs already bound.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from starlette.datastructures import Headers
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.domain.errors import PayloadTooLargeError
from app.observability.context import bind_context, get_request_id, new_id
from app.observability.logging import get_logger
from app.observability.metrics import get_metrics

logger = get_logger(__name__)

RequestHandler = Callable[[Request], Awaitable[Response]]

#: Inbound headers accepted as a caller-supplied request ID, in priority order.
_REQUEST_ID_HEADERS = ("x-request-id", "x-correlation-id", "request-id")


class CorrelationMiddleware(BaseHTTPMiddleware):
    """Establish request/correlation IDs and echo them on the response.

    A client-supplied ``X-Request-ID`` is honoured so a caller can trace a
    request across service boundaries — but it is length-capped and sanitised
    first. An unvalidated header value flows into logs, and a log store that
    accepts arbitrary caller-controlled content is a log-injection surface.

    ``correlation_id`` defaults to ``request_id`` when the caller supplies
    neither, so the causal chain always has a root.
    """

    #: Long enough for a UUID or a W3C traceparent, short enough to be harmless.
    MAX_ID_LENGTH = 128

    async def dispatch(self, request: Request, call_next: RequestHandler) -> Response:
        request_id = self._extract_id(request) or new_id()
        correlation_id = self._sanitise(request.headers.get("x-correlation-id")) or request_id

        tokens = bind_context(request_id=request_id, correlation_id=correlation_id)
        # Stash on state so route handlers and dependencies can read them
        # without re-parsing headers.
        request.state.request_id = request_id
        request.state.correlation_id = correlation_id
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Correlation-ID"] = correlation_id
            return response
        finally:
            # Restore in `finally` so a failed request cannot leak its IDs into
            # whatever the event loop runs next on this task.
            tokens.restore()

    def _extract_id(self, request: Request) -> str | None:
        for header in _REQUEST_ID_HEADERS:
            if value := self._sanitise(request.headers.get(header)):
                return value
        return None

    def _sanitise(self, raw: str | None) -> str | None:
        """Keep only characters safe to embed in a log line, bounded in length."""
        if not raw:
            return None
        cleaned = "".join(c for c in raw[: self.MAX_ID_LENGTH] if c.isalnum() or c in "-_.")
        return cleaned or None


class MetricsMiddleware(BaseHTTPMiddleware):
    """Record request count, latency, and in-flight gauge.

    Labels use the matched **route template**, not the raw path. ``/jobs/abc123``
    and ``/jobs/def456`` must collapse to ``/jobs/{job_id}`` or the series count
    grows without bound — one per job, forever. This is the single most
    important cardinality control in the service.
    """

    async def dispatch(self, request: Request, call_next: RequestHandler) -> Response:
        metrics = get_metrics()
        metrics.http_requests_in_flight.inc()
        started = time.perf_counter()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            elapsed = time.perf_counter() - started
            metrics.http_requests_in_flight.dec()
            endpoint = self._route_template(request)
            metrics.http_requests_total.labels(
                method=request.method, endpoint=endpoint, status=str(status_code)
            ).inc()
            metrics.http_request_duration_seconds.labels(
                method=request.method, endpoint=endpoint
            ).observe(elapsed)

    @staticmethod
    def _route_template(request: Request) -> str:
        """Resolve the route template, or a safe constant when unmatched.

        Unmatched requests (404s, scanners probing random paths) report as
        ``__unmatched__``. Using the raw path here would let an external scanner
        mint unbounded series — a genuine denial-of-service vector against the
        monitoring stack, not a theoretical one.
        """
        route = request.scope.get("route")
        path_format = getattr(route, "path_format", None) or getattr(route, "path", None)
        return str(path_format) if path_format else "__unmatched__"


def _error_payload(code: str, message: str) -> dict[str, object]:
    """Build an error body matching :class:`app.api.errors.ErrorResponse`.

    Constructed by hand rather than imported because this middleware runs
    *outside* the exception-handler stack — a rejection here never reaches
    those handlers, so the shape has to be produced locally. It must stay in
    sync with the Pydantic model; the contract test in
    ``tests/unit/test_middleware.py`` asserts exactly that.
    """
    return {
        "error": {
            "code": code,
            "message": message,
            "error_class": "invalid_request",
            "retryable": False,
            "retry_after_seconds": None,
            "details": [],
        },
        "request_id": get_request_id(),
    }


class RequestSizeLimitMiddleware:
    """Reject oversized bodies before they are buffered into memory (§34).

    Written as raw ASGI rather than ``BaseHTTPMiddleware`` because enforcement
    requires wrapping ``receive``, which the higher-level base class does not
    expose.

    Two checks, because ``Content-Length`` is caller-supplied and optional:

    1. Declared ``Content-Length`` over the cap → reject immediately, before a
       single body byte is read. Cheapest possible path.
    2. Absent header (HTTP/1.1 chunked transfer) → count bytes as they stream
       and abort once the cap is crossed.

    The second check is the one that matters. Trusting the header alone means a
    chunked request declaring no length bypasses the limit completely, which
    turns the body parser into an unbounded memory allocator — a trivial
    single-request DoS.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        declared = headers.get("content-length")

        if declared is not None:
            try:
                declared_bytes = int(declared)
            except ValueError:
                await self._reject(
                    send, 400, "invalid_content_length", "Content-Length is not an integer."
                )
                return
            if declared_bytes > self.max_bytes:
                await self._reject_too_large(send, declared_bytes)
                return
            get_metrics().http_request_size_bytes.observe(declared_bytes)
            await self.app(scope, receive, send)
            return

        # No declared length: meter the stream itself.
        await self.app(scope, self._metered_receive(receive), send)

    def _metered_receive(self, receive: Receive) -> Receive:
        """Wrap ``receive`` to count body bytes and cut off past the cap.

        On breach we raise, which the error handlers convert into a 413. We
        cannot send a response ourselves from inside ``receive`` — the
        application already owns the send channel at that point.
        """
        total = 0

        async def wrapped() -> Message:
            nonlocal total
            message = await receive()
            if message["type"] == "http.request":
                total += len(message.get("body", b""))
                if total > self.max_bytes:
                    logger.info(
                        "request_body_too_large_streamed",
                        streamed_bytes=total,
                        limit_bytes=self.max_bytes,
                    )
                    raise PayloadTooLargeError(
                        f"Request body exceeds {self.max_bytes} bytes.",
                        component="api",
                        details={"limit_bytes": self.max_bytes},
                    )
                if not message.get("more_body", False):
                    get_metrics().http_request_size_bytes.observe(total)
            return message

        return wrapped

    async def _reject_too_large(self, send: Send, size: int) -> None:
        logger.info("request_body_too_large", declared_bytes=size, limit_bytes=self.max_bytes)
        await self._reject(
            send, 413, "request_too_large", f"Request body exceeds {self.max_bytes} bytes."
        )

    async def _reject(self, send: Send, status: int, code: str, message: str) -> None:
        get_metrics().errors_total.labels(error_type="invalid_request", component="api").inc()
        response = JSONResponse(status_code=status, content=_error_payload(code, message))
        await response(  # type: ignore[call-arg]
            {"type": "http"}, _empty_receive, send
        )


async def _empty_receive() -> Message:
    """Receive channel for responses generated without reading a request body."""
    return {"type": "http.disconnect"}


class AccessLogMiddleware(BaseHTTPMiddleware):
    """Emit one structured access log per request.

    Replaces uvicorn's plaintext access log, which cannot carry correlation IDs
    and would otherwise emit a second, differently-shaped line per request.

    Health and metrics endpoints are excluded: Kubernetes probes them every few
    seconds and Prometheus scrapes on an interval, which at any real replica
    count drowns the log store in lines nobody reads.
    """

    _EXCLUDED_PREFIXES = ("/health", "/metrics")

    async def dispatch(self, request: Request, call_next: RequestHandler) -> Response:
        if request.url.path.startswith(self._EXCLUDED_PREFIXES):
            return await call_next(request)

        started = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - started) * 1000

        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round(duration_ms, 2),
            client=request.client.host if request.client else None,
        )
        return response
