"""Structured API error envelope and exception handlers.

Every error leaving this service has the same shape, so clients can branch on
``error.code`` instead of parsing prose. The envelope carries ``request_id`` —
that single field turns a user's bug report into a log query.

What is deliberately *not* returned: stack traces, driver messages, internal
hostnames, SQL. Those go to logs. An error response that leaks internals is a
reconnaissance gift, and the client can do nothing with them anyway.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.domain.errors import AppError, ErrorClass, UnexpectedError, http_status_for
from app.observability.context import get_request_id
from app.observability.logging import get_logger
from app.observability.metrics import get_metrics

logger = get_logger(__name__)


class ErrorDetail(BaseModel):
    """A single field-level problem, used for validation failures."""

    field: str = Field(description="Dotted path to the offending field.")
    message: str
    type: str | None = None


class ErrorBody(BaseModel):
    code: str = Field(description="Stable machine-readable code. Branch on this, not the message.")
    message: str = Field(description="Human-readable summary. Safe to display.")
    error_class: ErrorClass = Field(description="Coarse classification driving client retry policy.")
    retryable: bool = Field(description="Whether retrying this exact request could succeed.")
    retry_after_seconds: float | None = Field(
        default=None, description="Server-directed backoff. Honour it."
    )
    details: list[ErrorDetail] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Top-level error envelope."""

    error: ErrorBody
    request_id: str | None = Field(
        default=None, description="Correlates this response with server logs and traces."
    )


def _render(
    *,
    status_code: int,
    code: str,
    message: str,
    error_class: ErrorClass,
    retryable: bool,
    retry_after_seconds: float | None = None,
    details: list[ErrorDetail] | None = None,
) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorBody(
            code=code,
            message=message,
            error_class=error_class,
            retryable=retryable,
            retry_after_seconds=retry_after_seconds,
            details=details or [],
        ),
        request_id=get_request_id(),
    )

    headers: dict[str, str] = {}
    if retry_after_seconds is not None:
        # Retry-After is integer seconds. Round up — rounding 0.4s down to 0
        # tells the client to retry immediately, which is the opposite of intent.
        headers["Retry-After"] = str(max(1, int(retry_after_seconds + 0.999)))

    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(mode="json"),
        headers=headers,
    )


async def app_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle deliberately-raised :class:`AppError`s."""
    assert isinstance(exc, AppError)
    status_code = http_status_for(exc)
    metrics = get_metrics()
    metrics.errors_total.labels(
        error_type=exc.error_class.value, component=exc.component
    ).inc()

    # 5xx is our fault and warrants a stack trace; 4xx is the client's and does
    # not. Logging every 400 at ERROR with a traceback is how alert fatigue starts.
    log = logger.bind(path=request.url.path, method=request.method, **exc.to_log_fields())
    if status_code >= 500:
        log.error("request_failed", exc_info=exc)
    else:
        log.info("request_rejected")

    return _render(
        status_code=status_code,
        code=exc.code,
        message=exc.message,
        error_class=exc.error_class,
        retryable=exc.is_retryable,
        retry_after_seconds=exc.retry_after_seconds,
    )


async def validation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Flatten Pydantic validation errors into the standard envelope."""
    assert isinstance(exc, RequestValidationError)
    details = [
        ErrorDetail(
            # loc[0] is the source ("body", "query"); the rest is the field path.
            field=".".join(str(p) for p in err["loc"][1:]) or str(err["loc"][0]),
            message=err["msg"],
            type=err.get("type"),
        )
        for err in exc.errors()
    ]

    get_metrics().errors_total.labels(
        error_type=ErrorClass.INVALID_REQUEST.value, component="api"
    ).inc()
    logger.info(
        "request_validation_failed",
        path=request.url.path,
        method=request.method,
        field_count=len(details),
    )

    return _render(
        status_code=422,
        code="validation_failed",
        message="Request failed validation.",
        error_class=ErrorClass.INVALID_REQUEST,
        retryable=False,
        details=details,
    )


async def http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Normalise Starlette HTTPExceptions (404s, 405s) into the same envelope."""
    assert isinstance(exc, StarletteHTTPException)
    error_class = (
        ErrorClass.INVALID_REQUEST if exc.status_code < 500 else ErrorClass.APPLICATION_FAILURE
    )
    return _render(
        status_code=exc.status_code,
        code=f"http_{exc.status_code}",
        message=str(exc.detail),
        error_class=error_class,
        retryable=exc.status_code in {502, 503, 504},
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last resort. Log everything, return nothing revealing.

    Reaching this handler is a bug — an unclassified escape. It is logged at
    ERROR with the full traceback and counted so it can be alerted on.
    """
    wrapped = UnexpectedError.wrap(exc, component="api")
    get_metrics().errors_total.labels(
        error_type=wrapped.error_class.value, component="api"
    ).inc()
    logger.error(
        "unhandled_exception",
        path=request.url.path,
        method=request.method,
        **wrapped.to_log_fields(),
        exc_info=exc,
    )
    return _render(
        status_code=500,
        code="internal_error",
        message="An internal error occurred.",
        error_class=ErrorClass.APPLICATION_FAILURE,
        retryable=False,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Wire every handler onto the app.

    Order matters only in that the bare ``Exception`` handler is the fallback;
    Starlette dispatches most-specific-first.
    """
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)


def openapi_error_responses(*status_codes: int) -> dict[int | str, dict[str, Any]]:
    """Declare the error envelope on a route's OpenAPI schema."""
    return {
        code: {"model": ErrorResponse, "description": _STATUS_DESCRIPTIONS.get(code, "Error")}
        for code in status_codes
    }


_STATUS_DESCRIPTIONS: dict[int, str] = {
    400: "Invalid request",
    401: "Authentication failed",
    403: "Not authorized",
    404: "Not found",
    409: "Conflict with current resource state",
    422: "Request failed validation",
    429: "Rate limited or budget exceeded",
    500: "Internal error",
    502: "Upstream model or tool failure",
    503: "Dependency unavailable or system shedding load",
    504: "Upstream timeout",
}
