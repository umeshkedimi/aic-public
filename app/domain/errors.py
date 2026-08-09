"""Error taxonomy and classification.

This module is the foundation of the retry system. The spec's rule "do not
blindly retry every exception" (§10, §11) is only enforceable if every error
carries a machine-readable classification, so retry decisions are a property of
the *error*, not a guess at the call site.

Why classification rather than a retry-on-exception-type list: the same Python
exception type means different things in different contexts. ``httpx.HTTPStatusError``
is retryable at 503 and emphatically not at 400 — retrying a malformed request
just burns budget and produces the same rejection. Classification happens at the
adapter boundary, where the context to make that call actually exists.

The retryability rule:

===================  =========  ===================================================
Class                Retryable  Rationale
===================  =========  ===================================================
TRANSIENT            yes        Condition is expected to clear on its own.
RATE_LIMITED         yes        Retry, but honour Retry-After — this is the one
                                class where backoff must be provider-directed.
TIMEOUT              qualified  Only when the operation is idempotent. A timeout
                                means *unknown outcome*, not failure: the write
                                may have landed.
DEPENDENCY_DOWN      yes        Circuit breaker governs whether we even try.
AUTHENTICATION       no         Credentials will not fix themselves. Retrying
                                risks triggering lockout.
INVALID_REQUEST      no         Deterministic rejection; identical retry, identical
                                failure.
MODEL_FAILURE        qualified  Malformed structured output is worth one reprompt;
                                a refusal is not.
RESOURCE_EXHAUSTED   no         Budget/quota exceeded. Retrying is precisely the
                                behaviour the budget exists to prevent.
APPLICATION_FAILURE  no         Our bug. Retrying hides it and doubles the damage.
===================  =========  ===================================================
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self


class ErrorClass(StrEnum):
    """Coarse classification that drives retry, alerting, and metric labels.

    Deliberately small and closed. It is used as a Prometheus label value, so
    an open-ended set would be a cardinality hazard.
    """

    TRANSIENT = "transient"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    DEPENDENCY_DOWN = "dependency_down"
    AUTHENTICATION = "authentication"
    INVALID_REQUEST = "invalid_request"
    MODEL_FAILURE = "model_failure"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    APPLICATION_FAILURE = "application_failure"

    @property
    def is_retryable(self) -> bool:
        """Whether an error of this class may be retried at all.

        TIMEOUT and MODEL_FAILURE report ``True`` here but carry an additional
        gate — see :attr:`AppError.is_retryable`, which also requires the
        operation to be idempotent. This property answers "could this ever be
        retried", not "retry it now".
        """
        return self in _RETRYABLE_CLASSES

    @property
    def requires_idempotency(self) -> bool:
        """Whether retrying is only safe when the operation is idempotent.

        A timeout leaves the outcome genuinely unknown. Retrying a non-idempotent
        operation after one risks a duplicate side effect — a second letter
        mailed, a second charge. That is worse than surfacing the failure.
        """
        return self in {ErrorClass.TIMEOUT, ErrorClass.MODEL_FAILURE}

    @property
    def http_status(self) -> int:
        """Status code to surface for this class at the API edge."""
        return _HTTP_STATUS_BY_CLASS[self]


_RETRYABLE_CLASSES = frozenset(
    {
        ErrorClass.TRANSIENT,
        ErrorClass.RATE_LIMITED,
        ErrorClass.TIMEOUT,
        ErrorClass.DEPENDENCY_DOWN,
        ErrorClass.MODEL_FAILURE,
    }
)

_HTTP_STATUS_BY_CLASS: dict[ErrorClass, int] = {
    ErrorClass.TRANSIENT: 503,
    ErrorClass.RATE_LIMITED: 429,
    ErrorClass.TIMEOUT: 504,
    ErrorClass.DEPENDENCY_DOWN: 503,
    ErrorClass.AUTHENTICATION: 401,
    ErrorClass.INVALID_REQUEST: 400,
    ErrorClass.MODEL_FAILURE: 502,
    ErrorClass.RESOURCE_EXHAUSTED: 429,
    ErrorClass.APPLICATION_FAILURE: 500,
}


class AppError(Exception):
    """Base for every error this system raises deliberately.

    Carries the classification, the component that produced it, and structured
    detail suitable for logs. ``message`` is safe to return to a client;
    ``details`` is not necessarily, so the API layer filters it.
    """

    error_class: ErrorClass = ErrorClass.APPLICATION_FAILURE
    #: Stable machine-readable code for clients to branch on. Set per subclass.
    code: str = "internal_error"

    def __init__(
        self,
        message: str,
        *,
        component: str = "unknown",
        details: dict[str, Any] | None = None,
        retry_after_seconds: float | None = None,
        idempotent_operation: bool = False,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.component = component
        self.details = details or {}
        self.retry_after_seconds = retry_after_seconds
        self.idempotent_operation = idempotent_operation
        self.__cause__ = cause

    @property
    def is_retryable(self) -> bool:
        """Whether *this specific error instance* should be retried.

        Combines the class-level rule with the idempotency of the operation
        that failed. This is the predicate the retry executor calls.
        """
        if not self.error_class.is_retryable:
            return False
        if self.error_class.requires_idempotency:
            return self.idempotent_operation
        return True

    def to_log_fields(self) -> dict[str, Any]:
        """Structured representation for logging."""
        return {
            "error_type": self.__class__.__name__,
            "error_class": self.error_class.value,
            "error_code": self.code,
            "component": self.component,
            "retryable": self.is_retryable,
            **({"retry_after": self.retry_after_seconds} if self.retry_after_seconds else {}),
            **self.details,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(message={self.message!r}, "
            f"class={self.error_class.value}, component={self.component!r})"
        )


# ---------------------------------------------------------------------------
# Retryable
# ---------------------------------------------------------------------------


class TransientError(AppError):
    """A self-clearing failure — connection reset, brief unavailability."""

    error_class = ErrorClass.TRANSIENT
    code = "transient_failure"


class RateLimitedError(AppError):
    """Upstream applied a rate limit.

    ``retry_after_seconds`` should be populated from the provider's
    ``Retry-After`` header when present. Ignoring a server-supplied backoff and
    substituting our own is how a rate limit becomes an outage.
    """

    error_class = ErrorClass.RATE_LIMITED
    code = "rate_limited"


class TimeoutError_(AppError):
    """Operation exceeded its deadline. Outcome is UNKNOWN, not failed.

    Named with a trailing underscore to avoid shadowing the builtin.
    """

    error_class = ErrorClass.TIMEOUT
    code = "timeout"


class DependencyDownError(AppError):
    """A dependency is unreachable, or its circuit breaker is open."""

    error_class = ErrorClass.DEPENDENCY_DOWN
    code = "dependency_unavailable"


class ModelOutputError(AppError):
    """The model returned something unusable — unparseable or schema-invalid.

    Retryable exactly once or twice with a repair prompt. Distinguished from
    :class:`ModelRefusalError` because a refusal will be repeated on retry.
    """

    error_class = ErrorClass.MODEL_FAILURE
    code = "model_output_invalid"


# ---------------------------------------------------------------------------
# Non-retryable
# ---------------------------------------------------------------------------


class AuthenticationError(AppError):
    """Credentials missing, invalid, or expired."""

    error_class = ErrorClass.AUTHENTICATION
    code = "authentication_failed"


class AuthorizationError(AppError):
    """Authenticated, but not permitted."""

    error_class = ErrorClass.AUTHENTICATION
    code = "not_authorized"

    @property
    def http_status_override(self) -> int:
        return 403


class ValidationError_(AppError):
    """Input failed validation. Deterministic — retrying changes nothing."""

    error_class = ErrorClass.INVALID_REQUEST
    code = "validation_failed"


class NotFoundError(AppError):
    """Requested resource does not exist."""

    error_class = ErrorClass.INVALID_REQUEST
    code = "not_found"

    @property
    def http_status_override(self) -> int:
        return 404


class ConflictError(AppError):
    """Request conflicts with current state — e.g. approving a cancelled job."""

    error_class = ErrorClass.INVALID_REQUEST
    code = "conflict"

    @property
    def http_status_override(self) -> int:
        return 409


class ModelRefusalError(AppError):
    """The model declined to answer. Deterministic for the same prompt."""

    error_class = ErrorClass.INVALID_REQUEST
    code = "model_refused"


class BudgetExceededError(AppError):
    """A cost, token, step, or time budget was exhausted (§37, §38).

    Non-retryable by construction: the budget exists to stop exactly the
    runaway that a retry would continue.
    """

    error_class = ErrorClass.RESOURCE_EXHAUSTED
    code = "budget_exceeded"


class BackpressureError(AppError):
    """Admission control rejected the request — the system is saturated (§15).

    Surfaced as 503 with Retry-After. Shedding load early is a deliberate
    availability trade: accepting work we cannot deliver on degrades everyone.
    """

    error_class = ErrorClass.RESOURCE_EXHAUSTED
    code = "system_overloaded"

    @property
    def http_status_override(self) -> int:
        return 503


class PayloadTooLargeError(AppError):
    """Request body exceeded the configured size cap (§34).

    Raised from the ASGI receive wrapper when a chunked request streams past
    the limit, since a rejection cannot be sent from inside ``receive`` itself.
    """

    error_class = ErrorClass.INVALID_REQUEST
    code = "request_too_large"

    @property
    def http_status_override(self) -> int:
        return 413


class ConfigurationError(AppError):
    """Invalid configuration detected at startup. Fail fast, do not serve."""

    error_class = ErrorClass.APPLICATION_FAILURE
    code = "misconfigured"


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------


def http_status_for(error: AppError) -> int:
    """Resolve the HTTP status for an error, honouring per-class overrides."""
    override = getattr(error, "http_status_override", None)
    if isinstance(override, int):
        return override
    return error.error_class.http_status


def classify_exception(
    exc: BaseException,
    *,
    component: str = "unknown",
    idempotent_operation: bool = False,
) -> AppError:
    """Convert an arbitrary exception into a classified :class:`AppError`.

    The catch-all at adapter boundaries. Anything unrecognised becomes
    APPLICATION_FAILURE — non-retryable — because an error we cannot classify is
    one we do not understand, and retrying something we do not understand is how
    a small bug becomes an incident.
    """
    if isinstance(exc, AppError):
        return exc

    # Stdlib/asyncio timeouts. TimeoutError is an alias of asyncio.TimeoutError
    # on 3.11+, so this single check covers both.
    if isinstance(exc, TimeoutError):
        return TimeoutError_(
            "operation timed out",
            component=component,
            idempotent_operation=idempotent_operation,
            cause=exc,
        )

    if isinstance(exc, (ConnectionError, OSError)):
        return TransientError(
            f"connection failure: {type(exc).__name__}",
            component=component,
            idempotent_operation=idempotent_operation,
            cause=exc,
        )

    return UnexpectedError(
        f"unhandled {type(exc).__name__}: {exc}",
        component=component,
        cause=exc,
    )


class UnexpectedError(AppError):
    """An exception we did not anticipate. Never retried; always alert-worthy."""

    error_class = ErrorClass.APPLICATION_FAILURE
    code = "unexpected_error"

    @classmethod
    def wrap(cls, exc: BaseException, *, component: str) -> Self:
        return cls(f"unhandled {type(exc).__name__}: {exc}", component=component, cause=exc)
