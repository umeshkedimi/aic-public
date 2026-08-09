"""Correlation context propagated across async boundaries.

Every log line, metric exemplar, and span in this system can be tied back to a
single request. That is what makes an 8-second latency question answerable.

Implemented with :mod:`contextvars` because the values must survive ``await``
points and follow a task into ``asyncio.gather``. A thread-local would not:
async tasks share a thread, so thread-locals leak context between concurrent
requests — a real and hard-to-debug correctness bug at any real concurrency.

The IDs and their scope:

``request_id``
    One inbound HTTP request. Generated at the edge if the client didn't supply
    one, echoed back on the response.
``correlation_id``
    Spans a *causal chain* that may cross several requests — an API call, the
    queued job it created, and a later approval call all share it.
``job_id`` / ``workflow_id``
    Set once execution moves into the worker.
``agent``
    The workflow node currently executing.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Any

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)
_job_id: ContextVar[str | None] = ContextVar("job_id", default=None)
_workflow_id: ContextVar[str | None] = ContextVar("workflow_id", default=None)
_agent: ContextVar[str | None] = ContextVar("agent", default=None)

_ALL_VARS: dict[str, ContextVar[str | None]] = {
    "request_id": _request_id,
    "correlation_id": _correlation_id,
    "job_id": _job_id,
    "workflow_id": _workflow_id,
    "agent": _agent,
}


def new_id() -> str:
    """Generate a correlation identifier.

    uuid4 hex without dashes — compact in logs, and URL-safe without escaping.
    """
    return uuid.uuid4().hex


# --- Accessors -------------------------------------------------------------


def get_request_id() -> str | None:
    return _request_id.get()


def get_correlation_id() -> str | None:
    return _correlation_id.get()


def get_job_id() -> str | None:
    return _job_id.get()


def get_workflow_id() -> str | None:
    return _workflow_id.get()


def get_agent() -> str | None:
    return _agent.get()


def current_context() -> dict[str, str]:
    """Snapshot the set correlation fields, omitting unset ones.

    Unset fields are omitted rather than emitted as null so log lines stay
    readable and log-store indexes don't fill with nulls.
    """
    return {name: value for name, var in _ALL_VARS.items() if (value := var.get()) is not None}


# --- Scoped binding --------------------------------------------------------


@dataclass(slots=True)
class _ContextTokens:
    """Reset tokens captured when entering a scope, replayed on exit."""

    tokens: list[tuple[ContextVar[str | None], Token[str | None]]]

    def restore(self) -> None:
        # Reverse order so nested scopes unwind correctly.
        for var, token in reversed(self.tokens):
            var.reset(token)


def bind_context(**fields: str | None) -> _ContextTokens:
    """Bind correlation fields, returning tokens for restoration.

    Prefer :func:`context_scope` unless you need manual lifetime control (as the
    HTTP middleware does, where bind and restore straddle a yield).
    """
    tokens: list[tuple[ContextVar[str | None], Token[str | None]]] = []
    for name, value in fields.items():
        var = _ALL_VARS.get(name)
        if var is None:
            raise KeyError(f"unknown correlation field: {name!r}")
        if value is not None:
            tokens.append((var, var.set(value)))
    return _ContextTokens(tokens)


@contextmanager
def context_scope(**fields: str | None) -> Iterator[None]:
    """Bind correlation fields for the duration of a block.

    Always restores, including on exception — otherwise a failed agent would
    leak its name into every subsequent log line on that task.
    """
    tokens = bind_context(**fields)
    try:
        yield
    finally:
        tokens.restore()


def clear_context() -> None:
    """Reset every correlation field. Used between test cases."""
    for var in _ALL_VARS.values():
        var.set(None)


def enrich(payload: dict[str, Any]) -> dict[str, Any]:
    """Merge correlation fields into a dict without clobbering explicit values.

    Explicit values win: if a caller passes ``agent="compliance"`` it means the
    event is *about* that agent, which is more specific than ambient context.
    """
    merged = current_context()
    merged.update(payload)
    return merged
