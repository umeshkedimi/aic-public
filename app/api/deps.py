"""FastAPI dependency providers.

Dependencies resolve from ``request.app.state``, which the lifespan handler
populates. Reaching resources this way — rather than importing a module-level
singleton — is what lets a test build an app with a stubbed database and have
every route pick it up with no patching.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.config.settings import Settings
from app.state import AppState


def get_app_state(request: Request) -> AppState:
    """Return the process's :class:`AppState`."""
    state: AppState = request.app.state.app_state
    return state


def get_settings_dep(request: Request) -> Settings:
    return get_app_state(request).settings


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Yield a transactional database session scoped to one request.

    The session commits on clean exit and rolls back on exception — the unit of
    work is the request. Handlers must not commit themselves; doing so splits
    one logical operation into several transactions and breaks the atomicity
    that idempotency depends on.
    """
    state = get_app_state(request)
    async with state.database.session() as session:
        yield session


AppStateDep = Annotated[AppState, Depends(get_app_state)]
SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]
