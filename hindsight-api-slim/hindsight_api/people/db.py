"""Small database helpers shared by the people modules."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from .context import PeopleContext


@asynccontextmanager
async def connection(ctx: PeopleContext, conn: Any | None = None) -> AsyncIterator[Any]:
    """The caller's open connection when given, else a fresh one from the engine's pool.

    Lets a function join a transaction the caller already opened (the dreamer records a dream-history
    row and a conclusion change in one transaction) while staying callable on its own.
    """
    if conn is not None:
        yield conn
        return
    async with ctx.conn() as acquired:
        yield acquired
