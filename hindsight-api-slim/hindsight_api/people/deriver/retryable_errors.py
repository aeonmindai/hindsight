# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/utils/retryable_errors.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""Classify exceptions as transient (safe to retry) or terminal.

Imports only exception taxonomies, so it is importable from anywhere and
unit-testable without a DB.
"""

import asyncio
from collections.abc import Iterator

import aiohttp
import asyncpg

__all__ = ["is_retryable_db_error", "is_retryable_error"]

_RETRYABLE_SQLSTATES = frozenset(
    {
        "40001",  # serialization_failure
        "40P01",  # deadlock_detected
        "55P03",  # lock_not_available (lock_timeout / NOWAIT)
        "57014",  # query_canceled (statement_timeout)
        "57P03",  # cannot_connect_now
        "08000",  # connection_exception family
        "08001",
        "08003",
        "08004",
        "08006",
    }
)

# Provider/network transport failures. SDK wrappers (anthropic APIConnectionError etc.) chain to
# these via __cause__.
_TRANSPORT_ERRORS = (
    aiohttp.ClientConnectionError,
    aiohttp.ServerTimeoutError,
    ConnectionError,
    asyncio.TimeoutError,
    TimeoutError,
)

# A pooled connection that went away mid-statement.
_LOST_CONNECTION_ERRORS = (
    asyncpg.exceptions.ConnectionDoesNotExistError,
    asyncpg.exceptions.ConnectionFailureError,
    asyncpg.exceptions.InterfaceError,
)


def _iter_cause_chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__


def is_retryable_db_error(exc: BaseException) -> bool:
    """True for transient DB failures: deadlock, serialization failure,
    lock/statement timeout, or a lost connection.

    Integrity (23xxx), data (22xxx), and programming (42xxx) errors are
    deliberately terminal.
    """
    for current in _iter_cause_chain(exc):
        if isinstance(current, _LOST_CONNECTION_ERRORS):
            return True
        if isinstance(current, asyncpg.PostgresError) and getattr(current, "sqlstate", None) in _RETRYABLE_SQLSTATES:
            return True
    return False


def is_retryable_error(exc: BaseException) -> bool:
    """Superset of ``is_retryable_db_error``: also transient network/provider
    transport failures (timeouts, connection refused/reset).

    Auth failures (401 from a rotated key) are deliberately terminal: they
    never self-heal, so retrying only delays the burn.
    """
    if is_retryable_db_error(exc):
        return True
    return any(isinstance(current, _TRANSPORT_ERRORS) for current in _iter_cause_chain(exc))
