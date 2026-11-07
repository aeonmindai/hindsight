# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/crud/collection.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""Per-pair dream state: the part of Honcho's ``Collection`` the dreamer uses.

Honcho keeps, for each (observer, observed) collection, an ``internal_metadata`` object with the
scheduler's ``dream`` key (last dream time, the explicit-conclusion count it saw, the work-list backlog)
and the working notes. The fork has no collections table, so that object lives in ``people_dream_state``,
one row per (bank, observer, observed), created on first write. The collection's ``explicit_count`` and
``newest_explicit_at`` are computed from the live explicit conclusions instead of kept as counters.
"""

from __future__ import annotations

import datetime
from typing import Any

from ..context import PeopleContext
from ..jsonb import from_jsonb, to_jsonb


async def get_collection_internal_metadata(
    ctx: PeopleContext, bank_id: str, observer: str, observed: str
) -> dict[str, Any]:
    """The pair's whole internal_metadata object ({} when the pair has no state yet)."""
    async with ctx.conn() as conn:
        raw = await conn.fetchval(
            f"SELECT internal_metadata FROM {ctx.table('dream_state')} "
            "WHERE bank_id = $1 AND observer = $2 AND observed = $3",
            bank_id,
            observer,
            observed,
        )
    value = from_jsonb(raw, {})
    return value if isinstance(value, dict) else {}


async def get_collection_internal_metadata_value(
    ctx: PeopleContext, bank_id: str, observer: str, observed: str, key: str, *, conn: Any | None = None
) -> Any | None:
    """Read one top-level internal_metadata value. Returns None when the pair or the key is missing."""
    sql = (
        f"SELECT internal_metadata -> $4 FROM {ctx.table('dream_state')} "
        "WHERE bank_id = $1 AND observer = $2 AND observed = $3"
    )
    if conn is not None:
        return from_jsonb(await conn.fetchval(sql, bank_id, observer, observed, key))
    async with ctx.conn() as acquired:
        return from_jsonb(await acquired.fetchval(sql, bank_id, observer, observed, key))


async def update_collection_internal_metadata(
    ctx: PeopleContext, bank_id: str, observer: str, observed: str, update_data: dict[str, Any]
) -> None:
    """Merge a patch into the pair's internal_metadata (JSONB ``||``, top-level keys replaced whole)."""
    table = ctx.table("dream_state")
    async with ctx.conn() as conn:
        await conn.execute(
            f"INSERT INTO {table} AS t (bank_id, observer, observed, internal_metadata) "
            "VALUES ($1, $2, $3, $4::jsonb) "
            "ON CONFLICT (bank_id, observer, observed) DO UPDATE "
            "SET internal_metadata = t.internal_metadata || EXCLUDED.internal_metadata",
            bank_id,
            observer,
            observed,
            to_jsonb(update_data),
        )


async def merge_collection_internal_metadata_value(
    ctx: PeopleContext,
    bank_id: str,
    observer: str,
    observed: str,
    key: str,
    patch: dict[str, Any],
    *,
    conn: Any | None = None,
    if_unchanged: tuple[str, str | None] | None = None,
) -> bool:
    """Merge ``patch`` into the object stored under one top-level key, server-side.

    ``update_collection_internal_metadata`` replaces a top-level value whole, so two writers of
    different sub-keys under the same key would overwrite each other. This merges one level deeper
    (``value || patch``) in a single statement, so writers of different sub-keys never collide.

    With ``if_unchanged=(subkey, value)``, the write happens only while the string stored under
    ``subkey`` still equals ``value`` (None: absent), so a writer that read before another reset the
    value can't write over it.

    Returns whether a row was written.
    """
    table = ctx.table("dream_state")
    patch_json = to_jsonb(patch)
    args: list[Any] = [bank_id, observer, observed, key, patch_json]
    sql = (
        f"INSERT INTO {table} AS t (bank_id, observer, observed, internal_metadata) "
        "VALUES ($1, $2, $3, jsonb_build_object($4::text, $5::jsonb)) "
        "ON CONFLICT (bank_id, observer, observed) DO UPDATE SET internal_metadata = t.internal_metadata || "
        "jsonb_build_object($4::text, COALESCE(t.internal_metadata -> $4::text, '{}'::jsonb) || $5::jsonb)"
    )
    if if_unchanged is not None:
        subkey, expected = if_unchanged
        args.extend([subkey, expected])
        sql += " WHERE (t.internal_metadata -> $4::text ->> $6::text) IS NOT DISTINCT FROM $7::text"
    sql += " RETURNING 1"
    if conn is not None:
        return await conn.fetchval(sql, *args) is not None
    async with ctx.conn() as acquired:
        return await acquired.fetchval(sql, *args) is not None


async def explicit_stats(
    ctx: PeopleContext, bank_id: str, observer: str, observed: str
) -> tuple[int, datetime.datetime | None]:
    """(explicit_count, newest_explicit_at): the pair's live explicit conclusions and the newest one's time."""
    async with ctx.conn() as conn:
        row = await conn.fetchrow(
            f"SELECT count(id) AS n, max(created_at) AS newest FROM {ctx.table('conclusions')} "
            "WHERE bank_id = $1 AND observer = $2 AND observed = $3 AND level = 'explicit' AND deleted_at IS NULL",
            bank_id,
            observer,
            observed,
        )
    return int(row["n"] or 0), row["newest"]


async def get_session_configuration(ctx: PeopleContext, bank_id: str, session_name: str) -> dict[str, Any] | None:
    """The ``configuration`` object a live session keeps in its internal metadata, or None when the session does not exist."""
    async with ctx.conn() as conn:
        raw = await conn.fetchval(
            f"SELECT internal_metadata -> 'configuration' FROM {ctx.table('sessions')} "
            "WHERE bank_id = $1 AND name = $2 AND deleted_at IS NULL",
            bank_id,
            session_name,
        )
    value = from_jsonb(raw)
    return value if isinstance(value, dict) else None
