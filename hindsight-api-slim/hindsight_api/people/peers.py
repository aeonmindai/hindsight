# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/crud/peer.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""CRUD helpers for peer records (one per person handle) and peer-scoped session queries."""

import logging
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from . import schemas
from .context import PeopleContext
from .db import connection
from .exceptions import ConflictException, ResourceNotFoundException, ValidationException
from .ids import generate_nanoid
from .jsonb import from_jsonb, to_jsonb
from .schemas import RESOURCE_NAME_PATTERN

logger = logging.getLogger(__name__)

# Matches the people_peers.name CHECK constraint and PeerCreate's max_length.
PEER_NAME_MAX_LENGTH = 512

T = TypeVar("T")


@dataclass
class GetOrCreateResult(Generic[T]):
    """What a get-or-create returns: the resource and whether anything was created."""

    resource: T
    created: bool = False


def _reject_impossible_peer_names(names: Collection[str]) -> None:
    """Reject names that cannot correspond to any stored row, before querying.

    ``PeerSpec`` accepts anything so existing names can be looked up, and the full new-name rules run
    later on the insert path. A couple of values cannot be a stored row *by construction*, and sending
    them to Postgres first fails before the clean error can happen:

    - NUL bytes: Postgres text cannot hold them.
    - Over-length names: the ``people_peers.name`` CHECK caps them at ``PEER_NAME_MAX_LENGTH``.

    Takes a ``Collection`` rather than an ``Iterable`` on purpose: it inspects the input twice, so a
    generator would be half-consumed and the second check would silently see nothing.

    Raises:
        ValidationException: On a NUL byte or an over-length name.
    """
    if any("\x00" in name for name in names):
        raise ValidationException("Peer name(s) must not contain NUL (0x00) bytes")
    too_long = sorted({n for n in names if len(n) > PEER_NAME_MAX_LENGTH})
    if too_long:
        raise ValidationException(f"Peer name(s) {too_long} must be at most {PEER_NAME_MAX_LENGTH} characters")


def _validate_new_peer_names(names: list[str]) -> None:
    """Validate peer names that are about to be created.

    Mirrors ``PeerCreate``'s contract for peers arriving through code rather than the peers route.

    Raises:
        ValidationException: On a non-conforming name.
    """
    # Length and NUL bytes are already refused before the lookup by
    # _reject_impossible_peer_names; RESOURCE_NAME_PATTERN's `+` rejects empty.
    offenders = sorted({n for n in names if not re.fullmatch(RESOURCE_NAME_PATTERN, n)})
    if offenders:
        raise ValidationException(f"Peer name(s) {offenders} must match pattern {RESOURCE_NAME_PATTERN}")


def _metadata_filter_clause(filters: dict[str, Any] | None, column: str, first_param: int) -> tuple[str, list[Any]]:
    """SQL for a metadata filter: every key/value pair must be contained in the ``column`` jsonb.

    Honcho's ``apply_filter`` also supports operators and logical groups; the people routes only use
    plain key/value filters, so anything else is refused rather than half-applied.
    """
    if not filters:
        return "", []
    for key, value in filters.items():
        if key in {"AND", "OR", "NOT"} or (isinstance(value, dict) and any(str(k).startswith("$") for k in value)):
            raise ValidationException(f"Unsupported filter {key!r}: only key/value metadata filters are supported")
    return f" AND {column} @> ${first_param}::jsonb", [to_jsonb(filters)]


def _peer_from_row(row: Any) -> schemas.Peer:
    return schemas.Peer(
        id=row["id"],
        name=row["name"],
        bank_id=row["bank_id"],
        created_at=row["created_at"],
        h_metadata=from_jsonb(row["metadata"], {}),
        internal_metadata=from_jsonb(row["internal_metadata"], {}),
        configuration=from_jsonb(row["configuration"], {}),
    )


def _session_from_row(row: Any) -> schemas.Session:
    return schemas.Session(
        id=row["id"],
        name=row["name"],
        is_active=row["is_active"],
        bank_id=row["bank_id"],
        h_metadata=from_jsonb(row["metadata"], {}),
        configuration=from_jsonb(row["internal_metadata"], {}).get("configuration", {}),
        created_at=row["created_at"],
    )


_PEER_COLUMNS = "bank_id, id, name, metadata, internal_metadata, configuration, created_at"


async def get_or_create_peers(
    ctx: PeopleContext,
    bank_id: str,
    peers: Sequence[schemas.PeerSpec | str],
    *,
    conn: Any | None = None,
    _retry: bool = False,
) -> GetOrCreateResult[list[schemas.Peer]]:
    """Get an existing list of peers or create new peers if they don't exist.

    Updates existing peers with metadata and configuration if provided. A peer a caller soft-deleted
    comes back as a new peer under the same handle.

    Args:
        ctx: People context
        bank_id: Bank the peers live in
        peers: Person handles, or peer specs with metadata and configuration
        conn: Optional open connection to join the caller's transaction

    Returns:
        GetOrCreateResult containing the list of peers (in the order asked for) and whether any were created

    Raises:
        ConflictException: If we fail to get or create the peers
        ValidationException: On an impossible name (NUL byte, over-length), or a non-conforming new name
    """
    specs = [schemas.PeerSpec(name=p) if isinstance(p, str) else p for p in peers]
    peer_names = list(dict.fromkeys(p.name for p in specs))
    # Before the lookup: these values cannot match a stored row and would fail
    # inside the query itself rather than as a clean validation error.
    _reject_impossible_peer_names(peer_names)
    if not peer_names:
        return GetOrCreateResult([], created=False)

    # Create a mapping of peer names to peer schemas for easy lookup
    peer_schema_map = {p.name: p for p in specs}

    async with connection(ctx, conn) as db:
        existing_rows = await db.fetch(
            f"SELECT {_PEER_COLUMNS} FROM {ctx.table('peers')} "
            "WHERE bank_id = $1 AND name = ANY($2::text[]) AND deleted_at IS NULL",
            bank_id,
            peer_names,
        )
        by_name: dict[str, schemas.Peer] = {}
        for row in existing_rows:
            existing_peer = _peer_from_row(row)
            peer_schema = peer_schema_map[existing_peer.name]
            # Update with metadata / configuration if provided AND different
            new_metadata = peer_schema.metadata
            new_configuration = peer_schema.configuration
            changed = False
            if new_metadata is not None and existing_peer.h_metadata != new_metadata:
                existing_peer.h_metadata = new_metadata
                changed = True
            if new_configuration is not None and existing_peer.configuration != new_configuration:
                existing_peer.configuration = new_configuration
                changed = True
            if changed:
                await db.execute(
                    f"UPDATE {ctx.table('peers')} SET metadata = $3::jsonb, configuration = $4::jsonb "
                    "WHERE bank_id = $1 AND id = $2",
                    bank_id,
                    existing_peer.id,
                    to_jsonb(existing_peer.h_metadata),
                    to_jsonb(existing_peer.configuration),
                )
            by_name[existing_peer.name] = existing_peer

        # Find which peers need to be created
        names_to_create = [n for n in peer_names if n not in by_name]

        # Names are validated on the *create* path only. `PeerSpec` deliberately carries no charset
        # pattern so already-existing names can be looked up without a spurious error, but a name we
        # are about to INSERT is a new peer, and new peers must obey the public contract.
        if names_to_create:
            _validate_new_peer_names(names_to_create)

        created = False
        for name in names_to_create:
            spec = peer_schema_map[name]
            # The handle is unique per bank. A concurrent creator wins the insert and we read its row; a
            # soft-deleted peer under the same handle is revived as a new peer with no old card.
            row = await db.fetchrow(
                f"INSERT INTO {ctx.table('peers')} (bank_id, id, name, metadata, configuration) "
                "VALUES ($1, $2, $3, $4::jsonb, $5::jsonb) "
                "ON CONFLICT (bank_id, name) DO UPDATE SET deleted_at = NULL, created_at = now(), "
                "metadata = EXCLUDED.metadata, configuration = EXCLUDED.configuration, internal_metadata = '{}'::jsonb "
                f"WHERE {ctx.table('peers')}.deleted_at IS NOT NULL "
                f"RETURNING {_PEER_COLUMNS}",
                bank_id,
                generate_nanoid(),
                name,
                to_jsonb(spec.metadata or {}),
                to_jsonb(spec.configuration or {}),
            )
            if row is None:
                # Created by a concurrent writer between our read and our insert: read its row.
                if _retry:
                    raise ConflictException(f"Unable to create or get peers: {peer_names}")
                return await get_or_create_peers(ctx, bank_id, peers, conn=db, _retry=True)
            by_name[name] = _peer_from_row(row)
            created = True

    # Return combined list of existing and new peers, in the order asked for
    return GetOrCreateResult([by_name[n] for n in peer_names], created=created)


async def get_peer(
    ctx: PeopleContext,
    bank_id: str,
    peer_name: str,
    *,
    conn: Any | None = None,
) -> schemas.Peer:
    """Get an existing peer.

    Takes a plain handle, not a create schema: this is a pure read, and validating an already
    existing name against ``PeerCreate``'s charset pattern would turn a lookup into an error for a
    legacy name.

    Raises:
        ResourceNotFoundException: If the peer does not exist
    """
    _reject_impossible_peer_names([peer_name])
    async with connection(ctx, conn) as db:
        row = await db.fetchrow(
            f"SELECT {_PEER_COLUMNS} FROM {ctx.table('peers')} WHERE bank_id = $1 AND name = $2 AND deleted_at IS NULL",
            bank_id,
            peer_name,
        )
    if row is None:
        raise ResourceNotFoundException(f"Peer {peer_name} not found in bank {bank_id}")
    return _peer_from_row(row)


async def list_person_handles(ctx: PeopleContext, bank_id: str) -> list[str]:
    """The handles of every live peer in the bank, in creation order."""
    async with ctx.conn() as db:
        rows = await db.fetch(
            f"SELECT name FROM {ctx.table('peers')} WHERE bank_id = $1 AND deleted_at IS NULL ORDER BY created_at, id",
            bank_id,
        )
    return [r["name"] for r in rows]


async def get_peers(
    ctx: PeopleContext,
    bank_id: str,
    filters: dict[str, Any] | None = None,
    reverse: bool = False,
    *,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[schemas.Peer], int]:
    """A page of peers of a bank ordered by creation time, and the total count.

    Args:
        filters: Filter peers by metadata (key/value containment)
        reverse: Whether to reverse the default creation order
    """
    clause, params = _metadata_filter_clause(filters, "metadata", 2)
    order = "DESC" if reverse else "ASC"
    async with ctx.conn() as db:
        total = await db.fetchval(
            f"SELECT count(*) FROM {ctx.table('peers')} WHERE bank_id = $1 AND deleted_at IS NULL{clause}",
            bank_id,
            *params,
        )
        n = len(params) + 2
        rows = await db.fetch(
            f"SELECT {_PEER_COLUMNS} FROM {ctx.table('peers')} WHERE bank_id = $1 AND deleted_at IS NULL{clause} "
            f"ORDER BY created_at {order}, id {order} LIMIT ${n} OFFSET ${n + 1}",
            bank_id,
            *params,
            limit,
            offset,
        )
    return [_peer_from_row(r) for r in rows], int(total or 0)


async def update_peer(
    ctx: PeopleContext,
    bank_id: str,
    peer_name: str,
    peer: schemas.PeerUpdate,
) -> schemas.Peer:
    """Get or create a peer, then apply metadata and configuration updates.

    If the peer does not exist it is created first. Provided metadata and configuration replace the
    existing values when present.

    Raises:
        ConflictException: If concurrent creation prevents fetching or creating the peer
    """
    peers_result = await get_or_create_peers(ctx, bank_id, [schemas.PeerSpec(name=peer_name)])
    existing = peers_result.resource[0]

    needs_update = False

    if peer.metadata is not None and existing.h_metadata != peer.metadata:
        existing.h_metadata = peer.metadata
        needs_update = True

    if peer.configuration is not None and existing.configuration != peer.configuration:
        existing.configuration = peer.configuration
        needs_update = True

    # Early exit if unchanged
    if not needs_update:
        logger.debug("Peer %s unchanged in bank %s, skipping update", peer_name, bank_id)
        return existing

    async with ctx.conn() as db:
        await db.execute(
            f"UPDATE {ctx.table('peers')} SET metadata = $3::jsonb, configuration = $4::jsonb "
            "WHERE bank_id = $1 AND id = $2",
            bank_id,
            existing.id,
            to_jsonb(existing.h_metadata),
            to_jsonb(existing.configuration),
        )

    logger.debug("Peer %s updated successfully", peer_name)
    return existing


async def get_sessions_for_peer(
    ctx: PeopleContext,
    bank_id: str,
    peer_name: str,
    filters: dict[str, Any] | None = None,
    reverse: bool = False,
    *,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[schemas.Session], int]:
    """A page of the sessions of a peer (through the session peers table) and the total count.

    Args:
        filters: Filter sessions by metadata
        reverse: Whether to reverse the default creation order
    """
    clause, params = _metadata_filter_clause(filters, "s.metadata", 3)
    order = "DESC" if reverse else "ASC"
    base = (
        f"FROM {ctx.table('sessions')} s JOIN {ctx.table('session_peers')} sp "
        "ON sp.bank_id = s.bank_id AND sp.session_name = s.name "
        f"WHERE s.bank_id = $1 AND sp.peer_name = $2 AND s.deleted_at IS NULL{clause}"
    )
    async with ctx.conn() as db:
        total = await db.fetchval(f"SELECT count(*) {base}", bank_id, peer_name, *params)
        n = len(params) + 3
        rows = await db.fetch(
            "SELECT s.bank_id, s.id, s.name, s.is_active, s.metadata, s.internal_metadata, s.created_at "
            f"{base} ORDER BY s.created_at {order}, s.id {order} LIMIT ${n} OFFSET ${n + 1}",
            bank_id,
            peer_name,
            *params,
            limit,
            offset,
        )
    return [_session_from_row(r) for r in rows], int(total or 0)


_MESSAGE_COLUMNS = (
    "bank_id, public_id, session_name, peer_name, content, metadata, token_count, created_at, "
    "source_url, said_by, said_at, channel"
)


def _message_from_row(row: Any) -> schemas.Message:
    return schemas.Message(
        public_id=row["public_id"],
        content=row["content"],
        peer_name=row["peer_name"],
        session_name=row["session_name"],
        h_metadata=from_jsonb(row["metadata"], {}),
        created_at=row["created_at"],
        bank_id=row["bank_id"],
        token_count=row["token_count"],
        source_url=row["source_url"],
        said_by=row["said_by"],
        said_at=row["said_at"],
        channel=row["channel"],
    )


async def get_sessions(
    ctx: PeopleContext,
    bank_id: str,
    filters: dict[str, Any] | None = None,
    reverse: bool = False,
    *,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[schemas.Session], int]:
    """A page of the live sessions of a bank ordered by creation time, and the total count."""
    clause, params = _metadata_filter_clause(filters, "metadata", 2)
    order = "DESC" if reverse else "ASC"
    async with ctx.conn() as db:
        total = await db.fetchval(
            f"SELECT count(*) FROM {ctx.table('sessions')} WHERE bank_id = $1 AND deleted_at IS NULL{clause}",
            bank_id,
            *params,
        )
        n = len(params) + 2
        rows = await db.fetch(
            "SELECT bank_id, id, name, is_active, metadata, internal_metadata, created_at "
            f"FROM {ctx.table('sessions')} WHERE bank_id = $1 AND deleted_at IS NULL{clause} "
            f"ORDER BY created_at {order}, id {order} LIMIT ${n} OFFSET ${n + 1}",
            bank_id,
            *params,
            limit,
            offset,
        )
    return [_session_from_row(r) for r in rows], int(total or 0)


async def get_messages(
    ctx: PeopleContext,
    bank_id: str,
    session_name: str,
    reverse: bool = False,
    *,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[schemas.Message], int]:
    """A page of the live messages of a session in the order they were said, and the total count.

    Raises:
        ResourceNotFoundException: If the session does not exist
    """
    order = "DESC" if reverse else "ASC"
    async with ctx.conn() as db:
        exists = await db.fetchval(
            f"SELECT 1 FROM {ctx.table('sessions')} WHERE bank_id = $1 AND name = $2 AND deleted_at IS NULL",
            bank_id,
            session_name,
        )
        if not exists:
            raise ResourceNotFoundException(f"Session {session_name} not found in bank {bank_id}")
        total = await db.fetchval(
            f"SELECT count(*) FROM {ctx.table('messages')} "
            "WHERE bank_id = $1 AND session_name = $2 AND deleted_at IS NULL",
            bank_id,
            session_name,
        )
        rows = await db.fetch(
            f"SELECT {_MESSAGE_COLUMNS} FROM {ctx.table('messages')} "
            "WHERE bank_id = $1 AND session_name = $2 AND deleted_at IS NULL "
            f"ORDER BY id {order} LIMIT $3 OFFSET $4",
            bank_id,
            session_name,
            limit,
            offset,
        )
    return [_message_from_row(r) for r in rows], int(total or 0)


async def search_messages(
    ctx: PeopleContext,
    bank_id: str,
    peer_name: str,
    query: str,
    filters: dict[str, Any] | None = None,
    limit: int = 10,
) -> list[schemas.Message]:
    """Search a peer's messages by words (full text) or by substring, best match first.

    ``filters`` supports ``session_id`` (a session name).
    """
    params: list[Any] = [
        bank_id,
        peer_name,
        query,
        "%" + query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%",
    ]
    extra = ""
    for key, value in (filters or {}).items():
        if key != "session_id":
            raise ValidationException(f"Unsupported message filter {key!r}: only session_id is supported")
        params.append(value)
        extra += f" AND session_name = ${len(params)}"
    params.append(limit)
    async with ctx.conn() as db:
        rows = await db.fetch(
            f"SELECT {_MESSAGE_COLUMNS} FROM {ctx.table('messages')} "
            "WHERE bank_id = $1 AND peer_name = $2 AND deleted_at IS NULL "
            "AND (to_tsvector('english', content) @@ plainto_tsquery('english', $3) OR content ILIKE $4)"
            f"{extra} ORDER BY ts_rank(to_tsvector('english', content), plainto_tsquery('english', $3)) DESC, "
            f"created_at DESC, id DESC LIMIT ${len(params)}",
            *params,
        )
    return [_message_from_row(r) for r in rows]
