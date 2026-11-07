"""Helpers for the people core tests: sessions, messages and conclusions written the way the deriver does."""

import json
from typing import Any

from hindsight_api.people import schemas
from hindsight_api.people.conclusions import CreateDocumentsResult, create_documents
from hindsight_api.people.context import PeopleContext
from hindsight_api.people.ids import generate_nanoid
from hindsight_api.people.peers import get_or_create_peers


async def add_session(ctx: PeopleContext, bank_id: str, name: str, peers: list[str], metadata: dict | None = None):
    """A session with its member peers, written with plain SQL (the deriver owns the real writer)."""
    await get_or_create_peers(ctx, bank_id, peers)
    async with ctx.conn() as db:
        await db.execute(
            f"INSERT INTO {ctx.table('sessions')} (bank_id, id, name, metadata) VALUES ($1, $2, $3, $4::jsonb)",
            bank_id,
            generate_nanoid(),
            name,
            json.dumps(metadata or {}),
        )
        for peer in peers:
            await db.execute(
                f"INSERT INTO {ctx.table('session_peers')} (bank_id, session_name, peer_name) VALUES ($1, $2, $3)",
                bank_id,
                name,
                peer,
            )


async def add_message(
    ctx: PeopleContext,
    bank_id: str,
    session: str,
    peer: str,
    content: str,
    *,
    seq: int,
    source_url: str | None = None,
    said_at: Any = None,
) -> str:
    public_id = generate_nanoid()
    async with ctx.conn() as db:
        await db.execute(
            f"INSERT INTO {ctx.table('messages')} (bank_id, public_id, session_name, peer_name, content, "
            "seq_in_session, token_count, source_url, said_by, said_at, channel) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $4, $9, 'slack')",
            bank_id,
            public_id,
            session,
            peer,
            content,
            seq,
            len(content.split()),
            source_url,
            said_at,
        )
    return public_id


async def make_doc(
    ctx: PeopleContext,
    content: str,
    *,
    level: str = "explicit",
    session: str | None = "s1",
    message_ids: list[int] | None = None,
    source_message_ids: list[str] | None = None,
    source_ids: list[str] | None = None,
    embedding: list[float] | None = None,
    times_derived: int = 1,
    premises: list[str] | None = None,
) -> schemas.DocumentCreate:
    """A conclusion to create, embedded with the engine's (deterministic test) embeddings."""
    if embedding is None:
        [embedding] = await ctx.embed([content])
    return schemas.DocumentCreate(
        content=content,
        session_name=session,
        level=level,  # type: ignore[arg-type]
        times_derived=times_derived,
        metadata=schemas.DocumentMetadata(
            message_ids=message_ids or [], message_created_at="2026-01-01T00:00:00Z", premises=premises
        ),
        embedding=embedding,
        source_ids=source_ids,
        source_message_ids=source_message_ids,
    )


async def save(
    ctx: PeopleContext,
    bank_id: str,
    docs: list[schemas.DocumentCreate],
    *,
    observer: str = "alice",
    observed: str = "alice",
    deduplicate: bool = False,
) -> CreateDocumentsResult:
    await get_or_create_peers(ctx, bank_id, [observer, observed])
    return await create_documents(ctx, bank_id, docs, observer=observer, observed=observed, deduplicate=deduplicate)
