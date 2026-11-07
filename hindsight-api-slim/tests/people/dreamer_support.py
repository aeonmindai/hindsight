"""Helpers for the dreamer tests: seeded pairs, open dream runs and conclusions saved the way a dream saves them."""

import datetime
import json
from typing import Any

from hindsight_api.people.context import PeopleContext
from hindsight_api.people.dream_history import DreamHistoryTarget, open_run
from hindsight_api.people.dreamer.agent_tools import save_documents_with_history
from hindsight_api.people.ids import generate_nanoid
from hindsight_api.people.peers import get_or_create_peers
from tests.people.people_helpers import add_session, make_doc

PARALLEL = [0.5] * 384  # the test embeddings are 384-dimensional; equal vectors are exact duplicates


async def seed(ctx: PeopleContext, bank_id: str, peer: str = "ann") -> tuple[str, list[str]]:
    """A session and two explicit conclusions about the peer; returns (session, ids)."""
    session_name = generate_nanoid()
    await add_session(ctx, bank_id, session_name, [peer])
    docs = [
        await make_doc(ctx, "Ann drinks coffee daily", session=session_name),
        await make_doc(ctx, "Ann works nights", session=session_name),
    ]
    result = await save_documents_with_history(
        ctx, bank_id, docs, observer=peer, observed=peer, deduplicate=False, history=None
    )
    by_content = {c.content: c.id for c in result.created_conclusions}
    return session_name, [by_content["Ann drinks coffee daily"], by_content["Ann works nights"]]


async def open_target(
    ctx: PeopleContext, bank_id: str, observer: str = "ann", observed: str | None = None, specialist: str = "deduction"
) -> DreamHistoryTarget:
    """An open dream run for the pair, as the orchestrator opens one."""
    await get_or_create_peers(ctx, bank_id, [observer, observed or observer])
    target = await open_run(
        ctx,
        run_id=generate_nanoid(),
        bank_id=bank_id,
        observer=observer,
        observed=observed or observer,
        dream_type="omni",
        trigger_reason="manual",
    )
    assert target is not None
    return DreamHistoryTarget(run_id=target.run_id, specialist=specialist)


async def deductive(ctx: PeopleContext, content: str, source_ids: list[str]) -> Any:
    return await make_doc(ctx, content, level="deductive", session=None, source_ids=source_ids)


async def age_run(ctx: PeopleContext, bank_id: str, run_id: str, age: datetime.timedelta) -> None:
    """Move a run's start into the past."""
    async with ctx.conn() as conn:
        await conn.execute(
            f"UPDATE {ctx.table('dream_runs')} SET started_at = $3 WHERE bank_id = $1 AND id = $2",
            bank_id,
            run_id,
            datetime.datetime.now(datetime.UTC) - age,
        )


async def scalar(ctx: PeopleContext, sql: str, *args: Any) -> Any:
    async with ctx.conn() as conn:
        return await conn.fetchval(sql, *args)


async def tombstone(ctx: PeopleContext, table: str, bank_id: str, name: str) -> None:
    """Set deleted_at on a peer or a session, the way a soft delete would."""
    async with ctx.conn() as conn:
        await conn.execute(
            f"UPDATE {ctx.table(table)} SET deleted_at = now() WHERE bank_id = $1 AND name = $2", bank_id, name
        )


async def set_state(ctx: PeopleContext, bank_id: str, observer: str, observed: str, state: dict[str, Any]) -> None:
    """Replace a pair's dream state (the `internal_metadata` object)."""
    async with ctx.conn() as conn:
        await conn.execute(
            f"INSERT INTO {ctx.table('dream_state')} (bank_id, observer, observed, internal_metadata) "
            "VALUES ($1, $2, $3, $4::jsonb) ON CONFLICT (bank_id, observer, observed) "
            "DO UPDATE SET internal_metadata = EXCLUDED.internal_metadata",
            bank_id,
            observer,
            observed,
            json.dumps(state),
        )


async def make_session(
    ctx: PeopleContext, bank_id: str, configuration: dict[str, Any] | None = None, peers: list[str] | None = None
) -> str:
    """A session (with its peers) carrying an optional configuration."""
    name = f"s-{generate_nanoid()}"
    await add_session(ctx, bank_id, name, peers or ["ann"])
    if configuration:
        async with ctx.conn() as conn:
            await conn.execute(
                f"UPDATE {ctx.table('sessions')} SET internal_metadata = jsonb_build_object('configuration', $3::jsonb) WHERE bank_id = $1 AND name = $2",
                bank_id,
                name,
                json.dumps(configuration),
            )
    return name


async def insert_conclusions(
    ctx: PeopleContext,
    bank_id: str,
    level: str,
    count: int,
    *,
    observer: str = "ann",
    observed: str = "ann",
    age_minutes: int = 0,
    session_name: str | None = None,
    content: str = "test",
) -> list[str]:
    """Insert ``count`` conclusions created ``age_minutes`` ago (plain SQL: no embedding, no checks)."""
    await get_or_create_peers(ctx, bank_id, [observer, observed])
    created_at = datetime.datetime.now(datetime.UTC) - datetime.timedelta(minutes=age_minutes)
    ids = [generate_nanoid() for _ in range(count)]
    async with ctx.conn() as conn:
        await conn.executemany(
            f"INSERT INTO {ctx.table('conclusions')} "
            "(bank_id, id, observer, observed, level, content, session_name, created_at) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
            [(bank_id, cid, observer, observed, level, content, session_name, created_at) for cid in ids],
        )
    return ids


async def insert_queue_item(
    ctx: PeopleContext,
    bank_id: str,
    work_unit_key: str,
    task_type: str,
    *,
    age_minutes: int = 0,
    processed: bool = False,
    error: str | None = None,
) -> None:
    async with ctx.conn() as conn:
        await conn.execute(
            f"INSERT INTO {ctx.table('queue')} (bank_id, work_unit_key, task_type, payload, processed, error, created_at) "
            "VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7)",
            bank_id,
            work_unit_key,
            task_type,
            json.dumps({"task_type": task_type}),
            processed,
            error,
            datetime.datetime.now(datetime.UTC) - datetime.timedelta(minutes=age_minutes),
        )


async def seed_rows(
    ctx: PeopleContext,
    bank_id: str,
    rows: list[tuple[str, str, datetime.datetime]],
    state: dict[str, Any] | None = None,
    *,
    peer: str = "ann",
) -> None:
    """Conclusions with fixed ids, levels and creation times, and an optional dream state, for the peer's pair.

    Ids are padded to 21 characters (the id format) so short names stay readable and sortable.
    """
    await get_or_create_peers(ctx, bank_id, [peer])
    async with ctx.conn() as conn:
        for doc_id, level, created_at in rows:
            await conn.execute(
                f"INSERT INTO {ctx.table('conclusions')} "
                "(bank_id, id, observer, observed, level, content, created_at) VALUES ($1, $2, $3, $3, $4, $5, $6)",
                bank_id,
                padded_id(doc_id),
                peer,
                level,
                f"content {doc_id}",
                created_at,
            )
    if state is not None:
        await set_state(ctx, bank_id, peer, peer, state)


def padded_id(name: str) -> str:
    """Conclusion ids must be 21 characters; padding keeps names sortable."""
    return name.ljust(21, "0")


async def stored_state(ctx: PeopleContext, bank_id: str, peer: str = "ann") -> dict[str, Any]:
    """The pair's whole dream state object."""
    from hindsight_api.people.dreamer import collection_state

    return await collection_state.get_collection_internal_metadata(ctx, bank_id, peer, peer)
