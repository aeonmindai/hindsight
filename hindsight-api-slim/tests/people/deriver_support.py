"""Seeding helpers for the deriver tests: people rows, messages with chosen token counts, queue items."""

from datetime import UTC, datetime
from typing import Any

from hindsight_api.people.context import PeopleContext
from hindsight_api.people.deriver.enqueue import create_representation_record
from hindsight_api.people.deriver.models import MESSAGE_COLUMNS, QUEUE_COLUMNS, Message, QueueItem
from hindsight_api.people.deriver.queue_manager import QueueManager, WorkerOwnership
from hindsight_api.people.ids import generate_nanoid
from hindsight_api.people.jsonb import to_jsonb
from hindsight_api.people.peers import get_or_create_peers
from hindsight_api.people.schemas import ResolvedConfiguration, get_configuration


async def seed_messages(
    ctx: PeopleContext,
    bank_id: str,
    specs: list[tuple[str, int]],
    *,
    session: str = "s1",
) -> list[int]:
    """Store messages ``(peer, token_count)`` in a session, in order; returns their ids."""
    await get_or_create_peers(ctx, bank_id, sorted({peer for peer, _ in specs}))
    ids: list[int] = []
    async with ctx.conn() as conn, conn.transaction():
        await conn.execute(
            f"INSERT INTO {PeopleContext.table('sessions')} (bank_id, id, name) VALUES ($1, $2, $3) "
            "ON CONFLICT (bank_id, name) DO NOTHING",
            bank_id,
            generate_nanoid(),
            session,
        )
        for peer in sorted({peer for peer, _ in specs}):
            await conn.execute(
                f"INSERT INTO {PeopleContext.table('session_peers')} (bank_id, session_name, peer_name) "
                "VALUES ($1, $2, $3) ON CONFLICT DO NOTHING",
                bank_id,
                session,
                peer,
            )
        for index, (peer, token_count) in enumerate(specs):
            seq = await conn.fetchval(
                f"SELECT COALESCE(MAX(seq_in_session), 0) + 1 FROM {PeopleContext.table('messages')} "
                "WHERE bank_id = $1 AND session_name = $2",
                bank_id,
                session,
            )
            ids.append(
                await conn.fetchval(
                    f"INSERT INTO {PeopleContext.table('messages')} "
                    "(bank_id, public_id, session_name, peer_name, content, token_count, seq_in_session, said_by) "
                    "VALUES ($1, $2, $3, $4, $5, $6, $7, $4) RETURNING id",
                    bank_id,
                    generate_nanoid(),
                    session,
                    peer,
                    f"Message {index} from {peer}",
                    token_count,
                    seq,
                )
            )
    return ids


async def seed_queue(
    ctx: PeopleContext,
    bank_id: str,
    message_ids: list[int],
    *,
    observed: str,
    observers: list[str] | None = None,
    session: str = "s1",
    created_ats: list[datetime] | None = None,
    configuration: ResolvedConfiguration | None = None,
) -> tuple[str, list[QueueItem]]:
    """Queue a representation item for each message id; returns the work unit key and the items."""
    conf = configuration or get_configuration(None, None, None)
    items: list[QueueItem] = []
    key = ""
    async with ctx.conn() as conn:
        session_id = await conn.fetchval(
            f"SELECT id FROM {PeopleContext.table('sessions')} WHERE bank_id = $1 AND name = $2", bank_id, session
        )
        for index, message_id in enumerate(message_ids):
            row = await conn.fetchrow(
                f"SELECT content, created_at FROM {PeopleContext.table('messages')} WHERE bank_id = $1 AND id = $2",
                bank_id,
                message_id,
            )
            record = create_representation_record(
                {
                    "bank_id": bank_id,
                    "session_name": session,
                    "message_id": message_id,
                    "content": row["content"],
                    "created_at": row["created_at"],
                },
                conf,
                session_id,
                observers=observers or [observed],
                observed=observed,
            )
            key = key or record["work_unit_key"]
            created_at = created_ats[index] if created_ats else datetime.now(UTC)
            queue_row = await conn.fetchrow(
                f"INSERT INTO {PeopleContext.table('queue')} "
                "(bank_id, session_id, work_unit_key, task_type, payload, message_id, created_at) "
                f"VALUES ($1, $2, $3, 'representation', $4::jsonb, $5, $6) RETURNING {QUEUE_COLUMNS}",
                bank_id,
                session_id,
                record["work_unit_key"],
                to_jsonb(record["payload"]),
                message_id,
                created_at,
            )
            items.append(QueueItem.from_record(queue_row))
    return key, items


async def seed_unit(
    ctx: PeopleContext,
    bank_id: str,
    token_counts: list[int],
    *,
    peer: str = "alice",
    session: str = "s1",
    created_ats: list[datetime] | None = None,
) -> tuple[str, list[QueueItem], list[int]]:
    """A representation work unit of messages from one peer with the given token counts."""
    message_ids = await seed_messages(ctx, bank_id, [(peer, tokens) for tokens in token_counts], session=session)
    key, items = await seed_queue(ctx, bank_id, message_ids, observed=peer, session=session, created_ats=created_ats)
    return key, items, message_ids


async def claim(
    ctx: PeopleContext, bank_id: str, key: str, worker_id: str = "test_worker"
) -> tuple[QueueManager, str, str]:
    """Claim a work unit for a fresh manager and register the worker as its owner."""
    qm = QueueManager(lambda: ctx)
    async with ctx.conn() as conn:
        claimed = await qm.claim_work_units(conn, bank_id, [key])
    qm.worker_ownership[worker_id] = WorkerOwnership(bank_id, key, claimed[key])
    return qm, worker_id, claimed[key]


async def items_of(ctx: PeopleContext, bank_id: str, key: str) -> list[QueueItem]:
    async with ctx.conn() as conn:
        rows = await conn.fetch(
            f"SELECT {QUEUE_COLUMNS} FROM {PeopleContext.table('queue')} "
            "WHERE bank_id = $1 AND work_unit_key = $2 ORDER BY id",
            bank_id,
            key,
        )
    return [QueueItem.from_record(row) for row in rows]


async def claim_rows(ctx: PeopleContext, bank_id: str, key: str | None = None) -> int:
    async with ctx.conn() as conn:
        if key is None:
            return await conn.fetchval(
                f"SELECT count(*) FROM {PeopleContext.table('active_queue_sessions')} WHERE bank_id = $1", bank_id
            )
        return await conn.fetchval(
            f"SELECT count(*) FROM {PeopleContext.table('active_queue_sessions')} "
            "WHERE bank_id = $1 AND work_unit_key = $2",
            bank_id,
            key,
        )


async def messages_of(ctx: PeopleContext, bank_id: str) -> list[Message]:
    async with ctx.conn() as conn:
        rows = await conn.fetch(
            f"SELECT {MESSAGE_COLUMNS} FROM {PeopleContext.table('messages')} m WHERE m.bank_id = $1 ORDER BY m.id",
            bank_id,
        )
    return [Message.from_record(row) for row in rows]


def retry_attempts(items: list[QueueItem]) -> Any:
    """The attempt counter on the oldest unprocessed item (None when there is none)."""
    unprocessed = [item for item in items if not item.processed]
    if not unprocessed:
        return None
    raw = (unprocessed[0].payload or {}).get("_retry_attempts")
    return None if raw is None else int(raw)
