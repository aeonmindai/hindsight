# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/deriver/enqueue.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""Put retained messages and dreams on the deriver queue.

Honcho's ``enqueue`` receives messages its API already stored. The fork's retain path hands over what
a person said (``IncomingMessage``), so ``enqueue_messages`` stores the people rows first (peer,
session, session peer, message) and then writes the queue items, all in one transaction: a message is
never stored without its derivation being queued. Honcho's summary, scope and deletion tasks are not
copied.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import asyncpg

from ...engine.token_encoding import count_tokens
from ..context import PeopleContext
from ..dreamer.payload import DreamPayload
from ..dreamer.schemas import DreamType
from ..ids import generate_nanoid
from ..jsonb import from_jsonb, to_jsonb
from ..peers import get_or_create_peers
from ..sanitization import strip_nul
from ..schemas import MessageConfiguration, PeerConfig, ResolvedConfiguration, SessionPeerConfig, get_configuration
from .queue_payload import create_payload
from .retryable_errors import is_retryable_error
from .work_unit import construct_work_unit_key

logger = logging.getLogger(__name__)

# Caps the queue writes one process runs at once, so a burst of enqueues
# cannot take every pooled DB connection from request handlers.
_ENQUEUE_DB_SLOTS = asyncio.Semaphore(4)

# people_messages.content is limited to 65535 characters; a longer retained item is stored as
# several messages of the same session, split at paragraph or line breaks.
MAX_MESSAGE_CHARS = 60000

# Transient failures of the enqueue transaction (deadlock, lost connection) are retried this many times.
_ENQUEUE_RETRIES = 2


@dataclass
class IncomingMessage:
    """What a person said, as the retain path hands it to the deriver.

    ``peer`` is the person handle who said it (``said_by``), ``session`` the conversation it belongs
    to, ``document_id`` the retained document it came from (a second enqueue of the same document and
    text is a no-op). ``message_id`` names a message already stored in ``people_messages``: it is queued
    without being stored again. ``configuration`` overrides the bank's default reasoning configuration
    for this message.
    """

    peer: str
    content: str
    session: str
    said_at: datetime | None = None
    channel: str | None = None
    source_url: str | None = None
    document_id: str | None = None
    message_id: int | None = None
    configuration: MessageConfiguration | None = None


def split_content(content: str, limit: int = MAX_MESSAGE_CHARS) -> list[str]:
    """Split text longer than ``limit`` characters at a paragraph, line or space break.

    A break is used when it leaves a part longer than half the limit; the break itself is dropped.
    Text with no such break is cut at the limit.
    """
    parts: list[str] = []
    rest = content
    while len(rest) > limit:
        part, rest = rest[:limit], rest[limit:]
        for separator in ("\n\n", "\n", " "):
            cut = (part + rest[: len(separator)]).rfind(separator)
            if cut > limit // 2:
                part, rest = (part + rest)[:cut], (part + rest)[cut + len(separator) :]
                break
        parts.append(part)
    if rest or not parts:
        parts.append(rest)
    return parts


async def enqueue_messages(ctx: PeopleContext, bank_id: str, messages: list[IncomingMessage]) -> list[int]:
    """
    Store retained messages and add them to the deriver queue for processing.

    Args:
        ctx: The people context
        bank_id: The bank the messages belong to
        messages: What was said, in order

    Returns:
        The ``people_messages.id`` of every message stored or reused, in order (a long message that was
        split contributes one id per part).
    """
    if not messages:
        return []

    # Cancel any pending dreams for the people who spoke: they are active again. This cancels dreams
    # for every observer pair where observed = the speaker, which covers both self-observation and
    # peer-to-peer observation.
    await _cancel_pending_dreams(bank_id, {message.peer for message in messages})

    async with _ENQUEUE_DB_SLOTS:
        for attempt in range(_ENQUEUE_RETRIES + 1):
            try:
                return await _store_and_enqueue(ctx, bank_id, messages)
            except Exception as exc:
                if attempt == _ENQUEUE_RETRIES or not is_retryable_error(exc):
                    raise
                await asyncio.sleep(0.1 * (attempt + 1))
    raise AssertionError("unreachable")


async def _cancel_pending_dreams(bank_id: str, handles: set[str]) -> None:
    # Imported here: the dream scheduler imports this module.
    from ..dreamer.dream_scheduler import get_dream_scheduler

    dream_scheduler = get_dream_scheduler()
    if dream_scheduler is None:
        return
    cancelled_dreams: set[str] = set()
    for handle in handles:
        cancelled = await dream_scheduler.cancel_dreams_for_observed(bank_id, handle)
        cancelled_dreams.update(cancelled)
    if cancelled_dreams:
        logger.info(f"Cancelled {len(cancelled_dreams)} pending dreams due to new activity")


async def _store_and_enqueue(ctx: PeopleContext, bank_id: str, messages: list[IncomingMessage]) -> list[int]:
    message_ids: list[int] = []
    async with ctx.conn() as conn, conn.transaction():
        queue_records: list[dict[str, Any]] = []
        # Messages of one session are handled together, in the order they were given.
        sessions: dict[str, list[IncomingMessage]] = {}
        for message in messages:
            sessions.setdefault(message.session, []).append(message)

        await get_or_create_peers(ctx, bank_id, sorted({m.peer for m in messages}), conn=conn)
        for session_name in sorted(sessions):
            session_messages = sessions[session_name]
            session_id = await _ensure_session(ctx, conn, bank_id, session_name)
            for peer in sorted({m.peer for m in session_messages}):
                await _ensure_session_peer(ctx, conn, bank_id, session_name, peer)

            stored: list[dict[str, Any]] = []
            for message in session_messages:
                stored.extend(await _store_message(ctx, conn, bank_id, session_name, message))
            message_ids.extend(row["message_id"] for row in stored)

            peers_with_configuration = await get_peers_with_configuration(ctx, conn, bank_id, session_name)
            for row in stored:
                if not row["queue"]:
                    continue
                conf = get_configuration(row["configuration"], None, None)
                queue_records.extend(generate_queue_records(row["message"], peers_with_configuration, session_id, conf))

        if queue_records:
            await _insert_queue_records(ctx, conn, queue_records)
    return message_ids


async def _ensure_session(ctx: PeopleContext, conn: Any, bank_id: str, session_name: str) -> str:
    sessions = PeopleContext.table("sessions")
    await conn.execute(
        f"INSERT INTO {sessions} (bank_id, id, name) VALUES ($1, $2, $3) ON CONFLICT (bank_id, name) DO NOTHING",
        bank_id,
        generate_nanoid(),
        session_name,
    )
    return await conn.fetchval(f"SELECT id FROM {sessions} WHERE bank_id = $1 AND name = $2", bank_id, session_name)


async def _ensure_session_peer(ctx: PeopleContext, conn: Any, bank_id: str, session_name: str, peer: str) -> None:
    """Add the speaker to the session, or bring them back if they had left."""
    session_peers = PeopleContext.table("session_peers")
    await conn.execute(
        f"""
        INSERT INTO {session_peers} (bank_id, session_name, peer_name) VALUES ($1, $2, $3)
        ON CONFLICT (bank_id, session_name, peer_name) DO UPDATE SET left_at = NULL
        """,
        bank_id,
        session_name,
        peer,
    )


async def _store_message(
    ctx: PeopleContext, conn: Any, bank_id: str, session_name: str, message: IncomingMessage
) -> list[dict[str, Any]]:
    """Store one retained message (as several rows when it is too long) and describe what to queue.

    Each entry: ``message_id``, ``message`` (the dict the queue payload is built from), ``queue``
    (False when the same document text was already stored) and ``configuration``.
    """
    messages_table = PeopleContext.table("messages")

    if message.message_id is not None:
        row = await conn.fetchrow(
            f"SELECT id, public_id, content, created_at, seq_in_session FROM {messages_table} "
            "WHERE bank_id = $1 AND id = $2 AND session_name = $3",
            bank_id,
            message.message_id,
            session_name,
        )
        if row is None:
            raise ValueError(f"message {message.message_id} does not exist in session {session_name!r}")
        return [
            {
                "message_id": row["id"],
                "queue": True,
                "configuration": message.configuration,
                "message": _queue_message(bank_id, session_name, message.peer, row),
            }
        ]

    content = strip_nul(message.content)
    if not content.strip():
        return []
    parts = split_content(content)
    stored: list[dict[str, Any]] = []
    for part_index, part in enumerate(parts):
        if message.document_id is not None:
            existing = await conn.fetchrow(
                f"SELECT id, public_id, content, created_at, seq_in_session FROM {messages_table} "
                "WHERE bank_id = $1 AND session_name = $2 AND deleted_at IS NULL "
                "AND internal_metadata->>'document_id' = $3 AND internal_metadata->>'part' = $4",
                bank_id,
                session_name,
                message.document_id,
                str(part_index),
            )
            if existing is not None and existing["content"] == part:
                stored.append(
                    {
                        "message_id": existing["id"],
                        "queue": False,
                        "configuration": message.configuration,
                        "message": _queue_message(bank_id, session_name, message.peer, existing),
                    }
                )
                continue

        await conn.execute("SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", f"{bank_id}\x1f{session_name}")
        seq = await conn.fetchval(
            f"SELECT COALESCE(MAX(seq_in_session), 0) + 1 FROM {messages_table} WHERE bank_id = $1 AND session_name = $2",
            bank_id,
            session_name,
        )
        internal_metadata: dict[str, Any] = {}
        if message.document_id is not None:
            internal_metadata = {"document_id": message.document_id, "part": str(part_index)}
        row = await conn.fetchrow(
            f"""
            INSERT INTO {messages_table}
                (bank_id, public_id, session_name, peer_name, content, internal_metadata, token_count,
                 seq_in_session, source_url, said_by, said_at, channel)
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $9, $10, $11, $12)
            RETURNING id, public_id, content, created_at, seq_in_session
            """,
            bank_id,
            generate_nanoid(),
            session_name,
            message.peer,
            part,
            to_jsonb(internal_metadata),
            count_tokens(part),
            seq,
            message.source_url,
            message.peer,
            message.said_at,
            message.channel,
        )
        stored.append(
            {
                "message_id": row["id"],
                "queue": True,
                "configuration": message.configuration,
                "message": _queue_message(bank_id, session_name, message.peer, row),
            }
        )
    return stored


def _queue_message(bank_id: str, session_name: str, peer: str, row: Any) -> dict[str, Any]:
    """The message dict the queue payload builders read."""
    return {
        "bank_id": bank_id,
        "session_name": session_name,
        "peer_name": peer,
        "message_id": row["id"],
        "message_public_id": row["public_id"],
        "content": row["content"],
        "created_at": row["created_at"],
        "seq_in_session": row["seq_in_session"],
    }


async def _insert_queue_records(ctx: PeopleContext, conn: Any, queue_records: list[dict[str, Any]]) -> None:
    """Insert a key-sorted batch of queue records on the caller's connection."""
    # Sorted by work_unit_key so concurrent enqueues take their locks in one canonical order.
    records = sorted(queue_records, key=lambda record: record["work_unit_key"])
    await conn.executemany(
        f"INSERT INTO {PeopleContext.table('queue')} (bank_id, session_id, work_unit_key, task_type, payload, message_id) "
        "VALUES ($1, $2, $3, $4, $5::jsonb, $6)",
        [
            (
                record["bank_id"],
                record["session_id"],
                record["work_unit_key"],
                record["task_type"],
                to_jsonb(record["payload"]),
                record["message_id"],
            )
            for record in records
        ],
    )


async def get_peers_with_configuration(
    ctx: PeopleContext, conn: Any, bank_id: str, session_name: str
) -> dict[str, list[Any]]:
    """
    Retrieve the people in a session with their configurations.

    Args:
        ctx: The people context
        conn: An open connection
        bank_id: The bank
        session_name: Name of the session

    Returns:
        Dictionary mapping person handles to [peer configuration, session peer configuration, is_active]
    """
    rows = await conn.fetch(
        f"""
        SELECT sp.peer_name, p.configuration AS peer_configuration,
               sp.configuration AS session_peer_configuration, (sp.left_at IS NULL) AS is_active
        FROM {PeopleContext.table("session_peers")} sp
        JOIN {PeopleContext.table("peers")} p ON p.bank_id = sp.bank_id AND p.name = sp.peer_name
        WHERE sp.bank_id = $1 AND sp.session_name = $2
        """,
        bank_id,
        session_name,
    )
    return {
        row["peer_name"]: [
            from_jsonb(row["peer_configuration"], {}),
            from_jsonb(row["session_peer_configuration"], {}),
            row["is_active"],
        ]
        for row in rows
    }


def create_representation_record(
    message: dict[str, Any],
    conf: ResolvedConfiguration,
    session_id: str | None = None,
    *,
    observers: list[str],
    observed: str,
) -> dict[str, Any]:
    """
    Create a queue record for representation task.

    Args:
        message: The message payload
        conf: Resolved configuration for this particular message
        session_id: Optional session ID
        observers: List of observer person handles
        observed: Handle of the sender

    Returns:
        Queue record dictionary with bank_id and message_id as separate fields
    """
    bank_id = message.get("bank_id")
    message_id = message.get("message_id")

    if not isinstance(bank_id, str):
        raise TypeError("bank_id is required and must be a string")
    if not isinstance(message_id, int):
        raise TypeError("message_id is required and must be an integer")

    processed_payload: dict[str, Any] = create_payload(
        message=message,
        configuration=conf,
        task_type="representation",
        observers=observers,
        observed=observed,
    )
    return {
        "work_unit_key": construct_work_unit_key(bank_id, processed_payload),
        "payload": processed_payload,
        "session_id": session_id,
        "task_type": "representation",
        "bank_id": bank_id,
        "message_id": message_id,
    }


def get_effective_observe_me(observed: str, peers_with_configuration: dict[str, list[Any]]) -> bool:
    """
    Determine the effective observe_me setting for a sender, considering session and peer configurations.

    Args:
        observed: Handle of the sender
        peers_with_configuration: Dictionary of peer configurations

    Returns:
        True if observe_me is enabled, False otherwise
    """
    # If the sender is not in peers_with_configuration, they left after sending a message.
    # We'll use the default behavior of observing the sender by instantiating the default
    # peer-level and session-level configs.
    configuration: list[Any] = peers_with_configuration.get(observed, [{}, {}])
    sender_session_peer_config = SessionPeerConfig(**configuration[1]) if configuration[1] else None
    sender_peer_config = PeerConfig(**configuration[0]) if configuration[0] else PeerConfig()

    # Session peer config takes precedence if it exists and has observe_me set
    if sender_session_peer_config and sender_session_peer_config.observe_me is not None:
        return sender_session_peer_config.observe_me

    # Otherwise use peer config
    return sender_peer_config.observe_me if sender_peer_config.observe_me is not None else True


def generate_queue_records(
    message: dict[str, Any],
    peers_with_configuration: dict[str, list[Any]],
    session_id: str,
    conf: ResolvedConfiguration,
) -> list[dict[str, Any]]:
    """
    Process a single message and generate queue records based on configurations.

    Args:
        message: The message payload
        peers_with_configuration: Dictionary of peer configurations
        session_id: Session ID
        conf: Resolved configuration for this particular message

    Returns:
        List of queue records for this message
    """
    observed = message["peer_name"]
    message_id: int = message["message_id"]

    records: list[dict[str, Any]] = []

    # Check if the sender should be observed based on peer configuration
    should_observe = get_effective_observe_me(observed, peers_with_configuration)

    if not conf.reasoning.enabled:
        return records

    # Collect all observers into a single list
    observers: list[str] = []

    if should_observe:
        # Self-observation: the sender observes themselves
        observers.append(observed)

        # Other people who want to observe
        for peer_name, peer_conf in peers_with_configuration.items():
            if peer_name == observed:
                continue

            # If the observer has left the session, skip them
            if not peer_conf[2]:
                continue

            session_peer_config = SessionPeerConfig(**peer_conf[1]) if peer_conf[1] else None

            if session_peer_config is None or not session_peer_config.observe_others:
                continue

            observers.append(peer_name)

    # Create a single record with all observers (if any)
    if observers:
        records.append(
            create_representation_record(
                message,
                conf,
                observed=observed,
                observers=observers,
                session_id=session_id,
            )
        )

    logger.debug(
        "message %s from %s created %s queue items with %s observers",
        message_id,
        observed,
        len(records),
        len(observers),
    )

    return records


def create_dream_record(
    bank_id: str,
    *,
    observer: str,
    observed: str,
    dream_type: DreamType,
    session_name: str | None = None,
    trigger_reason: str | None = None,
    delay_reason: str | None = None,
    documents_since_last_dream_at_schedule: int | None = None,
    document_threshold: int | None = None,
    rebuild: bool = False,
) -> dict[str, Any]:
    """
    Create a queue record for a dream task.

    Args:
        bank_id: The bank
        observer: Handle of the observer
        observed: Handle of the observed person
        dream_type: Type of dream to execute
        session_name: Name of the session to scope the dream to if specified
        trigger_reason: what tripped the schedule
        delay_reason: what governed when it fires
        documents_since_last_dream_at_schedule: count snapshot at schedule time
        document_threshold: DOCUMENT_THRESHOLD snapshot at schedule time
        rebuild: card_refresh only - rebuild the card without the prior card

    Returns:
        Queue record dictionary with bank_id and other fields
    """
    dream_payload = DreamPayload(
        dream_type=dream_type,
        observer=observer,
        observed=observed,
        session_name=session_name,
        trigger_reason=trigger_reason,
        delay_reason=delay_reason,
        documents_since_last_dream_at_schedule=documents_since_last_dream_at_schedule,
        document_threshold=document_threshold,
        rebuild=rebuild,
    ).model_dump(mode="json", exclude_none=True)

    return {
        "work_unit_key": construct_work_unit_key(bank_id, dream_payload),
        "payload": dream_payload,
        "session_id": None,
        "task_type": "dream",
        "bank_id": bank_id,
        "message_id": None,
    }


async def enqueue_dream(
    ctx: PeopleContext,
    bank_id: str,
    observer: str,
    observed: str,
    dream_type: DreamType,
    session_name: str | None = None,
    trigger_reason: str | None = None,
    delay_reason: str | None = None,
    documents_since_last_dream_at_schedule: int | None = None,
    document_threshold: int | None = None,
    rebuild: bool = False,
) -> None:
    """
    Enqueue a dream task for immediate processing by the deriver.

    Does not touch the dream bookkeeping: both guard fields are written atomically in process_dream
    on successful completion.

    Deduplication: If a dream with the same work_unit_key is already in-progress
    (has an active queue session) or pending in the queue, the enqueue is skipped.
    The work unit key includes the dream type, so e.g. a card_refresh dream
    never collides with a pending omni dream for the same pair.

    Args:
        ctx: The people context
        bank_id: The bank
        observer: Handle of the observer
        observed: Handle of the observed person
        dream_type: Type of dream to execute
        session_name: Name of the session to scope the dream to if specified
        rebuild: card_refresh only - rebuild the card without the prior card
    """
    queue = PeopleContext.table("queue")
    claims = PeopleContext.table("active_queue_sessions")
    try:
        dream_record = create_dream_record(
            bank_id,
            observer=observer,
            observed=observed,
            dream_type=dream_type,
            session_name=session_name,
            trigger_reason=trigger_reason,
            delay_reason=delay_reason,
            documents_since_last_dream_at_schedule=documents_since_last_dream_at_schedule,
            document_threshold=document_threshold,
            rebuild=rebuild,
        )
        work_unit_key = dream_record["work_unit_key"]

        async with _ENQUEUE_DB_SLOTS, ctx.conn() as conn:
            is_in_progress = await conn.fetchval(
                f"SELECT EXISTS (SELECT 1 FROM {claims} WHERE bank_id = $1 AND work_unit_key = $2)",
                bank_id,
                work_unit_key,
            )
            if is_in_progress:
                logger.debug(
                    "Skipping dream enqueue - already in progress: %s/%s/%s (type: %s)",
                    bank_id,
                    observer,
                    observed,
                    dream_type.value,
                )
                return

            is_pending = await conn.fetchval(
                f"SELECT EXISTS (SELECT 1 FROM {queue} WHERE bank_id = $1 AND work_unit_key = $2 AND NOT processed)",
                bank_id,
                work_unit_key,
            )
            if is_pending:
                logger.debug(
                    "Dream already pending in queue: %s/%s/%s (type: %s)",
                    bank_id,
                    observer,
                    observed,
                    dream_type.value,
                )
                return

            await _insert_queue_records(ctx, conn, [dream_record])

        logger.info(
            "Enqueued dream task for %s/%s/%s (type: %s)",
            bank_id,
            observer,
            observed,
            dream_type.value,
        )

    except asyncpg.UniqueViolationError:
        logger.debug(
            "Dream already enqueued by another process: %s/%s/%s (type: %s)",
            bank_id,
            observer,
            observed,
            dream_type.value,
        )
    except Exception:
        logger.exception("Failed to enqueue dream task!")
        raise
