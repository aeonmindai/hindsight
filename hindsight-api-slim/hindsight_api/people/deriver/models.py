"""Row shapes the deriver passes around.

Honcho hands SQLAlchemy ORM rows (``QueueItem``, ``Message``) from the queue manager to the deriver.
The fork reads plain asyncpg records, so these dataclasses carry the same attributes under the same
names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..jsonb import from_jsonb


@dataclass
class QueueItem:
    """One ``people_queue`` row."""

    id: int
    bank_id: str
    work_unit_key: str
    task_type: str
    payload: dict[str, Any]
    processed: bool = False
    error: str | None = None
    created_at: datetime | None = None
    session_id: str | None = None
    message_id: int | None = None

    @classmethod
    def from_record(cls, record: Any, prefix: str = "") -> QueueItem:
        """Build from an asyncpg record whose queue columns are named ``<prefix><column>``."""
        return cls(
            id=record[f"{prefix}id"],
            bank_id=record[f"{prefix}bank_id"],
            work_unit_key=record[f"{prefix}work_unit_key"],
            task_type=record[f"{prefix}task_type"],
            payload=from_jsonb(record[f"{prefix}payload"], {}),
            processed=record[f"{prefix}processed"],
            error=record[f"{prefix}error"],
            created_at=record[f"{prefix}created_at"],
            session_id=record[f"{prefix}session_id"],
            message_id=record[f"{prefix}message_id"],
        )


@dataclass
class Message:
    """One ``people_messages`` row, as the deriver reads it."""

    id: int
    bank_id: str
    public_id: str
    session_name: str
    peer_name: str
    content: str
    token_count: int
    created_at: datetime
    seq_in_session: int = 0
    said_at: datetime | None = None
    said_by: str | None = None
    source_url: str | None = None
    channel: str | None = None
    internal_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def spoken_at(self) -> datetime:
        """When the person said it, falling back to when the message was stored."""
        return self.said_at or self.created_at

    @classmethod
    def from_record(cls, record: Any) -> Message:
        return cls(
            id=record["id"],
            bank_id=record["bank_id"],
            public_id=record["public_id"],
            session_name=record["session_name"],
            peer_name=record["peer_name"],
            content=record["content"],
            token_count=record["token_count"],
            created_at=record["created_at"],
            seq_in_session=record["seq_in_session"],
            said_at=record["said_at"],
            said_by=record["said_by"],
            source_url=record["source_url"],
            channel=record["channel"],
            internal_metadata=from_jsonb(record["internal_metadata"], {}),
        )


MESSAGE_COLUMNS = (
    "m.id, m.bank_id, m.public_id, m.session_name, m.peer_name, m.content, m.token_count, m.created_at, "
    "m.seq_in_session, m.said_at, m.said_by, m.source_url, m.channel, m.internal_metadata"
)
QUEUE_COLUMNS = "id, bank_id, work_unit_key, task_type, payload, processed, error, created_at, session_id, message_id"
