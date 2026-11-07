"""The people feature inside the memory engine: background workers and the retain hook.

``HINDSIGHT_API_PEOPLE_ENABLED=true`` switches it on. The engine then

* starts the deriver's queue manager and the dream scheduler when it initialises and stops them when it
  closes (:class:`PeopleRuntime`);
* hands every retained item that says who said it (``said_by``) to the deriver after the facts are
  stored (:func:`after_retain`), so each speaker becomes a peer and what they said is read for what it
  tells about them;
* gives every peer a ``person:<handle>`` mental model after each consolidation (see
  :mod:`hindsight_api.people.person_models`).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from .context import PeopleContext

if TYPE_CHECKING:
    from ..engine.memory_engine import MemoryEngine

logger = logging.getLogger(__name__)


class PeopleRuntime:
    """Owns the deriver queue manager and the dream scheduler of one engine."""

    def __init__(self, engine: "MemoryEngine") -> None:
        self._engine = engine
        self._queue: Any = None
        self._scheduler: Any = None

    def _ctx(self) -> PeopleContext:
        return PeopleContext.internal(self._engine)

    async def start(self) -> None:
        from .deriver.queue_manager import QueueManager
        from .dreamer.dream_scheduler import DreamScheduler

        self._queue = QueueManager(self._ctx)
        await self._queue.start()
        self._scheduler = DreamScheduler(self._ctx)
        await self._scheduler.start()
        logger.info("People started: deriver queue and dream scheduler")

    async def stop(self) -> None:
        for worker in (self._scheduler, self._queue):
            if worker is not None:
                try:
                    await worker.stop()
                except Exception:
                    logger.warning("People worker did not stop cleanly", exc_info=True)
        self._scheduler = self._queue = None


def _parse_time(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def incoming_messages(contents: list[dict[str, Any]]) -> list[Any]:
    """The retained items that say who said them, as the deriver's ``IncomingMessage`` rows."""
    from .deriver.enqueue import IncomingMessage

    messages = []
    for item in contents:
        metadata = item.get("metadata") or {}
        speaker = metadata.get("said_by")
        text = item.get("content")
        if not speaker or not isinstance(text, str) or not text.strip():
            continue
        channel = metadata.get("channel")
        document_id = item.get("document_id")
        session = metadata.get("session") or (f"{channel}:{document_id}" if channel and document_id else None)
        messages.append(
            IncomingMessage(
                peer=speaker,
                content=text,
                session=session or document_id or channel or "retain",
                said_at=_parse_time(metadata.get("said_at")) or _parse_time(item.get("event_date")),
                channel=channel,
                source_url=metadata.get("source_url"),
                document_id=document_id,
            )
        )
    return messages


async def after_retain(engine: "MemoryEngine", bank_id: str, contents: list[dict[str, Any]]) -> None:
    """Queue what retained items say for the deriver. Never fails the retain that called it."""
    messages = incoming_messages(contents)
    if not messages:
        return
    try:
        from .deriver.enqueue import enqueue_messages

        await enqueue_messages(PeopleContext.internal(engine), bank_id, messages)
    except Exception:
        logger.warning(
            "[PEOPLE] could not queue %d retained messages in bank %s", len(messages), bank_id, exc_info=True
        )
