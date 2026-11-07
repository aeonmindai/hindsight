# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/deriver/consumer.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
import logging

from pydantic import ValidationError

from ..context import PeopleContext
from ..dreamer.payload import DreamPayload
from ..schemas import ResolvedConfiguration
from .deriver import process_representation_tasks_batch
from .models import Message, QueueItem
from .queue_payload import RETRY_ATTEMPTS_PAYLOAD_KEY

logger = logging.getLogger(__name__)


async def process_item(ctx: PeopleContext, queue_item: QueueItem) -> None:
    """Process a single item from the queue (a dream; representation items go through batches)."""
    task_type = queue_item.task_type
    # Drop the work-unit retry counter before payload validation: every payload
    # model sets extra="forbid", so leaving it in burns the item as
    # extra_forbidden on the reclaim that was supposed to retry it.
    queue_payload = dict(queue_item.payload or {})
    queue_payload.pop(RETRY_ATTEMPTS_PAYLOAD_KEY, None)
    bank_id = queue_item.bank_id

    if not bank_id:
        raise ValueError(f"Queue item {queue_item.id} has no bank_id")

    if task_type == "dream":
        # Imported here: the orchestrator imports the deriver's enqueue module.
        from ..dreamer.orchestrator import process_dream

        try:
            validated = DreamPayload(**queue_payload)
        except ValidationError as e:
            logger.error("Invalid dream payload received: %s. Payload: %s", str(e), queue_payload)
            raise ValueError(f"Invalid payload structure: {str(e)}") from e
        await process_dream(ctx, validated, bank_id, queue_item_id=queue_item.id)

    else:
        raise ValueError(f"Invalid task type: {task_type}")


async def process_representation_batch(
    ctx: PeopleContext,
    bank_id: str,
    messages: list[Message],
    message_level_configuration: ResolvedConfiguration | None,
    *,
    observers: list[str] | None,
    observed: str | None,
    queue_item_message_ids: list[int],
    session_id: str | None = None,
    queue_item_ids: list[int] | None = None,
    hit_batch_token_cap: bool = False,
    was_flush_enabled: bool = False,
    batch_max_tokens: int = 0,
) -> None:
    """
    Prepares and processes a batch of messages for representation tasks.

    Args:
        ctx: The people context
        bank_id: The bank the messages belong to
        messages: List of messages to process
        message_level_configuration: Resolved configuration for this batch
        observers: List of observers for the messages
        observed: The observed of the messages
        queue_item_message_ids: Message IDs from queue items
        session_id: Canonical session id from the queue.
        queue_item_ids: Queue rows that triggered this batch, when available.
        hit_batch_token_cap: whether the queue batcher clamped this batch to fit
        was_flush_enabled: snapshot of DERIVER.FLUSH_ENABLED at fetch time
        batch_max_tokens: DERIVER.REPRESENTATION_BATCH_TARGET_INPUT_TOKENS snapshot
    """
    if not messages or not messages[0]:
        logger.debug("process_representation_batch received no messages")
        return

    if observed is None or observers is None or len(observers) == 0:
        raise ValueError("observed and observers are required for representation tasks")

    await process_representation_tasks_batch(
        ctx,
        bank_id,
        messages,
        message_level_configuration,
        observers=observers,
        observed=observed,
        queue_item_message_ids=queue_item_message_ids,
        session_id=session_id,
        queue_item_ids=queue_item_ids,
        hit_batch_token_cap=hit_batch_token_cap,
        was_flush_enabled=was_flush_enabled,
        batch_max_tokens=batch_max_tokens,
    )
