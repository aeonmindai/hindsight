# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/deriver/deriver.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
import logging
import time
from dataclasses import dataclass
from typing import Any

from ..context import PeopleContext
from ..representation import RepresentationManager
from ..representation_utils import PromptRepresentation, Representation
from ..schemas import ResolvedConfiguration, get_configuration
from .models import Message
from .prompts import (
    estimate_deriver_prompt_tokens,
    estimate_tokens,
    format_deriver_message,
    minimal_deriver_prompt,
)
from .retryable_errors import is_retryable_error
from .settings import DERIVER

logger = logging.getLogger(__name__)

# Attempts of the deriver call when the answer comes back empty in a way that looks degraded.
_DEGRADED_RETRY_ATTEMPTS = 3


class EmptyRepresentationError(Exception):
    """The deriver's structured response for a batch was empty and degraded."""

    def __init__(self, message: str, *, parse_class: str, output_tokens: int = 0) -> None:
        super().__init__(message)
        self.parse_class = parse_class
        self.output_tokens = output_tokens

    @classmethod
    def for_prompt(cls, prompt: str, *, parse_class: str, output_tokens: int = 0) -> "EmptyRepresentationError":
        return cls(
            f"Deriver generated an empty representation ({parse_class}, output_tokens={output_tokens}, "
            f"prompt_tokens={estimate_tokens(prompt)})",
            parse_class=parse_class,
            output_tokens=output_tokens,
        )


class RepresentationSaveError(Exception):
    """Raised when every observer's representation save fails in a batch."""


@dataclass
class _SaveCounts:
    """Sum of the per-observer save results (duplicates the save skipped or replaced)."""

    exact_dup_existing_count: int = 0
    exact_dup_in_batch_count: int = 0
    semantic_dup_rejected_count: int = 0
    semantic_dup_replaced_count: int = 0

    def add(self, result: Any) -> None:
        for name in (
            "exact_dup_existing_count",
            "exact_dup_in_batch_count",
            "semantic_dup_rejected_count",
            "semantic_dup_replaced_count",
        ):
            setattr(self, name, getattr(self, name) + int(getattr(result, name, 0) or 0))


async def _session_id(ctx: PeopleContext, bank_id: str, session_name: str) -> str | None:
    async with ctx.conn() as conn:
        return await conn.fetchval(
            f"SELECT id FROM {PeopleContext.table('sessions')} WHERE bank_id = $1 AND name = $2",
            bank_id,
            session_name,
        )


async def process_representation_tasks_batch(
    ctx: PeopleContext,
    bank_id: str,
    messages: list[Message],
    message_level_configuration: ResolvedConfiguration | None,
    *,
    observers: list[str],
    observed: str,
    queue_item_message_ids: list[int],
    session_id: str | None = None,
    queue_item_ids: list[int] | None = None,
    hit_batch_token_cap: bool = False,
    was_flush_enabled: bool = False,
    batch_max_tokens: int = 0,
) -> None:
    """
    Process messages with minimal overhead - single LLM call, save to multiple observers.

    Args:
        ctx: The people context (engine and request context).
        bank_id: The bank the messages belong to.
        messages: List of messages to process (includes interleaving context).
        message_level_configuration: Optional configuration override.
        observers: List of observer person handles (whose conclusions to save to).
        observed: The observed person handle.
        queue_item_message_ids: Message IDs from queue items being processed
        session_id: Canonical session id from the queue, resolved if not provided.
        queue_item_ids: Queue rows that triggered this batch, when available.
        hit_batch_token_cap: queue batcher clamped this batch to fit
        was_flush_enabled: DERIVER.FLUSH_ENABLED snapshot at batch time
        batch_max_tokens: DERIVER.REPRESENTATION_BATCH_TARGET_INPUT_TOKENS snapshot
    """
    if not messages:
        return

    overall_start = time.perf_counter()

    messages.sort(key=lambda x: x.id)
    latest_message = messages[-1]
    earliest_message = messages[0]

    # Reuse the queue's canonical session ID, resolving it for direct/legacy callers.
    if message_level_configuration is None or session_id is None:
        if session_id is None:
            session_id = await _session_id(ctx, bank_id, latest_message.session_name)
        if message_level_configuration is None:
            message_level_configuration = get_configuration(None, None, None)

    # Skip if disabled
    if message_level_configuration.reasoning.enabled is False:
        return

    custom_instructions = message_level_configuration.reasoning.custom_instructions

    formatted_messages = "\n".join(
        format_deriver_message(idx, msg.peer_name, observed, msg.spoken_at, msg.content)
        for idx, msg in enumerate(messages)
    )
    prompt_message_ids = [msg.public_id for msg in messages]

    # Token usage: count only tokens from messages being processed
    prompt_tokens = estimate_deriver_prompt_tokens(custom_instructions)
    queue_item_message_ids_set = set(queue_item_message_ids)
    messages_tokens = sum(msg.token_count for msg in messages if msg.id in queue_item_message_ids_set)

    # Build prompt
    prompt = minimal_deriver_prompt(
        peer_id=observed,
        messages=formatted_messages,
        custom_instructions=custom_instructions,
    )

    context_prep_duration = (time.perf_counter() - overall_start) * 1000

    max_tokens = DERIVER.MAX_OUTPUT_TOKENS

    message_ids = [m.id for m in messages if m.peer_name == observed]

    def refuse_degraded_empty(content: Any, output_tokens: int) -> None:
        """Raise so a truncated or filtered empty answer consumes an attempt."""
        if not message_ids:
            return
        if not isinstance(content, PromptRepresentation):
            # A provider refusal comes back as its text, not the model.
            parse_class = "content_filtered"
        elif content.explicit:
            return
        elif output_tokens >= max_tokens:
            # The output budget ran out right after the opening of the list.
            parse_class = "truncated"
        else:
            return
        raise EmptyRepresentationError.for_prompt(prompt, parse_class=parse_class, output_tokens=output_tokens)

    # Single LLM call (retried when the answer is empty in a degraded way)
    llm_start = time.perf_counter()
    response: Any = None
    for attempt in range(_DEGRADED_RETRY_ATTEMPTS):
        response = await ctx.llm_background.call(
            messages=[{"role": "user", "content": prompt}],
            response_format=PromptRepresentation,
            max_completion_tokens=max_tokens,
            scope="people_deriver",
        )
        try:
            refuse_degraded_empty(response.content, response.usage.output_tokens)
            break
        except EmptyRepresentationError:
            if attempt == _DEGRADED_RETRY_ATTEMPTS - 1:
                raise
            logger.warning(
                "Deriver answer was empty and degraded; retrying (%d/%d)", attempt + 1, _DEGRADED_RETRY_ATTEMPTS
            )
    llm_duration = (time.perf_counter() - llm_start) * 1000
    output_tokens = response.usage.output_tokens

    # Convert to Representation and save
    observations = Representation.from_prompt_representation(
        response.content,
        message_ids,
        prompt_message_ids,
        latest_message.session_name,
        latest_message.spoken_at,
    )

    agg_representation_result = _SaveCounts()
    successful_observer_count = 0
    save_errors: list[tuple[str, Exception]] = []
    if observations.is_empty() or not message_ids:
        logger.warning(
            "Deriver generated zero observations for messages %s:%s in %s/%s! (output_tokens=%d)",
            earliest_message.id,
            latest_message.id,
            bank_id,
            latest_message.session_name,
            output_tokens,
        )
    else:
        # Save to all observers' conclusions
        for observer in observers:
            representation_manager = RepresentationManager(ctx, bank_id, observer=observer, observed=observed)

            try:
                representation_result = await representation_manager.save_representation(
                    observations,
                    message_ids,
                    latest_message.session_name,
                    latest_message.spoken_at,
                    message_level_configuration,
                )
                agg_representation_result.add(representation_result)
                successful_observer_count += 1
            except Exception as e:  # noqa: BLE001
                logger.exception("Failed to save representation for observer %s", observer)
                save_errors.append((observer, e))

    overall_duration = (time.perf_counter() - overall_start) * 1000
    total_observations = len(observations.explicit) + len(observations.deductive)
    logger.debug(
        "minimal_deriver %s/%s observed=%s: %d observations from %d messages (%d queued, %d prompt tokens, "
        "%d message tokens, %d output tokens) prep=%.0fms llm=%.0fms total=%.0fms observers=%d dups=%s "
        "queue_items=%s cap_hit=%s flush=%s batch_max_tokens=%s",
        bank_id,
        latest_message.session_name,
        observed,
        total_observations,
        len(messages),
        len(queue_item_message_ids),
        prompt_tokens,
        messages_tokens,
        output_tokens,
        context_prep_duration,
        llm_duration,
        overall_duration,
        successful_observer_count,
        agg_representation_result,
        queue_item_ids,
        hit_batch_token_cap,
        was_flush_enabled,
        batch_max_tokens,
    )

    retryable = next((exc for _, exc in save_errors if is_retryable_error(exc)), None)
    if retryable is not None:
        raise retryable
    if save_errors and successful_observer_count == 0:
        details = "; ".join(f"{observer}: {exc.__class__.__name__}: {exc}" for observer, exc in save_errors)
        raise RepresentationSaveError(
            f"save_representation failed for all {len(save_errors)} observer(s): " + details
        ) from save_errors[0][1]
