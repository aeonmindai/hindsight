# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/utils/queue_payload.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""Queue payload for representation tasks.

The dream payload lives with the dreamer (``people.dreamer.payload``). Honcho's summary, webhook,
deletion, scope and reconciler payloads are not copied: the fork has no such tasks.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from ..schemas import ResolvedConfiguration

# Queue mechanics, not task data: the deriver stores a per-work-unit transient
# failure count under this key so a retry budget survives work-unit reclaim.
# Every payload model below forbids extras, so anything that reads a raw
# queue payload must strip this key before validating. Lives here rather
# than in the deriver because both the writer (queue_manager) and the stripper
# (consumer) need it, and queue_manager imports consumer.
RETRY_ATTEMPTS_PAYLOAD_KEY = "_retry_attempts"


class BasePayload(BaseModel):
    """Base payload with common fields."""

    model_config = ConfigDict(extra="forbid")


class RepresentationPayload(BasePayload):
    """Payload for representation tasks."""

    task_type: Literal["representation"] = "representation"
    session_name: str
    content: str
    observers: list[str]
    observed: str
    created_at: datetime
    configuration: ResolvedConfiguration


class RepresentationPayloads(BasePayload):
    """Payload for a batch of representation tasks."""

    payloads: list[RepresentationPayload]


def create_payload(
    message: dict[str, Any],
    configuration: ResolvedConfiguration,
    task_type: Literal["representation"] = "representation",
    *,
    observers: list[str] | None = None,
    observed: str | None = None,
) -> dict[str, Any]:
    """
    Create a processed payload from a message for queue processing.

    Note: bank_id and message_id are not included in the returned payload as they are stored in
    dedicated columns on the queue table. The caller is responsible for extracting and passing these
    values separately.

    Args:
        message: The original message dictionary
        task_type: Type of task (only 'representation' exists in the fork)
        observers: List of observer person handles (required for representation tasks)
        observed: Handle of the observed person (*always* the person who sent the message) (required)

    Returns:
        Processed payload dictionary ready for queue processing (without bank_id and message_id)

    Raises:
        ValueError: If the payload doesn't match the expected schema
    """
    bank_id = message.get("bank_id")
    session_name = message.get("session_name")
    message_id = message.get("message_id")

    if not isinstance(bank_id, str):
        raise TypeError("Bank id must be a string")

    if not isinstance(session_name, str):
        raise TypeError("Session name must be a string")

    if not isinstance(message_id, int):
        raise TypeError("Message ID must be an integer")

    try:
        if task_type != "representation":
            raise ValueError(f"Unsupported task type: {task_type}")

        content = message.get("content")
        created_at = message.get("created_at")

        if not isinstance(content, str):
            raise TypeError("Message content must be a string")

        if not isinstance(created_at, datetime):
            raise TypeError("created_at must be a datetime object")

        if observers is None or len(observers) == 0:
            raise ValueError("observers is required for representation tasks")

        if observed is None:
            raise ValueError("observed is required for representation tasks")

        validated_payload = RepresentationPayload(
            content=content,
            session_name=session_name,
            created_at=created_at,
            observers=observers,
            observed=observed,
            configuration=configuration,
        )

        # mode='json' ensures datetime is converted to an ISO string
        payload = validated_payload.model_dump(mode="json", exclude_none=True)

    except Exception as e:
        raise ValueError(f"Failed to create valid payload: {str(e)}") from e

    return payload
