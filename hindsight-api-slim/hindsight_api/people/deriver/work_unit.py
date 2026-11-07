# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/utils/work_unit.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""Work unit utility functions for generating and parsing work unit keys.

A work unit is the set of queue items one worker drains in order. Keys keep Honcho's shape
(``representation:{bank}:{session}:{observed}``; a dream is ``dream:{type}:{bank}:{observer}:{observed}``,
built by the dreamer). Honcho restricts names to a character set that excludes ``:``; bank ids, session
names and person handles here are free text, so a key escapes ``%`` and ``:`` inside each part (``%25``,
``%3A``) and the parser undoes it. Names without those characters are unchanged.
"""

from typing import Any

from pydantic import BaseModel

from ..dreamer.payload import dream_work_unit_key, parse_dream_work_unit_key

# The task types that lead a work_unit_key.
TASK_TYPES: frozenset[str] = frozenset({"representation", "dream"})


class ParsedWorkUnit(BaseModel):
    """Parsed work unit components."""

    task_type: str
    bank_id: str | None
    session_name: str | None
    observer: str | None
    observed: str | None
    dream_type: str | None = None


def _escape(part: str) -> str:
    return part.replace("%", "%25").replace(":", "%3A")


def _unescape(part: str) -> str:
    return part.replace("%3A", ":").replace("%25", "%")


def construct_work_unit_key(bank_id: str, payload: dict[str, Any] | ParsedWorkUnit) -> str:
    """
    Generate a work unit key for a given task type, bank, and payload.

    Args:
        bank_id: The bank the work unit belongs to
        payload: Dictionary containing work unit information

    Returns:
        Formatted work unit key string

    Raises:
        ValueError: If required fields are missing or task type is invalid
    """
    if isinstance(payload, ParsedWorkUnit):
        payload = payload.model_dump()

    task_type: str | None = payload.get("task_type")
    if not bank_id or not task_type:
        raise ValueError("bank_id and task_type are required to generate a work_unit_key")

    if task_type == "dream":
        dream_type = payload.get("dream_type")
        if not dream_type:
            raise ValueError("dream_type is required for dream tasks")
        return dream_work_unit_key(
            bank_id, payload.get("observer", "None"), payload.get("observed", "None"), dream_type
        )

    if task_type == "representation":
        # Representation tasks don't include the observer in the key since we process once and save to
        # multiple observers' conclusions.
        observed = payload.get("observed", "None")
        session_name = payload.get("session_name", "None")
        return f"{task_type}:{_escape(bank_id)}:{_escape(session_name)}:{_escape(observed)}"

    raise ValueError(f"Invalid task type: {task_type}")


def parse_work_unit_key(work_unit_key: str) -> ParsedWorkUnit:
    """
    Parse a work unit key to extract its components.

    Args:
        work_unit_key: The work unit key string to parse

    Returns:
        ParsedWorkUnit with extracted components

    Raises:
        ValueError: If the work unit key format is invalid
    """
    parts = work_unit_key.split(":")
    task_type = parts[0]

    if task_type == "representation":
        if len(parts) != 4:
            raise ValueError(f"Invalid work_unit_key format for task_type {task_type}: {work_unit_key}")
        return ParsedWorkUnit(
            task_type=task_type,
            bank_id=_unescape(parts[1]),
            session_name=_unescape(parts[2]),
            observer=None,
            observed=_unescape(parts[3]),
        )

    if task_type == "dream":
        parsed_dream = parse_dream_work_unit_key(work_unit_key)
        if parsed_dream is None:
            raise ValueError(f"Invalid work_unit_key format for task_type {task_type}: {work_unit_key}")
        dream_type, bank_id, observer, observed = parsed_dream
        return ParsedWorkUnit(
            task_type=task_type,
            bank_id=bank_id,
            session_name=None,
            observer=observer,
            observed=observed,
            dream_type=dream_type,
        )

    raise ValueError(f"Invalid task type in work_unit_key: {task_type}")
