# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/utils/formatting.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""
Formatting helpers the dialectic needs: pattern escaping, turns with timestamps, custom instructions.
"""

from datetime import datetime

ILIKE_ESCAPE_CHAR = "\\"


def escape_ilike_pattern(text: str) -> str:
    """
    Escape SQL ILIKE/LIKE pattern special characters in user-provided text.

    SQL LIKE/ILIKE patterns treat '%' as "match any sequence" and '_' as
    "match any single character". Without escaping, a user searching for
    "100%" would match "100" followed by anything, not the literal "100%".

    This function escapes these wildcards so user input is treated literally.
    The escape character itself (backslash) is also escaped.

    Args:
        text: User-provided search text that may contain %, _, or backslash

    Returns:
        Escaped text safe for use in ILIKE patterns. Use with ``ESCAPE '\\'``.
    """
    return (
        text.replace(ILIKE_ESCAPE_CHAR, ILIKE_ESCAPE_CHAR + ILIKE_ESCAPE_CHAR)
        .replace("%", ILIKE_ESCAPE_CHAR + "%")
        .replace("_", ILIKE_ESCAPE_CHAR + "_")
    )


def format_new_turn_with_timestamp(new_turn: str, current_time: datetime, speaker: str) -> str:
    """
    Format new turn message with timestamp.

    Args:
        new_turn: The message content
        current_time: Message timestamp
        speaker: The speaker's name

    Returns:
        Formatted string like "2023-05-08 13:56:00 speaker: hello"
    """
    current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    return f"{current_time_str} {speaker}: {new_turn}"


def custom_instructions_section(
    custom_instructions: str | None,
    *,
    heading: str = "CUSTOM INSTRUCTIONS:",
    note: str | None = None,
) -> str:
    """
    Render operator-configured custom instructions as a prompt section.

    Args:
        custom_instructions: Resolved instructions; blank or None renders nothing
        heading: Section heading line
        note: Optional line between the heading and the instructions

    Returns:
        The rendered section, or an empty string when there are no instructions
    """
    normalized = (custom_instructions or "").strip()
    if not normalized:
        return ""
    lines = [heading, note, normalized] if note else [heading, normalized]
    return "\n".join(lines)
