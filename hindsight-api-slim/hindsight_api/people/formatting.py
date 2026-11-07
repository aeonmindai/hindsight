# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/utils/formatting.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""Datetime helpers shared by the people modules."""

from datetime import UTC, datetime


def format_datetime_utc(dt: datetime) -> str:
    """Format datetime to an ISO 8601 string with a Z suffix for UTC, without subsecond precision.

    A naive datetime is taken as UTC.

    Example:
        >>> dt = datetime(2023, 1, 1, 12, 0, 0, tzinfo=UTC)
        >>> format_datetime_utc(dt)
        '2023-01-01T12:00:00Z'
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    if dt.tzinfo != UTC:
        dt = dt.astimezone(UTC)
    dt = dt.replace(microsecond=0)
    return dt.isoformat().replace("+00:00", "Z")


def parse_datetime_iso(iso_string: str) -> datetime:
    """Parse an ISO 8601 datetime string, handling a Z suffix, timezone offsets and naive timestamps.

    Always returns a timezone-aware datetime (naive input is taken as UTC).

    Raises:
        ValueError: If the input string is invalid or contains suspicious content

    Example:
        >>> parse_datetime_iso('2023-01-01T12:00:00Z')
        datetime.datetime(2023, 1, 1, 12, 0, tzinfo=datetime.timezone.utc)
    """
    if not iso_string:
        raise ValueError("Invalid input: must be a non-empty string")

    iso_string = str(iso_string)

    # Reject strings with null bytes or suspicious characters
    if "\x00" in iso_string or "\r" in iso_string or "\n" in iso_string:
        raise ValueError("Invalid input: contains null bytes or line breaks")
    if any(ord(c) < 32 and c not in "\t" for c in iso_string):
        raise ValueError("Invalid input: contains non-printable characters")

    iso_string = iso_string.strip()
    if not iso_string:
        raise ValueError("Invalid input: empty after stripping whitespace")

    # Handle Z suffix (convert to +00:00)
    if iso_string.endswith(("Z", "z")):
        iso_string = iso_string[:-1] + "+00:00"

    try:
        result = datetime.fromisoformat(iso_string)
        if result.tzinfo is None:
            result = result.replace(tzinfo=UTC)
        return result
    except ValueError as e:
        raise ValueError(f"Invalid ISO 8601 datetime format: {e}") from e
