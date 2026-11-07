"""Source metadata and pointer fields carried by every retained item (Neo v2 contract 7).

Every retained item says where it came from and who said it:

* ``source_url``: a working link to the message, page or record the content came from;
* ``said_by``: the handle of the person (or agent) who said it;
* ``said_at``: when they said it (ISO 8601);
* ``channel``: where (``discord``, ``whatsapp``, ``slack``, ``mail``, ``linear``, ``omp``, ...).

A value that the CRM, Linear or Stripe holds is stored as a pointer, never as a copy:

* ``pointer_kind``: ``crm``, ``linear`` or ``stripe``;
* ``pointer_id``: the record's id in that system.

All of them live in the item's ``metadata`` (string to string), which retain already copies onto every
fact extracted from the item and recall already returns. This module folds the named fields into that
metadata, checks them, and gives the extraction prompt the pointer rule. The database refuses a pointer
kind outside the list and a pointer with no id (``memory_units_pointer_check``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

SOURCE_KEYS = ("source_url", "said_by", "said_at", "channel")
POINTER_KINDS = ("crm", "linear", "stripe")
POINTER_KEYS = ("pointer_kind", "pointer_id")


def _parse_said_at(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"said_at must be an ISO 8601 time, got {value!r}") from exc


def fold_source_fields(
    metadata: dict[str, str] | None,
    *,
    source_url: str | None = None,
    said_by: str | None = None,
    said_at: datetime | str | None = None,
    channel: str | None = None,
    pointer_kind: str | None = None,
    pointer_id: str | None = None,
) -> tuple[dict[str, str], datetime | None]:
    """Fold the named source and pointer fields into ``metadata``.

    Returns ``(metadata, said_at)``; ``said_at`` is parsed so the caller can default the item's time to
    it. A named field that disagrees with the same key in ``metadata`` is an error rather than a silent
    override. Raises ``ValueError`` for an unknown pointer kind or a pointer with only one half.
    """
    merged = dict(metadata or {})
    parsed_said_at = _parse_said_at(said_at)
    named: dict[str, str | None] = {
        "source_url": source_url,
        "said_by": said_by,
        "said_at": parsed_said_at.isoformat() if parsed_said_at else None,
        "channel": channel,
        "pointer_kind": pointer_kind,
        "pointer_id": pointer_id,
    }
    for key, value in named.items():
        if value is None or value == "":
            continue
        existing = merged.get(key)
        if existing is not None and existing != value:
            raise ValueError(f"{key} is given twice with different values (field {value!r}, metadata {existing!r})")
        merged[key] = str(value)

    kind, pointer = merged.get("pointer_kind"), merged.get("pointer_id")
    if kind or pointer:
        if kind not in POINTER_KINDS:
            raise ValueError(f"pointer_kind must be one of {', '.join(POINTER_KINDS)}, got {kind!r}")
        if not pointer:
            raise ValueError("pointer_id is required with pointer_kind: a pointer names the record it points to")
    if merged.get("said_at") and parsed_said_at is None:
        parsed_said_at = _parse_said_at(merged["said_at"])
    return merged, parsed_said_at


def missing_source_fields(metadata: dict[str, str] | None) -> list[str]:
    """The source fields an item does not carry (``HINDSIGHT_API_REQUIRE_SOURCE_METADATA`` refuses it)."""
    present = metadata or {}
    return [key for key in SOURCE_KEYS if not present.get(key)]


def pointer_prompt_section(metadata: dict[str, str] | None) -> str:
    """The extraction rule for an item that points at a record another system owns, or ``""``."""
    if not metadata or not metadata.get("pointer_kind"):
        return ""
    kind, pointer = metadata["pointer_kind"], metadata.get("pointer_id", "")
    return (
        f"\n\nPOINTER: this content mentions a record that {kind} owns (id {pointer}). {kind} is the source of "
        "truth for the record's current values (a deal stage, a status, an amount, a date, a plan). Do NOT "
        "store such a value as a fact: it goes stale and the copy would disagree with the record. Store only "
        f"what the person said about the record in their own words (who raised it, what they want done, what "
        f"they promised) and name the record as '{kind} record {pointer}'."
    )


def speaker_prompt_section(metadata: dict[str, str] | None) -> str:
    """The extraction rule for an item whose speaker is known, or ``""``."""
    if not metadata or not metadata.get("said_by"):
        return ""
    where = f" on {metadata['channel']}" if metadata.get("channel") else ""
    return (
        f"\n\nSPEAKER: {metadata['said_by']} said this{where}. Attribute first-person statements ('I', 'we', "
        f"'my') to {metadata['said_by']} by name, and write the facts about them so a reader who has never seen "
        "the message knows who each fact is about."
    )
