# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/utils/queue_payload.py
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/utils/work_unit.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""Dream queue payload and work-unit key.

The deriver's queue stores a dream as a ``people_queue`` row with ``task_type = 'dream'``: its payload
is a ``DreamPayload`` and its work-unit key is ``dream_work_unit_key`` (Honcho's
``dream:<dream_type>:<workspace>:<observer>:<observed>`` with the bank in place of the workspace).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from ..schemas import DreamType


class DreamPayload(BaseModel):
    """Payload for dream tasks."""

    model_config = ConfigDict(extra="forbid")

    task_type: Literal["dream"] = "dream"
    dream_type: DreamType
    observer: str
    observed: str
    session_name: str | None = None
    # Scheduling context captured at schedule time, so the dream can be attributed
    # back to why it was scheduled (which threshold tripped) and what governed when
    # it fired (idle delay vs. immediate vs. min-hours gate).
    trigger_reason: str | None = None
    delay_reason: str | None = None
    documents_since_last_dream_at_schedule: int | None = None
    document_threshold: int | None = None
    # card_refresh only: when True the existing peer card is NOT injected into
    # the specialist prompt and the card is rebuilt solely from observations
    # currently in the collection (used after removals, where the old card may
    # contain facts whose support was deleted).
    rebuild: bool = False


def _escape(part: str) -> str:
    """Bank ids and person handles are free text: ``%`` and ``:`` are escaped so a key splits cleanly."""
    return part.replace("%", "%25").replace(":", "%3A")


def _unescape(part: str) -> str:
    return part.replace("%3A", ":").replace("%25", "%")


def dream_work_unit_key(bank_id: str, observer: str, observed: str, dream_type: DreamType | str) -> str:
    """The work-unit key of a dream over one (observer, observed) pair."""
    kind = dream_type.value if isinstance(dream_type, DreamType) else dream_type
    return f"dream:{kind}:{_escape(bank_id)}:{_escape(observer)}:{_escape(observed)}"


def parse_dream_work_unit_key(work_unit_key: str) -> tuple[str, str, str, str] | None:
    """(dream_type, bank_id, observer, observed) of a dream key, or None for any other key.

    Parts are unescaped (see ``dream_work_unit_key``).
    """
    parts = work_unit_key.split(":")
    if len(parts) != 5 or parts[0] != "dream":
        return None
    return parts[1], _unescape(parts[2]), _unescape(parts[3]), _unescape(parts[4])
