"""Person models: one mental model, ``person:<handle>``, for every person the bank knows.

After each consolidation :func:`ensure_person_models` gives every peer without one a model, with no one
asking. The model is created with ``refresh_after_consolidation`` on, so the consolidation that follows
rewrites it (a changed fact shows in it after the next consolidation). Its first content is built from
what the deriver and the dreamer already concluded about the person, each line with its source.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from .context import PeopleContext

if TYPE_CHECKING:
    from ..models import RequestContext

logger = logging.getLogger(__name__)

PERSON_MODEL_PREFIX = "person:"

#: What a person model covers (RUN-2355 item 2). Used as the model's source query, so every refresh
#: rewrites the whole picture from the bank's memories and keeps a source on each line.
PERSON_SOURCE_QUERY = (
    "Write the picture of the person {handle}, as sections with short bullet lines: "
    "Who they are; Role and company; How they work and talk; Likes and dislikes; What they are working on; "
    "Open promises (to them and from them, with the date and who owes what); Who they work with. "
    "Use only what the memories say about {handle}. Every line ends with its source in brackets: "
    "the source link (source_url) and the date it was said (said_at), as recorded in the memory's metadata. "
    "When a fact changed, state the latest one and put the earlier one under 'Earlier' with its date and source. "
    "Leave out a value that the CRM, Linear or Stripe holds; name the record (its pointer_kind and pointer_id) instead."
)


def person_model_id(handle: str) -> str:
    """The mental model id for ``handle``: ``person:<handle>`` with characters that split a URL replaced."""
    return PERSON_MODEL_PREFIX + re.sub(r"[/\x00-\x1f]", "_", handle)


def handle_of(model_id: str) -> str | None:
    """The handle a person model id names, or ``None`` for any other mental model."""
    return model_id[len(PERSON_MODEL_PREFIX) :] if model_id.startswith(PERSON_MODEL_PREFIX) else None


async def person_conclusion_lines(ctx: PeopleContext, bank_id: str, handle: str, limit: int = 200) -> list[str]:
    """What is concluded about ``handle``, one line each with its source, newest first."""
    async with ctx.conn() as conn:
        rows = await conn.fetch(
            f"""
            SELECT level, content, source_url, said_at
            FROM {ctx.table("conclusions")}
            WHERE bank_id = $1 AND observed = $2 AND deleted_at IS NULL
            ORDER BY COALESCE(said_at, created_at) DESC
            LIMIT $3
            """,
            bank_id,
            handle,
            limit,
        )
    lines = []
    for row in rows:
        source = ", ".join(
            part for part in (row["source_url"], row["said_at"].date().isoformat() if row["said_at"] else None) if part
        )
        lines.append(f"- {row['content']}" + (f" [{source}]" if source else ""))
    return lines


async def person_model_content(ctx: PeopleContext, bank_id: str, handle: str) -> str:
    """The first content of a person model: the peer card and the conclusions, each with its source."""
    from .peer_card import get_peer_card

    parts = [f"# {handle}"]
    card = await get_peer_card(ctx, bank_id, observer=handle, observed=handle)
    if card:
        parts += ["", "## Peer card", *[f"- {line}" for line in card]]
    lines = await person_conclusion_lines(ctx, bank_id, handle)
    if lines:
        parts += ["", "## What is known", *lines]
    if len(parts) == 1:
        parts += ["", "Nothing is known about this person yet. The next consolidation fills this in."]
    return "\n".join(parts)


async def ensure_person_models(engine: Any, bank_id: str, request_context: "RequestContext | None" = None) -> list[str]:
    """Create ``person:<handle>`` for every peer of the bank that has none. Returns the handles created."""
    from .peers import list_person_handles

    ctx = PeopleContext.internal(engine)
    handles = await list_person_handles(ctx, bank_id)
    if not handles:
        return []
    wanted = {person_model_id(handle): handle for handle in handles}
    async with ctx.conn() as conn:
        existing = await conn.fetch(
            "SELECT id FROM " + _mental_models_table() + " WHERE bank_id = $1 AND id = ANY($2::text[])",
            bank_id,
            list(wanted),
        )
    missing = [handle for model_id, handle in wanted.items() if model_id not in {row["id"] for row in existing}]
    created: list[str] = []
    for handle in missing:
        try:
            await engine.create_mental_model(
                bank_id=bank_id,
                name=PERSON_MODEL_PREFIX + handle,
                source_query=PERSON_SOURCE_QUERY.format(handle=handle),
                content=await person_model_content(ctx, bank_id, handle),
                mental_model_id=person_model_id(handle),
                trigger={"refresh_after_consolidation": True},
                request_context=request_context or ctx.request_context,
            )
            created.append(handle)
        except Exception:
            # One person's model failing must not stop the others or the consolidation that called us.
            logger.warning(
                "[PEOPLE] could not create the person model for %s in bank %s", handle, bank_id, exc_info=True
            )
    return created


def _mental_models_table() -> str:
    from ..engine.schema import fq_table

    return fq_table("mental_models")
