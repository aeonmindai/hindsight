"""Person models: one mental model, ``person:<handle>``, for every person the bank knows.

After each consolidation :func:`ensure_person_models` gives every peer without one a model, with no one
asking. The model is created with a refresh every 6 hours (``refresh_cron``; the refresh runs only when the bank has new memories in the
model's scope), not after a consolidation: consolidation can lag or be held back for days and the model must not wait for it (a changed fact
shows in it at the next tick). Its first content is built from
what the deriver and the dreamer already concluded about the person, each line with its source.
"""

from __future__ import annotations

import json
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
    "Who they are; Role and company; How they want to be addressed and treated (what they asked for, corrected or objected to, in their words); Likes and dislikes; What they are working on; "
    "Open promises (to them and from them, with the date and who owes what); Who they work with. "
    "Use only what the memories say about {handle}. Never record statistics or rates about how they write (emoji, message length, punctuation, how often or how rarely they use anything), and never state a style as a percentage or a count; "
    "an emoji, a length or a tone is only worth a line when the person asked for or against it, and then say what they asked. Every line ends with its source in brackets: "
    "the source link (source_url) and the date it was said (said_at), as recorded in the memory's metadata. "
    "When a fact changed, state the latest one and put the earlier one under 'Earlier' with its date and source. "
    "Leave out a value that the CRM, Linear or Stripe holds; name the record (its pointer_kind and pointer_id) instead."
)

#: The tag Neo puts on a confidence (what someone asked him to keep between them) and on the messages it covers.
CONFIDENCE_TAG = "confidence"

#: How a person model is refreshed: on a clock, every 6 hours when something new was said (never tied to consolidation catching up: on 8 Nov a 58k
#: consolidation backlog kept every model from refreshing for a day), from the memories that are not confidences, and not
#: from other mental models (a model built from memories would otherwise carry a confidence into this one). A person
#: model is shown to every agent, so a confidence must never be in it.
PERSON_MODEL_REFRESH_CRON = "0 */6 * * *"
PERSON_MODEL_TRIGGER: dict[str, Any] = {
    "refresh_cron": PERSON_MODEL_REFRESH_CRON,
    "exclude_mental_models": True,
    "tag_groups": [{"not": {"tags": [CONFIDENCE_TAG], "match": "any_strict"}}],
}


def leaves_out_confidences(trigger: Any) -> bool:
    """Whether a stored trigger (a JSON string or a dict) already leaves confidences out of the refresh."""
    if isinstance(trigger, str):
        try:
            trigger = json.loads(trigger)
        except ValueError:
            return False
    if not isinstance(trigger, dict):
        return False
    return bool(trigger.get("exclude_mental_models")) and PERSON_MODEL_TRIGGER["tag_groups"] == trigger.get("tag_groups")


def refreshes_on_clock(trigger: Any) -> bool:
    """Whether a stored trigger already refreshes on the person-model clock and leaves confidences out."""
    if isinstance(trigger, str):
        try:
            trigger = json.loads(trigger)
        except ValueError:
            return False
    return (
        leaves_out_confidences(trigger)
        and trigger.get("refresh_cron") == PERSON_MODEL_REFRESH_CRON
        and not trigger.get("refresh_after_consolidation")
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
            "SELECT id, trigger, source_query FROM " + _mental_models_table() + " WHERE bank_id = $1 AND id = ANY($2::text[])",
            bank_id,
            list(wanted),
        )
    present = {row["id"] for row in existing}
    missing = [handle for model_id, handle in wanted.items() if model_id not in present]
    created: list[str] = []
    for handle in missing:
        try:
            await engine.create_mental_model(
                bank_id=bank_id,
                name=PERSON_MODEL_PREFIX + handle,
                source_query=PERSON_SOURCE_QUERY.format(handle=handle),
                content=await person_model_content(ctx, bank_id, handle),
                mental_model_id=person_model_id(handle),
                trigger=dict(PERSON_MODEL_TRIGGER),
                request_context=request_context or ctx.request_context,
            )
            created.append(handle)
        except Exception:
            # One person's model failing must not stop the others or the consolidation that called us.
            logger.warning(
                "[PEOPLE] could not create the person model for %s in bank %s", handle, bank_id, exc_info=True
            )
    # A model made before the exclusion existed would keep reading the confidences, and one still refreshing after consolidation would wait for it: bring its trigger up to date.
    # The same for the mission: a model made with an older source query is rewritten from the current one.
    for row in existing:
        handle = handle_of(row["id"])
        stale_query = handle is not None and row["source_query"] != PERSON_SOURCE_QUERY.format(handle=handle)
        if refreshes_on_clock(row["trigger"]) and not stale_query:
            continue
        try:
            await engine.update_mental_model(
                bank_id=bank_id,
                mental_model_id=row["id"],
                trigger=dict(PERSON_MODEL_TRIGGER),
                **({"source_query": PERSON_SOURCE_QUERY.format(handle=handle)} if stale_query else {}),
                request_context=request_context or ctx.request_context,
            )
        except Exception:
            logger.warning(
                "[PEOPLE] could not update the trigger of %s in bank %s", row["id"], bank_id, exc_info=True
            )
    return created


def _mental_models_table() -> str:
    from ..engine.schema import fq_table

    return fq_table("mental_models")
