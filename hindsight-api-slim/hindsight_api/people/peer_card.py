# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/crud/peer_card.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
import logging
from typing import Any, cast

from . import schemas
from .context import PeopleContext
from .db import connection
from .exceptions import ResourceNotFoundException
from .jsonb import from_jsonb, to_jsonb
from .peers import get_or_create_peers, get_peer
from .sanitization import strip_nul

logger = logging.getLogger(__name__)


async def get_peer_card(
    ctx: PeopleContext,
    bank_id: str,
    observer: str,
    observed: str | None = None,
    *,
    conn: Any | None = None,
) -> list[str] | None:
    """Get peer card from internal_metadata.

    The peer card is returned for the observer/observed relationship. With no ``observed`` it is the
    observer's own card.

    Args:
        ctx: People context
        bank_id: Bank the peers live in
        observer: Handle of the observer
        observed: Handle of the person described in the peer card (default: the observer)

    Returns:
        The peer's card text if present, otherwise None.

    Raises:
        ResourceNotFoundException: If the peer does not exist.
    """
    peer = await get_peer(ctx, bank_id, observer, conn=conn)
    return cast(
        list[str] | None,
        peer.internal_metadata.get(construct_peer_card_label(observer=observer, observed=observed or observer)),
    )


async def set_peer_card(
    ctx: PeopleContext,
    bank_id: str,
    peer_card: list[str],
    observer: str,
    observed: str | None = None,
    *,
    history: Any | None = None,
    conn: Any | None = None,
) -> None:
    """Set peer card for a peer.

    The peer card is set for the observer/observed relationship (the observer's own card when
    ``observed`` is omitted).

    Args:
        ctx: People context
        bank_id: Bank the peers live in
        peer_card: List of strings to set as the peer card
        observer: Handle of the observer
        observed: Handle of the person described in the peer card (default: the observer)
        history: When set (a dream history target), record the card before and after to this dream
            run in the same transaction
        conn: Optional open connection to join the caller's transaction
    """
    observed = observed or observer

    # jsonb rejects NUL in strings. Only the bytes are removed; what counts as a
    # valid entry is the caller's call (the API may set `[]` or free-form lines).
    peer_card = strip_nul(peer_card)

    # Ensure the peer exists (get-or-create)
    await get_or_create_peers(ctx, bank_id, [schemas.PeerSpec(name=observer)], conn=conn)

    label = construct_peer_card_label(observer=observer, observed=observed)
    async with connection(ctx, conn) as db:
        async with db.transaction():
            if history is not None:
                # Locked so a concurrent write can't land between this read and the
                # update, which would make `before` wrong.
                before_raw = await db.fetchval(
                    f"SELECT internal_metadata -> $3 FROM {ctx.table('peers')} "
                    "WHERE bank_id = $1 AND name = $2 AND deleted_at IS NULL FOR UPDATE",
                    bank_id,
                    observer,
                    label,
                )
                from .dream_history import record_peer_card  # the dreamer owns the history tables

                await record_peer_card(
                    db,
                    history,
                    bank_id,
                    before=cast(list[str] | None, from_jsonb(before_raw)),
                    after=peer_card,
                )

            updated = await db.execute(
                f"UPDATE {ctx.table('peers')} SET internal_metadata = internal_metadata || $3::jsonb "
                "WHERE bank_id = $1 AND name = $2 AND deleted_at IS NULL",
                bank_id,
                observer,
                to_jsonb({label: peer_card}),
            )
            if updated.endswith(" 0"):
                raise ResourceNotFoundException(f"Peer {observer} not found in bank {bank_id}")


def construct_peer_card_label(*, observer: str, observed: str) -> str:
    if observer == observed:
        return "peer_card"
    return f"{observed}_peer_card"
