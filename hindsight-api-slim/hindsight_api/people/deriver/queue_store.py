# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/crud/deriver.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""SQL the queue manager runs against ``people_queue`` and ``people_active_queue_sessions``.

Honcho's claim reads ``queue_item_batches``, a table that triggers keep in step with the queue. The
fork's queue is small, so the claim aggregates ``people_queue`` (joined to ``people_messages`` for
token counts) directly. Tenancy, pause and fairness queries are not copied.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from ..context import PeopleContext
from .settings import DERIVER


def stale_claim_cutoff() -> datetime:
    return datetime.now(UTC) - timedelta(minutes=DERIVER.STALE_SESSION_TIMEOUT_MINUTES)


def claim_heartbeat_interval_seconds() -> float:
    """How often an in-flight claim refreshes itself, well inside the stale cutoff.

    Derived from ``STALE_SESSION_TIMEOUT_MINUTES`` rather than configured
    separately: the two are halves of one invariant -- a heartbeat slower than the
    cutoff reaps live work -- and a second knob is a way for them to drift apart.
    A fifth of the window tolerates four consecutive failed beats before a claim
    is considered stale.
    """
    return max(5.0, DERIVER.STALE_SESSION_TIMEOUT_MINUTES * 60 / 5)


def claim_rows_query(
    limit: int,
    *,
    bank_id: str | None = None,
    exclude_keys: Sequence[str] = (),
    flush: bool = False,
) -> tuple[str, list[Any]]:
    """The oldest claimable work units: ``(sql, args)`` for rows of (bank_id, work_unit_key, task_type,
    total_tokens, oldest_created_at).

    A unit is claimable when it has unprocessed items and no claim row exists for it (stale claims
    included: cleanup deletes those). A representation unit waits for the token target or the age
    flush unless ``flush`` or ``DERIVER.FLUSH_ENABLED`` is set; other task types are always eligible.
    """
    queue = PeopleContext.table("queue")
    messages = PeopleContext.table("messages")
    claims = PeopleContext.table("active_queue_sessions")

    args: list[Any] = [bank_id, list(exclude_keys), limit]
    gate = ""
    target_tokens = DERIVER.REPRESENTATION_BATCH_WORK_UNIT_TARGET_TOKENS
    if not flush and not DERIVER.FLUSH_ENABLED and target_tokens > 0:
        args.append(target_tokens)
        gate = f"task_type <> 'representation' OR total_tokens >= ${len(args)}"
        max_age_seconds = DERIVER.REPRESENTATION_BATCH_MAX_AGE_SECONDS
        if max_age_seconds > 0:
            args.append(float(max_age_seconds))
            gate += f" OR oldest_created_at <= now() - make_interval(secs => ${len(args)})"
        gate = f"WHERE {gate}"

    sql = f"""
        WITH pending AS (
            SELECT q.bank_id, q.work_unit_key, q.task_type,
                   COALESCE(SUM(m.token_count), 0)::bigint AS total_tokens,
                   MIN(q.created_at) AS oldest_created_at
            FROM {queue} q
            LEFT JOIN {messages} m ON m.bank_id = q.bank_id AND m.id = q.message_id
            WHERE NOT q.processed
              AND ($1::text IS NULL OR q.bank_id = $1)
              AND NOT (q.work_unit_key = ANY($2::text[]))
              AND NOT EXISTS (
                  SELECT 1 FROM {claims} a WHERE a.bank_id = q.bank_id AND a.work_unit_key = q.work_unit_key
              )
            GROUP BY q.bank_id, q.work_unit_key, q.task_type
        )
        SELECT bank_id, work_unit_key, task_type, total_tokens, oldest_created_at
        FROM pending
        {gate}
        ORDER BY oldest_created_at ASC, work_unit_key ASC
        LIMIT $3
    """
    return sql, args


async def cleanup_stale_work_units(ctx: PeopleContext) -> None:
    """Delete claim rows whose last update is older than the stale timeout."""
    claims = PeopleContext.table("active_queue_sessions")
    async with ctx.conn() as conn, conn.transaction():
        stale = await conn.fetch(
            f"SELECT bank_id, id FROM {claims} WHERE last_updated < $1 ORDER BY last_updated FOR UPDATE SKIP LOCKED",
            stale_claim_cutoff(),
        )
        for row in stale:
            await conn.execute(f"DELETE FROM {claims} WHERE bank_id = $1 AND id = $2", row["bank_id"], row["id"])
