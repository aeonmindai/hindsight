# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/crud/dream_history.py
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/schemas/api.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""Representation History: what each dream changed in a representation.

A dream's writes record their own history rows in the same transaction as the
write, through the ``record_*`` helpers here. The orchestrator opens a run
before its specialists start and closes it when they finish, so a dream that
fails partway still shows the changes it committed.
"""

from __future__ import annotations

import datetime
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .context import PeopleContext
from .dreamer import settings as dream_settings
from .dreamer.schemas import (
    DreamConclusionRef,
    DreamCreatedConclusion,
    DreamPeerCardChange,
    DreamRemovedConclusion,
    DreamRun,
    DreamRunDetail,
)
from .exceptions import ResourceNotFoundException
from .jsonb import from_jsonb, to_jsonb
from .schemas import DocumentLevel

logger = logging.getLogger(__name__)

# A run still marked running after this long was cut off (worker crash or
# shutdown) without closing, and is reported as interrupted.
_STALE_RUN_AFTER = datetime.timedelta(hours=24)
_PRUNE_BATCH_SIZE = 500


@dataclass(frozen=True, slots=True)
class DreamHistoryTarget:
    """Where a dream's writes record their changes: an open run and its specialist."""

    run_id: str
    specialist: str | None = None


@dataclass(frozen=True, slots=True)
class CreatedConclusion:
    id: str
    level: DocumentLevel
    source_ids: list[str]


@dataclass(frozen=True, slots=True)
class RemovedConclusion:
    id: str
    level: DocumentLevel
    content: str
    session_name: str | None
    replaced_by: str | None = None


# A run is live while both of its peers are; a removed-conclusion copy is live while its session is.
# "r" is the run alias, "c" the change alias.
def _live_run_sql(peers: str) -> str:
    return (
        f"EXISTS (SELECT 1 FROM {peers} p WHERE p.bank_id = r.bank_id AND p.name = r.observer AND p.deleted_at IS NULL)"
        f" AND EXISTS (SELECT 1 FROM {peers} p WHERE p.bank_id = r.bank_id AND p.name = r.observed"
        " AND p.deleted_at IS NULL)"
    )


def _live_change_sql(sessions: str) -> str:
    return (
        "(c.session_name IS NULL OR EXISTS (SELECT 1 FROM "
        f"{sessions} s WHERE s.bank_id = c.bank_id AND s.name = c.session_name AND s.deleted_at IS NULL))"
    )


async def open_run(
    ctx: PeopleContext,
    *,
    run_id: str,
    bank_id: str,
    observer: str,
    observed: str,
    dream_type: str | None,
    trigger_reason: str | None,
) -> DreamHistoryTarget | None:
    """Record the start of a dream. Returns None when history is off or the insert fails.

    A failure here never stops the dream: it runs without recording history.
    """
    if not dream_settings.DREAM.HISTORY_ENABLED:
        return None
    try:
        async with ctx.conn() as conn:
            await conn.execute(
                f"INSERT INTO {ctx.table('dream_runs')} "
                "(bank_id, id, observer, observed, dream_type, trigger_reason) VALUES ($1, $2, $3, $4, $5, $6)",
                bank_id,
                run_id,
                observer,
                observed,
                dream_type,
                trigger_reason,
            )
    except Exception:
        logger.warning(
            "[%s] Could not open a dream history run; the dream runs unrecorded",
            run_id,
            exc_info=True,
        )
        return None
    return DreamHistoryTarget(run_id=run_id)


async def close_run(ctx: PeopleContext, bank_id: str, run_id: str, *, succeeded: bool) -> None:
    """Mark a run completed or failed. Never raises."""
    try:
        async with ctx.conn() as conn:
            await conn.execute(
                f"UPDATE {ctx.table('dream_runs')} SET status = $3, finished_at = now() "
                "WHERE bank_id = $1 AND id = $2 AND status = 'running'",
                bank_id,
                run_id,
                "completed" if succeeded else "failed",
            )
    except Exception:
        logger.warning("[%s] Could not close the dream history run", run_id, exc_info=True)


async def record_created(
    conn: Any,
    target: DreamHistoryTarget,
    bank_id: str,
    created: Sequence[CreatedConclusion],
) -> None:
    """Add `created` rows and bump the run's count. The caller commits."""
    if not created:
        return
    changes = PeopleContext.table("dream_run_changes")
    await conn.executemany(
        f"INSERT INTO {changes} (bank_id, run_id, kind, specialist, conclusion_id, level, source_ids) "
        "VALUES ($1, $2, 'created', $3, $4, $5, $6::jsonb)",
        [
            (
                bank_id,
                target.run_id,
                target.specialist,
                conclusion.id,
                conclusion.level,
                to_jsonb(conclusion.source_ids) if conclusion.source_ids else None,
            )
            for conclusion in created
        ],
    )
    await conn.execute(
        f"UPDATE {PeopleContext.table('dream_runs')} SET created_count = created_count + $3 "
        "WHERE bank_id = $1 AND id = $2",
        bank_id,
        target.run_id,
        len(created),
    )


async def record_removed(
    conn: Any,
    target: DreamHistoryTarget,
    bank_id: str,
    removed: Sequence[RemovedConclusion],
) -> None:
    """Add `removed` rows, copying each conclusion's content. The caller commits."""
    if not removed:
        return
    changes = PeopleContext.table("dream_run_changes")
    await conn.executemany(
        f"INSERT INTO {changes} "
        "(bank_id, run_id, kind, specialist, conclusion_id, level, content, session_name, replaced_by) "
        "VALUES ($1, $2, 'removed', $3, $4, $5, $6, $7, $8)",
        [
            (
                bank_id,
                target.run_id,
                target.specialist,
                conclusion.id,
                conclusion.level,
                conclusion.content,
                conclusion.session_name,
                conclusion.replaced_by,
            )
            for conclusion in removed
        ],
    )
    await conn.execute(
        f"UPDATE {PeopleContext.table('dream_runs')} SET removed_count = removed_count + $3 "
        "WHERE bank_id = $1 AND id = $2",
        bank_id,
        target.run_id,
        len(removed),
    )


async def record_peer_card(
    conn: Any,
    target: DreamHistoryTarget,
    bank_id: str,
    *,
    before: list[str] | None,
    after: list[str],
) -> None:
    """Add a `peer_card` row with the card before and after. The caller commits."""
    await conn.execute(
        f"INSERT INTO {PeopleContext.table('dream_run_changes')} "
        "(bank_id, run_id, kind, specialist, peer_card_before, peer_card_after) "
        "VALUES ($1, $2, 'peer_card', $3, $4::jsonb, $5::jsonb)",
        bank_id,
        target.run_id,
        target.specialist,
        to_jsonb(before) if before is not None else None,
        to_jsonb(after),
    )
    await conn.execute(
        f"UPDATE {PeopleContext.table('dream_runs')} SET peer_card_updated = true WHERE bank_id = $1 AND id = $2",
        bank_id,
        target.run_id,
    )


def _run_from_row(row: Any) -> DreamRun:
    return DreamRun.model_validate(dict(row))


def _runs_filter(ctx: PeopleContext, bank_id: str, observed: str | None, observer: str | None) -> tuple[str, list[Any]]:
    """The WHERE clause (and its arguments) of the live runs of a bank, for one pair or peer if asked."""
    args: list[Any] = [bank_id]
    where = ["r.bank_id = $1", _live_run_sql(ctx.table("peers"))]
    if observed is not None:
        args.append(observed)
        where.append(f"r.observed = ${len(args)}")
    if observer is not None:
        args.append(observer)
        where.append(f"r.observer = ${len(args)}")
    return " AND ".join(where), args


async def list_runs(
    ctx: PeopleContext,
    bank_id: str,
    *,
    observed: str | None = None,
    observer: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[DreamRun]:
    """Runs in a bank, newest first, optionally for one observed/observer peer."""
    where, args = _runs_filter(ctx, bank_id, observed, observer)
    args.extend([limit, offset])
    sql = (
        f"SELECT r.id, r.observer, r.observed, r.dream_type, r.trigger_reason, r.status, r.created_count, "
        f"r.removed_count, r.peer_card_updated, r.started_at, r.finished_at, r.pruned_at "
        f"FROM {ctx.table('dream_runs')} r WHERE {where} ORDER BY r.started_at DESC, r.id DESC "
        f"LIMIT ${len(args) - 1} OFFSET ${len(args)}"
    )
    async with ctx.conn() as conn:
        rows = await conn.fetch(sql, *args)
    return [_run_from_row(row) for row in rows]


async def count_runs(
    ctx: PeopleContext, bank_id: str, *, observed: str | None = None, observer: str | None = None
) -> int:
    """How many runs ``list_runs`` would list with the same filters."""
    where, args = _runs_filter(ctx, bank_id, observed, observer)
    async with ctx.conn() as conn:
        return int(await conn.fetchval(f"SELECT count(*) FROM {ctx.table('dream_runs')} r WHERE {where}", *args))


async def get_run_detail(ctx: PeopleContext, bank_id: str, run_id: str) -> DreamRunDetail:
    """A run and its changes, with conclusion content resolved for display.

    Created conclusions and sources are looked up live first. One a later
    dream removed resolves from that dream's copy; one deleted through the
    API, or by deleting its session, resolves to `unavailable`. A run whose
    observer or observed peer is deleted is not found.
    """
    runs = ctx.table("dream_runs")
    changes_table = ctx.table("dream_run_changes")
    async with ctx.conn() as conn:
        run = await conn.fetchrow(
            f"SELECT r.* FROM {runs} r WHERE r.bank_id = $1 AND r.id = $2 AND {_live_run_sql(ctx.table('peers'))}",
            bank_id,
            run_id,
        )
        if run is None:
            raise ResourceNotFoundException(f"Dream run {run_id} not found")

        changes = await conn.fetch(
            f"SELECT c.* FROM {changes_table} c WHERE c.bank_id = $1 AND c.run_id = $2 "
            f"AND {_live_change_sql(ctx.table('sessions'))} ORDER BY c.id",
            bank_id,
            run_id,
        )

        created_changes = [c for c in changes if c["kind"] == "created" and c["conclusion_id"]]
        to_resolve: set[str] = set()
        for change in created_changes:
            to_resolve.add(change["conclusion_id"])
            to_resolve.update(from_jsonb(change["source_ids"], []) or [])
        resolved = await _resolve_conclusions(
            ctx, conn, bank_id, observer=run["observer"], observed=run["observed"], conclusion_ids=to_resolve
        )

    created = [
        DreamCreatedConclusion(
            conclusion=resolved[change["conclusion_id"]],
            sources=[resolved[source_id] for source_id in from_jsonb(change["source_ids"], []) or []],
            specialist=change["specialist"],
        )
        for change in created_changes
    ]
    removed = [
        DreamRemovedConclusion(
            id=change["conclusion_id"],
            content=change["content"],
            level=change["level"],
            replaced_by=change["replaced_by"],
            specialist=change["specialist"],
        )
        for change in changes
        if change["kind"] == "removed" and change["conclusion_id"]
    ]
    card_changes = [c for c in changes if c["kind"] == "peer_card"]
    peer_card = (
        DreamPeerCardChange(
            before=from_jsonb(card_changes[0]["peer_card_before"]),
            after=from_jsonb(card_changes[-1]["peer_card_after"], []) or [],
        )
        if card_changes
        else None
    )
    base = _run_from_row(run).model_dump()
    return DreamRunDetail(**base, created=created, removed=removed, peer_card=peer_card)


async def _resolve_conclusions(
    ctx: PeopleContext,
    conn: Any,
    bank_id: str,
    *,
    observer: str,
    observed: str,
    conclusion_ids: set[str],
) -> dict[str, DreamConclusionRef]:
    resolved = {
        conclusion_id: DreamConclusionRef(id=conclusion_id, status="unavailable") for conclusion_id in conclusion_ids
    }
    if not conclusion_ids:
        return resolved

    live = await conn.fetch(
        f"SELECT k.id, k.content, k.level FROM {ctx.table('conclusions')} k "
        "WHERE k.bank_id = $1 AND k.id = ANY($2::text[]) AND k.observer = $3 AND k.observed = $4 "
        "AND k.deleted_at IS NULL "
        f"AND (k.session_name IS NULL OR EXISTS (SELECT 1 FROM {ctx.table('sessions')} s "
        "WHERE s.bank_id = k.bank_id AND s.name = k.session_name AND s.deleted_at IS NULL)) "
        "ORDER BY k.id",
        bank_id,
        sorted(conclusion_ids),
        observer,
        observed,
    )
    for row in live:
        resolved[row["id"]] = DreamConclusionRef(
            id=row["id"], content=row["content"], level=row["level"], status="current"
        )

    missing = [cid for cid, ref in resolved.items() if ref.status == "unavailable"]
    if missing:
        # The copy's own run must be live too: a source can come from another peer pair, whose runs
        # may be hidden.
        removed = await conn.fetch(
            f"SELECT c.conclusion_id, c.content, c.level, c.run_id FROM {ctx.table('dream_run_changes')} c "
            f"JOIN {ctx.table('dream_runs')} r ON r.bank_id = c.bank_id AND r.id = c.run_id "
            "WHERE c.bank_id = $1 AND c.kind = 'removed' AND c.conclusion_id = ANY($2::text[]) "
            f"AND {_live_change_sql(ctx.table('sessions'))} AND {_live_run_sql(ctx.table('peers'))} "
            "ORDER BY c.id",
            bank_id,
            missing,
        )
        for row in removed:
            resolved[row["conclusion_id"]] = DreamConclusionRef(
                id=row["conclusion_id"],
                content=row["content"],
                level=row["level"],
                status="removed",
                removed_in_run=row["run_id"],
            )
    return resolved


async def delete_session_history(conn: Any, bank_id: str, session_name: str) -> None:
    """Delete removed-conclusion copies that came from a session. The caller commits.

    Run it after the session's conclusions are deleted, so a dream still committing
    a copy of one of them has finished first.
    """
    # Locked in id order: an IN predicate alone imposes none.
    changes = PeopleContext.table("dream_run_changes")
    await conn.execute(
        f"DELETE FROM {changes} WHERE bank_id = $1 AND id IN ("
        f"SELECT id FROM {changes} WHERE bank_id = $1 AND session_name = $2 ORDER BY id FOR UPDATE)",
        bank_id,
        session_name,
    )


async def erase_conclusion_history(conn: Any, bank_id: str, conclusion_id: str) -> int:
    """Delete the copies history keeps of a removed conclusion. The caller commits.

    For a user-requested deletion: afterwards the conclusion reads as
    `unavailable` everywhere history references it. Returns the copies deleted.
    """
    runs = PeopleContext.table("dream_runs")
    changes = PeopleContext.table("dream_run_changes")
    # The copies' runs first, in id order: bank deletion and the prune lock runs before the changes
    # that cascade from them.
    await conn.fetch(
        f"SELECT r.id FROM {runs} r WHERE r.bank_id = $1 AND r.id IN ("
        f"SELECT run_id FROM {changes} WHERE bank_id = $1 AND kind = 'removed' AND conclusion_id = $2) "
        "ORDER BY r.id FOR UPDATE",
        bank_id,
        conclusion_id,
    )
    # Then the copies in id order, as session deletion and the prune take them.
    status = await conn.execute(
        f"DELETE FROM {changes} WHERE bank_id = $1 AND id IN ("
        f"SELECT id FROM {changes} WHERE bank_id = $1 AND kind = 'removed' AND conclusion_id = $2 "
        "ORDER BY id FOR UPDATE)",
        bank_id,
        conclusion_id,
    )
    return int(status.split()[-1])


async def delete_bank_history(conn: Any, bank_id: str) -> None:
    """Delete every run in a bank; their changes cascade. The caller commits.

    Runs are locked in id order, the same order the retention prune takes them.
    """
    runs = PeopleContext.table("dream_runs")
    await conn.execute(
        f"DELETE FROM {runs} WHERE bank_id = $1 AND id IN ("
        f"SELECT id FROM {runs} WHERE bank_id = $1 ORDER BY id FOR UPDATE)",
        bank_id,
    )


async def prune_dream_history(ctx: PeopleContext) -> int:
    """Mark stale running runs interrupted and apply the retention window.

    Past the window, a run keeps only its `created` changes whose conclusion
    still exists, so which dream made a live conclusion stays readable. Its
    other changes are deleted and the run gets `pruned_at`. A run with nothing
    left is deleted. Returns the number of runs deleted.
    """
    now = datetime.datetime.now(datetime.UTC)
    cutoff = now - datetime.timedelta(days=dream_settings.DREAM.HISTORY_RETENTION_DAYS)
    deleted = 0
    runs = ctx.table("dream_runs")
    changes = ctx.table("dream_run_changes")
    conclusions = ctx.table("conclusions")
    async with ctx.conn() as conn:
        # Runs across every bank.
        await conn.execute(
            f"UPDATE {runs} SET status = 'interrupted' WHERE (bank_id, id) IN ("
            f"SELECT bank_id, id FROM {runs} WHERE status = 'running' AND started_at < $1 "
            "ORDER BY bank_id, id FOR UPDATE)",
            now - _STALE_RUN_AFTER,
        )

        after: tuple[str, str] | None = None
        while True:
            # Runs are locked before their changes, the order bank deletion takes them in.
            async with conn.transaction():
                if after is None:
                    batch = await conn.fetch(
                        f"SELECT bank_id, id FROM {runs} WHERE started_at < $1 "
                        "ORDER BY bank_id, id LIMIT $2 FOR UPDATE",
                        cutoff,
                        _PRUNE_BATCH_SIZE,
                    )
                else:
                    batch = await conn.fetch(
                        f"SELECT bank_id, id FROM {runs} WHERE started_at < $1 AND (bank_id, id) > ($3, $4) "
                        "ORDER BY bank_id, id LIMIT $2 FOR UPDATE",
                        cutoff,
                        _PRUNE_BATCH_SIZE,
                        after[0],
                        after[1],
                    )
                if not batch:
                    break
                keys = [(row["bank_id"], row["id"]) for row in batch]
                after = keys[-1]
                bank_ids = [k[0] for k in keys]
                run_ids = [k[1] for k in keys]

                # Changes are locked in id order, as session deletion takes them.
                pruned = await conn.fetch(
                    f"DELETE FROM {changes} WHERE (bank_id, id) IN ("
                    f"SELECT ch.bank_id, ch.id FROM {changes} ch "
                    "WHERE (ch.bank_id, ch.run_id) IN (SELECT * FROM unnest($1::text[], $2::text[])) "
                    "AND NOT (ch.kind = 'created' AND EXISTS ("
                    f"SELECT 1 FROM {conclusions} k WHERE k.bank_id = ch.bank_id AND k.id = ch.conclusion_id "
                    "AND k.deleted_at IS NULL)) "
                    "ORDER BY ch.bank_id, ch.id FOR UPDATE OF ch) RETURNING bank_id, run_id",
                    bank_ids,
                    run_ids,
                )
                pruned_runs = sorted({(row["bank_id"], row["run_id"]) for row in pruned})
                if pruned_runs:
                    await conn.execute(
                        f"UPDATE {runs} SET pruned_at = $3 WHERE (bank_id, id) IN "
                        "(SELECT * FROM unnest($1::text[], $2::text[])) AND pruned_at IS NULL",
                        [k[0] for k in pruned_runs],
                        [k[1] for k in pruned_runs],
                        now,
                    )
                status = await conn.execute(
                    f"DELETE FROM {runs} r WHERE (r.bank_id, r.id) IN (SELECT * FROM unnest($1::text[], $2::text[])) "
                    f"AND NOT EXISTS (SELECT 1 FROM {changes} ch WHERE ch.bank_id = r.bank_id AND ch.run_id = r.id)",
                    bank_ids,
                    run_ids,
                )
                deleted += int(status.split()[-1])
            if len(batch) < _PRUNE_BATCH_SIZE:
                break
    return deleted
