# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/dreamer/dream_due.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""Read-only listing of the collections whose next dream is due. Enqueues nothing."""

from datetime import UTC, datetime, timedelta
from logging import getLogger
from typing import Any, NamedTuple, cast

from ..context import PeopleContext
from ..deriver.work_unit import parse_work_unit_key
from ..jsonb import from_jsonb
from ..schemas import DreamType, get_configuration
from .dream_scheduler import WORK_LIST_BACKLOG_KEY
from .payload import dream_work_unit_key
from .settings import DREAM

logger = getLogger(__name__)


class DueDream(NamedTuple):
    """One pair whose next dream is due, with everything needed to enqueue it."""

    bank_id: str
    observer: str
    observed: str
    dream_type: DreamType
    session_name: str


async def count_due_dreams(ctx: PeopleContext) -> int:
    """Count pairs whose next dream is due."""
    return len(await list_due_dreams(ctx))


async def list_due_dreams(ctx: PeopleContext) -> list[DueDream]:
    """List pairs past the threshold, the idle timeout, the min-hours gate, any earlier attempt, pending representation work, and the session's dream setting."""
    dream_types = [DreamType(dream_type) for dream_type in DREAM.ENABLED_TYPES if dream_type == DreamType.OMNI.value]
    if not DREAM.ENABLED or not dream_types:
        return []

    now = datetime.now(UTC)
    idle_cutoff = now - timedelta(minutes=DREAM.IDLE_TIMEOUT_MINUTES)
    # A pair's "collection" is its live explicit conclusions (count and newest time) plus the stored
    # dream state: how many explicit conclusions the last dream saw, and the work-list backlog.
    async with ctx.conn() as conn:
        rows = await conn.fetch(
            "SELECT p.bank_id, p.observer, p.observed, s.internal_metadata, p.explicit_count, p.newest_explicit_at "
            f"FROM (SELECT bank_id, observer, observed, count(id) AS explicit_count, "
            f"max(created_at) AS newest_explicit_at FROM {ctx.table('conclusions')} "
            "WHERE level = 'explicit' AND deleted_at IS NULL GROUP BY bank_id, observer, observed) p "
            f"LEFT JOIN {ctx.table('dream_state')} s "
            "ON s.bank_id = p.bank_id AND s.observer = p.observer AND s.observed = p.observed "
            "WHERE (p.explicit_count - COALESCE(((s.internal_metadata -> 'dream') ->> 'last_dream_document_count')"
            "::integer, 0) >= $1 AND p.newest_explicit_at <= $2) "
            "OR COALESCE(((s.internal_metadata -> 'dream') ->> 'work_list_backlog')::integer, 0) > 0",
            DREAM.DOCUMENT_THRESHOLD,
            idle_cutoff,
        )

    # A candidate's newest explicit conclusion, or, for a pair whose work list
    # is still behind, its last completed dream: an attempt after that time
    # means the dream is already queued or was tried.
    candidates: dict[str, tuple[DueDream, datetime]] = {}

    for row in rows:
        bank_id = cast(str, row["bank_id"])
        observer = cast(str, row["observer"])
        observed = cast(str, row["observed"])
        internal_metadata = cast("dict[str, Any] | None", from_jsonb(row["internal_metadata"]))
        explicit_count = cast(int, row["explicit_count"])
        newest_created_at = cast("datetime | None", row["newest_explicit_at"])

        dream_metadata: dict[str, Any] = (internal_metadata or {}).get("dream", {})
        since_last_dream = explicit_count - int(dream_metadata.get("last_dream_document_count", 0))
        # The work list's backlog is due even without new conclusions: without
        # them, the attempt check below would never let the pair through again.
        backlog_since = (
            _parse_last_dream_at(dream_metadata.get("last_dream_at"))
            if int(dream_metadata.get(WORK_LIST_BACKLOG_KEY, 0) or 0) > 0
            else None
        )
        if since_last_dream < DREAM.DOCUMENT_THRESHOLD and backlog_since is None:
            continue

        if newest_created_at is None or newest_created_at > idle_cutoff:
            continue

        last_dream_at = cast("str | None", dream_metadata.get("last_dream_at"))
        if last_dream_at and _within_min_hours_gate(last_dream_at, now):
            continue

        for dream_type in dream_types:
            work_unit_key = dream_work_unit_key(bank_id, observer, observed, dream_type)
            candidates[work_unit_key] = (
                DueDream(
                    bank_id=bank_id,
                    observer=observer,
                    observed=observed,
                    dream_type=dream_type,
                    # Hydrated later on
                    session_name="",
                ),
                max(newest_created_at, backlog_since) if backlog_since is not None else newest_created_at,
            )

    if not candidates:
        return []

    async with ctx.conn() as conn:
        attempt_rows = await conn.fetch(
            f"SELECT work_unit_key, max(created_at) AS newest FROM {ctx.table('queue')} "
            "WHERE task_type = 'dream' AND work_unit_key = ANY($1::text[]) GROUP BY work_unit_key",
            list(candidates.keys()),
        )
        newest_attempts: dict[str, datetime] = {
            cast(str, row["work_unit_key"]): cast(datetime, row["newest"]) for row in attempt_rows
        }

        banks = sorted({due_dream.bank_id for due_dream, _ in candidates.values()})
        pending_representation_keys = [
            row["work_unit_key"]
            for row in await conn.fetch(
                f"SELECT DISTINCT work_unit_key FROM {ctx.table('queue')} "
                "WHERE task_type = 'representation' AND processed = false AND bank_id = ANY($1::text[])",
                banks,
            )
        ]
    active_observed = {
        (parsed.bank_id, parsed.observed)
        for parsed in (parse_work_unit_key(key) for key in pending_representation_keys)
    }

    unattempted = [
        due_dream
        for work_unit_key, (due_dream, newest_created_at) in candidates.items()
        if (work_unit_key not in newest_attempts or newest_attempts[work_unit_key] < newest_created_at)
        and (due_dream.bank_id, due_dream.observed) not in active_observed
    ]
    if not unattempted:
        return []

    newest_sessions = await _newest_session_names(ctx, unattempted)
    with_sessions = [
        due_dream._replace(session_name=session_name)
        for due_dream in unattempted
        if (session_name := newest_sessions.get((due_dream.bank_id, due_dream.observer, due_dream.observed)))
        is not None
    ]
    if not with_sessions:
        return []

    return await _filter_dreams_enabled(ctx, with_sessions)


async def _newest_session_names(
    ctx: PeopleContext, candidates: list[DueDream]
) -> dict[tuple[str, str, str], str | None]:
    """The session of each candidate pair's newest explicit conclusion."""
    keys = sorted({(due.bank_id, due.observer, due.observed) for due in candidates})
    async with ctx.conn() as conn:
        rows = await conn.fetch(
            "SELECT k.bank_id, k.observer, k.observed, ("
            f"SELECT c.session_name FROM {ctx.table('conclusions')} c "
            "WHERE c.bank_id = k.bank_id AND c.observer = k.observer AND c.observed = k.observed "
            "AND c.level = 'explicit' AND c.deleted_at IS NULL ORDER BY c.created_at DESC LIMIT 1) AS session_name "
            "FROM unnest($1::text[], $2::text[], $3::text[]) AS k(bank_id, observer, observed)",
            [k[0] for k in keys],
            [k[1] for k in keys],
            [k[2] for k in keys],
        )
    return {
        (cast(str, row["bank_id"]), cast(str, row["observer"]), cast(str, row["observed"])): cast(
            "str | None", row["session_name"]
        )
        for row in rows
    }


async def _filter_dreams_enabled(ctx: PeopleContext, candidates: list[DueDream]) -> list[DueDream]:
    """Drop candidates whose resolved configuration has dreams turned off."""
    session_keys = sorted({(due.bank_id, due.session_name) for due in candidates})
    async with ctx.conn() as conn:
        session_rows = await conn.fetch(
            f"SELECT s.bank_id, s.name, s.internal_metadata -> 'configuration' AS configuration FROM {ctx.table('sessions')} s "
            "WHERE (s.bank_id, s.name) IN (SELECT * FROM unnest($1::text[], $2::text[]))",
            [k[0] for k in session_keys],
            [k[1] for k in session_keys],
        )
    sessions: dict[tuple[str, str], dict[str, Any] | None] = {}
    for row in session_rows:
        value = from_jsonb(row["configuration"])
        sessions[(row["bank_id"], row["name"])] = value if isinstance(value, dict) else None

    enabled: list[DueDream] = []
    for due_dream in candidates:
        configuration = get_configuration(None, sessions.get((due_dream.bank_id, due_dream.session_name)), None)
        if configuration.dream.enabled:
            enabled.append(due_dream)
    return enabled


def _parse_last_dream_at(last_dream_at: Any) -> datetime | None:
    """The stored `last_dream_at` as an aware datetime, or None if unusable."""
    try:
        last_dream_time = datetime.fromisoformat(last_dream_at)
    except (ValueError, TypeError):
        return None
    if last_dream_time.tzinfo is None:
        last_dream_time = last_dream_time.replace(tzinfo=UTC)
    return last_dream_time


def _within_min_hours_gate(last_dream_at: str, now: datetime) -> bool:
    """True when the last dream is too recent for another one."""
    last_dream_time = _parse_last_dream_at(last_dream_at)
    if last_dream_time is None:
        return False

    hours_since = (now - last_dream_time).total_seconds() / 3600
    return hours_since < DREAM.MIN_HOURS_BETWEEN_DREAMS
