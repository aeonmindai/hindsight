# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/dreamer/dream_scheduler.py
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/backlog.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
import asyncio
import contextlib
import time
from collections.abc import Callable
from datetime import UTC, datetime
from logging import getLogger

from ..context import PeopleContext
from ..dream_history import prune_dream_history
from ..schemas import DreamType, get_configuration
from . import collection_state
from .payload import dream_work_unit_key, parse_dream_work_unit_key
from .settings import DREAM

logger = getLogger(__name__)


_dream_scheduler: "DreamScheduler | None" = None


def set_dream_scheduler(dream_scheduler: "DreamScheduler | None") -> None:
    """Set the global dream scheduler reference."""
    global _dream_scheduler
    _dream_scheduler = dream_scheduler


def get_dream_scheduler() -> "DreamScheduler | None":
    """Get the global dream scheduler reference."""
    return _dream_scheduler


# How often the history retention prune runs, in seconds.
_PRUNE_INTERVAL_SECONDS = 3600


class DreamScheduler:
    """Schedules dreams: delayed timers for pairs that reached the threshold, and a poll for due pairs.

    ``ctx_factory`` returns the ``PeopleContext`` the scheduler works under (it is called when a timer
    fires and on every poll). ``start()`` registers the scheduler globally and begins polling the
    collections whose next dream is due; ``stop()`` cancels the poll and every pending timer.
    """

    def __init__(self, ctx_factory: Callable[[], PeopleContext]):
        self._ctx_factory = ctx_factory
        self.pending_dreams: dict[str, asyncio.Task[None]] = {}
        self._poll_task: asyncio.Task[None] | None = None
        self._next_prune: float | None = None

    async def start(self) -> None:
        """Register this scheduler and start the due-dream poll."""
        set_dream_scheduler(self)
        if self._poll_task is None and DREAM.ENABLED:
            self._poll_task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        """Stop polling and cancel all pending dreams."""
        if self._poll_task is not None:
            self._poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._poll_task
            self._poll_task = None
        await self.shutdown()
        if get_dream_scheduler() is self:
            set_dream_scheduler(None)

    async def _poll_loop(self) -> None:
        while True:
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("Error polling for due dreams: %s", e, exc_info=True)
            await asyncio.sleep(DREAM.DUE_POLL_INTERVAL_SECONDS)

    async def poll_once(self) -> int:
        """Enqueue the dreams that are due (at most MAX_ENQUEUED_PER_POLL). Returns how many were tried."""
        from ..deriver.enqueue import enqueue_dream
        from .dream_due import list_due_dreams

        ctx = self._ctx_factory()
        due_dreams = await list_due_dreams(ctx)
        cap = DREAM.MAX_ENQUEUED_PER_POLL
        for due_dream in due_dreams[:cap]:
            try:
                await enqueue_dream(
                    ctx,
                    due_dream.bank_id,
                    due_dream.observer,
                    due_dream.observed,
                    due_dream.dream_type,
                    session_name=due_dream.session_name,
                    trigger_reason="document_threshold",
                    delay_reason="api_poll",
                )
            except Exception as e:
                logger.error(
                    "Failed to enqueue dream for %s/%s/%s: %s",
                    due_dream.bank_id,
                    due_dream.observer,
                    due_dream.observed,
                    e,
                )
        if self._next_prune is None or time.monotonic() >= self._next_prune:
            self._next_prune = time.monotonic() + _PRUNE_INTERVAL_SECONDS
            try:
                await prune_dream_history(ctx)
            except Exception as e:
                logger.error("Error pruning dream history: %s", e, exc_info=True)
        return len(due_dreams[:cap])

    async def schedule_dream(
        self,
        work_unit_key: str,
        bank_id: str,
        delay_minutes: int,
        dream_type: DreamType,
        *,
        observer: str,
        observed: str,
        trigger_reason: str | None = None,
        delay_reason: str | None = None,
        documents_since_last_dream_at_schedule: int | None = None,
        document_threshold: int | None = None,
    ) -> None:
        """Schedule a dream for a pair after a delay.

        The scheduling context (``trigger_reason``, ``delay_reason`` and the counts) is captured at
        schedule time and threaded through the queue payload, so the dream can be attributed back to why
        it was scheduled.
        """
        if not DREAM.ENABLED:
            return

        # Cancel any existing dream for this pair
        await self.cancel_dream(work_unit_key)

        task = asyncio.create_task(
            self._delayed_dream(
                work_unit_key,
                bank_id,
                delay_minutes,
                dream_type,
                observer=observer,
                observed=observed,
                trigger_reason=trigger_reason,
                delay_reason=delay_reason,
                documents_since_last_dream_at_schedule=documents_since_last_dream_at_schedule,
                document_threshold=document_threshold,
            )
        )
        self.pending_dreams[work_unit_key] = task
        task.add_done_callback(lambda t: self.pending_dreams.pop(work_unit_key, None))

    async def cancel_dream(self, work_unit_key: str) -> bool:
        """Cancel a pending dream. Returns True if a dream was cancelled."""
        if work_unit_key in self.pending_dreams:
            task = self.pending_dreams.pop(work_unit_key)
            task.cancel()
            # Wait for the task to actually finish (including its done callback)
            with contextlib.suppress(asyncio.CancelledError):
                await task
            return True
        return False

    async def cancel_dreams_for_observed(self, bank_id: str, observed: str) -> set[str]:
        """
        Cancel all pending dreams where the observed peer matches.

        This handles both self-observation (observer=observed) and peer-to-peer
        observation (observer!=observed) dreams.

        Args:
            bank_id: The bank to match
            observed: The observed peer name to match

        Returns:
            Set of work_unit_keys that were cancelled
        """
        cancelled: set[str] = set()

        # Collect keys to cancel (can't modify dict while iterating)
        keys_to_cancel: list[str] = []
        for work_unit_key in self.pending_dreams:
            parsed = parse_dream_work_unit_key(work_unit_key)
            if parsed is not None and parsed[1] == bank_id and parsed[3] == observed:
                keys_to_cancel.append(work_unit_key)

        # Cancel each matching dream
        for key in keys_to_cancel:
            if await self.cancel_dream(key):
                cancelled.add(key)

        return cancelled

    async def _delayed_dream(
        self,
        work_unit_key: str,
        bank_id: str,
        delay_minutes: int,
        dream_type: DreamType,
        *,
        observer: str,
        observed: str,
        trigger_reason: str | None = None,
        delay_reason: str | None = None,
        documents_since_last_dream_at_schedule: int | None = None,
        document_threshold: int | None = None,
    ) -> None:
        try:
            await asyncio.sleep(delay_minutes * 60)

            await self.execute_dream(
                bank_id,
                dream_type,
                observer=observer,
                observed=observed,
                trigger_reason=trigger_reason,
                delay_reason=delay_reason,
                documents_since_last_dream_at_schedule=documents_since_last_dream_at_schedule,
                document_threshold=document_threshold,
            )
            logger.info("Executed dream for %s", work_unit_key)

        except asyncio.CancelledError:
            logger.debug("Dream task cancelled for %s", work_unit_key)
        except Exception as e:
            logger.error("Error in delayed dream for %s: %s", work_unit_key, e)

    async def execute_dream(
        self,
        bank_id: str,
        dream_type: DreamType,
        *,
        observer: str,
        observed: str,
        trigger_reason: str | None = None,
        delay_reason: str | None = None,
        documents_since_last_dream_at_schedule: int | None = None,
        document_threshold: int | None = None,
    ) -> None:
        """Execute the dream by enqueueing it."""
        from ..deriver.enqueue import enqueue_dream

        ctx = self._ctx_factory()

        # The dream is scoped to the session of the pair's newest explicit conclusion, and that
        # session's configuration decides whether dreams are enabled.
        async with ctx.conn() as conn:
            session_name = await conn.fetchval(
                f"SELECT session_name FROM {ctx.table('conclusions')} "
                "WHERE bank_id = $1 AND observer = $2 AND observed = $3 AND level = 'explicit' "
                "AND deleted_at IS NULL ORDER BY created_at DESC LIMIT 1",
                bank_id,
                observer,
                observed,
            )

        if not session_name:
            logger.warning(f"No documents found for {bank_id}/{observer}/{observed}, skipping dream")
            return

        configuration = get_configuration(
            None, await collection_state.get_session_configuration(ctx, bank_id, session_name), None
        )

        if not configuration.dream.enabled:
            logger.debug(f"Dreams disabled for {bank_id}/{session_name}, skipping dream")
            return

        await enqueue_dream(
            ctx,
            bank_id,
            observer,
            observed,
            dream_type,
            session_name=session_name,
            trigger_reason=trigger_reason,
            delay_reason=delay_reason,
            documents_since_last_dream_at_schedule=documents_since_last_dream_at_schedule,
            document_threshold=document_threshold,
        )

    async def shutdown(self) -> None:
        """Cancel all pending dreams during shutdown."""
        if self.pending_dreams:
            logger.info(f"Cancelling {len(self.pending_dreams)} pending dreams...")
            for task in self.pending_dreams.values():
                task.cancel()
            await asyncio.gather(*self.pending_dreams.values(), return_exceptions=True)
            self.pending_dreams.clear()


# Recorded in the scheduler's `dream` metadata: work-list conclusions the last
# dream left for the next one. The poll treats a pair with a backlog as
# due even when no new conclusions have arrived since.
WORK_LIST_BACKLOG_KEY = "work_list_backlog"


async def schedule_backlog_dream(ctx: PeopleContext, bank_id: str, *, observer: str, observed: str) -> None:
    """Schedule the next dream for a pair whose work list is still behind.

    The timer scheduler only checks a pair when new conclusions are saved, so a
    backlog left after ingestion stops would wait for the next message. The
    timer waits out `MIN_HOURS_BETWEEN_DREAMS`, the same gate a threshold dream
    passes. The poll needs no timer: it reads the recorded backlog.

    Like a threshold dream, the follow-up needs a session to resolve the dream
    configuration from: when the pair's newest explicit conclusion has none
    (created through the API without a session), the backlog waits for a
    manually scheduled dream.
    """
    dream_scheduler = get_dream_scheduler()
    if dream_scheduler is None:
        return
    work_unit_key = dream_work_unit_key(bank_id, observer, observed, DreamType.OMNI)
    await dream_scheduler.schedule_dream(
        work_unit_key,
        bank_id,
        DREAM.MIN_HOURS_BETWEEN_DREAMS * 60,
        DreamType.OMNI,
        observer=observer,
        observed=observed,
        trigger_reason="work_list_backlog",
        delay_reason="min_hours_gate",
    )


async def check_and_schedule_dream(ctx: PeopleContext, bank_id: str, observer: str, observed: str) -> bool:
    """
    From the moment a dream is scheduled until it completes or fails, no second
    dream may be enqueued for the same (bank, observer, observed) - and the
    baseline count advances only when consolidation actually happened.

    Check if a pair has reached the explicit-observation threshold and schedule a timer-based dream.

    This function only schedules a timer-based dream if:
    1. Dreams are enabled
    2. Explicit-observation threshold is reached (dreamer output does not count)
    3. Minimum hours between dreams have passed
    4. No dream is already pending in the queue for this pair (in-flight check)
    5. No dream is already scheduled for this pair

    Args:
        ctx: People context
        bank_id: Bank identifier
        observer: The observing peer
        observed: The observed peer

    Returns:
        True if a dream timer was scheduled, False otherwise
    """
    if not DREAM.ENABLED:
        return False

    internal_metadata = await collection_state.get_collection_internal_metadata(ctx, bank_id, observer, observed)
    dream_metadata = internal_metadata.get("dream", {})
    last_dream_document_count = dream_metadata.get("last_dream_document_count", 0)
    last_dream_at = dream_metadata.get("last_dream_at")

    # Count explicit-level docs only: dreamer output (deductive/inductive/
    # contradiction) would inflate the threshold and create a feedback loop.
    current_explicit_count, _ = await collection_state.explicit_stats(ctx, bank_id, observer, observed)

    documents_since_last_dream = current_explicit_count - last_dream_document_count

    logger.debug(
        "Dream check",
        extra={
            "bank_id": bank_id,
            "observer": observer,
            "observed": observed,
            "current_explicit_count": current_explicit_count,
            "last_dream_document_count": last_dream_document_count,
            "documents_since_last_dream": documents_since_last_dream,
            "document_threshold": DREAM.DOCUMENT_THRESHOLD,
        },
    )

    if documents_since_last_dream >= DREAM.DOCUMENT_THRESHOLD:
        # Capture *why* this schedule fired (threshold) and *how* it will fire (idle vs immediate). The
        # two gates were intentionally split: collapsing them into a single trigger_reason would lose
        # the scheduling semantics.
        trigger_reason = "document_threshold"
        delay_reason = "idle_timeout" if DREAM.IDLE_TIMEOUT_MINUTES > 0 else "immediate"
        if last_dream_at:
            try:
                last_dream_time = datetime.fromisoformat(last_dream_at)
                hours_since_last_dream = (datetime.now(UTC) - last_dream_time).total_seconds() / 3600

                if hours_since_last_dream < DREAM.MIN_HOURS_BETWEEN_DREAMS:
                    logger.debug(
                        f"Skipping dream for {observer}/{observed}: only {hours_since_last_dream:.1f} hours "
                        + f"since last dream (minimum: {DREAM.MIN_HOURS_BETWEEN_DREAMS})"
                    )
                    return False
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid last_dream_at timestamp: {last_dream_at}, error: {e}")

        # Queue is source of truth for in-flight dreams; mirrors uq_people_queue_dream_pending.
        enabled_dream_types = DREAM.ENABLED_TYPES
        pending_keys = [
            dream_work_unit_key(bank_id, observer, observed, dream_type) for dream_type in enabled_dream_types
        ]
        async with ctx.conn() as conn:
            pending_exists = await conn.fetchval(
                f"SELECT EXISTS (SELECT 1 FROM {ctx.table('queue')} "
                "WHERE bank_id = $1 AND task_type = 'dream' AND processed = false "
                "AND work_unit_key = ANY($2::text[]))",
                bank_id,
                pending_keys,
            )
        if pending_exists:
            logger.debug(
                "Skipping dream schedule for %s/%s: pending dream already in queue",
                observer,
                observed,
            )
            return False

        dream_scheduler = get_dream_scheduler()
        if dream_scheduler:
            for dream_type in enabled_dream_types:
                await dream_scheduler.schedule_dream(
                    dream_work_unit_key(bank_id, observer, observed, dream_type),
                    bank_id,
                    DREAM.IDLE_TIMEOUT_MINUTES,
                    dream_type=DreamType(dream_type),
                    observer=observer,
                    observed=observed,
                    trigger_reason=trigger_reason,
                    delay_reason=delay_reason,
                    documents_since_last_dream_at_schedule=documents_since_last_dream,
                    document_threshold=DREAM.DOCUMENT_THRESHOLD,
                )
                logger.debug(
                    "Scheduled dream",
                    extra={
                        "bank_id": bank_id,
                        "observer": observer,
                        "observed": observed,
                        "documents_since_last_dream": documents_since_last_dream,
                        "document_threshold": DREAM.DOCUMENT_THRESHOLD,
                        "dream_type": dream_type,
                    },
                )
            return True

    return False
