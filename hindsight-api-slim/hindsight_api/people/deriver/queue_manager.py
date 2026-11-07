# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/deriver/queue_manager.py
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/utils/sanitization.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
import asyncio
import contextlib
import random
import re
import time
from asyncio import Task
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, NamedTuple

from ..context import PeopleContext
from ..ids import generate_nanoid
from ..jsonb import from_jsonb, to_jsonb
from ..schemas import ResolvedConfiguration
from .consumer import process_item, process_representation_batch
from .models import MESSAGE_COLUMNS, QUEUE_COLUMNS, Message, QueueItem
from .queue_payload import RETRY_ATTEMPTS_PAYLOAD_KEY
from .queue_store import (
    claim_heartbeat_interval_seconds,
    claim_rows_query,
)
from .queue_store import (
    cleanup_stale_work_units as crud_cleanup_stale_work_units,
)
from .retryable_errors import is_retryable_error
from .settings import DERIVER
from .work_unit import ParsedWorkUnit, parse_work_unit_key

logger = getLogger(__name__)

# Total processing attempts per work unit for transient errors. Count is
# stored on the oldest unprocessed queue item so every deriver instance
# shares one budget.
MAX_RETRYABLE_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 1.0


class WorkerOwnership(NamedTuple):
    """Represents the instance of a work unit that a worker is processing."""

    bank_id: str
    work_unit_key: str
    aqs_id: str  # The ID of the active queue session that the worker is processing


@dataclass(frozen=True)
class QueueBatchResult:
    """Result of `QueueManager.get_queue_item_batch`.

    Beyond the batch contents it says whether the cumulative-token cap clamped the batch, and what
    the configured cap was.
    """

    messages_context: list[Message] = field(default_factory=list)
    items_to_process: list[QueueItem] = field(default_factory=list)
    configuration: ResolvedConfiguration | None = None
    hit_batch_token_cap: bool = False
    was_flush_enabled: bool = False
    batch_max_tokens: int = 0


def _resolve_batch_configuration(
    items_to_process: list[QueueItem],
) -> tuple[list[QueueItem], ResolvedConfiguration | None]:
    """Keep only the initial homogeneous configuration prefix for a batch."""
    if not items_to_process:
        return [], None

    raw_config = items_to_process[0].payload.get("configuration")
    resolved_config = None if raw_config is None else ResolvedConfiguration.model_validate(raw_config)

    valid_items: list[QueueItem] = []
    for item in items_to_process:
        item_raw_config = item.payload.get("configuration")
        item_config = None if item_raw_config is None else ResolvedConfiguration.model_validate(item_raw_config)
        if item_config != resolved_config:
            break
        valid_items.append(item)

    return valid_items, resolved_config


# NUL and UTF-16 surrogate code points: the only characters Postgres can't store in text. A Python str can
# hold surrogates (a JSON "\\ud83d" escape decodes to one), and they can't be UTF-8 encoded on the way to
# the driver.
_UNSTORABLE = re.compile("[\x00\ud800-\udfff]")


def _escape_unstorable(text: str) -> str:
    """Replace NUL and lone surrogates with their escape text, e.g. ``\\x00``.

    For diagnostic text such as a stored error message, where the bad character is evidence worth keeping.
    """
    return _UNSTORABLE.sub(
        lambda m: "\\x00" if m.group() == "\x00" else f"\\u{ord(m.group()):04x}",
        text,
    )


def _rowcount(status: str) -> int:
    """The row count of an asyncpg command status (``"UPDATE 3"``)."""
    try:
        return int(status.rsplit(" ", 1)[-1])
    except ValueError:
        return 0


class QueueManager:
    def __init__(self, ctx_factory: Callable[[], PeopleContext], *, bank_id: str | None = None):
        """``bank_id`` restricts the manager to one bank's work (default: every bank in the schema)."""
        self._ctx_factory = ctx_factory
        self.bank_id = bank_id
        self.shutdown_event: asyncio.Event = asyncio.Event()
        self.active_tasks: set[asyncio.Task[Any]] = set()
        self.worker_ownership: dict[str, WorkerOwnership] = {}
        self.queue_empty_flag: asyncio.Event = asyncio.Event()
        self._polling_task: asyncio.Task[None] | None = None

        # Current adaptive polling interval; grows while idle/erroring and
        # resets to the base interval as soon as work is claimed.
        self._current_poll_interval: float = DERIVER.POLLING_SLEEP_INTERVAL_SECONDS

        # Monotonic timestamp of the last stale-work-unit cleanup ATTEMPT.
        # None -> the first poll always runs cleanup (recovers rows left stale
        # by a crashed predecessor immediately).
        self._last_stale_cleanup_attempt: float | None = None
        # Jittered gate width (seconds) sampled ONCE per attempt, so the deadline
        # for the next run is fixed when the timestamp is set rather than
        # re-rolled on every poll.
        self._stale_cleanup_gate_seconds: float = 0.0

        self.workers: int = DERIVER.WORKERS
        self.semaphore: asyncio.Semaphore = asyncio.Semaphore(self.workers)

    @property
    def ctx(self) -> PeopleContext:
        return self._ctx_factory()

    def add_task(self, task: asyncio.Task[Any]) -> None:
        """Track a task so shutdown can wait for it, and surface its exception when it ends."""
        self.active_tasks.add(task)
        task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task[Any]) -> None:
        self.active_tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error(
                "Queue task exception",
                exc_info=(type(error), error, error.__traceback__),
            )

    def track_worker_work_unit(self, worker_id: str, bank_id: str, work_unit_key: str, aqs_id: str) -> None:
        """Track a work unit owned by a specific worker"""
        self.worker_ownership[worker_id] = WorkerOwnership(bank_id, work_unit_key, aqs_id)

    def untrack_worker_work_unit(self, worker_id: str, work_unit_key: str) -> None:
        """Remove a work unit from worker tracking"""
        ownership = self.worker_ownership.get(worker_id)
        if ownership and ownership.work_unit_key == work_unit_key:
            del self.worker_ownership[worker_id]

    async def _touch_claim(self, bank_id: str, work_unit_key: str, aqs_id: str) -> bool:
        """Refresh one claim's last_updated. False when the row is already gone."""
        async with self.ctx.conn() as conn:
            status = await conn.execute(
                f"UPDATE {PeopleContext.table('active_queue_sessions')} SET last_updated = now() "
                "WHERE bank_id = $1 AND work_unit_key = $2 AND id = $3",
                bank_id,
                work_unit_key,
                aqs_id,
            )
        return _rowcount(status) > 0

    async def _heartbeat_claim(self, bank_id: str, work_unit_key: str, worker_id: str) -> None:
        """Keep this worker's claim alive for as long as it holds the work unit.

        A claim is otherwise refreshed only when a queue ITEM completes
        (``mark_queue_items_as_processed``). A work unit that is one long item -- a
        dream -- therefore never refreshes it, and ``cleanup_stale_work_units`` reaps
        the claim out from under it once it runs past ``STALE_SESSION_TIMEOUT_MINUTES``.
        Another worker then claims the same unit and runs it concurrently.

        Beating from here rather than from inside each long task covers every task
        type at once, including ones added later.

        The beat shares the event loop with the work, so anything that blocks the
        loop -- CPU-bound code, a sync ``time.sleep``, a blocking library call --
        silences every claim in this process at once. That coupling is deliberate: a
        thread-based beat would keep a wedged process's claims alive forever, so no
        other worker could take its work over. Blocking work belongs in
        ``asyncio.to_thread``.
        """
        interval = claim_heartbeat_interval_seconds()
        while True:
            await asyncio.sleep(interval)
            ownership = self.worker_ownership.get(worker_id)
            if ownership is None or ownership.work_unit_key != work_unit_key:
                return
            try:
                if not await self._touch_claim(bank_id, work_unit_key, ownership.aqs_id):
                    # Already reaped: a concurrent claimer may now own this unit.
                    logger.warning(
                        "Claim for work unit %s (AQS %s) no longer exists; it was reaped while still being processed",
                        work_unit_key,
                        ownership.aqs_id,
                    )
                    return
            except Exception:
                # Never let a heartbeat failure kill the work it is protecting; the
                # next beat retries, and four may fail before the claim goes stale.
                logger.warning(
                    "Failed to refresh claim for work unit %s",
                    work_unit_key,
                    exc_info=True,
                )

    def create_worker_id(self) -> str:
        """Generate a unique worker ID for this processing task"""
        return generate_nanoid()

    def get_total_owned_work_units(self) -> int:
        """Get the total number of work units owned by all workers"""
        return len(self.worker_ownership)

    async def start(self) -> None:
        """Start the polling loop in the background (the server owns signals and the event loop)."""
        if self._polling_task is not None and not self._polling_task.done():
            return
        self.shutdown_event.clear()
        logger.debug(f"Starting QueueManager with {self.workers} workers")
        self._polling_task = asyncio.create_task(self._run(), name="people-deriver-queue-manager")

    async def _run(self) -> None:
        try:
            await self._sleep_startup_jitter()
            await self.polling_loop()
        finally:
            await self.cleanup()

    async def stop(self) -> None:
        """Stop polling, wait for work in flight to reach a stopping point, and release the claims."""
        await self.shutdown()
        task, self._polling_task = self._polling_task, None
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        await self.cleanup()

    async def shutdown(self) -> None:
        """Handle graceful shutdown"""
        logger.info("Queue manager shutting down...")
        self.shutdown_event.set()

        if self.active_tasks:
            logger.info(f"Waiting for {len(self.active_tasks)} active tasks to complete...")
            await asyncio.gather(*self.active_tasks, return_exceptions=True)

    async def cleanup(self) -> None:
        """Clean up owned work units"""
        total_work_units = self.get_total_owned_work_units()
        if total_work_units > 0:
            logger.debug(f"Cleaning up {total_work_units} owned work units...")
            try:
                claims = PeopleContext.table("active_queue_sessions")
                async with self.ctx.conn() as conn:
                    for ownership in list(self.worker_ownership.values()):
                        await conn.execute(
                            f"DELETE FROM {claims} WHERE bank_id = $1 AND id = $2",
                            ownership.bank_id,
                            ownership.aqs_id,
                        )
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")
            finally:
                self.worker_ownership.clear()

    ##########################
    # Polling and Scheduling #
    ##########################

    async def _maybe_cleanup_stale_work_units(self) -> None:
        """Run stale-work-unit cleanup at most once per (jittered) interval.

        Staleness is a minutes-timescale condition (STALE_SESSION_TIMEOUT_MINUTES),
        but the polling loop fires on a seconds timescale -- running cleanup
        unconditionally per poll multiplies into unnecessary write transactions.
        Concurrent cleaners on other instances remain safe via FOR UPDATE SKIP
        LOCKED. The gate tracks the last ATTEMPT (set before running), so a failing
        cleanup waits a full interval instead of retrying every poll against a DB
        that is already struggling. An interval of 0 preserves run-every-poll
        behavior.
        """
        interval = DERIVER.STALE_WORK_UNIT_CLEANUP_INTERVAL_SECONDS
        if (
            interval > 0.0
            and self._last_stale_cleanup_attempt is not None
            and time.monotonic() - self._last_stale_cleanup_attempt < self._stale_cleanup_gate_seconds
        ):
            return
        # Record the attempt and fix the next deadline before running, so the
        # gate width is stable for this cycle and a failing cleanup still waits.
        self._last_stale_cleanup_attempt = time.monotonic()
        self._stale_cleanup_gate_seconds = self._jitter(interval)
        await self.cleanup_stale_work_units()

    async def cleanup_stale_work_units(self) -> None:
        """Clean up stale work units"""
        await crud_cleanup_stale_work_units(self.ctx)

    async def get_and_claim_work_units(
        self,
        *,
        bank_id: str | None = None,
        exclude_keys: Collection[str] = (),
        flush: bool = False,
        limit: int | None = None,
    ) -> list[WorkerOwnership]:
        """
        Get available work units that aren't being processed by any other worker, and claim them.

        A representation work unit is only claimable once its pending messages add up to
        REPRESENTATION_BATCH_WORK_UNIT_TARGET_TOKENS or its oldest pending item exceeds
        REPRESENTATION_BATCH_MAX_AGE_SECONDS, unless FLUSH_ENABLED (or ``flush``) is set.
        Returns the claims taken, oldest work unit first.
        """
        if limit is None:
            limit = max(0, self.workers - self.get_total_owned_work_units())
        if limit == 0:
            return []

        if bank_id is None:
            bank_id = self.bank_id
        sql, args = claim_rows_query(limit, bank_id=bank_id, exclude_keys=list(exclude_keys), flush=flush)
        work_unit_target_tokens = DERIVER.REPRESENTATION_BATCH_WORK_UNIT_TARGET_TOKENS

        claimed: list[WorkerOwnership] = []
        async with self.ctx.conn() as conn, conn.transaction():
            available_rows = await conn.fetch(sql, *args)
            for row in available_rows:
                if (
                    not flush
                    and not DERIVER.FLUSH_ENABLED
                    and DERIVER.REPRESENTATION_BATCH_MAX_AGE_SECONDS > 0
                    and row["task_type"] == "representation"
                    and int(row["total_tokens"]) < work_unit_target_tokens
                ):
                    logger.info(
                        "age-flushing work unit %s (tokens=%s < %s, oldest=%s)",
                        row["work_unit_key"],
                        row["total_tokens"],
                        work_unit_target_tokens,
                        row["oldest_created_at"],
                    )
            # Claim per bank: the unique (bank_id, work_unit_key) is the durable claim marker.
            keys_by_bank: dict[str, list[str]] = {}
            for row in available_rows:
                keys_by_bank.setdefault(row["bank_id"], []).append(row["work_unit_key"])
            claimed_mapping = {
                bank: await self.claim_work_units(conn, bank, keys) for bank, keys in keys_by_bank.items()
            }
            claimed = [
                WorkerOwnership(
                    row["bank_id"], row["work_unit_key"], claimed_mapping[row["bank_id"]][row["work_unit_key"]]
                )
                for row in available_rows
                if row["work_unit_key"] in claimed_mapping[row["bank_id"]]
            ]

        logger.debug(f"Claimed {len(claimed)} work units: {[c.work_unit_key for c in claimed]}")
        return claimed

    async def claim_work_units(self, conn: Any, bank_id: str, work_unit_keys: Sequence[str]) -> dict[str, str]:
        """
        Claim work units and return a mapping of work_unit_key to aqs_id.
        Returns only the work units that were successfully claimed: a unit another claimer won
        inserts nothing and is not a claim.
        """
        claims = PeopleContext.table("active_queue_sessions")
        claimed_mapping: dict[str, str] = {}
        for key in work_unit_keys:
            aqs_id = await conn.fetchval(
                f"INSERT INTO {claims} (bank_id, id, work_unit_key) VALUES ($1, $2, $3) "
                "ON CONFLICT (bank_id, work_unit_key) DO NOTHING RETURNING id",
                bank_id,
                generate_nanoid(),
                key,
            )
            if aqs_id is not None:
                claimed_mapping[key] = aqs_id
        logger.debug(f"Claimed {len(claimed_mapping)} work units: {list(claimed_mapping.keys())}")
        return claimed_mapping

    def _reset_poll_interval(self) -> None:
        """Snap the polling interval back to the base after finding work."""
        self._current_poll_interval = DERIVER.POLLING_SLEEP_INTERVAL_SECONDS

    def _jitter(self, seconds: float) -> float:
        """Scatter a sleep by a random factor in [1 - ratio, 1 + ratio].

        Spreads poll loops that would otherwise re-converge. A ratio of 0 returns
        the input unchanged.
        """
        ratio = DERIVER.POLLING_JITTER_RATIO
        if ratio <= 0.0:
            return seconds
        # Scheduling jitter, not security/crypto - stdlib random is appropriate.
        return seconds * random.uniform(1.0 - ratio, 1.0 + ratio)  # nosec B311

    async def _sleep(self, seconds: float) -> None:
        """Sleep, returning early when shutdown is requested."""
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self.shutdown_event.wait(), timeout=seconds)

    async def _sleep_startup_jitter(self) -> None:
        """Sleep a random delay before the first poll so instances that start
        together don't poll in lockstep. Interruptible by shutdown so a signal
        during the delay exits promptly. No-op when the window is 0.0.
        """
        window = DERIVER.POLLING_STARTUP_JITTER_SECONDS
        if window <= 0.0:
            return
        # Scheduling jitter, not security/crypto - stdlib random is appropriate.
        delay = random.uniform(0.0, window)  # nosec B311
        logger.debug(f"Startup poll jitter: sleeping {delay:.1f}s before first poll")
        await self._sleep(delay)

    def _advance_poll_interval(self) -> float:
        """Return the current idle/backoff sleep, then grow it toward the cap."""
        interval = self._current_poll_interval
        if DERIVER.POLLING_BACKOFF_ENABLED:
            self._current_poll_interval = min(
                self._current_poll_interval * DERIVER.POLLING_BACKOFF_MULTIPLIER,
                DERIVER.POLLING_SLEEP_MAX_INTERVAL_SECONDS,
            )
        return self._jitter(interval)

    async def polling_loop(self) -> None:
        """Main polling loop to find and process new work units"""
        logger.debug("Starting polling loop")
        try:
            while not self.shutdown_event.is_set():
                if self.queue_empty_flag.is_set():
                    # The empty-poll branch below already slept this cycle's
                    # interval; just clear the flag and re-query (no second
                    # sleep - that would double the effective idle interval).
                    self.queue_empty_flag.clear()
                    continue

                # Check if we have capacity before querying. There is work to do
                # (workers are busy), so keep the base interval for fast pickup
                # when capacity frees rather than backing off.
                if self.semaphore.locked():
                    await self._sleep(self._jitter(DERIVER.POLLING_SLEEP_INTERVAL_SECONDS))
                    continue

                try:
                    await self._maybe_cleanup_stale_work_units()
                    claimed_work_units = await self.get_and_claim_work_units()
                    if claimed_work_units:
                        self._reset_poll_interval()
                        for claim in claimed_work_units:
                            # Create a new task for processing this work unit
                            if not self.shutdown_event.is_set():
                                # Track worker ownership
                                worker_id = self.create_worker_id()
                                self.track_worker_work_unit(worker_id, claim.bank_id, claim.work_unit_key, claim.aqs_id)

                                task: Task[int] = asyncio.create_task(
                                    self.process_work_unit(claim.work_unit_key, worker_id, bank_id=claim.bank_id)
                                )
                                self.add_task(task)
                    else:
                        self.queue_empty_flag.set()
                        await self._sleep(self._advance_poll_interval())
                except Exception:
                    logger.exception("Error in polling loop")
                    # Back off so a down/saturated DB isn't hammered every cycle.
                    await self._sleep(self._advance_poll_interval())
        finally:
            logger.info("Polling loop stopped")

    async def process_available(self, bank_id: str | None = None, *, flush: bool = True) -> int:
        """Drain the queue once: claim every claimable work unit and process it to the end.

        ``flush`` (the default) bypasses the batch token threshold and the age gate, so everything
        pending is processed now; pass ``flush=False`` to apply the gate like the polling loop does.
        A work unit that is released with items left (a transient error awaiting its retry) is not
        claimed again by this call. Returns the number of queue items processed.
        """
        processed = 0
        skipped: set[str] = set()
        while not self.shutdown_event.is_set():
            claimed = await self.get_and_claim_work_units(
                bank_id=bank_id, exclude_keys=skipped, flush=flush, limit=max(1, self.workers)
            )
            if not claimed:
                break

            async def run(claim: WorkerOwnership) -> int:
                worker_id = self.create_worker_id()
                self.track_worker_work_unit(worker_id, claim.bank_id, claim.work_unit_key, claim.aqs_id)
                count = await self.process_work_unit(claim.work_unit_key, worker_id, bank_id=claim.bank_id)
                if not await self._is_drained(claim.bank_id, claim.work_unit_key):
                    skipped.add(claim.work_unit_key)
                return count

            processed += sum(await asyncio.gather(*(run(claim) for claim in claimed)))
        return processed

    ######################
    # Queue Worker Logic #
    ######################

    async def _handle_processing_error(
        self,
        error: Exception,
        items: list[QueueItem],
        bank_id: str,
        work_unit_key: str,
        context: str,
    ) -> bool:
        """
        Handle a processing error. Returns True when the caller should stop
        processing and release the work unit for a later re-claim.

        Transient errors (is_retryable_error) get up to MAX_RETRYABLE_ATTEMPTS
        attempts per work unit: items stay unprocessed with no error recorded.
        The attempt count lives on the oldest unprocessed queue item so a
        different deriver instance continues the same budget after reclaim.
        Reprocessing is at-least-once, not idempotent: the batch is re-derived
        by a fresh LLM call, so identical text collapses via exact dedup and
        near-identical text via semantic dedup. Retries can therefore inflate
        times_derived -- acceptable because the alternative is dropping the batch.

        Terminal errors mark only the first queue item as errored so we don't
        potentially throw away a batch. This allows us to incrementally attempt
        to process the batch while still maintaining progress in a work unit.
        If that mark fails, the work unit is released as for a transient error.

        Args:
            error: The exception that occurred
            items: The queue items that were being processed
            bank_id: The bank of the work unit
            work_unit_key: The work unit key for the queue items
            context: Context string describing what was being processed (e.g., "processing representation batch")
        """
        if is_retryable_error(error):
            try:
                attempts = await self._get_work_unit_retry_attempts(bank_id, work_unit_key) + 1
                if attempts < MAX_RETRYABLE_ATTEMPTS:
                    await self._set_work_unit_retry_attempts(bank_id, work_unit_key, attempts)
                    logger.warning(
                        "Transient error %s for work unit %s (attempt %d/%d); leaving items unprocessed for retry",
                        context,
                        work_unit_key,
                        attempts,
                        MAX_RETRYABLE_ATTEMPTS,
                        exc_info=error,
                    )
                    return True
            except Exception:
                logger.exception(
                    "Retry-counter I/O failed for work unit %s; releasing %s without recording an attempt",
                    work_unit_key,
                    context,
                )
                return True

        error_msg = f"{error.__class__.__name__}: {str(error)}"
        try:
            if items:
                # Clear retry metadata only after the terminal mark commits so a
                # failed mark leaves the shared budget intact for the next claim.
                await self.mark_queue_item_as_errored(items[0], work_unit_key, error_msg)
            await self._clear_work_unit_retry_attempts(bank_id, work_unit_key)
        except Exception as mark_error:
            logger.error(
                f"Failed to mark queue items as errored for work unit {work_unit_key}: {mark_error}",
                exc_info=True,
            )
            mark_failed = True
        else:
            mark_failed = False

        logger.error(
            f"Error {context} for work unit {work_unit_key}: {error}",
            exc_info=True,
        )
        # An unmarked item is still the next one in the work unit, so carrying on
        # would re-run it at once, LLM call included, in a tight loop. Release the
        # work unit instead and let a later poll re-claim it after the backoff.
        return mark_failed

    async def process_work_unit(self, work_unit_key: str, worker_id: str, *, bank_id: str | None = None) -> int:
        """Process all queue items for a specific work unit by routing to the correct handler.

        Returns the number of queue items processed.
        """
        logger.debug(f"Starting to process work unit {work_unit_key}")
        ownership0 = self.worker_ownership.get(worker_id)
        if bank_id is None:
            bank_id = ownership0.bank_id if ownership0 is not None else None
        if bank_id is None:
            raise ValueError("process_work_unit needs a bank_id (or a tracked worker that owns the work unit)")
        work_unit: ParsedWorkUnit | None = None
        queue_item_count = 0
        heartbeat = None
        try:
            async with self.semaphore:
                try:
                    work_unit = parse_work_unit_key(work_unit_key)
                except ValueError as error:
                    logger.exception("Invalid work unit key %s", work_unit_key)
                    await self._mark_invalid_work_unit_as_errored(bank_id, work_unit_key, error)
                    return 0

                # Hold the claim open for as long as this worker owns the unit; the
                # finally below cancels it. Started here rather than inside the long
                # task types, so every task type gets it. See _heartbeat_claim for
                # what goes wrong without it.
                heartbeat = asyncio.create_task(self._heartbeat_claim(bank_id, work_unit_key, worker_id))

                while not self.shutdown_event.is_set():
                    # Get worker ownership info for verification
                    ownership = self.worker_ownership.get(worker_id)
                    if not ownership or ownership.work_unit_key != work_unit_key:
                        logger.warning(
                            f"Worker {worker_id} lost ownership of work unit {work_unit_key}, stopping processing {work_unit_key}"
                        )
                        break
                    try:
                        if work_unit.task_type == "representation":
                            batch_result = await self.get_queue_item_batch(
                                bank_id, work_unit.task_type, work_unit_key, ownership.aqs_id
                            )
                            messages_context = batch_result.messages_context
                            items_to_process = batch_result.items_to_process
                            message_level_configuration = batch_result.configuration
                            logger.debug(
                                f"Worker {worker_id} retrieved {len(messages_context)} messages and {len(items_to_process)} queue items for work unit {work_unit_key} (AQS ID: {ownership.aqs_id})"
                            )
                            if not items_to_process:
                                logger.debug(
                                    f"No more queue items to process for work unit {work_unit_key} for worker {worker_id}"
                                )
                                break

                            try:
                                # Extract observers from the payload (handle both old and new format)
                                payload = items_to_process[0].payload
                                observers = payload.get("observers")
                                if observers is None:
                                    # Legacy format: single observer string
                                    legacy_observer = payload.get("observer")
                                    if legacy_observer:
                                        observers = [legacy_observer]
                                    else:
                                        observers = []

                                queue_item_message_ids = [
                                    item.message_id for item in items_to_process if item.message_id is not None
                                ]
                                await process_representation_batch(
                                    self.ctx,
                                    bank_id,
                                    messages_context,
                                    message_level_configuration,
                                    observers=observers,
                                    observed=work_unit.observed,
                                    queue_item_message_ids=queue_item_message_ids,
                                    session_id=items_to_process[0].session_id,
                                    queue_item_ids=[item.id for item in items_to_process],
                                    hit_batch_token_cap=batch_result.hit_batch_token_cap,
                                    was_flush_enabled=batch_result.was_flush_enabled,
                                    batch_max_tokens=batch_result.batch_max_tokens,
                                )
                                await self.mark_queue_items_as_processed(items_to_process, work_unit_key)
                                queue_item_count += len(items_to_process)
                            except Exception as e:
                                if await self._handle_processing_error(
                                    e,
                                    items_to_process,
                                    bank_id,
                                    work_unit_key,
                                    f"processing {work_unit.task_type} batch",
                                ):
                                    # Release the work unit (via the finally
                                    # below) and let a later poll re-claim it.
                                    await self._sleep(self._jitter(RETRY_BACKOFF_SECONDS))
                                    break

                        else:
                            queue_item = await self.get_next_queue_item(
                                bank_id, work_unit.task_type, work_unit_key, ownership.aqs_id
                            )
                            if not queue_item:
                                logger.debug(
                                    f"No more queue items to process for work unit {work_unit_key} for worker {worker_id}"
                                )
                                break

                            try:
                                await process_item(self.ctx, queue_item)
                                await self.mark_queue_items_as_processed([queue_item], work_unit_key)
                                queue_item_count += 1
                            except Exception as e:
                                if await self._handle_processing_error(
                                    e,
                                    [queue_item],
                                    bank_id,
                                    work_unit_key,
                                    "processing queue item",
                                ):
                                    await self._sleep(self._jitter(RETRY_BACKOFF_SECONDS))
                                    break

                    except Exception as e:
                        logger.error(
                            f"Error in processing loop for work unit {work_unit_key}: {e}",
                            exc_info=True,
                        )
                        # Back off and release the unit rather than retry in
                        # place: an error raised before the iteration's first
                        # await would otherwise spin without yielding and stall
                        # every worker on this event loop.
                        await self._sleep(self._jitter(RETRY_BACKOFF_SECONDS))
                        break

                    # Check for shutdown after processing each batch
                    if self.shutdown_event.is_set():
                        logger.debug(
                            "Shutdown requested, stopping processing for work unit %s",
                            work_unit_key,
                        )
                        break

        finally:
            # Cancel before the claim is released below. An in-flight beat
            # could not resurrect a deleted row -- an UPDATE matches nothing --
            # but it would log a misleading "reaped while processing" warning
            # on an ordinary teardown.
            if heartbeat is not None:
                heartbeat.cancel()
            # Also release ownership on parse failure or cancellation while
            # waiting for the semaphore. A failed DB cleanup must not leak the
            # in-memory slot; its claim can be recovered by stale-claim cleanup.
            try:
                ownership = self.worker_ownership.get(worker_id)
                if ownership and ownership.work_unit_key == work_unit_key:
                    await self._cleanup_work_unit(bank_id, ownership.aqs_id, work_unit_key)
            finally:
                self.untrack_worker_work_unit(worker_id, work_unit_key)
        return queue_item_count

    async def get_next_queue_item(
        self, bank_id: str, task_type: str, work_unit_key: str, aqs_id: str
    ) -> QueueItem | None:
        """Get the next queue item to process for a specific work unit."""
        if task_type == "representation":
            raise ValueError("representation tasks are not supported for get_next_queue_item")
        async with self.ctx.conn() as conn:
            # The claim row verifies that this worker still owns the work unit.
            row = await conn.fetchrow(
                f"""
                SELECT {", ".join(f"q.{c.strip()} AS {c.strip()}" for c in QUEUE_COLUMNS.split(","))}
                FROM {PeopleContext.table("queue")} q
                JOIN {PeopleContext.table("active_queue_sessions")} a
                  ON a.bank_id = q.bank_id AND a.work_unit_key = q.work_unit_key
                WHERE q.bank_id = $1 AND q.work_unit_key = $2 AND NOT q.processed AND a.id = $3
                ORDER BY q.id
                LIMIT 1
                """,
                bank_id,
                work_unit_key,
                aqs_id,
            )
        return QueueItem.from_record(row) if row is not None else None

    async def get_queue_item_batch(
        self,
        bank_id: str,
        task_type: str,
        work_unit_key: str,
        aqs_id: str,
    ) -> "QueueBatchResult":
        """
        Batch processing for representation tasks.

        Returns a `QueueBatchResult` carrying:
        - messages_context: unique messages (conversation turns) forming the context window
        - items_to_process: queue items for the current work_unit_key within that window
        - configuration: Resolved configuration for the batch
        - hit_batch_token_cap: True when the cumulative-token window clamped the batch
        - was_flush_enabled: snapshot of `DERIVER.FLUSH_ENABLED` at fetch time
        - batch_max_tokens: snapshot of the cap actually applied to this batch
        """
        if task_type != "representation":
            raise ValueError(f"{task_type} tasks are not supported for get_queue_item_batch")

        batch_max_tokens = DERIVER.REPRESENTATION_BATCH_TARGET_INPUT_TOKENS
        was_flush_enabled = DERIVER.FLUSH_ENABLED
        parsed_key = parse_work_unit_key(work_unit_key)
        messages_context: list[Message] = []
        items_to_process: list[QueueItem] = []

        queue = PeopleContext.table("queue")
        messages_table = PeopleContext.table("messages")
        claims = PeopleContext.table("active_queue_sessions")
        queue_select = ", ".join(f"q.{c.strip()} AS q_{c.strip()}" for c in QUEUE_COLUMNS.split(","))

        async with self.ctx.conn() as conn:
            # Step 1: Verify worker still owns the work_unit_key. Failing returns an empty batch, which
            # ends the loop and releases the unit.
            owned = await conn.fetchval(
                f"SELECT 1 FROM {claims} WHERE bank_id = $1 AND work_unit_key = $2 AND id = $3",
                bank_id,
                work_unit_key,
                aqs_id,
            )
            if not owned:
                return QueueBatchResult(was_flush_enabled=was_flush_enabled, batch_max_tokens=batch_max_tokens)

            # Step 2: Build a single SQL query that:
            # 1. Finds the earliest unprocessed message for this work_unit_key
            # 2. Optionally includes the preceding message if from a different peer (for context)
            # 3. Gets ALL messages from that point forward (for conversational context)
            # 4. Tracks cumulative tokens
            # 5. Returns messages up to token limit, always including the first unprocessed message
            rows = await conn.fetch(
                f"""
                WITH min_unprocessed AS (
                    -- the minimum message id with an unprocessed queue item across the session
                    SELECT MIN(m.id) AS id
                    FROM {queue} q
                    JOIN {messages_table} m ON m.bank_id = q.bank_id AND m.id = q.message_id
                    WHERE q.bank_id = $1 AND q.work_unit_key = $3 AND NOT q.processed
                      AND m.session_name = $2
                ),
                preceding AS (
                    -- the immediately preceding message, only when it is from a different peer than the
                    -- observed one: it gives conversational context (the question that prompted the response)
                    SELECT m.id
                    FROM {messages_table} m
                    WHERE m.bank_id = $1 AND m.session_name = $2 AND m.peer_name <> $4
                      AND m.id = (
                          SELECT MAX(p.id) FROM {messages_table} p
                          WHERE p.bank_id = $1 AND p.session_name = $2 AND p.id < (SELECT id FROM min_unprocessed)
                      )
                ),
                start AS (
                    SELECT COALESCE((SELECT id FROM preceding), (SELECT id FROM min_unprocessed)) AS id
                ),
                inner_cte AS (
                    SELECT m.id AS message_id,
                           SUM(m.token_count) OVER (ORDER BY m.id) AS cumulative_token_count
                    FROM {messages_table} m
                    WHERE m.bank_id = $1 AND m.session_name = $2 AND m.id >= (SELECT id FROM start)
                ),
                cte AS (
                    -- cap_exceeded is identical on every row: whether the cap excluded at least one message
                    SELECT message_id, cumulative_token_count,
                           bool_or(cumulative_token_count > $5) OVER () AS cap_exceeded
                    FROM inner_cte
                )
                SELECT {MESSAGE_COLUMNS}, {queue_select}, cte.cap_exceeded
                FROM cte
                JOIN {messages_table} m ON m.bank_id = $1 AND m.id = cte.message_id
                LEFT JOIN {queue} q
                  ON q.bank_id = $1 AND q.work_unit_key = $3 AND NOT q.processed AND q.message_id = cte.message_id
                WHERE cte.cumulative_token_count <= $5
                   OR cte.message_id = (SELECT id FROM min_unprocessed)  -- always include the first unprocessed message
                ORDER BY cte.message_id, q.id
                """,
                bank_id,
                parsed_key.session_name,
                work_unit_key,
                parsed_key.observed,
                batch_max_tokens,
            )
            if not rows:
                return QueueBatchResult(was_flush_enabled=was_flush_enabled, batch_max_tokens=batch_max_tokens)

            # cap_exceeded is window-aggregated over the CTE - same value on every row.
            cap_exceeded_from_query: bool = bool(rows[0]["cap_exceeded"]) if batch_max_tokens > 0 else False

            seen_messages: set[int] = set()
            for row in rows:
                if row["id"] not in seen_messages:
                    messages_context.append(Message.from_record(row))
                    seen_messages.add(row["id"])
                if row["q_id"] is not None:
                    items_to_process.append(QueueItem.from_record(row, prefix="q_"))

        # The QUEUE-ITEM boundary (not the messages_context tail) is
        # what matters for cap detection. messages_context includes
        # non-queue interleaving context messages - if SQL kept some
        # trailing context past the last queued item, the config
        # filter trims that context but doesn't touch the queue items.
        last_queued_id_before: int | None = (
            max(qi.message_id for qi in items_to_process if qi.message_id is not None) if items_to_process else None
        )

        items_to_process, resolved_config = _resolve_batch_configuration(items_to_process)
        if items_to_process:
            max_queue_item_message_id = max(qi.message_id for qi in items_to_process if qi.message_id is not None)
            messages_context = [m for m in messages_context if m.id <= max_queue_item_message_id]

        last_queued_id_after: int | None = (
            max(qi.message_id for qi in items_to_process if qi.message_id is not None) if items_to_process else None
        )

        # Detect if `batch_max_tokens` clamped this returned batch: the cap was binding when the config
        # filter did not shrink the QUEUE-ITEM boundary (SQL chose the trailing queue item, not config)
        # and the CTE saw at least one message past the cap.
        if batch_max_tokens > 0 and last_queued_id_before is not None and last_queued_id_before == last_queued_id_after:
            hit_batch_token_cap = cap_exceeded_from_query
        else:
            hit_batch_token_cap = False

        return QueueBatchResult(
            messages_context=messages_context,
            items_to_process=items_to_process,
            configuration=resolved_config,
            hit_batch_token_cap=hit_batch_token_cap,
            was_flush_enabled=was_flush_enabled,
            batch_max_tokens=batch_max_tokens,
        )

    async def _oldest_unprocessed_item(self, conn: Any, bank_id: str, work_unit_key: str) -> QueueItem | None:
        row = await conn.fetchrow(
            f"SELECT {QUEUE_COLUMNS} FROM {PeopleContext.table('queue')} "
            "WHERE bank_id = $1 AND work_unit_key = $2 AND NOT processed ORDER BY id LIMIT 1",
            bank_id,
            work_unit_key,
        )
        return QueueItem.from_record(row) if row is not None else None

    async def _get_work_unit_retry_attempts(self, bank_id: str, work_unit_key: str) -> int:
        """Read the shared transient-failure attempt count for a work unit."""
        async with self.ctx.conn() as conn:
            item = await self._oldest_unprocessed_item(conn, bank_id, work_unit_key)
        if item is None:
            return 0
        raw = (item.payload or {}).get(RETRY_ATTEMPTS_PAYLOAD_KEY, 0)
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return 0

    async def _set_work_unit_retry_attempts(self, bank_id: str, work_unit_key: str, attempts: int) -> None:
        """Persist the shared attempt count on the oldest unprocessed item."""
        queue = PeopleContext.table("queue")
        async with self.ctx.conn() as conn, conn.transaction():
            row = await conn.fetchrow(
                f"SELECT id, payload FROM {queue} "
                "WHERE bank_id = $1 AND work_unit_key = $2 AND NOT processed ORDER BY id LIMIT 1 FOR UPDATE",
                bank_id,
                work_unit_key,
            )
            if row is None:
                return
            new_payload = dict(from_jsonb(row["payload"], {}) or {})
            new_payload[RETRY_ATTEMPTS_PAYLOAD_KEY] = attempts
            await conn.execute(
                f"UPDATE {queue} SET payload = $3::jsonb WHERE bank_id = $1 AND id = $2",
                bank_id,
                row["id"],
                to_jsonb(new_payload),
            )

    async def _clear_work_unit_retry_attempts(self, bank_id: str, work_unit_key: str) -> None:
        """Drop the shared attempt count from remaining unprocessed items.

        One statement on purpose: a multi-row ``SELECT ... FOR UPDATE`` here
        would take locks on the queue in scan order, which is a deadlock partner
        for any other multi-row writer on the same table. The JSONB ``-``
        operator does the strip server-side, so no rows are locked ahead of the
        write and there is no lock order to get wrong.
        """
        async with self.ctx.conn() as conn:
            await conn.execute(
                f"UPDATE {PeopleContext.table('queue')} SET payload = payload - $3::text "
                "WHERE bank_id = $1 AND work_unit_key = $2 AND NOT processed AND payload ? $3::text",
                bank_id,
                work_unit_key,
                RETRY_ATTEMPTS_PAYLOAD_KEY,
            )

    async def _mark_invalid_work_unit_as_errored(self, bank_id: str, work_unit_key: str, error: ValueError) -> None:
        """Quarantine pending rows sharing a key that cannot be parsed.

        Unlike a processing failure, an invalid key makes every item in the
        work unit unprocessable. Match the raw key without parsing it again.
        """
        error_msg = _escape_unstorable(f"{error.__class__.__name__}: {error}")[:65535]
        async with self.ctx.conn() as conn:
            await conn.execute(
                f"UPDATE {PeopleContext.table('queue')} SET processed = true, error = $3 "
                "WHERE bank_id = $1 AND work_unit_key = $2 AND NOT processed",
                bank_id,
                work_unit_key,
                error_msg,
            )

    async def mark_queue_items_as_processed(self, items: list[QueueItem], work_unit_key: str) -> None:
        if not items:
            return
        bank_id = items[0].bank_id
        item_ids = [item.id for item in items]
        async with self.ctx.conn() as conn, conn.transaction():
            await conn.execute(
                f"UPDATE {PeopleContext.table('queue')} SET processed = true "
                "WHERE bank_id = $1 AND id = ANY($2::bigint[]) AND work_unit_key = $3",
                bank_id,
                item_ids,
                work_unit_key,
            )
            await conn.execute(
                f"UPDATE {PeopleContext.table('active_queue_sessions')} SET last_updated = now() "
                "WHERE bank_id = $1 AND work_unit_key = $2",
                bank_id,
                work_unit_key,
            )

    async def mark_queue_item_as_errored(self, item: QueueItem, work_unit_key: str, error: str) -> None:
        """Mark queue item as processed with an error"""
        if not item:
            return
        async with self.ctx.conn() as conn, conn.transaction():
            await conn.execute(
                f"UPDATE {PeopleContext.table('queue')} SET processed = true, error = $4 "
                "WHERE bank_id = $1 AND id = $2 AND work_unit_key = $3",
                item.bank_id,
                item.id,
                work_unit_key,
                # Escape before truncating: an exception message can echo a NUL or
                # lone surrogate from its input, and Postgres would reject the
                # UPDATE, leaving the item unprocessed and claimable forever.
                _escape_unstorable(error)[:65535],
            )
            await conn.execute(
                f"UPDATE {PeopleContext.table('active_queue_sessions')} SET last_updated = now() "
                "WHERE bank_id = $1 AND work_unit_key = $2",
                item.bank_id,
                work_unit_key,
            )

    async def _is_drained(self, bank_id: str, work_unit_key: str) -> bool:
        """Whether the work unit has no unprocessed items left."""
        async with self.ctx.conn() as conn:
            remaining = await conn.fetchval(
                f"SELECT EXISTS (SELECT 1 FROM {PeopleContext.table('queue')} "
                "WHERE bank_id = $1 AND work_unit_key = $2 AND NOT processed)",
                bank_id,
                work_unit_key,
            )
        return not remaining

    async def _cleanup_work_unit(self, bank_id: str, aqs_id: str, work_unit_key: str) -> bool:
        """
        Clean up a specific work unit session by both work_unit_key and AQS ID.
        """
        async with self.ctx.conn() as conn:
            status = await conn.execute(
                f"DELETE FROM {PeopleContext.table('active_queue_sessions')} "
                "WHERE bank_id = $1 AND id = $2 AND work_unit_key = $3",
                bank_id,
                aqs_id,
                work_unit_key,
            )
        return _rowcount(status) > 0


async def process_available(ctx: PeopleContext, bank_id: str | None = None, *, flush: bool = True) -> int:
    """Drain the queue once and return the number of queue items processed.

    Runs the deriver synchronously for callers that need the conclusions now (tests, consolidation).
    ``flush`` (the default) bypasses the batch token threshold and the age gate; see
    ``QueueManager.process_available``. Safe to call while a ``QueueManager`` polls: claims are shared
    through ``people_active_queue_sessions``.
    """
    return await QueueManager(lambda: ctx).process_available(bank_id, flush=flush)
