# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: tests/deriver/test_queue_processing.py
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: tests/deriver/test_work_unit_lifecycle.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""Bounded retry of transient errors, terminal errors, and work unit ownership (leaks, poison units)."""

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import asyncpg
import pytest
from pydantic import ValidationError

from hindsight_api.people.context import PeopleContext
from hindsight_api.people.deriver.consumer import process_item
from hindsight_api.people.deriver.enqueue import create_dream_record, enqueue_dream
from hindsight_api.people.deriver.models import QueueItem
from hindsight_api.people.deriver.queue_manager import MAX_RETRYABLE_ATTEMPTS, QueueManager
from hindsight_api.people.deriver.queue_payload import RETRY_ATTEMPTS_PAYLOAD_KEY
from hindsight_api.people.deriver.settings import DERIVER
from hindsight_api.people.dreamer.payload import DreamPayload
from hindsight_api.people.dreamer.schemas import DreamType
from hindsight_api.people.jsonb import to_jsonb
from tests.people.deriver_support import claim, claim_rows, items_of, retry_attempts, seed_unit

BATCH = "hindsight_api.people.deriver.queue_manager.process_representation_batch"
RETRY_BACKOFF = "hindsight_api.people.deriver.queue_manager.RETRY_BACKOFF_SECONDS"


def _retryable_error() -> Exception:
    """A deadlock: Postgres sqlstate 40P01."""
    return asyncpg.exceptions.DeadlockDetectedError("deadlock detected")


async def _seed_claimed(
    ctx: PeopleContext, bank_id: str, n_messages: int = 1
) -> tuple[QueueManager, str, str, list[QueueItem]]:
    """A claimed representation work unit owned by a test worker."""
    key, items, _ = await seed_unit(ctx, bank_id, [10] * n_messages)
    qm, worker_id, _ = await claim(ctx, bank_id, key)
    return qm, key, worker_id, items


class _CountingBatches:
    """Wrap ``get_queue_item_batch`` to count how often the unit was fetched."""

    def __init__(self, qm: QueueManager) -> None:
        self.count = 0
        self._original = qm.get_queue_item_batch

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.count += 1
        return await self._original(*args, **kwargs)


@pytest.mark.asyncio
class TestQueueRetry:
    """A transient failure (deadlock, lost connection, provider transport) leaves the batch's queue items
    unprocessed and releases the work unit for re-claim, up to MAX_RETRYABLE_ATTEMPTS per work unit;
    terminal failures burn one item."""

    async def test_retryable_error_leaves_items_unprocessed(
        self, ctx: PeopleContext, bank_id: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A transient error stops the work unit after ONE batch fetch (no tight loop), leaves items
        unprocessed with no error, and releases the claim."""
        monkeypatch.setattr(RETRY_BACKOFF, 0.0)
        qm, key, worker_id, _ = await _seed_claimed(ctx, bank_id, n_messages=2)
        initial_semaphore_value = qm.semaphore._value
        counting = _CountingBatches(qm)

        with (
            patch.object(qm, "get_queue_item_batch", new=counting),
            patch(BATCH, side_effect=_retryable_error()),
        ):
            await qm.process_work_unit(key, worker_id)

        assert counting.count == 1
        items = await items_of(ctx, bank_id, key)
        assert all(not item.processed for item in items)
        assert all(item.error is None for item in items)
        assert await claim_rows(ctx, bank_id, key) == 0
        assert retry_attempts(items) == 1
        assert qm.semaphore._value == initial_semaphore_value

    async def test_retry_exhaustion_is_terminal(
        self, ctx: PeopleContext, bank_id: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """At the attempt cap a transient error burns the first item like the terminal path and clears the
        counter."""
        monkeypatch.setattr(RETRY_BACKOFF, 0.0)
        qm, key, worker_id, _ = await _seed_claimed(ctx, bank_id)
        await qm._set_work_unit_retry_attempts(bank_id, key, MAX_RETRYABLE_ATTEMPTS - 1)

        with patch(BATCH, side_effect=_retryable_error()):
            await qm.process_work_unit(key, worker_id)

        items = await items_of(ctx, bank_id, key)
        assert len(items) == 1
        assert items[0].processed
        assert items[0].error is not None
        assert "DeadlockDetectedError" in items[0].error
        assert retry_attempts(items) is None

    async def test_non_retryable_error_burns_immediately(self, ctx: PeopleContext, bank_id: str) -> None:
        """A non-retryable error marks the first item errored on the first attempt."""
        qm, key, worker_id, _ = await _seed_claimed(ctx, bank_id)

        with patch(BATCH, side_effect=ValueError("bad batch")):
            await qm.process_work_unit(key, worker_id)

        items = await items_of(ctx, bank_id, key)
        assert len(items) == 1
        assert items[0].processed
        assert items[0].error is not None
        assert "ValueError" in items[0].error
        assert retry_attempts(items) is None

    async def test_a_terminal_error_burns_one_item_and_the_rest_continue(
        self, ctx: PeopleContext, bank_id: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only the first queue item of a failed batch is errored; the rest of the unit is still processed."""
        monkeypatch.setattr(DERIVER, "REPRESENTATION_BATCH_TARGET_INPUT_TOKENS", 100)
        key, _, _ = await seed_unit(ctx, bank_id, [80, 80, 80])
        qm, worker_id, _ = await claim(ctx, bank_id, key)
        calls: list[int] = []

        async def fail_once(_ctx: Any, _bank: Any, messages: list[Any], *_a: Any, **_k: Any) -> None:
            calls.append(len(messages))
            if len(calls) == 1:
                raise ValueError("bad batch")

        with patch(BATCH, fail_once):
            await qm.process_work_unit(key, worker_id)

        items = await items_of(ctx, bank_id, key)
        assert [(i.processed, i.error is not None) for i in items] == [(True, True), (True, False), (True, False)]
        assert len(calls) == 3

    async def test_error_text_postgres_rejects_still_marks_item_errored(self, ctx: PeopleContext, bank_id: str) -> None:
        """An exception message echoing NUL or a lone surrogate still reaches the terminal error state.
        Unescaped, Postgres rejects the UPDATE and the item is never marked, so it is processed again."""
        qm, key, worker_id, _ = await _seed_claimed(ctx, bank_id)

        with (
            patch(BATCH, side_effect=ValueError("bad value 'a\x00b' and '\ud83d'")),
            # The error is logged verbatim, and pytest-xdist can't ship a captured log holding a lone
            # surrogate back from its worker.
            patch("hindsight_api.people.deriver.queue_manager.logger"),
        ):
            await asyncio.wait_for(qm.process_work_unit(key, worker_id), timeout=10)

        items = await items_of(ctx, bank_id, key)
        assert len(items) == 1
        assert items[0].processed
        assert items[0].error == "ValueError: bad value 'a\\x00b' and '\\ud83d'"

    async def test_failed_error_mark_releases_work_unit(
        self, ctx: PeopleContext, bank_id: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the terminal mark itself fails, the item is still unprocessed and next in line, so the work
        unit is released (one batch fetch) rather than re-running the same item in a tight loop."""
        monkeypatch.setattr(RETRY_BACKOFF, 0.0)
        qm, key, worker_id, _ = await _seed_claimed(ctx, bank_id)
        counting = _CountingBatches(qm)

        with (
            patch.object(qm, "get_queue_item_batch", new=counting),
            patch.object(qm, "mark_queue_item_as_errored", side_effect=RuntimeError("mark failed")),
            patch(BATCH, side_effect=ValueError("bad batch")),
        ):
            await asyncio.wait_for(qm.process_work_unit(key, worker_id), timeout=10)

        assert counting.count == 1
        items = await items_of(ctx, bank_id, key)
        assert all(not item.processed for item in items)
        assert await claim_rows(ctx, bank_id, key) == 0

    async def test_counter_cleared_after_success(self, ctx: PeopleContext, bank_id: str) -> None:
        """A success wipes the accumulated attempt count for the work unit."""
        qm, key, worker_id, _ = await _seed_claimed(ctx, bank_id)
        await qm._set_work_unit_retry_attempts(bank_id, key, 1)

        with patch(BATCH, new=AsyncMock(return_value=None)):
            await qm.process_work_unit(key, worker_id)

        items = await items_of(ctx, bank_id, key)
        assert all(item.processed for item in items)
        assert all(item.error is None for item in items)
        assert retry_attempts(items) is None

    async def test_clearing_the_counter_leaves_the_rest_of_the_payload(self, ctx: PeopleContext, bank_id: str) -> None:
        qm, key, _, items = await _seed_claimed(ctx, bank_id)
        await qm._set_work_unit_retry_attempts(bank_id, key, 2)
        assert retry_attempts(await items_of(ctx, bank_id, key)) == 2
        assert await qm._get_work_unit_retry_attempts(bank_id, key) == 2

        await qm._clear_work_unit_retry_attempts(bank_id, key)

        (after,) = await items_of(ctx, bank_id, key)
        assert RETRY_ATTEMPTS_PAYLOAD_KEY not in after.payload
        assert after.payload == items[0].payload
        assert await qm._get_work_unit_retry_attempts(bank_id, key) == 0

    async def test_retry_budget_survives_reclaim_by_another_manager(
        self, ctx: PeopleContext, bank_id: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A second QueueManager continues the durable attempt budget."""
        monkeypatch.setattr(RETRY_BACKOFF, 0.0)
        qm1, key, worker_id, _ = await _seed_claimed(ctx, bank_id)

        with patch(BATCH, side_effect=_retryable_error()):
            await qm1.process_work_unit(key, worker_id)

        assert retry_attempts(await items_of(ctx, bank_id, key)) == 1
        assert await claim_rows(ctx, bank_id, key) == 0

        # Seed the remaining budget so the next reclaim is the terminal attempt.
        qm2 = QueueManager(lambda: ctx)
        await qm2._set_work_unit_retry_attempts(bank_id, key, MAX_RETRYABLE_ATTEMPTS - 1)
        qm2, worker_id_2, _ = await claim(ctx, bank_id, key, "test_worker_2")

        with patch(BATCH, side_effect=_retryable_error()):
            await qm2.process_work_unit(key, worker_id_2)

        items = await items_of(ctx, bank_id, key)
        assert len(items) == 1
        assert items[0].processed
        assert items[0].error is not None
        assert "DeadlockDetectedError" in items[0].error

    async def test_retry_counter_read_and_write_follow_the_oldest_unprocessed_item(
        self, ctx: PeopleContext, bank_id: str
    ) -> None:
        qm, key, _, items = await _seed_claimed(ctx, bank_id, n_messages=2)
        assert await qm._get_work_unit_retry_attempts(bank_id, key) == 0
        await qm.mark_queue_items_as_processed(items[:1], key)

        await qm._set_work_unit_retry_attempts(bank_id, key, 2)

        after = await items_of(ctx, bank_id, key)
        assert RETRY_ATTEMPTS_PAYLOAD_KEY not in after[0].payload
        assert after[1].payload[RETRY_ATTEMPTS_PAYLOAD_KEY] == 2

    async def test_process_item_strips_retry_counter_before_validation(self, ctx: PeopleContext) -> None:
        """A reclaimed dream item must survive its own retry counter.

        The counter is written onto an *unprocessed* item so the budget outlives a work-unit reclaim, which
        means the next claim re-reads it. Every payload model sets ``extra="forbid"``, so without the strip
        in ``process_item`` the reclaim raises extra_forbidden -> ValueError -> not retryable -> the item is
        burned terminally on the very attempt that was supposed to retry it."""
        raw: dict[str, Any] = {
            "task_type": "dream",
            "dream_type": "omni",
            "observer": "alice",
            "observed": "bob",
            RETRY_ATTEMPTS_PAYLOAD_KEY: 1,
        }
        # Pin the premise: the payload model must keep rejecting the key, so this fails loudly if someone
        # "fixes" the burn with extra="allow" instead of stripping.
        with pytest.raises(ValidationError) as exc_info:
            DreamPayload.model_validate(raw)
        assert any(err["type"] == "extra_forbidden" for err in exc_info.value.errors())

        queue_item = QueueItem(
            id=7,
            bank_id="bank-1",
            work_unit_key="dream:omni:bank-1:alice:bob",
            task_type="dream",
            payload=raw,
        )

        with patch("hindsight_api.people.dreamer.orchestrator.process_dream", new_callable=AsyncMock) as process:
            await process_item(ctx, queue_item)

        process.assert_awaited_once()
        _ctx, payload, bank = process.await_args.args
        assert isinstance(payload, DreamPayload) and (payload.observer, payload.observed) == ("alice", "bob")
        assert payload.dream_type == DreamType.OMNI
        assert bank == "bank-1"
        assert process.await_args.kwargs == {"queue_item_id": 7}
        # The strip must happen on a copy: the counter has to stay on the row so the budget still advances
        # if this attempt fails again.
        assert raw[RETRY_ATTEMPTS_PAYLOAD_KEY] == 1

    async def test_process_item_refuses_an_unknown_task_type(self, ctx: PeopleContext) -> None:
        queue_item = QueueItem(id=1, bank_id="b", work_unit_key="k", task_type="summary", payload={})
        with pytest.raises(ValueError, match="Invalid task type"):
            await process_item(ctx, queue_item)


@pytest.mark.asyncio
class TestDreamUnits:
    async def test_a_dream_item_is_processed_once_and_marked_done(self, ctx: PeopleContext, bank_id: str) -> None:
        await enqueue_dream(ctx, bank_id, "alice", "bob", DreamType.OMNI)
        key = create_dream_record(bank_id, observer="alice", observed="bob", dream_type=DreamType.OMNI)["work_unit_key"]
        qm, worker_id, _ = await claim(ctx, bank_id, key)

        with patch("hindsight_api.people.deriver.queue_manager.process_item", new_callable=AsyncMock) as process:
            count = await qm.process_work_unit(key, worker_id)

        assert count == 1
        process.assert_awaited_once()
        assert process.await_args.args[1].task_type == "dream"
        (item,) = await items_of(ctx, bank_id, key)
        assert item.processed and item.error is None
        assert await claim_rows(ctx, bank_id, key) == 0

    async def test_a_failing_dream_is_errored_and_not_retried(self, ctx: PeopleContext, bank_id: str) -> None:
        await enqueue_dream(ctx, bank_id, "alice", "bob", DreamType.OMNI)
        key = create_dream_record(bank_id, observer="alice", observed="bob", dream_type=DreamType.OMNI)["work_unit_key"]
        qm, worker_id, _ = await claim(ctx, bank_id, key)

        with patch(
            "hindsight_api.people.deriver.queue_manager.process_item",
            new_callable=AsyncMock,
            side_effect=RuntimeError("dream failed"),
        ) as process:
            await qm.process_work_unit(key, worker_id)

        process.assert_awaited_once()
        (item,) = await items_of(ctx, bank_id, key)
        assert item.processed and item.error == "RuntimeError: dream failed"


async def _bad_unit(ctx: PeopleContext, bank_id: str, key: str, created_at: datetime | None = None) -> None:
    async with ctx.conn() as conn:
        for processed in (False, False, True):
            await conn.execute(
                f"INSERT INTO {PeopleContext.table('queue')} (bank_id, work_unit_key, task_type, payload, processed"
                + (", created_at" if created_at else "")
                + ") VALUES ($1, $2, 'representation', $3::jsonb, $4"
                + (", $5" if created_at else "")
                + ")",
                bank_id,
                key,
                to_jsonb({}),
                processed,
                *([created_at] if created_at else []),
            )


@pytest.mark.asyncio
class TestWorkUnitLifecycle:
    """Regressions for ownership leaks and poison work units."""

    async def _claimed_dream(self, ctx: PeopleContext, bank_id: str) -> tuple[QueueManager, str, str]:
        await enqueue_dream(ctx, bank_id, "alice", "bob", DreamType.OMNI)
        key = create_dream_record(bank_id, observer="alice", observed="bob", dream_type=DreamType.OMNI)["work_unit_key"]
        qm, worker_id, _ = await claim(ctx, bank_id, key)
        return qm, key, worker_id

    async def _assert_claim_released(self, ctx: PeopleContext, bank_id: str, qm: QueueManager, key: str) -> None:
        assert qm.get_total_owned_work_units() == 0
        assert not qm.semaphore.locked()
        assert await claim_rows(ctx, bank_id, key) == 0

    @pytest.mark.parametrize(
        "bad_key",
        [
            "representation:bank:session",
            "representation:bank:session:observed:extra",
            "dream:omni:bank:alice",
            "dream:omni:bank:a:b:extra",
            "unknown:bank:session:None:None",
        ],
    )
    async def test_malformed_key_is_quarantined_and_valid_work_continues(
        self, ctx: PeopleContext, bank_id: str, monkeypatch: pytest.MonkeyPatch, bad_key: str
    ) -> None:
        """One malformed key must not poison a one-worker process or its restart."""
        monkeypatch.setattr(DERIVER, "WORKERS", 1)
        monkeypatch.setattr(DERIVER, "FLUSH_ENABLED", True)
        await _bad_unit(ctx, bank_id, bad_key, created_at=datetime.now(UTC) - timedelta(minutes=1))
        await enqueue_dream(ctx, bank_id, "alice", "bob", DreamType.OMNI)
        good_key = create_dream_record(bank_id, observer="alice", observed="bob", dream_type=DreamType.OMNI)[
            "work_unit_key"
        ]

        manager = QueueManager(lambda: ctx, bank_id=bank_id)
        (claimed,) = await manager.get_and_claim_work_units()
        assert claimed.work_unit_key == bad_key
        manager.track_worker_work_unit("bad-worker", bank_id, bad_key, claimed.aqs_id)
        await manager.process_work_unit(bad_key, "bad-worker")

        await self._assert_claim_released(ctx, bank_id, manager, bad_key)
        bad_items = await items_of(ctx, bank_id, bad_key)
        assert [i.processed for i in bad_items] == [True, True, True]
        assert all(i.error is not None and i.error.startswith("ValueError:") for i in bad_items[:2])
        assert bad_items[2].error is None  # already processed before: untouched
        (good_item,) = await items_of(ctx, bank_id, good_key)
        assert not good_item.processed and good_item.error is None

        (claimed,) = await manager.get_and_claim_work_units()
        assert claimed.work_unit_key == good_key
        manager.track_worker_work_unit("good-worker", bank_id, good_key, claimed.aqs_id)
        with patch("hindsight_api.people.deriver.queue_manager.process_item", new_callable=AsyncMock) as process:
            await manager.process_work_unit(good_key, "good-worker")
            process.assert_awaited_once()
        await self._assert_claim_released(ctx, bank_id, manager, good_key)
        (good_item,) = await items_of(ctx, bank_id, good_key)
        assert good_item.processed
        assert await manager.get_and_claim_work_units() == []

    async def test_cleanup_failure_still_releases_in_memory_ownership(
        self, ctx: PeopleContext, bank_id: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(DERIVER, "WORKERS", 1)
        manager, key, worker = await self._claimed_dream(ctx, bank_id)
        with (
            patch.object(manager, "get_next_queue_item", new=AsyncMock(return_value=None)),
            patch.object(
                manager, "_cleanup_work_unit", new=AsyncMock(side_effect=ConnectionError("cleanup unavailable"))
            ),
            pytest.raises(ConnectionError, match="cleanup unavailable"),
        ):
            await manager.process_work_unit(key, worker)
        assert manager.get_total_owned_work_units() == 0
        assert not manager.semaphore.locked()

    async def test_cancellation_waiting_for_semaphore_releases_claim(
        self, ctx: PeopleContext, bank_id: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(DERIVER, "WORKERS", 1)
        manager, key, worker = await self._claimed_dream(ctx, bank_id)
        await manager.semaphore.acquire()
        task = asyncio.create_task(manager.process_work_unit(key, worker))
        try:
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            manager.semaphore.release()
        await self._assert_claim_released(ctx, bank_id, manager, key)

    async def test_failed_quarantine_releases_claim_without_discarding_items(
        self, ctx: PeopleContext, bank_id: str
    ) -> None:
        key = "representation:bank:session"
        await _bad_unit(ctx, bank_id, key)
        manager, worker, _ = await claim(ctx, bank_id, key)
        with (
            patch.object(
                manager,
                "_mark_invalid_work_unit_as_errored",
                new=AsyncMock(side_effect=ConnectionError("quarantine unavailable")),
            ),
            pytest.raises(ConnectionError, match="quarantine unavailable"),
        ):
            await manager.process_work_unit(key, worker)
        await self._assert_claim_released(ctx, bank_id, manager, key)
        items = await items_of(ctx, bank_id, key)
        assert [i.processed for i in items] == [False, False, True]
        assert all(i.error is None for i in items)

    async def test_cancellation_during_processing_releases_claim(self, ctx: PeopleContext, bank_id: str) -> None:
        manager, key, worker = await self._claimed_dream(ctx, bank_id)
        started = asyncio.Event()

        async def wait_for_cancellation(*_: object) -> None:
            started.set()
            await asyncio.Event().wait()

        with patch.object(manager, "get_next_queue_item", new=AsyncMock(side_effect=wait_for_cancellation)):
            task = asyncio.create_task(manager.process_work_unit(key, worker))
            await asyncio.wait_for(started.wait(), timeout=5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        await self._assert_claim_released(ctx, bank_id, manager, key)

    async def test_cancellation_during_cleanup_releases_in_memory_ownership(
        self, ctx: PeopleContext, bank_id: str
    ) -> None:
        manager, key, worker = await self._claimed_dream(ctx, bank_id)
        with (
            patch.object(manager, "get_next_queue_item", new=AsyncMock(return_value=None)),
            patch.object(manager, "_cleanup_work_unit", new=AsyncMock(side_effect=asyncio.CancelledError)),
            pytest.raises(asyncio.CancelledError),
        ):
            await manager.process_work_unit(key, worker)
        assert manager.get_total_owned_work_units() == 0
        assert not manager.semaphore.locked()

    async def test_losing_ownership_stops_the_unit(self, ctx: PeopleContext, bank_id: str) -> None:
        key, items, _ = await seed_unit(ctx, bank_id, [10, 10])
        qm, worker_id, _ = await claim(ctx, bank_id, key)
        qm.worker_ownership.pop(worker_id)
        with patch(BATCH, new_callable=AsyncMock) as batch:
            assert await qm.process_work_unit(key, worker_id, bank_id=bank_id) == 0
        batch.assert_not_awaited()

    async def test_process_work_unit_needs_to_know_the_bank(self, ctx: PeopleContext) -> None:
        with pytest.raises(ValueError, match="needs a bank_id"):
            await QueueManager(lambda: ctx).process_work_unit("representation:b:s:a", "nobody")

    async def test_the_heartbeat_runs_while_a_unit_is_processed(
        self, ctx: PeopleContext, bank_id: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A long item keeps its claim fresh: the beat touches the claim while the work runs."""
        monkeypatch.setattr("hindsight_api.people.deriver.queue_manager.claim_heartbeat_interval_seconds", lambda: 0.05)
        manager, key, worker = await self._claimed_dream(ctx, bank_id)
        beats = 0
        real_touch = manager._touch_claim

        async def counting_touch(*args: Any) -> bool:
            nonlocal beats
            beats += 1
            return await real_touch(*args)

        async def slow_item(*_: Any) -> None:
            await asyncio.sleep(0.4)

        with (
            patch.object(manager, "_touch_claim", side_effect=counting_touch),
            patch("hindsight_api.people.deriver.queue_manager.process_item", side_effect=slow_item),
        ):
            await manager.process_work_unit(key, worker)

        assert beats >= 3
        (item,) = await items_of(ctx, bank_id, key)
        assert item.processed

    async def test_task_exception_is_logged_when_task_finishes(
        self, ctx: PeopleContext, caplog: pytest.LogCaptureFixture
    ) -> None:
        async def fail() -> None:
            raise RuntimeError("unexpected worker failure")

        manager = QueueManager(lambda: ctx)
        task = asyncio.create_task(fail())
        with caplog.at_level("ERROR", logger="hindsight_api.people.deriver.queue_manager"):
            manager.add_task(task)
            await asyncio.gather(task, return_exceptions=True)
            await asyncio.sleep(0)
        assert not manager.active_tasks
        assert "unexpected worker failure" in caplog.text

    async def test_cancelled_task_is_not_logged_as_an_error(
        self, ctx: PeopleContext, caplog: pytest.LogCaptureFixture
    ) -> None:
        manager = QueueManager(lambda: ctx)
        task = asyncio.create_task(asyncio.sleep(60))
        manager.add_task(task)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)
        assert not manager.active_tasks
        assert not [record for record in caplog.records if record.levelname == "ERROR"]
