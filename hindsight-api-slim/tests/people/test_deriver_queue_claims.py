# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: tests/deriver/test_queue_processing.py
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: tests/deriver/test_claim_heartbeat.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""Claiming work units: the token gate, age flush, ordering, claim rows, heartbeat and polling."""

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api.people.context import PeopleContext
from hindsight_api.people.deriver.enqueue import create_dream_record, enqueue_dream
from hindsight_api.people.deriver.queue_manager import QueueManager
from hindsight_api.people.deriver.queue_store import (
    claim_heartbeat_interval_seconds,
    cleanup_stale_work_units,
)
from hindsight_api.people.deriver.settings import DERIVER
from hindsight_api.people.dreamer.schemas import DreamType
from tests.people.deriver_support import claim, claim_rows, items_of, seed_unit

WORK_UNIT_KEY = "representation:bank-hb:obs:obsd"


@pytest.fixture(autouse=True)
def _default_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Honcho's defaults for the claim gate, whatever another test left on the shared settings."""
    monkeypatch.setattr(DERIVER, "FLUSH_ENABLED", False)
    monkeypatch.setattr(DERIVER, "REPRESENTATION_BATCH_WORK_UNIT_TARGET_TOKENS", 512)
    monkeypatch.setattr(DERIVER, "REPRESENTATION_BATCH_MAX_AGE_SECONDS", 1800)
    monkeypatch.setattr(DERIVER, "WORKERS", 8)


async def _other_bank(ctx: PeopleContext) -> str:
    bank = f"test-people-{uuid.uuid4().hex[:10]}"
    await ctx.engine.ensure_bank_profile(bank_id=bank, request_context=ctx.request_context)
    return bank


@pytest.mark.asyncio
class TestClaiming:
    async def test_get_and_claim_work_units(self, ctx: PeopleContext, bank_id: str) -> None:
        key_a, _, _ = await seed_unit(ctx, bank_id, [600], peer="alice")
        key_b, _, _ = await seed_unit(ctx, bank_id, [600], peer="bob")

        claimed = await QueueManager(lambda: ctx).get_and_claim_work_units(bank_id=bank_id)

        assert {c.work_unit_key for c in claimed} == {key_a, key_b}
        assert all(c.bank_id == bank_id for c in claimed)
        assert key_a.startswith("representation:") and key_a.endswith(":alice")
        assert await claim_rows(ctx, bank_id) == 2
        assert await claim_rows(ctx, bank_id, key_a) == 1

    async def test_a_claimed_work_unit_is_not_claimed_again(self, ctx: PeopleContext, bank_id: str) -> None:
        key, _, _ = await seed_unit(ctx, bank_id, [600])
        qm = QueueManager(lambda: ctx)

        first = await qm.get_and_claim_work_units(bank_id=bank_id)
        second = await qm.get_and_claim_work_units(bank_id=bank_id)

        assert [c.work_unit_key for c in first] == [key]
        assert second == []

    async def test_claim_work_unit_conflict_returns_nothing(self, ctx: PeopleContext, bank_id: str) -> None:
        key, _, _ = await seed_unit(ctx, bank_id, [600])
        qm = QueueManager(lambda: ctx)
        async with ctx.conn() as conn:
            assert list(await qm.claim_work_units(conn, bank_id, [key])) == [key]
            assert await qm.claim_work_units(conn, bank_id, [key]) == {}

    async def test_concurrent_claimers_get_disjoint_units(self, ctx: PeopleContext, bank_id: str) -> None:
        keys = {(await seed_unit(ctx, bank_id, [600], peer=f"p{i}"))[0] for i in range(6)}

        results = await asyncio.gather(
            *(QueueManager(lambda: ctx).get_and_claim_work_units(bank_id=bank_id) for _ in range(3))
        )

        claimed = [c.work_unit_key for result in results for c in result]
        assert len(claimed) == len(set(claimed)) == len(keys)
        assert set(claimed) == keys

    async def test_claims_are_limited_by_the_free_workers(
        self, ctx: PeopleContext, bank_id: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(DERIVER, "WORKERS", 2)
        for i in range(4):
            await seed_unit(ctx, bank_id, [600], peer=f"p{i}")
        qm = QueueManager(lambda: ctx)

        claimed = await qm.get_and_claim_work_units(bank_id=bank_id)
        assert len(claimed) == 2
        # Workers that are busy leave no room for more claims.
        for index, c in enumerate(claimed):
            qm.track_worker_work_unit(f"w{index}", c.bank_id, c.work_unit_key, c.aqs_id)
        assert await qm.get_and_claim_work_units(bank_id=bank_id) == []

    async def test_the_oldest_work_unit_is_claimed_first(
        self, ctx: PeopleContext, bank_id: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(DERIVER, "WORKERS", 1)
        now = datetime.now(UTC)
        await seed_unit(ctx, bank_id, [600], peer="new", created_ats=[now])
        old_key, _, _ = await seed_unit(ctx, bank_id, [600], peer="old", created_ats=[now - timedelta(minutes=5)])

        claimed = await QueueManager(lambda: ctx).get_and_claim_work_units(bank_id=bank_id)

        assert [c.work_unit_key for c in claimed] == [old_key]

    async def test_only_the_asked_bank_is_claimed(self, ctx: PeopleContext, bank_id: str) -> None:
        other = await _other_bank(ctx)
        key, _, _ = await seed_unit(ctx, bank_id, [600])
        other_key, _, _ = await seed_unit(ctx, other, [600])

        claimed = await QueueManager(lambda: ctx).get_and_claim_work_units(bank_id=bank_id)
        assert [c.work_unit_key for c in claimed] == [key]

        rest = await QueueManager(lambda: ctx).get_and_claim_work_units(bank_id=other)
        assert [(c.bank_id, c.work_unit_key) for c in rest] == [(other, other_key)]

    async def test_excluded_keys_are_skipped(self, ctx: PeopleContext, bank_id: str) -> None:
        key_a, _, _ = await seed_unit(ctx, bank_id, [600], peer="alice")
        key_b, _, _ = await seed_unit(ctx, bank_id, [600], peer="bob")
        claimed = await QueueManager(lambda: ctx).get_and_claim_work_units(bank_id=bank_id, exclude_keys={key_a})
        assert [c.work_unit_key for c in claimed] == [key_b]

    async def test_processed_items_make_no_work(self, ctx: PeopleContext, bank_id: str) -> None:
        key, items, _ = await seed_unit(ctx, bank_id, [600])
        async with ctx.conn() as conn:
            await conn.execute(
                f"UPDATE {PeopleContext.table('queue')} SET processed = true WHERE bank_id = $1", bank_id
            )
        assert await QueueManager(lambda: ctx).get_and_claim_work_units(bank_id=bank_id) == []


@pytest.mark.asyncio
class TestBatchThreshold:
    """A representation unit waits for the token target or the age flush."""

    async def test_a_unit_below_the_token_target_waits(self, ctx: PeopleContext, bank_id: str) -> None:
        key, _, _ = await seed_unit(ctx, bank_id, [100, 100, 100])
        claimed = await QueueManager(lambda: ctx).get_and_claim_work_units(bank_id=bank_id)
        assert key not in {c.work_unit_key for c in claimed}

    async def test_a_unit_at_exactly_the_token_target_is_claimed(self, ctx: PeopleContext, bank_id: str) -> None:
        key, _, _ = await seed_unit(ctx, bank_id, [256, 256])
        claimed = await QueueManager(lambda: ctx).get_and_claim_work_units(bank_id=bank_id)
        assert [c.work_unit_key for c in claimed] == [key]

    async def test_a_unit_one_token_short_waits(self, ctx: PeopleContext, bank_id: str) -> None:
        key, _, _ = await seed_unit(ctx, bank_id, [256, 255])
        assert await QueueManager(lambda: ctx).get_and_claim_work_units(bank_id=bank_id) == []

    async def test_a_single_large_message_is_claimed(self, ctx: PeopleContext, bank_id: str) -> None:
        key, _, _ = await seed_unit(ctx, bank_id, [5000])
        claimed = await QueueManager(lambda: ctx).get_and_claim_work_units(bank_id=bank_id)
        assert [c.work_unit_key for c in claimed] == [key]

    async def test_an_old_sub_threshold_unit_is_age_flushed(self, ctx: PeopleContext, bank_id: str) -> None:
        old = datetime.now(UTC) - timedelta(hours=2)
        key, items, _ = await seed_unit(ctx, bank_id, [100, 100, 100], created_ats=[old, old, old])
        qm = QueueManager(lambda: ctx)

        claimed = await qm.get_and_claim_work_units(bank_id=bank_id)

        assert [c.work_unit_key for c in claimed] == [key]
        batch = await qm.get_queue_item_batch(bank_id, "representation", key, claimed[0].aqs_id)
        assert [item.id for item in batch.items_to_process] == [item.id for item in items]

    async def test_age_flush_zero_keeps_waiting_for_old_items(
        self, ctx: PeopleContext, bank_id: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(DERIVER, "REPRESENTATION_BATCH_MAX_AGE_SECONDS", 0)
        old = datetime.now(UTC) - timedelta(hours=2)
        await seed_unit(ctx, bank_id, [100, 100], created_ats=[old, old])
        assert await QueueManager(lambda: ctx).get_and_claim_work_units(bank_id=bank_id) == []

    async def test_flush_enabled_bypasses_age_and_token_thresholds(
        self, ctx: PeopleContext, bank_id: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(DERIVER, "FLUSH_ENABLED", True)
        key, _, _ = await seed_unit(ctx, bank_id, [100, 100])
        claimed = await QueueManager(lambda: ctx).get_and_claim_work_units(bank_id=bank_id)
        assert [c.work_unit_key for c in claimed] == [key]

    async def test_a_zero_token_target_disables_the_gate(
        self, ctx: PeopleContext, bank_id: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(DERIVER, "REPRESENTATION_BATCH_WORK_UNIT_TARGET_TOKENS", 0)
        key, _, _ = await seed_unit(ctx, bank_id, [1])
        claimed = await QueueManager(lambda: ctx).get_and_claim_work_units(bank_id=bank_id)
        assert [c.work_unit_key for c in claimed] == [key]

    async def test_the_flush_argument_bypasses_the_gate_for_one_call(self, ctx: PeopleContext, bank_id: str) -> None:
        key, _, _ = await seed_unit(ctx, bank_id, [10])
        qm = QueueManager(lambda: ctx)
        assert await qm.get_and_claim_work_units(bank_id=bank_id) == []
        assert [c.work_unit_key for c in await qm.get_and_claim_work_units(bank_id=bank_id, flush=True)] == [key]

    async def test_age_flush_uses_the_oldest_unprocessed_item(self, ctx: PeopleContext, bank_id: str) -> None:
        now = datetime.now(UTC)
        key, _, _ = await seed_unit(ctx, bank_id, [100, 100], created_ats=[now - timedelta(hours=2), now])
        claimed = await QueueManager(lambda: ctx).get_and_claim_work_units(bank_id=bank_id)
        assert [c.work_unit_key for c in claimed] == [key]

    async def test_age_flush_ignores_old_processed_items(self, ctx: PeopleContext, bank_id: str) -> None:
        now = datetime.now(UTC)
        key, items, _ = await seed_unit(ctx, bank_id, [100, 100], created_ats=[now - timedelta(hours=2), now])
        async with ctx.conn() as conn:
            await conn.execute(
                f"UPDATE {PeopleContext.table('queue')} SET processed = true WHERE bank_id = $1 AND id = $2",
                bank_id,
                items[0].id,
            )
        assert await QueueManager(lambda: ctx).get_and_claim_work_units(bank_id=bank_id) == []

    async def test_a_dream_is_never_gated(self, ctx: PeopleContext, bank_id: str) -> None:
        await enqueue_dream(ctx, bank_id, "alice", "bob", DreamType.OMNI)
        record = create_dream_record(bank_id, observer="alice", observed="bob", dream_type=DreamType.OMNI)
        claimed = await QueueManager(lambda: ctx).get_and_claim_work_units(bank_id=bank_id)
        assert [c.work_unit_key for c in claimed] == [record["work_unit_key"]]


@pytest.mark.asyncio
class TestNextQueueItem:
    async def _dream_unit(self, ctx: PeopleContext, bank_id: str) -> tuple[QueueManager, str, str]:
        await enqueue_dream(ctx, bank_id, "alice", "bob", DreamType.OMNI)
        key = create_dream_record(bank_id, observer="alice", observed="bob", dream_type=DreamType.OMNI)["work_unit_key"]
        qm, _, aqs_id = await claim(ctx, bank_id, key)
        return qm, key, aqs_id

    async def test_returns_the_oldest_unprocessed_item_of_the_unit(self, ctx: PeopleContext, bank_id: str) -> None:
        qm, key, aqs_id = await self._dream_unit(ctx, bank_id)
        (item,) = await items_of(ctx, bank_id, key)

        found = await qm.get_next_queue_item(bank_id, "dream", key, aqs_id)
        assert found is not None and found.id == item.id and found.payload["observed"] == "bob"

        await qm.mark_queue_items_as_processed([item], key)
        assert await qm.get_next_queue_item(bank_id, "dream", key, aqs_id) is None

    async def test_needs_the_claim_that_owns_the_unit(self, ctx: PeopleContext, bank_id: str) -> None:
        qm, key, _ = await self._dream_unit(ctx, bank_id)
        assert await qm.get_next_queue_item(bank_id, "dream", key, "not-the-claim") is None

    async def test_representation_units_use_batches(self, ctx: PeopleContext, bank_id: str) -> None:
        qm = QueueManager(lambda: ctx)
        with pytest.raises(ValueError, match="not supported for get_next_queue_item"):
            await qm.get_next_queue_item(bank_id, "representation", "k", "a")
        with pytest.raises(ValueError, match="not supported for get_queue_item_batch"):
            await qm.get_queue_item_batch(bank_id, "dream", "k", "a")


@pytest.mark.asyncio
class TestClaimLifecycle:
    async def _stale_claim(self, ctx: PeopleContext, bank_id: str) -> str:
        """A claim already old enough for the reaper; returns its id."""
        aqs_id = f"stale-{uuid.uuid4().hex[:8]}"
        async with ctx.conn() as conn:
            await conn.execute(
                f"INSERT INTO {PeopleContext.table('active_queue_sessions')} (bank_id, id, work_unit_key, last_updated) "
                "VALUES ($1, $2, $3, now() - make_interval(mins => $4))",
                bank_id,
                aqs_id,
                WORK_UNIT_KEY,
                DERIVER.STALE_SESSION_TIMEOUT_MINUTES + 1,
            )
        return aqs_id

    async def _exists(self, ctx: PeopleContext, bank_id: str, aqs_id: str) -> int:
        async with ctx.conn() as conn:
            return await conn.fetchval(
                f"SELECT count(*) FROM {PeopleContext.table('active_queue_sessions')} WHERE bank_id = $1 AND id = $2",
                bank_id,
                aqs_id,
            )

    async def test_heartbeat_fires_well_inside_the_stale_window(self) -> None:
        """The two halves of the invariant must not be able to drift apart."""
        window = DERIVER.STALE_SESSION_TIMEOUT_MINUTES * 60
        assert claim_heartbeat_interval_seconds() < window / 2, (
            "a heartbeat at or near the stale window cannot keep a claim alive"
        )

    async def test_a_heartbeat_saves_a_claim_from_the_reaper(self, ctx: PeopleContext, bank_id: str) -> None:
        """The property that matters: a claim old enough to be reaped survives a beat."""
        aqs_id = await self._stale_claim(ctx, bank_id)
        qm = QueueManager(lambda: ctx)

        assert await qm._touch_claim(bank_id, WORK_UNIT_KEY, aqs_id) is True

        await cleanup_stale_work_units(ctx)
        assert await self._exists(ctx, bank_id, aqs_id) == 1, "heartbeat did not save the claim from the reaper"

    async def test_without_a_heartbeat_the_same_claim_is_reaped(self, ctx: PeopleContext, bank_id: str) -> None:
        """Counterpart, so the test above cannot pass vacuously."""
        aqs_id = await self._stale_claim(ctx, bank_id)
        await cleanup_stale_work_units(ctx)
        assert await self._exists(ctx, bank_id, aqs_id) == 0

    async def test_touch_claim_reports_an_already_reaped_claim(self, ctx: PeopleContext, bank_id: str) -> None:
        """A beat matching no row means another worker may now own the unit."""
        aqs_id = await self._stale_claim(ctx, bank_id)
        await cleanup_stale_work_units(ctx)
        assert await QueueManager(lambda: ctx)._touch_claim(bank_id, WORK_UNIT_KEY, aqs_id) is False

    async def test_a_stale_claim_frees_its_work_unit(self, ctx: PeopleContext, bank_id: str) -> None:
        key, _, _ = await seed_unit(ctx, bank_id, [600])
        async with ctx.conn() as conn:
            await conn.execute(
                f"INSERT INTO {PeopleContext.table('active_queue_sessions')} (bank_id, id, work_unit_key, last_updated) "
                "VALUES ($1, 'crashed', $2, now() - make_interval(mins => 60))",
                bank_id,
                key,
            )
        qm = QueueManager(lambda: ctx)
        assert await qm.get_and_claim_work_units(bank_id=bank_id) == []

        await qm.cleanup_stale_work_units()

        assert [c.work_unit_key for c in await qm.get_and_claim_work_units(bank_id=bank_id)] == [key]

    async def test_cleanup_work_unit_removes_only_the_matching_claim(self, ctx: PeopleContext, bank_id: str) -> None:
        key, _, _ = await seed_unit(ctx, bank_id, [600])
        qm, _, aqs_id = await claim(ctx, bank_id, key)

        assert await qm._cleanup_work_unit(bank_id, "another-claim", key) is False
        assert await claim_rows(ctx, bank_id, key) == 1
        assert await qm._cleanup_work_unit(bank_id, aqs_id, key) is True
        assert await claim_rows(ctx, bank_id, key) == 0

    async def test_stale_cleanup_is_gated_by_its_interval(
        self, ctx: PeopleContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(DERIVER, "POLLING_JITTER_RATIO", 0.0)
        monkeypatch.setattr(DERIVER, "STALE_WORK_UNIT_CLEANUP_INTERVAL_SECONDS", 3600.0)
        qm = QueueManager(lambda: ctx)
        with patch.object(qm, "cleanup_stale_work_units", new_callable=AsyncMock) as cleanup:
            await qm._maybe_cleanup_stale_work_units()  # the first poll always cleans
            await qm._maybe_cleanup_stale_work_units()
            assert cleanup.await_count == 1

            monkeypatch.setattr(DERIVER, "STALE_WORK_UNIT_CLEANUP_INTERVAL_SECONDS", 0.0)
            qm._stale_cleanup_gate_seconds = 0.0
            await qm._maybe_cleanup_stale_work_units()  # interval 0 means every poll
            await qm._maybe_cleanup_stale_work_units()
            assert cleanup.await_count == 3


class TestPollingJitter:
    """Polling jitter: desynchronize poll loops without changing the schedule."""

    def test_jitter_stays_within_ratio_bounds(self, ctx: PeopleContext, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(DERIVER, "POLLING_JITTER_RATIO", 0.5)
        qm = QueueManager(lambda: ctx)
        samples = [qm._jitter(10.0) for _ in range(1000)]
        assert all(5.0 <= s <= 15.0 for s in samples)
        # A 0.5 ratio over 1000 samples should produce real spread, not a constant.
        assert max(samples) - min(samples) > 1.0

    def test_jitter_ratio_zero_is_identity(self, ctx: PeopleContext, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(DERIVER, "POLLING_JITTER_RATIO", 0.0)
        qm = QueueManager(lambda: ctx)
        assert all(qm._jitter(7.5) == 7.5 for _ in range(50))

    def test_advance_jitters_return_but_keeps_deterministic_schedule(
        self, ctx: PeopleContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(DERIVER, "POLLING_JITTER_RATIO", 0.5)
        monkeypatch.setattr(DERIVER, "POLLING_BACKOFF_ENABLED", True)
        monkeypatch.setattr(DERIVER, "POLLING_BACKOFF_MULTIPLIER", 2.0)
        monkeypatch.setattr(DERIVER, "POLLING_SLEEP_INTERVAL_SECONDS", 1.0)
        monkeypatch.setattr(DERIVER, "POLLING_SLEEP_MAX_INTERVAL_SECONDS", 30.0)
        qm = QueueManager(lambda: ctx)

        # The underlying schedule advances 1 -> 2 -> 4 -> ... -> 30 deterministically;
        # each returned sleep is jittered within [0.5x, 1.5x] of the pre-advance step.
        expected_schedule = [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]
        for step in expected_schedule:
            returned = qm._advance_poll_interval()
            assert 0.5 * step <= returned <= 1.5 * step

    @pytest.mark.asyncio
    async def test_startup_jitter_disabled_returns_immediately(
        self, ctx: PeopleContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(DERIVER, "POLLING_STARTUP_JITTER_SECONDS", 0.0)
        qm = QueueManager(lambda: ctx)
        # Window 0.0 must not sleep at all.
        await asyncio.wait_for(qm._sleep_startup_jitter(), timeout=1.0)

    @pytest.mark.asyncio
    async def test_startup_jitter_interrupted_by_shutdown(
        self, ctx: PeopleContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(DERIVER, "POLLING_STARTUP_JITTER_SECONDS", 300.0)
        qm = QueueManager(lambda: ctx)
        qm.shutdown_event.set()
        # A shutdown already signalled must short-circuit the (long) jitter sleep.
        await asyncio.wait_for(qm._sleep_startup_jitter(), timeout=1.0)


@pytest.mark.asyncio
class TestStartStop:
    async def test_the_polling_loop_drains_the_queue_and_stop_releases_everything(
        self, ctx: PeopleContext, bank_id: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(DERIVER, "POLLING_STARTUP_JITTER_SECONDS", 0.0)
        monkeypatch.setattr(DERIVER, "POLLING_SLEEP_INTERVAL_SECONDS", 0.05)
        monkeypatch.setattr(DERIVER, "POLLING_SLEEP_MAX_INTERVAL_SECONDS", 0.2)
        monkeypatch.setattr(DERIVER, "POLLING_JITTER_RATIO", 0.0)
        key, items, _ = await seed_unit(ctx, bank_id, [600, 600])
        seen: list[Any] = []

        async def fake_batch(_ctx: Any, _bank: Any, messages: list[Any], *_args: Any, **_kwargs: Any) -> None:
            seen.append([m.id for m in messages])

        qm = QueueManager(lambda: ctx, bank_id=bank_id)
        with patch("hindsight_api.people.deriver.queue_manager.process_representation_batch", fake_batch):
            await qm.start()
            for _ in range(200):
                if all(item.processed for item in await items_of(ctx, bank_id, key)):
                    break
                await asyncio.sleep(0.05)
            await qm.stop()

        assert all(item.processed for item in await items_of(ctx, bank_id, key))
        assert seen and sum(len(ids) for ids in seen) >= 2
        assert await claim_rows(ctx, bank_id) == 0
        assert qm.worker_ownership == {} and qm.active_tasks == set()

    async def test_start_twice_runs_one_loop_and_stop_is_repeatable(
        self, ctx: PeopleContext, bank_id: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(DERIVER, "POLLING_STARTUP_JITTER_SECONDS", 0.0)
        qm = QueueManager(lambda: ctx, bank_id=bank_id)
        await qm.start()
        first = qm._polling_task
        await qm.start()
        assert qm._polling_task is first
        await qm.stop()
        await qm.stop()
        assert first is not None and first.done()
