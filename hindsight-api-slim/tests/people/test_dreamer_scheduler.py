# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: tests/dreamer/test_dream_scheduler.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""The dream scheduler: cancelling timers, the threshold check, the due-dream poll (ported from Honcho's
tests/dreamer/test_dream_scheduler.py)."""

from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api.people.context import PeopleContext
from hindsight_api.people.dreamer.dream_scheduler import (
    DreamScheduler,
    check_and_schedule_dream,
    get_dream_scheduler,
    set_dream_scheduler,
)
from hindsight_api.people.dreamer.payload import dream_work_unit_key, parse_dream_work_unit_key
from hindsight_api.people.dreamer.settings import DREAM
from hindsight_api.people.schemas import DreamType
from tests.people.dreamer_support import insert_conclusions, insert_queue_item, make_session, scalar, set_state


@pytest.fixture
def dream_scheduler(ctx: PeopleContext, monkeypatch: pytest.MonkeyPatch):
    """A fresh DreamScheduler for each test."""
    monkeypatch.setattr(DREAM, "ENABLED", True)
    scheduler = DreamScheduler(lambda: ctx)
    set_dream_scheduler(scheduler)
    yield scheduler
    set_dream_scheduler(None)


def _key(bank: str, observer: str, observed: str) -> str:
    return dream_work_unit_key(bank, observer, observed, "omni")


async def _schedule(scheduler: DreamScheduler, bank: str, observer: str, observed: str) -> str:
    key = _key(bank, observer, observed)
    with patch.object(scheduler, "execute_dream", new_callable=AsyncMock):
        await scheduler.schedule_dream(
            key, bank, delay_minutes=60, dream_type=DreamType.OMNI, observer=observer, observed=observed
        )
    return key


class TestWorkUnitKey:
    def test_round_trips_names_with_colons_and_percent(self):
        key = dream_work_unit_key("bank:1", "al%ice", "bo:b", DreamType.OMNI)
        assert parse_dream_work_unit_key(key) == ("omni", "bank:1", "al%ice", "bo:b")

    def test_other_keys_do_not_parse(self):
        assert parse_dream_work_unit_key("representation:bank:s:bob") is None


class TestCancelDreamsForObserved:
    """Peer-to-peer observation dreams are cancelled on activity too, not only self-observation ones."""

    async def test_cancel_self_observation_dream(self, dream_scheduler: DreamScheduler):
        key = await _schedule(dream_scheduler, "test_bank", "bob", "bob")
        assert key in dream_scheduler.pending_dreams

        cancelled = await dream_scheduler.cancel_dreams_for_observed("test_bank", "bob")

        assert key in cancelled
        assert key not in dream_scheduler.pending_dreams

    async def test_cancel_peer_to_peer_observation_dream(self, dream_scheduler: DreamScheduler):
        """Alice observes Bob and Bob sends a message: the (alice, bob) dream is cancelled."""
        key = await _schedule(dream_scheduler, "test_bank", "alice", "bob")

        cancelled = await dream_scheduler.cancel_dreams_for_observed("test_bank", "bob")

        assert key in cancelled
        assert key not in dream_scheduler.pending_dreams

    async def test_cancel_multiple_observers_same_observed(self, dream_scheduler: DreamScheduler):
        keys = {
            await _schedule(dream_scheduler, "test_bank", "bob", "bob"),
            await _schedule(dream_scheduler, "test_bank", "alice", "bob"),
            await _schedule(dream_scheduler, "test_bank", "carol", "bob"),
        }

        cancelled = await dream_scheduler.cancel_dreams_for_observed("test_bank", "bob")

        assert cancelled == keys
        assert dream_scheduler.pending_dreams == {}

    async def test_does_not_cancel_dreams_for_different_observed(self, dream_scheduler: DreamScheduler):
        bob_key = await _schedule(dream_scheduler, "test_bank", "alice", "bob")
        carol_key = await _schedule(dream_scheduler, "test_bank", "alice", "carol")

        cancelled = await dream_scheduler.cancel_dreams_for_observed("test_bank", "bob")

        assert cancelled == {bob_key}
        assert carol_key in dream_scheduler.pending_dreams

    async def test_does_not_cancel_dreams_for_different_bank(self, dream_scheduler: DreamScheduler):
        key_a = await _schedule(dream_scheduler, "bank_a", "bob", "bob")
        key_b = await _schedule(dream_scheduler, "bank_b", "bob", "bob")

        cancelled = await dream_scheduler.cancel_dreams_for_observed("bank_a", "bob")

        assert cancelled == {key_a}
        assert key_b in dream_scheduler.pending_dreams

    async def test_a_second_schedule_replaces_the_first(self, dream_scheduler: DreamScheduler):
        key = await _schedule(dream_scheduler, "test_bank", "bob", "bob")
        first = dream_scheduler.pending_dreams[key]

        await _schedule(dream_scheduler, "test_bank", "bob", "bob")

        assert first.cancelled() or first.done()
        assert dream_scheduler.pending_dreams[key] is not first

    async def test_disabled_dreams_schedule_nothing(self, dream_scheduler: DreamScheduler, monkeypatch):
        monkeypatch.setattr(DREAM, "ENABLED", False)
        await _schedule(dream_scheduler, "test_bank", "bob", "bob")
        assert dream_scheduler.pending_dreams == {}


class TestThresholdFilter:
    """The threshold counts only explicit-level conclusions.

    Counting dreamer output (deductive/inductive/contradiction) would create a feedback loop where each
    dream's output inflates the trigger for the next dream.
    """

    @pytest.fixture(autouse=True)
    def _pin_dream_config(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(DREAM, "DOCUMENT_THRESHOLD", 50)
        monkeypatch.setattr(DREAM, "ENABLED_TYPES", ["omni"])
        monkeypatch.setattr(DREAM, "MIN_HOURS_BETWEEN_DREAMS", 8)

    async def test_mixed_levels_below_explicit_threshold(
        self, dream_scheduler: DreamScheduler, ctx: PeopleContext, bank_id: str
    ):
        """30 explicit + 40 deductive + 10 inductive must not trigger: 80 in total, 30 explicit < 50."""
        await insert_conclusions(ctx, bank_id, "explicit", 30)
        await insert_conclusions(ctx, bank_id, "deductive", 40)
        await insert_conclusions(ctx, bank_id, "inductive", 10)

        with patch.object(dream_scheduler, "schedule_dream", new_callable=AsyncMock):
            scheduled = await check_and_schedule_dream(ctx, bank_id, "ann", "ann")

        assert scheduled is False

    async def test_explicit_only_at_threshold(self, dream_scheduler: DreamScheduler, ctx: PeopleContext, bank_id: str):
        await insert_conclusions(ctx, bank_id, "explicit", 60)

        with patch.object(dream_scheduler, "schedule_dream", new_callable=AsyncMock) as mock_schedule:
            scheduled = await check_and_schedule_dream(ctx, bank_id, "ann", "ann")

        assert scheduled is True
        [call] = mock_schedule.call_args_list
        assert call.args[0] == _key(bank_id, "ann", "ann")
        assert call.kwargs["trigger_reason"] == "document_threshold"
        assert call.kwargs["delay_reason"] == "idle_timeout"
        assert call.kwargs["documents_since_last_dream_at_schedule"] == 60

    async def test_contradiction_excluded_from_count(
        self, dream_scheduler: DreamScheduler, ctx: PeopleContext, bank_id: str
    ):
        await insert_conclusions(ctx, bank_id, "contradiction", 100)
        await insert_conclusions(ctx, bank_id, "explicit", 10)

        with patch.object(dream_scheduler, "schedule_dream", new_callable=AsyncMock):
            scheduled = await check_and_schedule_dream(ctx, bank_id, "ann", "ann")

        assert scheduled is False

    async def test_count_is_measured_from_the_last_dream(
        self, dream_scheduler: DreamScheduler, ctx: PeopleContext, bank_id: str
    ):
        await insert_conclusions(ctx, bank_id, "explicit", 60)
        await set_state(ctx, bank_id, "ann", "ann", {"dream": {"last_dream_document_count": 40}})

        with patch.object(dream_scheduler, "schedule_dream", new_callable=AsyncMock):
            assert await check_and_schedule_dream(ctx, bank_id, "ann", "ann") is False

    async def test_a_recent_dream_blocks_the_schedule(
        self, dream_scheduler: DreamScheduler, ctx: PeopleContext, bank_id: str
    ):
        import datetime

        await insert_conclusions(ctx, bank_id, "explicit", 60)
        recent = (datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=1)).isoformat()
        await set_state(ctx, bank_id, "ann", "ann", {"dream": {"last_dream_at": recent}})

        with patch.object(dream_scheduler, "schedule_dream", new_callable=AsyncMock):
            assert await check_and_schedule_dream(ctx, bank_id, "ann", "ann") is False

    async def test_a_pending_queued_dream_blocks_the_schedule(
        self, dream_scheduler: DreamScheduler, ctx: PeopleContext, bank_id: str
    ):
        await insert_conclusions(ctx, bank_id, "explicit", 60)
        await insert_queue_item(ctx, bank_id, _key(bank_id, "ann", "ann"), "dream", processed=False)

        with patch.object(dream_scheduler, "schedule_dream", new_callable=AsyncMock):
            assert await check_and_schedule_dream(ctx, bank_id, "ann", "ann") is False

    async def test_no_running_scheduler_schedules_nothing(self, ctx: PeopleContext, bank_id: str, monkeypatch):
        monkeypatch.setattr(DREAM, "ENABLED", True)
        set_dream_scheduler(None)
        await insert_conclusions(ctx, bank_id, "explicit", 60)

        assert await check_and_schedule_dream(ctx, bank_id, "ann", "ann") is False


class TestExecuteAndPoll:
    """A fired timer and the poll both end in a dream on the queue, once."""

    async def _queued(self, ctx: PeopleContext, bank_id: str) -> int:
        return await scalar(
            ctx,
            f"SELECT count(*) FROM {ctx.table('queue')} WHERE bank_id = $1 AND task_type = 'dream'",
            bank_id,
        )

    async def test_execute_dream_enqueues_with_the_newest_session(
        self, dream_scheduler: DreamScheduler, ctx: PeopleContext, bank_id: str
    ):
        session = await make_session(ctx, bank_id)
        await insert_conclusions(ctx, bank_id, "explicit", 3, session_name=session)

        await dream_scheduler.execute_dream(
            bank_id, DreamType.OMNI, observer="ann", observed="ann", trigger_reason="document_threshold"
        )

        assert await self._queued(ctx, bank_id) == 1
        payload = await scalar(
            ctx, f"SELECT payload::text FROM {ctx.table('queue')} WHERE bank_id = $1 AND task_type = 'dream'", bank_id
        )
        assert f'"session_name": "{session}"' in payload
        assert '"trigger_reason": "document_threshold"' in payload

    async def test_execute_dream_skips_a_pair_without_conclusions(
        self, dream_scheduler: DreamScheduler, ctx: PeopleContext, bank_id: str
    ):
        await dream_scheduler.execute_dream(bank_id, DreamType.OMNI, observer="ann", observed="ann")
        assert await self._queued(ctx, bank_id) == 0

    async def test_execute_dream_skips_a_session_with_dreams_disabled(
        self, dream_scheduler: DreamScheduler, ctx: PeopleContext, bank_id: str
    ):
        session = await make_session(ctx, bank_id, {"dream": {"enabled": False}})
        await insert_conclusions(ctx, bank_id, "explicit", 3, session_name=session)

        await dream_scheduler.execute_dream(bank_id, DreamType.OMNI, observer="ann", observed="ann")

        assert await self._queued(ctx, bank_id) == 0

    async def test_poll_enqueues_a_due_pair_once(
        self, dream_scheduler: DreamScheduler, ctx: PeopleContext, bank_id: str, monkeypatch
    ):
        monkeypatch.setattr(DREAM, "DOCUMENT_THRESHOLD", 50)
        monkeypatch.setattr(DREAM, "ENABLED_TYPES", ["omni"])
        monkeypatch.setattr(DREAM, "IDLE_TIMEOUT_MINUTES", 60)
        session = await make_session(ctx, bank_id)
        await insert_conclusions(ctx, bank_id, "explicit", 60, age_minutes=120, session_name=session)

        await dream_scheduler.poll_once()
        assert await self._queued(ctx, bank_id) == 1

        # The queued dream is pending: the next poll does not add a second one.
        await dream_scheduler.poll_once()
        assert await self._queued(ctx, bank_id) == 1

    async def test_start_registers_and_stop_unregisters(self, ctx: PeopleContext, monkeypatch):
        monkeypatch.setattr(DREAM, "ENABLED", True)
        monkeypatch.setattr(DREAM, "DUE_POLL_INTERVAL_SECONDS", 3600)
        scheduler = DreamScheduler(lambda: ctx)
        await scheduler.start()
        try:
            assert get_dream_scheduler() is scheduler
            assert scheduler._poll_task is not None
        finally:
            await scheduler.stop()
        assert get_dream_scheduler() is None
        assert scheduler._poll_task is None

    async def test_stop_cancels_pending_dreams(self, dream_scheduler: DreamScheduler):
        await _schedule(dream_scheduler, "test_bank", "bob", "bob")
        await dream_scheduler.stop()
        assert dream_scheduler.pending_dreams == {}
