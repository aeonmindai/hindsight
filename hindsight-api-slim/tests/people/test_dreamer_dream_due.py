# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: tests/dreamer/test_dream_due.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""Which pairs have a dream due (ported from Honcho's tests/dreamer/test_dream_due.py)."""

import datetime
from typing import Any

import pytest

from hindsight_api.people.context import PeopleContext
from hindsight_api.people.dreamer.dream_due import DueDream, list_due_dreams
from hindsight_api.people.dreamer.payload import dream_work_unit_key
from hindsight_api.people.dreamer.settings import DREAM
from hindsight_api.people.schemas import DreamType
from tests.people.dreamer_support import insert_conclusions, insert_queue_item, make_session, set_state


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


async def _due(ctx: PeopleContext, bank_id: str) -> list[DueDream]:
    """The due dreams of this test's bank (the listing spans every bank)."""
    return [d for d in await list_due_dreams(ctx) if d.bank_id == bank_id]


async def _insert_docs(
    ctx: PeopleContext,
    bank_id: str,
    level: str,
    count: int,
    *,
    age_minutes: int = 0,
    session_name: str | None = None,
    sessionless: bool = False,
) -> None:
    if session_name is None and not sessionless:
        session_name = await make_session(ctx, bank_id)
    await insert_conclusions(ctx, bank_id, level, count, age_minutes=age_minutes, session_name=session_name)


async def _insert_dream_item(
    ctx: PeopleContext, bank_id: str, *, age_minutes: int, processed: bool, error: str | None = None
) -> None:
    await insert_queue_item(
        ctx,
        bank_id,
        dream_work_unit_key(bank_id, "ann", "ann", DreamType.OMNI),
        "dream",
        age_minutes=age_minutes,
        processed=processed,
        error=error,
    )


@pytest.fixture(autouse=True)
def _pin_dream_config(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(DREAM, "ENABLED", True)
    monkeypatch.setattr(DREAM, "DOCUMENT_THRESHOLD", 50)
    monkeypatch.setattr(DREAM, "ENABLED_TYPES", ["omni"])
    monkeypatch.setattr(DREAM, "IDLE_TIMEOUT_MINUTES", 60)
    monkeypatch.setattr(DREAM, "MIN_HOURS_BETWEEN_DREAMS", 8)


class TestCountDueDreams:
    async def test_below_threshold_is_not_due(self, ctx: PeopleContext, bank_id: str):
        await _insert_docs(ctx, bank_id, "explicit", 30, age_minutes=90)

        assert await _due(ctx, bank_id) == []

    async def test_derived_levels_do_not_count(self, ctx: PeopleContext, bank_id: str):
        await _insert_docs(ctx, bank_id, "explicit", 30, age_minutes=90)
        await _insert_docs(ctx, bank_id, "deductive", 40, age_minutes=90)

        assert await _due(ctx, bank_id) == []

    async def test_threshold_met_but_not_idle_is_not_due(self, ctx: PeopleContext, bank_id: str):
        """A pair still receiving conclusions is not idle yet."""
        await _insert_docs(ctx, bank_id, "explicit", 60, age_minutes=1)

        assert await _due(ctx, bank_id) == []

    async def test_threshold_met_and_idle_is_due(self, ctx: PeopleContext, bank_id: str):
        await _insert_docs(ctx, bank_id, "explicit", 60, age_minutes=90)

        [due] = await _due(ctx, bank_id)
        assert (due.observer, due.observed, due.dream_type) == ("ann", "ann", DreamType.OMNI)
        assert due.session_name.startswith("s-")

    async def test_pending_representation_work_blocks_a_due_dream(self, ctx: PeopleContext, bank_id: str):
        await _insert_docs(ctx, bank_id, "explicit", 60, age_minutes=90)

        session_name = await make_session(ctx, bank_id)
        from hindsight_api.people.deriver.work_unit import construct_work_unit_key

        representation_key = construct_work_unit_key(
            bank_id, {"task_type": "representation", "session_name": session_name, "observed": "ann"}
        )
        await insert_queue_item(ctx, bank_id, representation_key, "representation", processed=False)

        assert await _due(ctx, bank_id) == []

        # Once the representation work is processed, the dream is due again.
        async with ctx.conn() as conn:
            await conn.execute(
                f"UPDATE {ctx.table('queue')} SET processed = true WHERE bank_id = $1 AND task_type = 'representation'",
                bank_id,
            )
        assert len(await _due(ctx, bank_id)) == 1

    async def test_documents_since_last_dream_uses_stored_count(self, ctx: PeopleContext, bank_id: str):
        await _insert_docs(ctx, bank_id, "explicit", 60, age_minutes=90)
        await set_state(ctx, bank_id, "ann", "ann", {"dream": {"last_dream_document_count": 40}})

        assert await _due(ctx, bank_id) == []

    async def test_min_hours_gate_blocks_a_recent_dream(self, ctx: PeopleContext, bank_id: str):
        last_dream_at = (_now() - datetime.timedelta(hours=2)).isoformat()
        await _insert_docs(ctx, bank_id, "explicit", 60, age_minutes=90)
        await set_state(ctx, bank_id, "ann", "ann", {"dream": {"last_dream_at": last_dream_at}})

        assert await _due(ctx, bank_id) == []

    async def test_naive_last_dream_at_is_read_as_utc(self, ctx: PeopleContext, bank_id: str):
        """A stored timestamp with no offset must gate, not raise."""
        naive = (_now() - datetime.timedelta(hours=2)).replace(tzinfo=None).isoformat()
        await _insert_docs(ctx, bank_id, "explicit", 60, age_minutes=90)
        await set_state(ctx, bank_id, "ann", "ann", {"dream": {"last_dream_at": naive}})

        assert await _due(ctx, bank_id) == []

    async def test_pending_dream_item_blocks(self, ctx: PeopleContext, bank_id: str):
        await _insert_docs(ctx, bank_id, "explicit", 60, age_minutes=90)
        await _insert_dream_item(ctx, bank_id, age_minutes=10, processed=False)

        assert await _due(ctx, bank_id) == []

    async def test_failed_dream_waits_for_new_documents(self, ctx: PeopleContext, bank_id: str):
        """Without this the count never returns to zero."""
        await _insert_docs(ctx, bank_id, "explicit", 60, age_minutes=90)
        await _insert_dream_item(ctx, bank_id, age_minutes=80, processed=True, error="boom")

        assert await _due(ctx, bank_id) == []

    async def test_failed_dream_retries_after_new_documents(self, ctx: PeopleContext, bank_id: str):
        await _insert_docs(ctx, bank_id, "explicit", 60, age_minutes=90)
        await _insert_dream_item(ctx, bank_id, age_minutes=80, processed=True, error="boom")
        await _insert_docs(ctx, bank_id, "explicit", 1, age_minutes=70)

        assert len(await _due(ctx, bank_id)) == 1

    async def test_sessionless_documents_are_not_due(self, ctx: PeopleContext, bank_id: str):
        """The deriver's own enqueue path refuses these, so they must not count."""
        await _insert_docs(ctx, bank_id, "explicit", 60, age_minutes=90, sessionless=True)

        assert await _due(ctx, bank_id) == []

    async def test_newest_document_decides_the_session(self, ctx: PeopleContext, bank_id: str):
        await _insert_docs(ctx, bank_id, "explicit", 60, age_minutes=120)

        assert len(await _due(ctx, bank_id)) == 1

        await _insert_docs(ctx, bank_id, "explicit", 1, age_minutes=90, sessionless=True)

        assert await _due(ctx, bank_id) == []

    async def test_session_with_dreams_disabled_is_not_due(self, ctx: PeopleContext, bank_id: str):
        """A dream the enqueue path would refuse must not be counted."""
        session_name = await make_session(ctx, bank_id, {"dream": {"enabled": False}})
        await _insert_docs(ctx, bank_id, "explicit", 60, age_minutes=90, session_name=session_name)

        assert await _due(ctx, bank_id) == []

    async def test_dreams_disabled_globally_returns_zero(self, ctx: PeopleContext, bank_id: str, monkeypatch):
        await _insert_docs(ctx, bank_id, "explicit", 60, age_minutes=90)

        monkeypatch.setattr(DREAM, "ENABLED", False)
        assert await list_due_dreams(ctx) == []

    async def test_card_refresh_is_never_counted(self, ctx: PeopleContext, bank_id: str, monkeypatch):
        await _insert_docs(ctx, bank_id, "explicit", 60, age_minutes=90)

        monkeypatch.setattr(DREAM, "ENABLED_TYPES", ["card_refresh"])
        assert await list_due_dreams(ctx) == []

    async def test_banks_are_isolated(self, ctx: PeopleContext, bank_id: str, people_engine: Any):
        """The same pair names in two banks: only the bank with the work is due."""
        other = f"{bank_id}-other"
        await people_engine.ensure_bank_profile(bank_id=other, request_context=ctx.request_context)
        await _insert_docs(ctx, bank_id, "explicit", 60, age_minutes=90)
        await _insert_docs(ctx, other, "explicit", 5, age_minutes=90)

        assert len(await _due(ctx, bank_id)) == 1
        assert await _due(ctx, other) == []


class TestWorkListBacklog:
    """A pair whose work list is still behind is due without new conclusions."""

    @staticmethod
    async def _dreamed(ctx: PeopleContext, bank_id: str, *, backlog: int, dreamed_hours_ago: float = 9) -> None:
        last_dream_at = _now() - datetime.timedelta(hours=dreamed_hours_ago)
        await set_state(
            ctx,
            bank_id,
            "ann",
            "ann",
            {
                "dream": {
                    "last_dream_at": last_dream_at.isoformat(),
                    # Below the threshold: only the backlog can make it due.
                    "last_dream_document_count": 55,
                    "work_list_backlog": backlog,
                }
            },
        )
        minutes_ago = int(dreamed_hours_ago * 60)
        await _insert_docs(ctx, bank_id, "explicit", 60, age_minutes=minutes_ago + 30)
        await _insert_dream_item(ctx, bank_id, age_minutes=minutes_ago + 5, processed=True)

    async def test_backlog_is_due_once_the_gate_passes(self, ctx: PeopleContext, bank_id: str):
        await self._dreamed(ctx, bank_id, backlog=5)
        assert len(await _due(ctx, bank_id)) == 1

    async def test_no_backlog_waits_for_new_conclusions(self, ctx: PeopleContext, bank_id: str):
        await self._dreamed(ctx, bank_id, backlog=0)
        assert await _due(ctx, bank_id) == []

    async def test_backlog_respects_the_min_hours_gate(self, ctx: PeopleContext, bank_id: str):
        await self._dreamed(ctx, bank_id, backlog=5, dreamed_hours_ago=2)
        assert await _due(ctx, bank_id) == []

    async def test_backlog_dream_already_queued_is_not_due_again(self, ctx: PeopleContext, bank_id: str):
        await self._dreamed(ctx, bank_id, backlog=5)
        await _insert_dream_item(ctx, bank_id, age_minutes=1, processed=False)
        assert await _due(ctx, bank_id) == []
