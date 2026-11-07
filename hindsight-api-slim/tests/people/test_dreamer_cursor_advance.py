# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: tests/dreamer/test_cursor_advance.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""How a finished dream moves the cursor and records the scheduler's guard fields (ported from Honcho's
tests/dreamer/test_cursor_advance.py)."""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.people.context import PeopleContext
from hindsight_api.people.dreamer.dream_scheduler import schedule_backlog_dream
from hindsight_api.people.dreamer.orchestrator import DreamResult, process_dream
from hindsight_api.people.dreamer.payload import DreamPayload
from hindsight_api.people.dreamer.settings import DREAM
from hindsight_api.people.dreamer.working_memory import (
    NOTES_KEY,
    DreamState,
    NotesCursor,
    WorkItem,
    WorkList,
)
from hindsight_api.people.schemas import DreamType
from tests.people.dreamer_support import insert_conclusions, set_state, stored_state


def _result(deduction: bool, induction: bool) -> DreamResult:
    return DreamResult(
        run_id="r1",
        specialists_run=["deduction", "induction"],
        deduction_success=deduction,
        induction_success=induction,
        surprisal_enabled=False,
        surprisal_conclusion_count=0,
        total_iterations=0,
        total_duration_ms=0.0,
        input_tokens=0,
        output_tokens=0,
    )


async def _run_process_dream(
    ctx: PeopleContext, bank_id: str, result: DreamResult, state: dict[str, Any], explicit_count: int
) -> AsyncMock:
    """Process a dream whose specialists are mocked; returns the mocked advance_cursor."""
    await insert_conclusions(ctx, bank_id, "explicit", explicit_count, observer="a", observed="a")
    if state:
        await set_state(ctx, bank_id, "a", "a", state)
    advance = AsyncMock()
    with (
        patch("hindsight_api.people.dreamer.orchestrator.run_dream", new=AsyncMock(return_value=result)),
        patch("hindsight_api.people.dreamer.orchestrator.advance_cursor", new=advance),
    ):
        await process_dream(ctx, DreamPayload(dream_type=DreamType.OMNI, observer="a", observed="a"), bank_id)
    return advance


@pytest.mark.parametrize(
    ("deduction", "induction", "recorded", "moves_position"),
    [
        (True, True, True, True),
        (True, False, True, False),
        (False, True, True, False),
        (False, False, False, False),
    ],
)
async def test_cursor_moves_only_when_both_specialists_succeeded(
    ctx: PeopleContext,
    bank_id: str,
    deduction: bool,
    induction: bool,
    recorded: bool,
    moves_position: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(DREAM, "BRIEF_ENABLED", True)
    advance = await _run_process_dream(ctx, bank_id, _result(deduction, induction), {}, 0)
    assert advance.await_count == (1 if recorded else 0)
    if recorded:
        assert advance.await_args is not None
        kwargs = advance.await_args.kwargs
        assert kwargs["all_specialists_succeeded"] is moves_position
        assert kwargs["any_specialist_succeeded"] is True


async def test_the_guard_fields_advance_with_a_successful_dream(
    ctx: PeopleContext, bank_id: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(DREAM, "BRIEF_ENABLED", False)
    monkeypatch.setattr(DREAM, "NOTES_MODE", "off")
    await _run_process_dream(ctx, bank_id, _result(True, True), {}, 4)
    dream_state = (await stored_state(ctx, bank_id, "a"))["dream"]
    assert dream_state["last_dream_document_count"] == 4
    assert dream_state["work_list_backlog"] == 0
    assert datetime.fromisoformat(dream_state["last_dream_at"]).tzinfo is not None


async def test_a_dream_where_nothing_ran_never_moves_the_cursor(ctx: PeopleContext, bank_id: str, monkeypatch):
    """Both specialists failing never moves the cursor."""
    monkeypatch.setattr(DREAM, "BRIEF_ENABLED", False)
    monkeypatch.setattr(DREAM, "NOTES_MODE", "off")
    advance = await _run_process_dream(ctx, bank_id, _result(False, False), {}, 4)
    assert advance.await_count == 0


async def test_working_memory_off_clears_a_stored_position(
    ctx: PeopleContext, bank_id: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(DREAM, "BRIEF_ENABLED", False)
    monkeypatch.setattr(DREAM, "NOTES_MODE", "off")
    stored = {
        "entries": [{"id": "n1"}],
        "cursor": {"through_created_at": "2026-10-01T00:00:00+00:00", "through_id": "x", "dreams": 2},
    }
    advance = await _run_process_dream(ctx, bank_id, _result(True, True), {NOTES_KEY: stored}, 4)
    assert advance.await_count == 0
    written = (await stored_state(ctx, bank_id, "a"))[NOTES_KEY]
    assert written["entries"] == [{"id": "n1"}]
    assert written["cursor"]["through_id"] is None
    assert written["cursor"]["dreams"] == 2


async def test_unshown_backlog_stays_counted_as_new(ctx: PeopleContext, bank_id: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(DREAM, "BRIEF_ENABLED", True)
    result = _result(True, True)
    result.state = DreamState(
        cursor=NotesCursor(),
        work_list=WorkList(items=[WorkItem(id="x", content="c", created_at=datetime.now(UTC))], remaining=30),
    )
    schedule = AsyncMock()
    with patch("hindsight_api.people.dreamer.orchestrator.schedule_backlog_dream", new=schedule):
        await _run_process_dream(ctx, bank_id, result, {}, 100)
    written = await stored_state(ctx, bank_id, "a")
    assert written["dream"]["last_dream_document_count"] == 70
    assert written["dream"]["work_list_backlog"] == 30
    # Nothing else checks the pair until new conclusions arrive.
    schedule.assert_awaited_once_with(ctx, bank_id, observer="a", observed="a")


async def test_failed_dream_schedules_no_follow_up(ctx: PeopleContext, bank_id: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(DREAM, "BRIEF_ENABLED", True)
    result = _result(False, False)
    result.state = DreamState(
        cursor=NotesCursor(),
        work_list=WorkList(items=[WorkItem(id="x", content="c", created_at=datetime.now(UTC))], remaining=30),
    )
    schedule = AsyncMock()
    with patch("hindsight_api.people.dreamer.orchestrator.schedule_backlog_dream", new=schedule):
        await _run_process_dream(ctx, bank_id, result, {}, 100)
    assert (await stored_state(ctx, bank_id, "a"))["dream"]["work_list_backlog"] == 0
    schedule.assert_not_awaited()


async def test_backlog_timer_waits_out_the_min_hours_gate(ctx: PeopleContext, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(DREAM, "MIN_HOURS_BETWEEN_DREAMS", 8)
    scheduler = MagicMock(schedule_dream=AsyncMock())
    with patch("hindsight_api.people.dreamer.dream_scheduler.get_dream_scheduler", return_value=scheduler):
        await schedule_backlog_dream(ctx, "w", observer="a", observed="b")
    args = scheduler.schedule_dream.await_args
    assert args is not None
    assert args.args[2] == 8 * 60
    assert args.kwargs["trigger_reason"] == "work_list_backlog"
