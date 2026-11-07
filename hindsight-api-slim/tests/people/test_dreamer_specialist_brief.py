# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: tests/dreamer/test_specialist_brief.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""The memory brief and the work list in a specialist's prompts (ported from Honcho's tests/dreamer/test_specialist_brief.py)."""

import datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api.people.conclusions import DocumentLevelSummary
from hindsight_api.people.dreamer.settings import DREAM
from hindsight_api.people.dreamer.specialists import (
    BaseSpecialist,
    CardRefreshSpecialist,
    DeductionSpecialist,
    InductionSpecialist,
    SpecialistResult,
)
from hindsight_api.people.dreamer.tool_loop import ToolLoopResult
from hindsight_api.people.dreamer.working_memory import DreamState, NotesCursor, WorkItem, WorkList
from tests.people.dialectic_support import bare_context

_SUMMARY = DocumentLevelSummary(counts_by_level={"explicit": 7}, earliest=None, latest=None)


async def _run(specialist: BaseSpecialist, **run_kwargs: Any) -> tuple[list[dict[str, Any]], SpecialistResult]:
    """Run the specialist with the model and the database mocked; return the messages sent and the result."""
    loop = AsyncMock(return_value=ToolLoopResult(content="DONE", iterations=1, input_tokens=1, output_tokens=1))
    with (
        patch("hindsight_api.people.dreamer.specialists.get_peer", new=AsyncMock()),
        patch("hindsight_api.people.dreamer.specialists.peer_card_crud.get_peer_card", new=AsyncMock()),
        patch(
            "hindsight_api.people.dreamer.specialists.conclusions_crud.summarize_documents_by_level",
            new=AsyncMock(return_value=_SUMMARY),
        ),
        patch("hindsight_api.people.dreamer.specialists.create_tool_executor", new=AsyncMock(return_value=AsyncMock())),
        patch("hindsight_api.people.dreamer.specialists.execute_tool_loop", new=loop),
    ):
        result = await specialist.run(
            bare_context(), bank_id="w", observer="alice", observed="alice", session_name=None, **run_kwargs
        )
    return loop.call_args.kwargs["messages"], result


@pytest.mark.parametrize("specialist", [DeductionSpecialist(), InductionSpecialist()])
async def test_brief_adds_layout_and_counts(specialist: BaseSpecialist, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(DREAM, "BRIEF_ENABLED", True)
    (system, user), _ = await _run(specialist)
    assert "## HOW MEMORY IS ORGANIZED" in system["content"]
    assert "`get_recent_observations`" in system["content"]
    assert user["content"].startswith("Memory counts: 7 explicit,")


async def test_brief_off_by_default():
    (system, user), _ = await _run(DeductionSpecialist())
    assert "HOW MEMORY IS ORGANIZED" not in system["content"]
    assert "Memory counts" not in user["content"]


async def test_card_refresh_never_gets_the_brief(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(DREAM, "BRIEF_ENABLED", True)
    (system, user), _ = await _run(CardRefreshSpecialist())
    assert "HOW MEMORY IS ORGANIZED" not in system["content"]
    assert "Memory counts" not in user["content"]


async def test_specialist_result_reports_its_iteration_limit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(DREAM, "DEDUCTION_MAX_TOOL_ITERATIONS", 9)
    _, result = await _run(DeductionSpecialist())
    assert result.iteration_limit == 9
    assert result.hit_iteration_limit is False
    assert result.budget_line_enabled is False


async def test_brief_includes_the_work_list(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(DREAM, "BRIEF_ENABLED", True)
    work_list = WorkList(
        items=[
            WorkItem(
                id="n1",
                content="moved to Lisbon",
                created_at=datetime.datetime(2026, 10, 1, tzinfo=datetime.UTC),
            )
        ],
        remaining=0,
    )
    state = DreamState(cursor=NotesCursor(), work_list=work_list)
    (_, user), result = await _run(InductionSpecialist(), state=state)
    assert "## NEW SINCE THE LAST DREAM" in user["content"]
    # The list replaces "start with get_recent_observations", which would re-fetch it.
    assert "Start from the conclusions under NEW SINCE THE LAST DREAM" in user["content"]
    assert "Start with `get_recent_observations`" not in user["content"]
    assert "[id:n1] [2026-10-01] moved to Lisbon" in user["content"]
    assert result.new_conclusions_shown == 1


@pytest.mark.parametrize("specialist", [DeductionSpecialist(), InductionSpecialist()])
@pytest.mark.parametrize("hints", [None, ["coffee"]])
def test_start_instruction_follows_the_work_list(specialist: BaseSpecialist, hints: list[str] | None):
    plain = specialist.build_user_prompt("alice", hints)
    listed = specialist.build_user_prompt("alice", hints, work_list_shown=True)
    assert "Start from the conclusions under NEW SINCE THE LAST DREAM" in listed
    assert "get_recent_observations" not in listed
    assert "NEW SINCE THE LAST DREAM" not in plain
