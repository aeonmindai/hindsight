# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: tests/dreamer/test_write_window.py
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: tests/dreamer/test_specialist_iteration_limits.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""The write window, the budget line and the specialists' iteration limits (ported from Honcho's
tests/dreamer/test_write_window.py and test_specialist_iteration_limits.py)."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api.people.dreamer.agent_tools import get_tool_handler
from hindsight_api.people.dreamer.settings import DREAM
from hindsight_api.people.dreamer.specialists import (
    DeductionSpecialist,
    InductionSpecialist,
    _handler_resolver,
    dream_budget_line,
    write_window_size,
)
from hindsight_api.people.dreamer.tool_loop import ToolLoopResult, ToolResult, set_current_iteration
from tests.people.dialectic_support import bare_context


def test_defaults_match_previous_hardcoded_limits():
    assert DeductionSpecialist().get_max_iterations() == 12
    assert InductionSpecialist().get_max_iterations() == 10


def test_specialists_read_their_own_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(DREAM, "DEDUCTION_MAX_TOOL_ITERATIONS", 20)
    monkeypatch.setattr(DREAM, "INDUCTION_MAX_TOOL_ITERATIONS", 7)
    # The card-refresh limit must not leak into deduction or induction.
    monkeypatch.setattr(DREAM, "MAX_TOOL_ITERATIONS", 3)

    assert DeductionSpecialist().get_max_iterations() == 20
    assert InductionSpecialist().get_max_iterations() == 7


@pytest.mark.parametrize(("limit", "window"), [(12, 2), (10, 2), (5, 1), (1, 1), (20, 3)])
def test_default_window_is_15_percent_rounded_min_one(limit: int, window: int):
    assert write_window_size(limit) == window


def test_window_override_leaves_a_search_round(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(DREAM, "WRITE_WINDOW_ITERATIONS", 4)
    assert write_window_size(12) == 4
    assert write_window_size(3) == 2
    assert write_window_size(1) == 1


def test_budget_line_wording():
    line = dream_budget_line(2)
    assert line(7, 12) == "Tool calls used: 7 of 12. Calls 11 and 12 are for writing; searches will be refused."
    assert "Searches are now refused" in (line(10, 12) or "")
    assert line(12, 12) is None
    assert dream_budget_line(1)(3, 10) == "Tool calls used: 3 of 10. Call 10 is for writing; searches will be refused."
    assert "Calls 8 to 10 are" in (dream_budget_line(3)(2, 10) or "")


async def test_resolver_refuses_searches_only_inside_the_window():
    resolve = _handler_resolver(11, {})

    set_current_iteration(10)
    assert resolve("search_memory") is get_tool_handler("search_memory")

    set_current_iteration(11)
    refuse = resolve("search_memory")
    assert refuse is not None and refuse is not get_tool_handler("search_memory")
    result = await refuse(AsyncMock(), {})
    assert isinstance(result, ToolResult) and result.metadata == {"refused": True}
    # Writes are never refused.
    assert resolve("create_observations_deductive") is get_tool_handler("create_observations_deductive")
    set_current_iteration(0)


async def _run(specialist: Any, content: str = "DONE") -> tuple[Any, dict[str, Any]]:
    loop = AsyncMock(return_value=ToolLoopResult(content=content, iterations=1, input_tokens=1, output_tokens=1))
    make_executor = AsyncMock(return_value=AsyncMock())
    with (
        patch("hindsight_api.people.dreamer.specialists.get_peer", new=AsyncMock()),
        patch("hindsight_api.people.dreamer.specialists.peer_card_crud.get_peer_card", new=AsyncMock()),
        patch("hindsight_api.people.dreamer.specialists.create_tool_executor", new=make_executor),
        patch("hindsight_api.people.dreamer.specialists.execute_tool_loop", new=loop),
    ):
        result = await specialist.run(
            bare_context(), bank_id="w", observer="alice", observed="alice", session_name=None
        )
    return result, {
        **loop.call_args.kwargs,
        "handler_resolver": make_executor.call_args.kwargs["handler_resolver"],
    }


@pytest.mark.parametrize(
    ("specialist", "write_tools"),
    [
        (
            DeductionSpecialist(),
            ["create_observations_deductive", "delete_observations", "update_peer_card"],
        ),
        (InductionSpecialist(), ["create_observations_inductive"]),
    ],
)
async def test_budget_enabled_wires_the_window(
    specialist: Any, write_tools: list[str], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(DREAM, "BUDGET_ENABLED", True)
    _, kwargs = await _run(specialist)
    assert kwargs["budget_line"] is not None
    assert [t["name"] for t in kwargs["final_call_tools"]] == write_tools
    assert kwargs["handler_resolver"] is not None
    assert "## BUDGET" in kwargs["messages"][0]["content"]


async def test_budget_off_by_default():
    _, kwargs = await _run(DeductionSpecialist())
    assert kwargs["budget_line"] is None
    assert kwargs["final_call_tools"] is None
    assert kwargs["handler_resolver"] is None
    assert "## BUDGET" not in kwargs["messages"][0]["content"]


async def test_write_only_final_call_reports_empty_content(monkeypatch: pytest.MonkeyPatch):
    # A final call that only makes tool calls has no text; run_dream slices the result's content, so
    # it must stay a string.
    monkeypatch.setattr(DREAM, "BUDGET_ENABLED", True)
    result, _ = await _run(DeductionSpecialist(), content="")
    assert result.success is True
    assert result.content == ""
