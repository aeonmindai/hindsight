# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: tests/dreamer/test_dream_history_wiring.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""Representation History wiring: orchestrator, specialist and tool handlers (ported from Honcho's
tests/dreamer/test_dream_history_wiring.py)."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api.people.context import PeopleContext
from hindsight_api.people.dream_history import DreamHistoryTarget, get_run_detail, list_runs
from hindsight_api.people.dreamer import orchestrator
from hindsight_api.people.dreamer.agent_tools import (
    ToolContext,
    _handle_create_observations_deductive,
    _handle_delete_observations,
    _handle_update_peer_card,
)
from hindsight_api.people.dreamer.settings import DREAM
from hindsight_api.people.dreamer.specialists import SPECIALISTS, DeductionSpecialist, SpecialistResult
from hindsight_api.people.dreamer.tool_loop import ToolLoopResult
from hindsight_api.people.peers import get_or_create_peers
from tests.people.dialectic_support import bare_context
from tests.people.dreamer_support import open_target, seed


async def test_dream_tool_writes_record_to_the_run(ctx: PeopleContext, bank_id: str):
    _session, explicit_ids = await seed(ctx, bank_id)
    run = await open_target(ctx, bank_id)
    tool_ctx = ToolContext(
        pctx=ctx,
        bank_id=bank_id,
        observer="ann",
        observed="ann",
        session_name=None,
        include_observation_ids=True,
        db_lock=asyncio.Lock(),
        agent_type="deduction",
        history=DreamHistoryTarget(run_id=run.run_id, specialist="deduction"),
    )

    created_result = await _handle_create_observations_deductive(
        tool_ctx,
        {
            "observations": [
                {
                    "content": "Ann relies on caffeine at night",
                    "source_ids": explicit_ids,
                    "premises": ["Ann drinks coffee", "Ann works nights"],
                }
            ]
        },
    )
    assert created_result.metadata["created_count"] == 1
    deleted_result = await _handle_delete_observations(tool_ctx, {"observation_ids": [explicit_ids[0]]})
    assert deleted_result.metadata["deleted_count"] == 1
    await _handle_update_peer_card(tool_ctx, {"content": ["IDENTITY: Name: Ann"]})

    detail = await get_run_detail(ctx, bank_id, run.run_id)
    [created] = detail.created
    assert created.conclusion.content == "Ann relies on caffeine at night"
    assert [source.id for source in created.sources] == explicit_ids
    assert created.specialist == "deduction"
    [removed] = detail.removed
    assert (removed.id, removed.content) == (explicit_ids[0], "Ann drinks coffee daily")
    assert detail.peer_card is not None
    assert detail.peer_card.after == ["IDENTITY: Name: Ann"]
    assert (detail.created_count, detail.removed_count) == (1, 1)


async def test_tool_writes_without_a_run_record_no_history(ctx: PeopleContext, bank_id: str):
    _session, explicit_ids = await seed(ctx, bank_id)
    tool_ctx = ToolContext(
        pctx=ctx,
        bank_id=bank_id,
        observer="ann",
        observed="ann",
        session_name=None,
        include_observation_ids=True,
        db_lock=asyncio.Lock(),
    )
    await _handle_delete_observations(tool_ctx, {"observation_ids": [explicit_ids[0]]})
    assert await list_runs(ctx, bank_id) == []


async def test_specialist_hands_its_name_and_the_run_to_the_executor():
    loop = AsyncMock(return_value=ToolLoopResult(content="DONE", iterations=1, input_tokens=1, output_tokens=1))
    make_executor = AsyncMock(return_value=AsyncMock())
    with (
        patch("hindsight_api.people.dreamer.specialists.get_peer", new=AsyncMock()),
        patch("hindsight_api.people.dreamer.specialists.peer_card_crud.get_peer_card", new=AsyncMock()),
        patch("hindsight_api.people.dreamer.specialists.create_tool_executor", new=make_executor),
        patch("hindsight_api.people.dreamer.specialists.execute_tool_loop", new=loop),
    ):
        await DeductionSpecialist().run(
            bare_context(),
            bank_id="w",
            observer="alice",
            observed="alice",
            session_name=None,
            parent_run_id="run-1",
            history=DreamHistoryTarget(run_id="run-1"),
        )
        await DeductionSpecialist().run(
            bare_context(), bank_id="w", observer="alice", observed="alice", session_name=None
        )

    with_history, without_history = make_executor.call_args_list
    assert with_history.kwargs["history"] == DreamHistoryTarget(run_id="run-1", specialist="deduction")
    assert without_history.kwargs["history"] is None


def _result(name: str, *, success: bool) -> SpecialistResult:
    return SpecialistResult(
        run_id="r",
        specialist_type=name,
        iterations=1,
        tool_calls_count=0,
        input_tokens=1,
        output_tokens=1,
        duration_ms=1.0,
        success=success,
        content="done",
    )


@pytest.mark.parametrize(("induction_succeeds", "status"), [(True, "completed"), (False, "failed")])
async def test_run_dream_opens_and_closes_its_run(
    ctx: PeopleContext, bank_id: str, monkeypatch: pytest.MonkeyPatch, induction_succeeds: bool, status: str
):
    monkeypatch.setattr(DREAM.SURPRISAL, "ENABLED", False)
    await get_or_create_peers(ctx, bank_id, ["ann"])
    seen: dict[str, Any] = {}

    def _specialist(name: str, *, success: bool) -> Any:
        async def run(_ctx: PeopleContext, **kwargs: Any) -> SpecialistResult:
            seen[name] = kwargs["history"]
            return _result(name, success=success)

        return type("Stub", (), {"run": staticmethod(run)})()

    with patch.dict(
        SPECIALISTS,
        {
            "deduction": _specialist("deduction", success=True),
            "induction": _specialist("induction", success=induction_succeeds),
        },
    ):
        result = await orchestrator.run_dream(ctx, bank_id, "ann", "ann", dream_type="omni", trigger_reason="manual")

    assert result is not None
    assert seen["deduction"] == seen["induction"] == DreamHistoryTarget(run_id=result.run_id)
    run = await get_run_detail(ctx, bank_id, result.run_id)
    assert (run.status, run.dream_type, run.trigger_reason) == (status, "omni", "manual")
    assert run.finished_at is not None


async def test_run_dream_without_history_passes_none(ctx: PeopleContext, bank_id: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(DREAM.SURPRISAL, "ENABLED", False)
    monkeypatch.setattr(DREAM, "HISTORY_ENABLED", False)
    seen: list[Any] = []

    async def run(_ctx: PeopleContext, **kwargs: Any) -> SpecialistResult:
        seen.append(kwargs["history"])
        return _result("x", success=True)

    stub = type("Stub", (), {"run": staticmethod(run)})()
    with patch.dict(SPECIALISTS, {"deduction": stub, "induction": stub}):
        await orchestrator.run_dream(ctx, bank_id, "ann", "ann")

    assert seen == [None, None]
    assert await list_runs(ctx, bank_id) == []


async def test_run_card_refresh_dream_opens_and_closes_its_run(ctx: PeopleContext, bank_id: str):
    await get_or_create_peers(ctx, bank_id, ["ann"])
    seen: list[Any] = []

    async def fake_run(self: Any, _ctx: PeopleContext, **kwargs: Any) -> SpecialistResult:
        seen.append(kwargs["history"])
        return _result("card_refresh", success=True)

    with patch("hindsight_api.people.dreamer.specialists.CardRefreshSpecialist.run", new=fake_run):
        result = await orchestrator.run_card_refresh_dream(
            ctx, bank_id, "ann", "ann", rebuild=True, dream_type="card_refresh", trigger_reason="manual"
        )

    assert result is not None
    assert seen == [DreamHistoryTarget(run_id=result.run_id)]
    run = await get_run_detail(ctx, bank_id, result.run_id)
    assert (run.status, run.dream_type) == ("completed", "card_refresh")
