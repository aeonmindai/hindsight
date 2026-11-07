# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: tests/dreamer/test_dreamer_integration.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""The dream end to end, with a scripted model and the real database (ported from Honcho's
tests/dreamer/test_dreamer_integration.py): the completion write of the guard fields, the session filter of
a fired timer, the in-flight guard, and a whole dream whose specialists run their tool loops."""

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api.people import peer_card as peer_card_crud
from hindsight_api.people.context import PeopleContext
from hindsight_api.people.deriver.enqueue import enqueue_dream
from hindsight_api.people.dream_history import get_run_detail, list_runs
from hindsight_api.people.dreamer.dream_scheduler import (
    DreamScheduler,
    check_and_schedule_dream,
    set_dream_scheduler,
)
from hindsight_api.people.dreamer.orchestrator import DreamResult, process_dream, run_dream
from hindsight_api.people.dreamer.payload import DreamPayload
from hindsight_api.people.dreamer.settings import DREAM
from hindsight_api.people.schemas import DreamType
from tests.people.conftest import FakeLLM
from tests.people.dialectic_support import tool_turn
from tests.people.dreamer_support import (
    insert_conclusions,
    make_session,
    scalar,
    seed,
    set_state,
    stored_state,
)


def _make_dream_result() -> DreamResult:
    """A minimal non-null DreamResult for happy-path tests."""
    return DreamResult(
        run_id="test_run_01",
        specialists_run=["deduction", "induction"],
        deduction_success=True,
        induction_success=True,
        surprisal_enabled=False,
        surprisal_conclusion_count=0,
        total_iterations=3,
        total_duration_ms=1234.5,
        input_tokens=100,
        output_tokens=50,
    )


def _payload() -> DreamPayload:
    return DreamPayload(dream_type=DreamType.OMNI, observer="ann", observed="ann")


async def _dream_meta(ctx: PeopleContext, bank_id: str) -> dict[str, Any]:
    return (await stored_state(ctx, bank_id)).get("dream", {})


class TestLastDreamAtCompletionWrite:
    """`last_dream_at` is written at dream completion, not at enqueue time."""

    async def test_happy_path_writes_last_dream_at(self, ctx: PeopleContext, bank_id: str):
        with patch(
            "hindsight_api.people.dreamer.orchestrator.run_dream", new=AsyncMock(return_value=_make_dream_result())
        ):
            await process_dream(ctx, _payload(), bank_id)

        dream_meta = await _dream_meta(ctx, bank_id)
        assert "last_dream_at" in dream_meta
        # Must be a tz-aware UTC ISO timestamp: a naive one would corrupt the min-hours gate math.
        parsed = datetime.fromisoformat(dream_meta["last_dream_at"])
        assert parsed.tzinfo is not None
        assert parsed.utcoffset() == timedelta(0)

    async def test_failure_path_leaves_last_dream_at_null(self, ctx: PeopleContext, bank_id: str):
        """run_dream returns None: the guard must not falsely advance."""
        with patch("hindsight_api.people.dreamer.orchestrator.run_dream", new=AsyncMock(return_value=None)):
            await process_dream(ctx, _payload(), bank_id)

        assert "last_dream_at" not in await _dream_meta(ctx, bank_id)

    async def test_completion_writes_guard_pair_together(self, ctx: PeopleContext, bank_id: str):
        """Both guard fields advance together; the baseline is the explicit count at completion."""
        await insert_conclusions(ctx, bank_id, "explicit", 7)

        with patch(
            "hindsight_api.people.dreamer.orchestrator.run_dream", new=AsyncMock(return_value=_make_dream_result())
        ):
            await process_dream(ctx, _payload(), bank_id)

        dream_meta = await _dream_meta(ctx, bank_id)
        assert "last_dream_at" in dream_meta
        assert dream_meta.get("last_dream_document_count") == 7

    async def test_derived_conclusions_do_not_move_the_baseline(self, ctx: PeopleContext, bank_id: str):
        await insert_conclusions(ctx, bank_id, "explicit", 12)
        await insert_conclusions(ctx, bank_id, "deductive", 30)

        with patch(
            "hindsight_api.people.dreamer.orchestrator.run_dream", new=AsyncMock(return_value=_make_dream_result())
        ):
            await process_dream(ctx, _payload(), bank_id)

        assert (await _dream_meta(ctx, bank_id)).get("last_dream_document_count") == 12

    async def test_exception_path_leaves_last_dream_at_null(self, ctx: PeopleContext, bank_id: str):
        """run_dream raises: process_dream swallows it (the task is marked processed) and writes no guard."""
        with patch(
            "hindsight_api.people.dreamer.orchestrator.run_dream",
            new=AsyncMock(side_effect=RuntimeError("simulated specialist crash")),
        ):
            await process_dream(ctx, _payload(), bank_id)

        assert "last_dream_at" not in await _dream_meta(ctx, bank_id)

    async def test_the_scheduler_key_keeps_other_state(self, ctx: PeopleContext, bank_id: str):
        """The completion write merges into the pair's state: notes stored under their own key survive."""
        await set_state(ctx, bank_id, "ann", "ann", {"dream_notes": {"entries": [{"id": "n1"}]}})

        with patch(
            "hindsight_api.people.dreamer.orchestrator.run_dream", new=AsyncMock(return_value=_make_dream_result())
        ):
            await process_dream(ctx, _payload(), bank_id)

        state = await stored_state(ctx, bank_id)
        assert state["dream_notes"] == {"entries": [{"id": "n1"}]}
        assert "last_dream_at" in state["dream"]


class TestEnqueueDreamLeavesMetadataAlone:
    """enqueue_dream must not touch the pair's dream state: the guard advances only at completion."""

    async def test_enqueue_does_not_modify_dream_metadata(self, ctx: PeopleContext, bank_id: str):
        prior = {"dream": {"last_dream_at": "2026-04-17T12:00:00+00:00", "last_dream_document_count": 99}}
        await set_state(ctx, bank_id, "ann", "ann", prior)

        await enqueue_dream(ctx, bank_id, "ann", "ann", DreamType.OMNI, session_name=None)

        assert await _dream_meta(ctx, bank_id) == prior["dream"]


class TestExecuteDreamSessionFilter:
    """The session lookup filters to explicit conclusions, symmetric with the count query.

    Otherwise a derived conclusion could win ORDER BY created_at DESC and the dream would be scoped to a
    session that was not in the triggering cohort.
    """

    async def test_session_name_picked_from_latest_explicit_doc(self, ctx: PeopleContext, bank_id: str):
        session_a = await make_session(ctx, bank_id)
        session_b = await make_session(ctx, bank_id)
        # Session A (older) has the only explicit conclusion; session B (newer) has dreamer output.
        await insert_conclusions(ctx, bank_id, "explicit", 1, age_minutes=10, session_name=session_a)
        await insert_conclusions(ctx, bank_id, "deductive", 1, age_minutes=1, session_name=session_b)

        captured: dict[str, Any] = {}

        async def capture_enqueue_dream(
            _ctx: Any, bank_id: str, observer: str, observed: str, dream_type: Any, **kwargs
        ):
            captured.update(observer=observer, observed=observed, dream_type=dream_type, **kwargs)

        scheduler = DreamScheduler(lambda: ctx)
        with (
            patch.object(DREAM, "ENABLED", True),
            patch("hindsight_api.people.deriver.enqueue.enqueue_dream", side_effect=capture_enqueue_dream),
        ):
            await scheduler.execute_dream(bank_id, DreamType.OMNI, observer="ann", observed="ann")

        assert captured, "enqueue_dream must be called: execute_dream returned early"
        assert captured["session_name"] == session_a


class TestGuardPairCoherence:
    """From the moment a dream is scheduled until it completes or fails, no second dream may be enqueued
    for the same pair, and the baseline count advances only when consolidation actually happened."""

    @pytest.fixture
    def _scheduler(self, ctx: PeopleContext, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(DREAM, "ENABLED", True)
        monkeypatch.setattr(DREAM, "DOCUMENT_THRESHOLD", 50)
        monkeypatch.setattr(DREAM, "ENABLED_TYPES", ["omni"])
        scheduler = DreamScheduler(lambda: ctx)
        set_dream_scheduler(scheduler)
        yield scheduler
        set_dream_scheduler(None)

    async def test_pending_queue_item_blocks_second_schedule(
        self, ctx: PeopleContext, bank_id: str, _scheduler: DreamScheduler
    ):
        """In-flight window: enqueue fires a dream, more explicit conclusions arrive past the threshold
        again, but the pending queue row blocks a second schedule."""
        await insert_conclusions(ctx, bank_id, "explicit", 50)
        await enqueue_dream(ctx, bank_id, "ann", "ann", DreamType.OMNI, session_name=None)

        async def pending() -> int:
            return await scalar(
                ctx,
                f"SELECT count(*) FROM {ctx.table('queue')} "
                "WHERE bank_id = $1 AND task_type = 'dream' AND processed = false",
                bank_id,
            )

        assert await pending() == 1
        await insert_conclusions(ctx, bank_id, "explicit", 50)

        assert await check_and_schedule_dream(ctx, bank_id, "ann", "ann") is False
        assert await pending() == 1

    async def test_silent_failure_allows_retry_on_same_corpus(
        self, ctx: PeopleContext, bank_id: str, _scheduler: DreamScheduler
    ):
        """A failed dream (run_dream returns None) leaves both guard fields untouched, so the same
        corpus is scheduled again instead of silently consuming the baseline."""
        await insert_conclusions(ctx, bank_id, "explicit", 50)

        with patch("hindsight_api.people.dreamer.orchestrator.run_dream", new=AsyncMock(return_value=None)):
            await process_dream(ctx, _payload(), bank_id)

        dream_meta = await _dream_meta(ctx, bank_id)
        assert dream_meta.get("last_dream_document_count", 0) == 0
        assert "last_dream_at" not in dream_meta

        with patch.object(_scheduler, "schedule_dream", new_callable=AsyncMock) as mock_schedule:
            scheduled = await check_and_schedule_dream(ctx, bank_id, "ann", "ann")

        assert scheduled is True
        assert mock_schedule.called


class TestWholeDream:
    """A dream whose specialists run their tool loops on a scripted model, writing to the real database."""

    @staticmethod
    def _script(coffee_id: str, nights_id: str, calls: list[str]) -> Any:
        """Deduction: read, write a deductive conclusion, update the card, finish. Induction: write one pattern, finish."""

        def script(index: int, messages: list[dict[str, Any]], response_format: Any) -> Any:
            system = messages[0]["content"]
            specialist = "deduction" if "deductive reasoning agent" in system else "induction"
            turn = sum(1 for c in calls if c == specialist)
            calls.append(specialist)
            if specialist == "deduction":
                steps = [
                    tool_turn("get_recent_observations", limit=5),
                    tool_turn(
                        "create_observations_deductive",
                        observations=[
                            {
                                "content": "Ann relies on caffeine to work nights",
                                "source_ids": [coffee_id, nights_id],
                                "premises": ["Ann drinks coffee daily", "Ann works nights"],
                            }
                        ],
                    ),
                    tool_turn("update_peer_card", content=["IDENTITY: Name: Ann", "ATTRIBUTE: Works nights"]),
                    tool_turn("done", summary="wrote one deduction"),
                ]
            else:
                steps = [
                    tool_turn(
                        "create_observations_inductive",
                        observations=[
                            {
                                "content": "Ann tends to organize her day around night shifts",
                                "source_ids": [coffee_id, nights_id],
                                "sources": ["Ann drinks coffee daily", "Ann works nights"],
                                "pattern_type": "behavior",
                                "confidence": "medium",
                            }
                        ],
                    ),
                    tool_turn("done"),
                ]
            return steps[min(turn, len(steps) - 1)]

        return script

    async def test_a_dream_writes_conclusions_card_and_history(
        self, ctx: PeopleContext, bank_id: str, fake_llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(DREAM, "ENABLED", True)
        monkeypatch.setattr(DREAM.SURPRISAL, "ENABLED", False)
        session, (coffee_id, nights_id) = await seed(ctx, bank_id)
        calls: list[str] = []
        fake_llm.script = self._script(coffee_id, nights_id, calls)

        result = await run_dream(ctx, bank_id, "ann", "ann", session, dream_type="omni", trigger_reason="manual")

        assert result is not None
        assert (result.deduction_success, result.induction_success) == (True, True)
        assert calls.count("deduction") == 4 and calls.count("induction") == 2
        levels = {
            row["level"]: row["n"]
            for row in await _fetch(
                ctx,
                f"SELECT level, count(*) AS n FROM {ctx.table('conclusions')} "
                "WHERE bank_id = $1 AND deleted_at IS NULL GROUP BY level",
                bank_id,
            )
        }
        assert levels == {"explicit": 2, "deductive": 1, "inductive": 1}
        assert await peer_card_crud.get_peer_card(ctx, bank_id, "ann", "ann") == [
            "IDENTITY: Name: Ann",
            "ATTRIBUTE: Works nights",
        ]

        # The dream recorded what it did, under the specialist that did it.
        [run] = await list_runs(ctx, bank_id)
        assert (run.status, run.created_count, run.peer_card_updated) == ("completed", 2, True)
        detail = await get_run_detail(ctx, bank_id, run.id)
        assert sorted(c.specialist or "" for c in detail.created) == ["deduction", "induction"]
        deductive = next(c for c in detail.created if c.specialist == "deduction")
        assert deductive.conclusion.content == "Ann relies on caffeine to work nights"
        assert [s.id for s in deductive.sources] == [coffee_id, nights_id]
        assert detail.peer_card is not None and detail.peer_card.before is None

    async def test_a_dream_moves_the_guard_through_process_dream(
        self, ctx: PeopleContext, bank_id: str, fake_llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(DREAM, "ENABLED", True)
        monkeypatch.setattr(DREAM.SURPRISAL, "ENABLED", False)
        session, (coffee_id, nights_id) = await seed(ctx, bank_id)
        fake_llm.script = self._script(coffee_id, nights_id, [])

        await process_dream(
            ctx, DreamPayload(dream_type=DreamType.OMNI, observer="ann", observed="ann", session_name=session), bank_id
        )

        dream_meta = await _dream_meta(ctx, bank_id)
        assert dream_meta["last_dream_document_count"] == 2
        assert "last_dream_at" in dream_meta

    async def test_a_failing_model_fails_the_run_but_not_the_dream_task(
        self, ctx: PeopleContext, bank_id: str, fake_llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(DREAM, "ENABLED", True)
        monkeypatch.setattr(DREAM.SURPRISAL, "ENABLED", False)
        session, _ = await seed(ctx, bank_id)

        def boom(index: int, messages: list[dict[str, Any]], response_format: Any) -> Any:
            raise RuntimeError("provider down")

        fake_llm.script = boom

        result = await run_dream(ctx, bank_id, "ann", "ann", session)

        assert result is not None
        assert (result.deduction_success, result.induction_success) == (False, False)
        [run] = await list_runs(ctx, bank_id)
        assert run.status == "failed"

    async def test_dreams_disabled_in_the_session_skip_the_run(
        self, ctx: PeopleContext, bank_id: str, fake_llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(DREAM, "ENABLED", True)
        session = await make_session(ctx, bank_id, {"dream": {"enabled": False}})

        assert await run_dream(ctx, bank_id, "ann", "ann", session) is None
        assert fake_llm.calls == []
        assert await list_runs(ctx, bank_id) == []


async def _fetch(ctx: PeopleContext, sql: str, *args: Any) -> list[Any]:
    async with ctx.conn() as conn:
        return list(await conn.fetch(sql, *args))
