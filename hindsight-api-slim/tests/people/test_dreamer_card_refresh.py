# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: tests/dreamer/test_card_refresh.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""The card_refresh dream type (ported from Honcho's tests/dreamer/test_card_refresh.py).

Covers queue plumbing (payload round trip, work-unit key isolation from omni, enqueue alongside a pending
omni dream), process_dream dispatch of DreamType.CARD_REFRESH (and that it does NOT advance the omni dream
guard pair), the specialist's tool restriction, the low tool-iteration cap, and rebuild mode omitting the
prior peer card from the prompt.
"""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api.people.context import PeopleContext
from hindsight_api.people.deriver.enqueue import enqueue_dream
from hindsight_api.people.dreamer.orchestrator import DreamResult, process_dream
from hindsight_api.people.dreamer.payload import DreamPayload, dream_work_unit_key
from hindsight_api.people.dreamer.settings import DREAM
from hindsight_api.people.dreamer.specialists import CardRefreshSpecialist
from hindsight_api.people.dreamer.tool_loop import ToolLoopResult
from hindsight_api.people.schemas import DreamType
from tests.people.dialectic_support import bare_context
from tests.people.dreamer_support import stored_state

OBSERVATION_MUTATION_TOOLS = {
    "create_observations",
    "create_observations_deductive",
    "create_observations_inductive",
    "delete_observations",
}


def _make_card_refresh_result() -> DreamResult:
    return DreamResult(
        run_id="test_run_card",
        specialists_run=["card_refresh"],
        deduction_success=True,
        induction_success=False,
        surprisal_enabled=False,
        surprisal_conclusion_count=0,
        total_iterations=2,
        total_duration_ms=42.0,
        input_tokens=10,
        output_tokens=5,
    )


class TestQueuePlumbing:
    def test_payload_roundtrip_carries_rebuild(self):
        payload = DreamPayload(dream_type=DreamType.CARD_REFRESH, observer="alice", observed="bob", rebuild=True)
        validated = DreamPayload(**payload.model_dump(mode="json"))
        assert validated.dream_type == DreamType.CARD_REFRESH
        assert validated.rebuild is True

        # Default is False, including for older payloads missing the field.
        assert DreamPayload(dream_type=DreamType.OMNI, observer="a", observed="b").rebuild is False

    def test_payload_refuses_unknown_fields(self):
        with pytest.raises(ValueError):
            DreamPayload(dream_type=DreamType.OMNI, observer="a", observed="b", surprise="x")  # type: ignore[call-arg]

    def test_work_unit_key_does_not_collide_with_omni(self):
        omni_key = dream_work_unit_key("ws", "alice", "bob", "omni")
        card_key = dream_work_unit_key("ws", "alice", "bob", DreamType.CARD_REFRESH)

        assert omni_key != card_key
        assert card_key == "dream:card_refresh:ws:alice:bob"

    async def test_enqueue_alongside_pending_omni(self, ctx: PeopleContext, bank_id: str):
        """A pending omni dream must not dedupe away a card_refresh enqueue: the work-unit keys differ by dream type."""
        await enqueue_dream(ctx, bank_id, "ann", "ann", DreamType.OMNI)
        await enqueue_dream(ctx, bank_id, "ann", "ann", DreamType.CARD_REFRESH, rebuild=True)

        async with ctx.conn() as conn:
            rows = await conn.fetch(
                f"SELECT payload::text AS payload FROM {ctx.table('queue')} "
                "WHERE bank_id = $1 AND task_type = 'dream' AND processed = false",
                bank_id,
            )
        import json

        payloads = [json.loads(row["payload"]) for row in rows]
        assert len(payloads) == 2
        assert {p["dream_type"] for p in payloads} == {"omni", "card_refresh"}
        card = next(p for p in payloads if p["dream_type"] == "card_refresh")
        assert card["rebuild"] is True


class TestProcessDreamDispatch:
    async def test_dispatches_card_refresh(self, ctx: PeopleContext, bank_id: str):
        payload = DreamPayload(
            dream_type=DreamType.CARD_REFRESH, observer="ann", observed="ann", rebuild=True, trigger_reason="manual"
        )

        with patch(
            "hindsight_api.people.dreamer.orchestrator.run_card_refresh_dream",
            new=AsyncMock(return_value=_make_card_refresh_result()),
        ) as mock_run:
            await process_dream(ctx, payload, bank_id)

        assert mock_run.await_args is not None
        kwargs = mock_run.await_args.kwargs
        assert kwargs["bank_id"] == bank_id
        assert kwargs["observer"] == "ann"
        assert kwargs["observed"] == "ann"
        assert kwargs["rebuild"] is True
        assert kwargs["dream_type"] == "card_refresh"
        assert kwargs["trigger_reason"] == "manual"

    async def test_card_refresh_does_not_advance_dream_guard(self, ctx: PeopleContext, bank_id: str):
        """The omni guard pair (last_dream_at / last_dream_document_count) must not move on a card
        refresh: it would delay real consolidation."""
        payload = DreamPayload(dream_type=DreamType.CARD_REFRESH, observer="ann", observed="ann")

        with patch(
            "hindsight_api.people.dreamer.orchestrator.run_card_refresh_dream",
            new=AsyncMock(return_value=_make_card_refresh_result()),
        ):
            await process_dream(ctx, payload, bank_id)

        dream_meta: dict[str, Any] = (await stored_state(ctx, bank_id, "ann")).get("dream", {})
        assert "last_dream_at" not in dream_meta
        assert "last_dream_document_count" not in dream_meta


class TestCardRefreshSpecialist:
    def test_tools_exclude_observation_mutation(self):
        for rebuild in (False, True):
            specialist = CardRefreshSpecialist(rebuild=rebuild)
            tool_names = {t["name"] for t in specialist.get_tools()}
            assert tool_names == {"get_recent_observations", "search_memory", "update_peer_card"}
            assert not tool_names & OBSERVATION_MUTATION_TOOLS

    def test_tools_without_peer_card_strip_update(self):
        specialist = CardRefreshSpecialist()
        tool_names = {t["name"] for t in specialist.get_tools(peer_card_enabled=False)}
        assert "update_peer_card" not in tool_names
        assert not tool_names & OBSERVATION_MUTATION_TOOLS

    def test_low_iteration_cap(self, monkeypatch: pytest.MonkeyPatch):
        specialist = CardRefreshSpecialist()
        assert specialist.get_max_iterations() == min(6, DREAM.MAX_TOOL_ITERATIONS)

        monkeypatch.setattr(DREAM, "MAX_TOOL_ITERATIONS", 4)
        assert specialist.get_max_iterations() == 4

        monkeypatch.setattr(DREAM, "MAX_TOOL_ITERATIONS", 30)
        assert specialist.get_max_iterations() == 6

    def test_rebuild_flag_controls_card_injection(self):
        assert CardRefreshSpecialist(rebuild=False).inject_peer_card is True
        assert CardRefreshSpecialist(rebuild=True).inject_peer_card is False

    def test_rebuild_prompts_instruct_observation_only_build(self):
        specialist = CardRefreshSpecialist(rebuild=True)
        system_prompt = specialist.build_system_prompt("alice")
        assert "REBUILD MODE" in system_prompt
        assert "solely from the observations" in system_prompt

        user_prompt = specialist.build_user_prompt("alice", hints=None, peer_card=None)
        assert "Rebuild the peer card" in user_prompt
        assert "CURRENT PEER CARD" not in user_prompt

    async def _run_specialist(
        self, specialist: CardRefreshSpecialist, stored_card: list[str]
    ) -> tuple[AsyncMock, AsyncMock]:
        """Run the specialist with a fully mocked model layer; returns the (get_peer_card, tool loop) mocks."""
        mock_get_peer_card = AsyncMock(return_value=stored_card)
        mock_loop = AsyncMock(
            return_value=ToolLoopResult(content="done", iterations=1, input_tokens=10, output_tokens=5)
        )

        with (
            patch("hindsight_api.people.dreamer.specialists.get_peer", new=AsyncMock()),
            patch("hindsight_api.people.dreamer.specialists.peer_card_crud.get_peer_card", new=mock_get_peer_card),
            patch(
                "hindsight_api.people.dreamer.specialists.create_tool_executor", new=AsyncMock(return_value=AsyncMock())
            ),
            patch("hindsight_api.people.dreamer.specialists.execute_tool_loop", new=mock_loop),
        ):
            result = await specialist.run(
                bare_context(), bank_id="bank", observer="alice", observed="alice", session_name=None
            )
        assert result.success is True
        return mock_get_peer_card, mock_loop

    # Sentinel card entry that cannot collide with the prompt's own examples (the shared PEER CARD
    # section contains e.g. "IDENTITY: Name: Alice").
    STORED_CARD: list[str] = ["IDENTITY: Name: Zorblax-Prime", "ATTRIBUTE: Location: Ganymede"]

    async def test_refresh_mode_injects_existing_card(self):
        mock_get_peer_card, mock_loop = await self._run_specialist(
            CardRefreshSpecialist(rebuild=False), self.STORED_CARD
        )

        mock_get_peer_card.assert_awaited_once()
        assert mock_loop.await_args is not None
        kwargs = mock_loop.await_args.kwargs
        user_message = kwargs["messages"][1]["content"]
        assert "IDENTITY: Name: Zorblax-Prime" in user_message
        assert "CURRENT PEER CARD" in user_message
        # Restricted tool offering and low iteration cap reach the model call.
        tool_names = {t["name"] for t in kwargs["tools"]}
        assert not tool_names & OBSERVATION_MUTATION_TOOLS
        assert kwargs["max_tool_iterations"] == min(6, DREAM.MAX_TOOL_ITERATIONS)

    async def test_rebuild_mode_omits_existing_card(self):
        mock_get_peer_card, mock_loop = await self._run_specialist(
            CardRefreshSpecialist(rebuild=True), self.STORED_CARD
        )

        # The stored card is never even fetched, let alone injected.
        mock_get_peer_card.assert_not_awaited()
        assert mock_loop.await_args is not None
        kwargs = mock_loop.await_args.kwargs
        for message in kwargs["messages"]:
            assert "IDENTITY: Name: Zorblax-Prime" not in message["content"]
        # No CURRENT PEER CARD block in the user prompt (the system prompt's shared taxonomy section
        # legitimately mentions the phrase).
        assert "CURRENT PEER CARD" not in kwargs["messages"][1]["content"]
