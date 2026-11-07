# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: tests/dreamer/test_model_config_usage.py
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: tests/dreamer/test_dream_v2_rollups.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""The specialists' prompts, tools and model call (ported from Honcho's tests/dreamer/test_model_config_usage.py
and the prompt/payload parts of test_dream_v2_rollups.py)."""

from unittest.mock import AsyncMock, patch

from hindsight_api.people.dreamer.payload import DreamPayload
from hindsight_api.people.dreamer.specialists import DeductionSpecialist, InductionSpecialist
from hindsight_api.people.dreamer.tool_loop import ToolLoopResult
from hindsight_api.people.schemas import DreamType
from tests.people.conftest import FakeLLM
from tests.people.dialectic_support import bare_context


def test_deduction_prompt_uses_identity_markers_framing() -> None:
    """Deduction prompt must frame the peer card as an identity store with the entity-agnostic prefix
    taxonomy, not as a human bio sheet."""
    prompt = DeductionSpecialist().build_system_prompt("alice", peer_card_enabled=True)

    assert "identity store" in prompt
    assert "stable identity markers" in prompt
    for prefix in ("IDENTITY:", "ATTRIBUTE:", "RELATIONSHIP:", "INSTRUCTION:"):
        assert prefix in prompt
    # Cross-entity examples confirm the prompt is not biased toward humans.
    assert "codebase" in prompt
    assert "team" in prompt
    # Behavioral content must be explicitly excluded.
    assert "TRAIT:" in prompt
    # The old human-shaped REQUIRED enumeration must be gone.
    assert "Family members and relationships" not in prompt
    assert "Core preferences and traits" not in prompt


def test_deduction_prompt_omits_peer_card_when_disabled() -> None:
    prompt = DeductionSpecialist().build_system_prompt("alice", peer_card_enabled=False)
    assert "PEER CARD" not in prompt
    assert "IDENTITY:" not in prompt


def test_dreamer_system_prompts_delay_observed_observee_for_cache_prefix() -> None:
    for specialist in (DeductionSpecialist(), InductionSpecialist()):
        prompt = specialist.build_system_prompt("alice", peer_card_enabled=True)
        other_prompt = specialist.build_system_prompt("bob", peer_card_enabled=True)

        assert prompt == other_prompt
        assert "the target observee" in prompt


def test_dreamer_user_prompts_include_target_observee() -> None:
    for specialist in (DeductionSpecialist(), InductionSpecialist()):
        prompt = specialist.build_user_prompt(observed="alice", hints=None, peer_card=None)

        assert "Target observee:\nalice" in prompt


def test_induction_prompt_has_no_peer_card_section() -> None:
    """Induction no longer writes to the peer card; its prompt must not reference it."""
    prompt = InductionSpecialist().build_system_prompt("alice", peer_card_enabled=True)
    assert "PEER CARD" not in prompt
    assert "update_peer_card" not in prompt


def test_induction_specialist_cannot_update_peer_card() -> None:
    """Induction must have can_update_peer_card=False and no update_peer_card tool."""
    specialist = InductionSpecialist()
    assert specialist.can_update_peer_card is False

    tool_names = {t["name"] for t in specialist.get_tools()}
    assert "update_peer_card" not in tool_names
    # Sanity: induction still has the discovery and create tools it actually needs.
    assert "create_observations_inductive" in tool_names
    assert "search_memory" in tool_names


def test_deduction_specialist_can_update_peer_card() -> None:
    specialist = DeductionSpecialist()
    assert specialist.can_update_peer_card is True

    tool_names = {t["name"] for t in specialist.get_tools(peer_card_enabled=True)}
    assert "update_peer_card" in tool_names

    disabled_names = {t["name"] for t in specialist.get_tools(peer_card_enabled=False)}
    assert "update_peer_card" not in disabled_names


async def test_deduction_specialist_runs_on_the_background_provider() -> None:
    """Every dream call goes through the engine's Claude-configured background provider."""
    background = FakeLLM()
    interactive = FakeLLM()
    ctx = bare_context(background)
    ctx.engine._reflect_llm_config = interactive  # type: ignore[attr-defined]
    loop = AsyncMock(return_value=ToolLoopResult(content="done", iterations=1, input_tokens=10, output_tokens=5))

    with (
        patch("hindsight_api.people.dreamer.specialists.get_peer", new=AsyncMock()),
        patch(
            "hindsight_api.people.dreamer.specialists.peer_card_crud.get_peer_card", new=AsyncMock(return_value=None)
        ),
        patch("hindsight_api.people.dreamer.specialists.create_tool_executor", new=AsyncMock(return_value=AsyncMock())),
        patch("hindsight_api.people.dreamer.specialists.execute_tool_loop", new=loop),
    ):
        result = await DeductionSpecialist().run(
            ctx, bank_id="bank", observer="alice", observed="alice", session_name="session"
        )

    assert loop.await_args is not None
    assert loop.await_args.args[0] is background
    assert loop.await_args.kwargs["scope"] == "people_dreamer_deduction"
    assert loop.await_args.kwargs["max_tokens"] == 8192
    assert result.content == "done"


async def test_rollups_come_from_tool_result_metadata_not_tool_names() -> None:
    """Two create calls that made 7 conclusions together report 7, not 2; a refused search is not a search."""
    calls = [
        {
            "tool_name": "create_observations_deductive",
            "tool_result_metadata": {"created_count": 5, "levels": ["deductive"] * 5},
        },
        {
            "tool_name": "create_observations_deductive",
            "tool_result_metadata": {"created_count": 2, "levels": ["deductive"] * 2},
        },
        {"tool_name": "delete_observations", "tool_result_metadata": {"deleted_count": 1, "levels": ["explicit"]}},
        {"tool_name": "update_peer_card", "tool_result_metadata": {"peer_card_updated": True}},
        {"tool_name": "search_memory", "tool_result_metadata": {"results_count": 3}},
        {"tool_name": "search_memory", "tool_result_metadata": {"refused": True}},
        {"tool_name": "update_notes", "tool_result_metadata": {"notes_added": 2, "pending_resolved": 1}},
    ]
    loop = AsyncMock(
        return_value=ToolLoopResult(
            content="DONE", tool_calls_made=calls, iterations=2, input_tokens=1, output_tokens=1
        )
    )
    with (
        patch("hindsight_api.people.dreamer.specialists.get_peer", new=AsyncMock()),
        patch("hindsight_api.people.dreamer.specialists.peer_card_crud.get_peer_card", new=AsyncMock()),
        patch("hindsight_api.people.dreamer.specialists.create_tool_executor", new=AsyncMock(return_value=AsyncMock())),
        patch("hindsight_api.people.dreamer.specialists.execute_tool_loop", new=loop),
    ):
        result = await DeductionSpecialist().run(
            bare_context(), bank_id="w", observer="alice", observed="alice", session_name=None
        )

    assert result.tool_calls_count == 7
    assert result.created_observation_count == 7
    assert result.deleted_observation_count == 1
    assert result.peer_card_updated is True
    assert result.search_tool_calls_count == 1
    assert result.refused_search_count == 1
    assert (result.notes_added, result.pending_resolved) == (2, 1)
    assert result.created_counts_by_level == {"deductive": 7}
    assert result.deleted_counts_by_level == {"explicit": 1}


class TestDreamPayloadSchedulerFields:
    def test_payload_defaults(self):
        payload = DreamPayload(dream_type=DreamType.OMNI, observer="o", observed="user")
        assert payload.trigger_reason is None
        assert payload.delay_reason is None
        assert payload.documents_since_last_dream_at_schedule is None
        assert payload.document_threshold is None

    def test_payload_threads_scheduler_reasons(self):
        payload = DreamPayload(
            dream_type=DreamType.OMNI,
            observer="o",
            observed="user",
            trigger_reason="document_threshold",
            delay_reason="idle_timeout",
            documents_since_last_dream_at_schedule=55,
            document_threshold=50,
        )
        # Round-trip through serialization and back mimics what happens between the scheduler's
        # enqueue and the consumer's dequeue.
        restored = DreamPayload(**payload.model_dump(mode="json"))
        assert restored.trigger_reason == "document_threshold"
        assert restored.delay_reason == "idle_timeout"
        assert restored.documents_since_last_dream_at_schedule == 55
        assert restored.document_threshold == 50
