# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: tests/dialectic/test_model_config_usage.py
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: tests/dialectic/test_structured_output.py
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: tests/test_workspace_chat.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""The dialectic agent's tool loop and answers, with a scripted model and no database."""

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from hindsight_api.engine.response_models import LLMToolCallResult
from hindsight_api.people.dialectic import core, settings
from hindsight_api.people.dialectic.core import (
    DialecticAgent,
    StructuredResponseError,
    chunk_text,
    dialectic_budget_line,
)
from hindsight_api.people.dialectic.evidence import EvidenceAccumulator
from hindsight_api.people.dialectic.settings import DialecticLevelSettings
from hindsight_api.people.dialectic.tools import (
    WORKSPACE_DIALECTIC_TOOLS,
    WORKSPACE_RECALL_TOOLS,
    tool_name,
)
from hindsight_api.people.dialectic.workspace import WorkspaceDialecticAgent
from tests.people.conftest import FakeLLM
from tests.people.dialectic_support import answer_ready, bare_context, tool_turn


class FoodPreferences(BaseModel):
    favorite: str
    confidence: float


class Recorder:
    """A tool executor that records its calls and answers from a table."""

    def __init__(self, outputs: dict[str, str] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.outputs = outputs or {}

    async def __call__(self, name: str, tool_input: dict[str, Any]) -> str:
        self.calls.append((name, tool_input))
        return self.outputs.get(name, f"result of {name}")


def _agent(
    script: Any, level: str = "low", recorder: Recorder | None = None, **kwargs: Any
) -> tuple[DialecticAgent, FakeLLM, Recorder]:
    llm = FakeLLM(script)
    recorder = recorder or Recorder()
    agent = DialecticAgent(bare_context(llm), "bank", None, "observer", "observed", reasoning_level=level, **kwargs)  # type: ignore[arg-type]
    return agent, llm, recorder


@pytest.fixture(autouse=True)
def _no_database(monkeypatch: pytest.MonkeyPatch) -> None:
    """The agent's prefetch and tool executor are the only parts that touch the database."""

    async def no_prefetch(self: DialecticAgent, query: str) -> None:
        return None

    monkeypatch.setattr(DialecticAgent, "_prefetch_relevant_observations", no_prefetch)


def _use(monkeypatch: pytest.MonkeyPatch, recorder: Recorder) -> None:
    async def make(self: DialecticAgent) -> Recorder:
        return recorder

    monkeypatch.setattr(DialecticAgent, "_create_tool_executor", make)


class TestTheLevelModel:
    async def test_answer_uses_the_interactive_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent, llm, recorder = _agent(["answer"], level="medium")
        _use(monkeypatch, recorder)
        # A turn without a call is the answer.
        assert await agent.answer("What do you know?") == "answer"
        assert [c["kind"] for c in llm.calls] == ["tools"]
        assert llm.calls[0]["scope"] == core.TOOL_SCOPE
        assert llm.calls[0]["max_completion_tokens"] == settings.MAX_OUTPUT_TOKENS

    async def test_minimal_uses_its_own_output_cap_and_tools(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent, llm, recorder = _agent([tool_turn("search_memory", query="x"), "short"], level="minimal")
        _use(monkeypatch, recorder)
        # One tool round, then the final call (minimal allows a single round).
        assert await agent.answer("q") == "short"
        assert llm.calls[0]["max_completion_tokens"] == 250
        offered = {tool_name(t) for t in llm.calls[0]["tools"]}
        assert offered == {"search_memory", "search_messages", "answer_ready"}

    async def test_every_turn_declares_the_same_functions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent, llm, recorder = _agent([tool_turn("search_memory", query="x"), answer_ready(), "done"])
        _use(monkeypatch, recorder)
        await agent.answer("q")
        declared = [tuple(tool_name(t) for t in c["declared_tools"]) for c in llm.calls if c["kind"] == "tools"]
        assert len(declared) == 2 and declared[0] == declared[1]
        assert declared[0][-1] == "answer_ready"


class TestTheToolLoop:
    async def test_research_then_final_answer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent, llm, recorder = _agent(
            [
                tool_turn("search_memory", query="sushi"),
                tool_turn("grep_messages", text="sushi"),
                answer_ready(),
                "Alice likes sushi [source: https://x/1, said_at: 2026-01-01].",
            ]
        )
        _use(monkeypatch, recorder)
        evidence = EvidenceAccumulator()
        agent.evidence = evidence

        text = await agent.answer("What does Alice like?")

        assert text.startswith("Alice likes sushi")
        assert recorder.calls == [("search_memory", {"query": "sushi"}), ("grep_messages", {"text": "sushi"})]
        assert [c["kind"] for c in llm.calls] == ["tools", "tools", "tools", "call"]
        final = llm.calls[-1]
        assert final["scope"] == core.ANSWER_SCOPE
        assert final["response_format"] is None
        # The final call sees the research as plain turns and is asked to answer.
        roles = [m["role"] for m in final["messages"]]
        assert roles[0] == "system" and roles[-1] == "user"
        assert any("Result of `search_memory`" in str(m["content"]) for m in final["messages"])
        assert "provide your final response now" in final["messages"][-1]["content"]
        assert not any(m.get("tool_calls") or m["role"] == "tool" for m in final["messages"])
        assert [t.tool_name for t in evidence.tool_calls] == ["search_memory", "grep_messages"]

    async def test_the_loop_stops_at_the_level_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent, llm, recorder = _agent(
            [tool_turn("search_memory", query="a"), tool_turn("search_memory", query="b"), "never asked", "final"],
            level="medium",
        )
        _use(monkeypatch, recorder)
        assert await agent.answer("q") == "never asked"
        # medium allows two tool rounds, then the final call (the script's third answer).
        assert [c["kind"] for c in llm.calls] == ["tools", "tools", "call"]
        assert len(recorder.calls) == 2

    async def test_a_failing_tool_is_not_logged_as_a_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = Recorder({"grep_messages": "ERROR: 'text' parameter is required"})
        agent, llm, _ = _agent([tool_turn("grep_messages"), answer_ready(), "answer"], recorder=recorder)
        _use(monkeypatch, recorder)
        evidence = EvidenceAccumulator()
        agent.evidence = evidence
        await agent.answer("q")
        assert evidence.tool_calls == []
        # The error went back to the model as the tool's result.
        assert llm.calls[1]["messages"][-1] == {
            "role": "tool",
            "tool_call_id": "call_grep_messages_2",
            "content": "ERROR: 'text' parameter is required",
        }

    async def test_the_input_cap_ends_the_research(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent, llm, recorder = _agent([tool_turn("search_memory", query="a"), "unused", "final"])
        _use(monkeypatch, recorder)
        recorder.outputs["search_memory"] = "x" * 4000
        # Room for the prompt and a little more: the 4000-character tool result then fills the budget.
        monkeypatch.setattr(settings, "MAX_INPUT_TOKENS", core._estimate_tokens(agent.messages) + 300)
        assert await agent.answer("q") == "unused"
        assert [c["kind"] for c in llm.calls] == ["tools", "call"]

    @pytest.mark.parametrize(
        ("used", "limit", "expected"),
        [
            (1, 5, "Tool calls used: 1 of 5."),
            (4, 5, "Tool calls used: 4 of 5. On the next call, answer from what you have."),
            (5, 5, None),
        ],
    )
    def test_budget_line(self, used: int, limit: int, expected: str | None) -> None:
        assert dialectic_budget_line(used, limit) == expected

    async def test_the_budget_line_follows_each_round(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent, llm, recorder = _agent([tool_turn("search_memory", query="a"), answer_ready(), "done"], level="max")
        _use(monkeypatch, recorder)
        monkeypatch.setattr(settings, "BUDGET_ENABLED", True)
        await agent.answer("q")
        assert llm.calls[1]["messages"][-1] == {"role": "user", "content": "Tool calls used: 1 of 10."}

    async def test_the_budget_line_is_off_by_default_and_for_small_levels(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent, llm, recorder = _agent([tool_turn("search_memory", query="a"), answer_ready(), "done"], level="max")
        _use(monkeypatch, recorder)
        await agent.answer("q")
        assert llm.calls[1]["messages"][-1]["role"] == "tool"
        small = _agent(["x"], level="medium")[0]
        monkeypatch.setattr(settings, "BUDGET_ENABLED", True)
        assert small._budget_line(settings.LEVELS["medium"]) is None


class TestStructuredOutput:
    """A final answer that fails response_model surfaces as a 422, not a 500."""

    async def test_answer_passes_response_model_and_serializes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        parsed = FoodPreferences(favorite="sushi", confidence=0.9)
        agent, llm, recorder = _agent([answer_ready(), parsed])
        _use(monkeypatch, recorder)
        result = await agent.answer("query", response_model=FoodPreferences)
        assert llm.calls[-1]["response_format"] is FoodPreferences
        assert FoodPreferences.model_validate_json(result) == parsed

    async def test_a_turn_without_a_call_still_goes_through_the_response_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent, llm, recorder = _agent(["I found sushi.", {"favorite": "sushi", "confidence": 0.5}])
        _use(monkeypatch, recorder)
        result = await agent.answer("query", response_model=FoodPreferences)
        assert [c["kind"] for c in llm.calls] == ["tools", "call"]
        assert FoodPreferences.model_validate_json(result).favorite == "sushi"

    async def test_answer_without_response_model_returns_plain_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent, llm, recorder = _agent([answer_ready(), "Alice likes sushi."])
        _use(monkeypatch, recorder)
        assert await agent.answer("query") == "Alice likes sushi."
        assert llm.calls[-1]["response_format"] is None

    @pytest.mark.parametrize("content", ["", "I can't help with that."])
    async def test_answer_rejects_unparsed_content(self, monkeypatch: pytest.MonkeyPatch, content: str) -> None:
        """A backend that degrades to a plain string instead of raising (an empty answer or a refusal) must still
        surface as a 422."""
        agent, _, recorder = _agent([answer_ready(), content])
        _use(monkeypatch, recorder)
        with pytest.raises(StructuredResponseError) as caught:
            await agent.answer("query", response_model=FoodPreferences)
        assert caught.value.status_code == 422

    async def test_answer_maps_a_schema_mismatch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The provider validates the answer against the response model and raises when it does not fit.
        agent, _, recorder = _agent([answer_ready(), {"favorite": "sushi"}])
        _use(monkeypatch, recorder)
        with pytest.raises(StructuredResponseError) as caught:
            await agent.answer("query", response_model=FoodPreferences)
        assert isinstance(caught.value.__cause__, ValidationError)

    async def test_answer_without_response_model_does_not_map_validation_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent, llm, recorder = _agent([answer_ready(), "x"])
        _use(monkeypatch, recorder)

        async def failing(**kwargs: Any) -> None:
            FoodPreferences.model_validate_json("I found sushi.")

        llm.call = failing  # type: ignore[method-assign]
        with pytest.raises(ValidationError):
            await agent.answer("query")


class TestStreaming:
    async def test_the_final_answer_is_yielded_in_pieces_that_rejoin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent, llm, recorder = _agent(
            [tool_turn("search_memory", query="a"), answer_ready(), "first line\nsecond line\n"]
        )
        _use(monkeypatch, recorder)
        chunks = [chunk async for chunk in agent.answer_stream("q")]
        assert chunks == ["first line\n", "second line\n"]
        assert [c["kind"] for c in llm.calls] == ["tools", "tools", "call"]

    async def test_stream_passes_response_model_and_chunks_rejoin_to_json(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent, llm, recorder = _agent([answer_ready(), FoodPreferences(favorite="sushi", confidence=0.9)])
        _use(monkeypatch, recorder)
        chunks = [chunk async for chunk in agent.answer_stream("query", response_model=FoodPreferences)]
        assert llm.calls[-1]["response_format"] is FoodPreferences
        assert FoodPreferences.model_validate_json("".join(chunks)).favorite == "sushi"

    def test_chunking_keeps_every_character(self) -> None:
        text = "a\n\nb\nc"
        assert "".join(chunk_text(text)) == text
        assert chunk_text("") == []


class TestForcedFirstTurn:
    """The bank agent must recall before it answers (Honcho's TestWorkspaceToolChoice)."""

    @pytest.mark.parametrize("level", ["minimal", "low", "medium", "high", "max"])
    def test_first_turn_requires_a_tool_call(self, level: str) -> None:
        agent = WorkspaceDialecticAgent(bare_context(), "bank", reasoning_level=level)  # type: ignore[arg-type]
        assert agent._tool_choice(settings.LEVELS[level]) == "required"  # type: ignore[index]

    def test_pair_agent_keeps_the_configured_choice(self) -> None:
        agent, _, _ = _agent(["x"])
        assert agent._tool_choice(settings.LEVELS["low"]) == settings.LEVELS["low"].TOOL_CHOICE

    def test_a_configured_non_auto_choice_is_passed_through(self) -> None:
        agent = WorkspaceDialecticAgent(bare_context(), "bank")
        assert agent._tool_choice(DialecticLevelSettings(MAX_TOOL_ITERATIONS=5, TOOL_CHOICE="none")) == "none"

    def test_forced_turn_is_satisfied_only_by_recall_tools(self) -> None:
        gate = WorkspaceDialecticAgent(bare_context(), "bank")._force_tools_until()
        assert gate is not None and gate == WORKSPACE_RECALL_TOOLS
        assert gate <= {tool_name(t) for t in WORKSPACE_DIALECTIC_TOOLS}
        assert {"get_peer_card", "get_reasoning_chain"}.isdisjoint(gate)
        assert _agent(["x"])[0]._force_tools_until() is None

    @pytest.mark.parametrize("level", ["low", "medium", "high", "max"])
    def test_workspace_gets_extra_tool_rounds(self, level: str) -> None:
        level_settings = settings.LEVELS[level]  # type: ignore[index]
        agent = WorkspaceDialecticAgent(bare_context(), "bank", reasoning_level=level)  # type: ignore[arg-type]
        assert agent._max_tool_iterations(level_settings) == (
            level_settings.MAX_TOOL_ITERATIONS + settings.WORKSPACE_EXTRA_TOOL_ITERATIONS
        )
        pair = _agent(["x"], level=level)[0]
        assert pair._max_tool_iterations(level_settings) == level_settings.MAX_TOOL_ITERATIONS

    def test_minimal_keeps_its_single_round(self) -> None:
        agent = WorkspaceDialecticAgent(bare_context(), "bank", reasoning_level="minimal")
        assert agent._max_tool_iterations(settings.LEVELS["minimal"]) == settings.LEVELS["minimal"].MAX_TOOL_ITERATIONS

    def test_workspace_loadouts_carry_no_orientation_stats_tool(self) -> None:
        assert all(tool_name(t) != "get_workspace_stats" for t in WORKSPACE_DIALECTIC_TOOLS)

    async def test_answer_ready_is_held_back_until_a_recall_tool_ran(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm = FakeLLM(
            [
                tool_turn("get_peer_card", observer="a", observed="a"),
                tool_turn("search_memory", observed=["a"], query="x"),
                answer_ready(),
                "answer",
            ]
        )
        agent = WorkspaceDialecticAgent(bare_context(llm), "bank", reasoning_level="low")
        recorder = Recorder()

        async def make(self: DialecticAgent) -> Recorder:
            return recorder

        async def no_prefetch(self: DialecticAgent, query: str) -> None:
            return None

        monkeypatch.setattr(WorkspaceDialecticAgent, "_create_tool_executor", make)
        monkeypatch.setattr(WorkspaceDialecticAgent, "_prefetch_relevant_observations", no_prefetch)

        assert await agent.answer("q") == "answer"
        offered = [{tool_name(t) for t in c["tools"]} for c in llm.calls if c["kind"] == "tools"]
        # Turn 1 and 2 may only recall; get_peer_card is not a recall tool, so turn 2 is still held.
        assert offered[0] == WORKSPACE_RECALL_TOOLS
        assert offered[1] == WORKSPACE_RECALL_TOOLS
        assert "answer_ready" in offered[2]

    async def test_a_failed_recall_does_not_lift_the_gate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm = FakeLLM(
            [tool_turn("search_messages", query="x"), tool_turn("search_messages", query="y"), answer_ready(), "answer"]
        )
        agent = WorkspaceDialecticAgent(bare_context(llm), "bank", reasoning_level="low")
        recorder = Recorder({"search_messages": "ERROR: Tool search_messages failed"})

        async def make(self: DialecticAgent) -> Recorder:
            return recorder

        async def no_prefetch(self: DialecticAgent, query: str) -> None:
            return None

        monkeypatch.setattr(WorkspaceDialecticAgent, "_create_tool_executor", make)
        monkeypatch.setattr(WorkspaceDialecticAgent, "_prefetch_relevant_observations", no_prefetch)
        await agent.answer("q")
        offered = [{tool_name(t) for t in c["tools"]} for c in llm.calls if c["kind"] == "tools"]
        assert "answer_ready" not in offered[1]


class TestAReplyWithoutCalls:
    async def test_tool_turn_result_type_is_what_the_provider_returns(self) -> None:
        # The engine's provider answers a free turn with exactly one call; a reply with none is the answer.
        turn = LLMToolCallResult(content="plain", tool_calls=[], finish_reason="stop")
        agent, llm, _ = _agent([turn])
        assert await agent.answer("q") == "plain"
