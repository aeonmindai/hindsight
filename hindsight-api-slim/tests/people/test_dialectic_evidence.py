# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: tests/dialectic/test_evidence.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""What the dialectic agent read while answering, collected passively and reported with its sources."""

from typing import Any

import pytest

from hindsight_api.people.context import PeopleContext
from hindsight_api.people.dialectic.core import DialecticAgent
from hindsight_api.people.dialectic.evidence import EvidenceAccumulator
from hindsight_api.people.dialectic.workspace import WorkspaceDialecticAgent
from tests.people.conftest import FakeLLM
from tests.people.dialectic_support import World, answer_ready, tool_turn, world  # noqa: F401  (world is a fixture)
from tests.people.dialectic_support import at as _at


def _agent(world: World, fake_llm: FakeLLM, script: list[Any], **kwargs: Any) -> DialecticAgent:
    fake_llm.script = script
    return DialecticAgent(world.ctx, world.bank_id, None, "alice", "alice", **kwargs)


class TestPrefetchEvidence:
    async def test_records_conclusions_the_prefetch_loaded(self, world: World, fake_llm: FakeLLM) -> None:
        evidence = EvidenceAccumulator()
        agent = _agent(world, fake_llm, ["answer"], evidence=evidence)
        assert await agent.answer("What does Alice eat?") == "answer"
        # The explicit search and the derived search each returned alice's own conclusions.
        assert {world.conclusions[k] for k in ("sushi", "berlin", "deadline", "foodie")} == set(evidence.conclusions)
        # The prefetch block went to the model, with sources.
        prompt = fake_llm.calls[0]["messages"][1]["content"]
        assert "Relevant Observations (prefetched)" in prompt
        assert "source: https://chat.example/s1/1" in prompt

    async def test_prefetch_covers_derived_conclusions_too(self, world: World, fake_llm: FakeLLM) -> None:
        evidence = EvidenceAccumulator()
        await _agent(world, fake_llm, ["answer"], evidence=evidence).answer("foodie")
        level = evidence.conclusions[world.conclusions["foodie"]].level
        assert level == "deductive"
        assert evidence.conclusions[world.conclusions["foodie"]].source_ids == [
            world.conclusions["sushi"],
            world.conclusions["berlin"],
        ]

    async def test_built_evidence_reports_the_prefetched_conclusions(self, world: World, fake_llm: FakeLLM) -> None:
        evidence = EvidenceAccumulator()
        await _agent(world, fake_llm, ["answer"], evidence=evidence).answer("sushi")
        built = evidence.build()
        assert len(built.conclusions) == 4
        assert built.tool_calls == []
        by_id = {c.id: c for c in built.conclusions}
        sushi = by_id[world.conclusions["sushi"]]
        assert (sushi.observer_id, sushi.observed_id, sushi.session_id) == ("alice", "alice", "s1")
        assert sushi.source_url == "https://chat.example/s1/1" and sushi.said_at == _at(1)

    async def test_records_nothing_when_the_prefetch_block_is_dropped(
        self, world: World, fake_llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Evidence names what the agent saw, not what was fetched for it."""

        async def failing(self: PeopleContext, text: str) -> list[float]:
            raise RuntimeError("embedder down")

        monkeypatch.setattr(PeopleContext, "embed_query", failing)
        evidence = EvidenceAccumulator()
        agent = _agent(world, fake_llm, ["answer"], evidence=evidence)
        assert await agent.answer("sushi") == "answer"
        assert not evidence.conclusions
        assert "prefetched" not in fake_llm.calls[0]["messages"][1]["content"]

    async def test_session_history_is_evidence_too(self, world: World, fake_llm: FakeLLM) -> None:
        evidence = EvidenceAccumulator()
        fake_llm.script = ["answer"]
        agent = DialecticAgent(world.ctx, world.bank_id, "s1", "alice", "alice", evidence=evidence)
        await agent.answer("sushi")
        system = fake_llm.calls[0]["messages"][0]["content"]
        assert "## SESSION HISTORY" in system
        assert "alice: I love sushi from Nobu" in system
        assert "source: https://chat.example/s1/1" in system
        assert {world.messages[k] for k in ("m_sushi", "m_nice", "m_berlin", "m_cool")} == set(evidence.messages)


class TestToolLoopEvidence:
    async def test_the_agents_tool_executor_records_into_the_same_accumulator(
        self, world: World, fake_llm: FakeLLM
    ) -> None:
        evidence = EvidenceAccumulator()
        agent = _agent(world, fake_llm, ["x"], evidence=evidence)
        run = await agent._create_tool_executor()
        await run("grep_messages", {"text": "Berlin"})
        assert world.messages["m_berlin"] in evidence.messages

    async def test_records_the_tool_calls_the_loop_made(self, world: World, fake_llm: FakeLLM) -> None:
        evidence = EvidenceAccumulator()
        agent = _agent(
            world,
            fake_llm,
            [
                tool_turn("grep_messages", text="Berlin"),
                tool_turn("search_memory", query="Berlin"),
                answer_ready(),
                "ok",
            ],
            evidence=evidence,
        )
        await agent.answer("Where does Alice live?")
        built = evidence.build()
        assert [(c.tool_name, c.tool_input) for c in built.tool_calls] == [
            ("grep_messages", {"text": "Berlin"}),
            ("search_memory", {"query": "Berlin"}),
        ]
        assert world.messages["m_berlin"] in {m.id for m in built.messages}

    async def test_records_the_tool_calls_a_streamed_answer_made(self, world: World, fake_llm: FakeLLM) -> None:
        evidence = EvidenceAccumulator()
        agent = _agent(
            world, fake_llm, [tool_turn("grep_messages", text="Berlin"), answer_ready(), "ok\n"], evidence=evidence
        )
        assert [chunk async for chunk in agent.answer_stream("Where does Alice live?")] == ["ok\n"]
        assert [c.tool_name for c in evidence.tool_calls] == ["grep_messages"]

    async def test_evidence_lists_message_provenance_without_content(self, world: World, fake_llm: FakeLLM) -> None:
        evidence = EvidenceAccumulator()
        agent = _agent(
            world, fake_llm, [tool_turn("grep_messages", text="Berlin"), answer_ready(), "ok"], evidence=evidence
        )
        await agent.answer("q")
        ref = evidence.build().messages[0].model_dump()
        assert "content" not in ref
        assert ref["source_url"].startswith("https://chat.example/s1/")
        assert ref["channel"] == "slack" and ref["said_at"] is not None


class TestEvidenceOptedOut:
    async def test_collects_nothing_and_answers_the_same(self, world: World, fake_llm: FakeLLM) -> None:
        agent = _agent(world, fake_llm, [tool_turn("grep_messages", text="Berlin"), answer_ready(), "same"])
        assert agent.evidence is None
        assert await agent.answer("q") == "same"
        run = await agent._create_tool_executor()
        assert "Berlin" in await run("grep_messages", {"text": "Berlin"})


class TestScopedEvidence:
    async def test_stays_empty_when_recall_fails_closed(self, world: World, fake_llm: FakeLLM) -> None:
        evidence = EvidenceAccumulator()
        agent = _agent(world, fake_llm, ["answer"], evidence=evidence, session_allowlist=[])
        await agent.answer("sushi")
        assert not evidence.conclusions and not evidence.messages

    async def test_only_reports_conclusions_inside_the_allowlist(self, world: World, fake_llm: FakeLLM) -> None:
        evidence = EvidenceAccumulator()
        agent = _agent(world, fake_llm, ["answer"], evidence=evidence, session_allowlist=["s2"])
        await agent.answer("deadline")
        assert set(evidence.conclusions) == {world.conclusions["deadline"]}
        run = await agent._create_tool_executor()
        await run("grep_messages", {"text": "e"})
        assert {m.session_id for m in evidence.messages.values()} == {"s2"}


class TestSources:
    async def test_sources_flatten_conclusions_messages_and_facts(self, world: World, fake_llm: FakeLLM) -> None:
        from hindsight_api.people.dialectic.evidence import EvidenceFact

        evidence = EvidenceAccumulator()
        agent = _agent(world, fake_llm, ["answer"], evidence=evidence)
        await agent.answer("sushi")
        run = await agent._create_tool_executor()
        await run("grep_messages", {"text": "Berlin"})
        evidence.add_facts(
            [
                EvidenceFact(
                    id="f1",
                    text="Alice runs",
                    fact_type="world",
                    source_url="https://x/9",
                    said_at="2026-02-01T00:00:00Z",
                )
            ]
        )
        sources = evidence.build().sources()
        assert {s.kind for s in sources} == {"conclusion", "message", "fact"}
        sushi = next(s for s in sources if s.id == world.conclusions["sushi"])
        assert (sushi.source_url, sushi.said_at, sushi.said_by) == (
            "https://chat.example/s1/1",
            _at(1).isoformat(),
            "alice",
        )
        assert sushi.text == "Alice loves sushi"
        message = next(s for s in sources if s.kind == "message")
        assert message.text is None and message.said_by == "alice"
        assert [s.said_at or "" for s in sources] == sorted(s.said_at or "" for s in sources)


class TestWorkspaceAgentEvidence:
    async def test_its_tool_executor_records_into_the_same_accumulator(self, world: World, fake_llm: FakeLLM) -> None:
        evidence = EvidenceAccumulator()
        agent = WorkspaceDialecticAgent(world.ctx, world.bank_id, evidence=evidence)
        run = await agent._create_tool_executor()
        await run("grep_messages", {"text": "Zebra"})
        assert world.messages["m_zebra"] in evidence.messages
