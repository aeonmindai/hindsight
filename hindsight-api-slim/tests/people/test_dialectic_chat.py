"""Asking about a person (or across a bank) end to end: the answer, and the sources it can cite."""

from typing import Any

import pytest
from pydantic import BaseModel

from hindsight_api.people.dialectic import settings as dialectic_settings
from hindsight_api.people.dialectic.chat import agentic_chat, agentic_chat_stream, workspace_chat, workspace_chat_stream
from hindsight_api.people.dialectic.core import DialecticResult, dialectic_chat
from hindsight_api.people.dialectic.evidence import EvidenceAccumulator
from hindsight_api.people.dialectic.tools import tool_name
from hindsight_api.people.exceptions import ResourceNotFoundException
from hindsight_api.people.peer_card import set_peer_card
from tests.people.conftest import FakeLLM
from tests.people.dialectic_support import World, answer_ready, tool_turn, world  # noqa: F401  (world is a fixture)
from tests.people.dialectic_support import at as _at


class TestDialecticChat:
    async def test_the_answer_cites_what_the_agent_read(self, world: World, fake_llm: FakeLLM) -> None:  # noqa: F811
        answer = "Alice moved to Berlin [source: https://chat.example/s1/3, said_at: 2026-03-05T12:00:00+00:00]."
        fake_llm.script = [tool_turn("grep_messages", text="Berlin"), answer_ready(), answer]

        result = await dialectic_chat(world.ctx, world.bank_id, "alice", "alice", "Where does Alice live?")

        assert isinstance(result, DialecticResult)
        assert result.text == answer
        by_id = {s.id: s for s in result.sources}
        berlin = by_id[world.conclusions["berlin"]]
        assert (berlin.kind, berlin.source_url, berlin.said_at) == (
            "conclusion",
            "https://chat.example/s1/3",
            _at(5).isoformat(),
        )
        message = by_id[world.messages["m_berlin"]]
        assert (message.kind, message.source_url, message.said_at, message.said_by) == (
            "message",
            "https://chat.example/s1/3",
            _at(5).isoformat(),
            "alice",
        )
        # Every source says when it was said.
        assert all(s.said_at for s in result.sources)
        assert [t.tool_name for t in result.evidence.tool_calls] == ["grep_messages"]

    async def test_the_prompt_asks_for_citations_and_the_prefetch_shows_them(
        self, world: World, fake_llm: FakeLLM
    ) -> None:  # noqa: F811
        fake_llm.script = ["ok"]
        await dialectic_chat(world.ctx, world.bank_id, "alice", "alice", "sushi")
        system, user = (m["content"] for m in fake_llm.calls[0]["messages"][:2])
        assert "[source: <url>, said_at: <time>]" in system
        assert "Query: sushi" in user
        assert f"[id:{world.conclusions['sushi']}] Alice loves sushi [source: https://chat.example/s1/1" in user

    async def test_a_directional_query_reads_the_observers_view_only(self, world: World, fake_llm: FakeLLM) -> None:  # noqa: F811
        fake_llm.script = ["ok"]
        await dialectic_chat(world.ctx, world.bank_id, "bob", "alice", "Alice")
        system, user = (m["content"] for m in fake_llm.calls[0]["messages"][:2])
        assert "perspective of bob's understanding of alice" in system
        assert "Bob thinks Alice is a foodie" in user
        assert "Alice loves sushi" not in user

    async def test_peer_cards_go_in_the_prompt(self, world: World, fake_llm: FakeLLM) -> None:  # noqa: F811
        await set_peer_card(world.ctx, world.bank_id, ["IDENTITY: Alice Smith"], "alice", "alice")
        await set_peer_card(world.ctx, world.bank_id, ["ATTRIBUTE: Likes fish"], "bob", "alice")
        fake_llm.script = ["ok"]
        await dialectic_chat(world.ctx, world.bank_id, "bob", "alice", "q")
        system = fake_llm.calls[0]["messages"][0]["content"]
        assert "<observed_peer_card>\nATTRIBUTE: Likes fish\n</observed_peer_card>" in system

    async def test_the_memory_brief_is_off_by_default_and_on_when_enabled(  # noqa: F811
        self, world: World, fake_llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_llm.script = ["ok"]
        await dialectic_chat(world.ctx, world.bank_id, "alice", "alice", "q")
        assert "HOW MEMORY IS ORGANIZED" not in fake_llm.calls[0]["messages"][0]["content"]
        assert "Memory counts" not in fake_llm.calls[0]["messages"][1]["content"]

        monkeypatch.setattr(dialectic_settings, "BRIEF_ENABLED", True)
        await dialectic_chat(world.ctx, world.bank_id, "alice", "alice", "q")
        system, user = (m["content"] for m in fake_llm.calls[1]["messages"][:2])
        assert "## HOW MEMORY IS ORGANIZED" in system
        assert "Memory counts: 3 explicit, 1 deductive, 0 inductive, 0 contradiction (4 conclusions" in user

        # Counts span every session of the pair, so a session allowlist rules them out.
        await dialectic_chat(world.ctx, world.bank_id, "alice", "alice", "q", session_allowlist=["s1"])
        assert "Memory counts" not in fake_llm.calls[2]["messages"][1]["content"]

    async def test_response_model_returns_json(self, world: World, fake_llm: FakeLLM) -> None:  # noqa: F811
        class City(BaseModel):
            city: str

        fake_llm.script = [answer_ready(), {"city": "Berlin"}]
        result = await dialectic_chat(
            world.ctx, world.bank_id, "alice", "alice", "city?", response_model=City, reasoning_level="minimal"
        )
        assert City.model_validate_json(result.text).city == "Berlin"

    async def test_the_session_is_in_the_prompt_and_restricts_nothing_by_itself(
        self, world: World, fake_llm: FakeLLM
    ) -> None:  # noqa: F811
        fake_llm.script = ["ok"]
        result = await dialectic_chat(world.ctx, world.bank_id, "alice", "alice", "q", session_name="s1")
        assert "## SESSION HISTORY" in fake_llm.calls[0]["messages"][0]["content"]
        assert world.conclusions["deadline"] in {s.id for s in result.sources}  # s2's conclusion is still recalled

    async def test_a_session_allowlist_restricts_the_sources(self, world: World, fake_llm: FakeLLM) -> None:  # noqa: F811
        fake_llm.script = [tool_turn("search_messages", query="Berlin deadline sushi"), answer_ready(), "ok"]
        result = await dialectic_chat(world.ctx, world.bank_id, "alice", "alice", "q", session_allowlist=["s2"])
        assert {s.session for s in result.sources if s.kind in ("conclusion", "message")} == {"s2"}

    @pytest.mark.parametrize(
        ("observer", "observed", "session"),
        [("nobody", "alice", None), ("alice", "nobody", None), ("alice", "alice", "no-such-session")],
    )
    async def test_missing_people_and_sessions_are_not_found(  # noqa: F811
        self, world: World, fake_llm: FakeLLM, observer: str, observed: str, session: str | None
    ) -> None:
        with pytest.raises(ResourceNotFoundException):
            await dialectic_chat(world.ctx, world.bank_id, observer, observed, "q", session_name=session)
        assert fake_llm.calls == []

    async def test_agentic_chat_and_its_stream(self, world: World, fake_llm: FakeLLM) -> None:  # noqa: F811
        fake_llm.script = ["one\ntwo\n"]
        evidence = EvidenceAccumulator()
        assert (
            await agentic_chat(world.ctx, world.bank_id, None, "q", "alice", "alice", evidence=evidence) == "one\ntwo\n"
        )
        assert evidence.conclusions
        chunks = [c async for c in agentic_chat_stream(world.ctx, world.bank_id, None, "q", "alice", "alice")]
        assert chunks == ["one\n", "two\n"]

    async def test_the_stream_reports_a_missing_person_when_iterated(self, world: World) -> None:  # noqa: F811
        with pytest.raises(ResourceNotFoundException):
            [c async for c in agentic_chat_stream(world.ctx, world.bank_id, None, "q", "nobody", "nobody")]


class TestWorkspaceChat:
    async def test_the_overview_is_prefetched_and_the_first_turn_must_recall(
        self, world: World, fake_llm: FakeLLM
    ) -> None:  # noqa: F811
        await set_peer_card(world.ctx, world.bank_id, ["IDENTITY: Alice Smith"], "alice", "alice")
        fake_llm.script = [tool_turn("search_messages", query="Zebra"), answer_ready(), "Bob works on Zebra."]
        evidence = EvidenceAccumulator()

        answer = await workspace_chat(world.ctx, world.bank_id, None, "Who works on Zebra?", evidence=evidence)

        assert answer == "Bob works on Zebra."
        system, user = (m["content"] for m in fake_llm.calls[0]["messages"][:2])
        assert "bank dialectic" in system
        assert "## Bank overview (prefetched)" in user
        assert "People: 2" in user and "Sessions: 3" in user and "Messages: 7" in user
        assert "- alice (4 messages, last active 2026-03-20)\n    - IDENTITY: Alice Smith" in user
        offered = [{tool_name(t) for t in c["tools"]} for c in fake_llm.calls if c["kind"] == "tools"]
        assert "answer_ready" not in offered[0] and "answer_ready" in offered[1]
        # The bank's message search is not scoped to one person's sessions.
        assert world.messages["m_zebra"] in evidence.messages

    async def test_an_empty_bank_has_no_overview(self, ctx: Any, people_engine: Any, fake_llm: FakeLLM) -> None:
        import uuid

        bank = f"test-people-{uuid.uuid4().hex[:10]}"
        await people_engine.ensure_bank_profile(bank_id=bank, request_context=ctx.request_context)
        fake_llm.script = ["nothing known"]
        assert await workspace_chat(ctx, bank, None, "anything?") == "nothing known"
        assert "prefetched" not in fake_llm.calls[0]["messages"][1]["content"]

    async def test_cards_are_not_prefetched_under_an_allowlist(self, world: World, fake_llm: FakeLLM) -> None:  # noqa: F811
        await set_peer_card(world.ctx, world.bank_id, ["IDENTITY: Alice Smith"], "alice", "alice")
        fake_llm.script = ["ok"]
        await workspace_chat(world.ctx, world.bank_id, None, "q", session_allowlist=["s2"])
        user = fake_llm.calls[0]["messages"][1]["content"]
        assert "People: 2" in user and "Sessions: 1" in user and "Messages: 2" in user
        assert "IDENTITY" not in user
        offered = {tool_name(t) for t in fake_llm.calls[0]["tools"]}
        assert {"get_peer_card", "get_reasoning_chain", "recall_facts"}.isdisjoint(offered)

    async def test_stream(self, world: World, fake_llm: FakeLLM) -> None:  # noqa: F811
        fake_llm.script = [tool_turn("grep_messages", text="Zebra"), answer_ready(), "a\nb"]
        chunks = [c async for c in workspace_chat_stream(world.ctx, world.bank_id, None, "q")]
        assert chunks == ["a\n", "b"]
