# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: tests/test_workspace_chat.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""The dialectic's read tools over a seeded bank: conclusions, messages, reasoning chains, peer cards and facts."""

from types import SimpleNamespace
from typing import Any

import pytest

from hindsight_api.people.dialectic import messages as message_store
from hindsight_api.people.dialectic.evidence import EvidenceAccumulator
from hindsight_api.people.dialectic.tools import (
    create_workspace_tool_executor,
    format_workspace_stats,
    get_active_peers,
    get_workspace_stats,
)
from hindsight_api.people.peer_card import set_peer_card
from tests.people.dialectic_support import World, tools_for, world  # noqa: F401  (world is a fixture)
from tests.people.dialectic_support import at as _at
from tests.people.people_helpers import add_message, add_session


class TestSearchMemory:
    async def test_conclusions_come_with_their_source_and_time(self, world: World) -> None:
        evidence = EvidenceAccumulator()
        run = await tools_for(world, evidence=evidence)
        output = await run("search_memory", {"query": "Alice loves sushi", "top_k": 20})
        assert "Alice loves sushi" in output
        assert f"[id:{world.conclusions['sushi']}]" in output
        assert "source: https://chat.example/s1/1" in output
        assert f"said_at: {_at(1).isoformat()}" in output
        assert f"message_ids: {world.messages['m_sushi']}" in output
        # The deductive conclusion shows under its own level heading.
        assert "DEDUCTIVE:" in output and "Alice is a foodie who relocated" in output
        assert world.conclusions["sushi"] in evidence.conclusions
        assert evidence.conclusions[world.conclusions["sushi"]].source_url == "https://chat.example/s1/1"

    async def test_only_the_pairs_own_conclusions_are_returned(self, world: World) -> None:
        run = await tools_for(world)
        output = await run("search_memory", {"query": "project"})
        assert "Zebra" not in output
        assert "foodie" in output.replace("Bob thinks Alice is a foodie", "")  # alice/alice's own deduction

    async def test_top_k_bounds_the_results(self, world: World) -> None:
        run = await tools_for(world)
        output = await run("search_memory", {"query": "Alice", "top_k": 1})
        assert output.startswith("Found 1 observations")

    async def test_bad_top_k_falls_back_to_the_default(self, world: World) -> None:
        run = await tools_for(world)
        output = await run("search_memory", {"query": "Alice", "top_k": "Infinity"})
        assert output.startswith("Found 4 observations")

    async def test_query_is_required(self, world: World) -> None:
        run = await tools_for(world)
        assert (await run("search_memory", {})).startswith("ERROR:")

    async def test_no_conclusions_falls_back_to_message_search(self, world: World) -> None:
        evidence = EvidenceAccumulator()
        run = await tools_for(world, "carol", "carol", evidence=evidence)
        # carol is in no session, so no messages are visible to her either.
        assert "No observations found" in await run("search_memory", {"query": "sushi"})

        await add_session(world.ctx, world.bank_id, "s4", ["carol"])
        await add_message(
            world.ctx,
            world.bank_id,
            "s4",
            "carol",
            "Carol bakes sourdough on Sundays",
            seq=1,
            source_url="https://chat.example/s4/1",
            said_at=_at(12),
        )
        output = await run("search_memory", {"query": "sourdough"})
        assert output.startswith("No observations yet. Message search results:")
        assert "Carol bakes sourdough" in output
        assert "source: https://chat.example/s4/1" in output
        assert len(evidence.messages) == 1

    async def test_a_session_allowlist_serves_only_explicit_conclusions_of_those_sessions(self, world: World) -> None:
        run = await tools_for(world, allowlist=["s2"])
        output = await run("search_memory", {"query": "Alice"})
        assert "deadline is April 22" in output
        assert "loves sushi" not in output
        assert "foodie" not in output  # derived levels span sessions: never served under an allowlist

    async def test_an_empty_allowlist_fails_closed(self, world: World) -> None:
        evidence = EvidenceAccumulator()
        run = await tools_for(world, allowlist=[], evidence=evidence)
        output = await run("search_memory", {"query": "Alice"})
        assert "No observations found" in output and "no messages found" in output
        assert not evidence.conclusions and not evidence.messages


class TestWorkspaceSearchMemory:
    async def _run(self, world: World, tool_input: dict[str, Any], **kwargs: Any) -> str:
        run = await create_workspace_tool_executor(world.ctx, world.bank_id, agent_type="workspace_dialectic", **kwargs)
        return await run("search_memory", tool_input)

    async def test_requires_observed(self, world: World) -> None:
        assert (await self._run(world, {"query": "x"})).startswith("ERROR: 'observed' is required")

    async def test_observer_defaults_to_observed(self, world: World) -> None:
        output = await self._run(world, {"observed": ["alice"], "query": "Alice loves sushi"})
        assert "[alice->alice]" in output and "Alice loves sushi" in output

    async def test_a_bare_string_is_one_person(self, world: World) -> None:
        output = await self._run(world, {"observed": "bob", "query": "Zebra"})
        assert "[bob->bob]" in output and "Bob works on Zebra" in output

    async def test_empty_observed_is_an_error(self, world: World) -> None:
        assert (await self._run(world, {"observed": [], "query": "x"})).startswith("ERROR: 'observed' must name")

    async def test_does_not_return_conclusions_from_other_pairs(self, world: World) -> None:
        output = await self._run(world, {"observed": ["alice"], "observer": "bob", "query": "foodie"})
        assert "[bob->alice]" in output
        assert "Bob thinks Alice is a foodie" in output
        assert "Alice loves sushi" not in output

    async def test_fans_out_over_several_people(self, world: World) -> None:
        output = await self._run(world, {"observed": ["alice", "bob"], "query": "Zebra sushi", "top_k": 20})
        assert "[alice->alice]" in output and "[bob->bob]" in output
        assert "Bob works on Zebra" in output and "Alice loves sushi" in output

    async def test_rejects_more_people_than_the_fanout_limit(self, world: World) -> None:
        output = await self._run(world, {"observed": [f"p{i}" for i in range(6)], "query": "x"})
        assert output.startswith("ERROR: at most 5 people per search")

    async def test_splits_top_k_across_people_with_a_floor(self, world: World) -> None:
        output = await self._run(world, {"observed": ["alice", "bob"], "query": "x", "top_k": 6})
        # 6 // 2 = 3 is under the floor of 5, so each person gets 5 (alice has 4 in her own pair, bob 1).
        assert "Found 4 observations" in output and "Found 1 observations" in output

    async def test_a_failure_on_every_person_is_reported_as_an_error(
        self, world: World, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("store down")

        monkeypatch.setattr("hindsight_api.people.dialectic.tools.search_memory", boom)
        output = await self._run(world, {"observed": ["alice", "bob"], "query": "x"})
        assert output.startswith("ERROR:") and "store down" in output

    async def test_a_partial_failure_still_returns_what_succeeded(
        self, world: World, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from hindsight_api.people.dialectic import tools as tools_module

        real = tools_module.search_memory

        async def flaky(ctx: Any, bank_id: str, observer: str, observed: str, *args: Any, **kwargs: Any) -> Any:
            if observed == "bob":
                raise RuntimeError("bob store down")
            return await real(ctx, bank_id, observer, observed, *args, **kwargs)

        monkeypatch.setattr(tools_module, "search_memory", flaky)
        output = await self._run(world, {"observed": ["alice", "bob"], "query": "Alice loves sushi"})
        assert "Alice loves sushi" in output
        assert "[bob->bob]\nERROR:" in output

    async def test_message_tools_are_bank_flat(self, world: World) -> None:
        run = await create_workspace_tool_executor(world.ctx, world.bank_id, agent_type="workspace_dialectic")
        # No observer: every session is visible, so Bob's s3 message is found without naming anyone.
        output = await run("grep_messages", {"text": "Zebra"})
        assert "My secret project is called Zebra" in output
        assert "source: https://chat.example/s3/1" in output

    async def test_pair_only_tools_are_unknown_to_the_bank_executor_unless_safe(self, world: World) -> None:
        run = await create_workspace_tool_executor(world.ctx, world.bank_id, agent_type="workspace_dialectic")
        assert (await run("get_recent_observations", {})).startswith("ERROR: Unknown tool")


class TestGetPeerCardByName:
    async def test_returns_the_peer_card(self, world: World) -> None:
        await set_peer_card(world.ctx, world.bank_id, ["IDENTITY: Alice"], "alice", "alice")
        run = await create_workspace_tool_executor(world.ctx, world.bank_id)
        output = await run("get_peer_card", {"observer": "alice", "observed": "alice"})
        assert output == "Peer card for alice:\n- IDENTITY: Alice"

    async def test_no_card(self, world: World) -> None:
        run = await create_workspace_tool_executor(world.ctx, world.bank_id)
        assert await run("get_peer_card", {"observer": "alice", "observed": "bob"}) == "No peer card available for bob"

    async def test_unknown_peer_is_answered_not_raised(self, world: World) -> None:
        run = await create_workspace_tool_executor(world.ctx, world.bank_id)
        output = await run("get_peer_card", {"observer": "nobody", "observed": "nobody"})
        assert output == "No peer named 'nobody' exists in this bank"

    async def test_session_allowlist_refuses(self, world: World) -> None:
        await set_peer_card(world.ctx, world.bank_id, ["IDENTITY: Alice"], "alice", "alice")
        run = await create_workspace_tool_executor(world.ctx, world.bank_id, session_allowlist=["s1"])
        output = await run("get_peer_card", {"observer": "alice", "observed": "alice"})
        assert "unavailable for session-scoped queries" in output

    @pytest.mark.parametrize("tool_input", [{}, {"observer": "alice"}, {"observed": "alice"}])
    async def test_missing_params_returns_error(self, world: World, tool_input: dict[str, str]) -> None:
        run = await create_workspace_tool_executor(world.ctx, world.bank_id)
        assert (await run("get_peer_card", tool_input)).startswith("ERROR: 'observer' and 'observed' are required")


class TestGetObservationContext:
    async def test_retrieves_messages_with_the_ones_around_them(self, world: World) -> None:
        evidence = EvidenceAccumulator()
        run = await tools_for(world, evidence=evidence)
        output = await run("get_observation_context", {"message_ids": [world.messages["m_berlin"]]})
        assert output.startswith("Retrieved 3 messages with context:")
        assert "Nice, I have never been" in output and "Cool, send photos" in output
        assert "I moved to Berlin last week" in output
        assert f"id: {world.messages['m_berlin']}" in output
        assert len(evidence.messages) == 3

    async def test_nonexistent_message_ids(self, world: World) -> None:
        run = await tools_for(world)
        assert (await run("get_observation_context", {"message_ids": ["nope"]})).startswith("No messages found for IDs")

    async def test_respects_the_observers_sessions(self, world: World) -> None:
        run = await tools_for(world, "alice", "alice")
        assert (await run("get_observation_context", {"message_ids": [world.messages["m_zebra"]]})).startswith(
            "No messages found"
        )

    async def test_respects_the_session_allowlist(self, world: World) -> None:
        run = await tools_for(world, allowlist=["s2"])
        assert (await run("get_observation_context", {"message_ids": [world.messages["m_berlin"]]})).startswith(
            "No messages found"
        )
        output = await run("get_observation_context", {"message_ids": [world.messages["m_moved"]]})
        assert "rescheduled to April 22" in output

    async def test_context_stays_inside_the_targets_session(self, world: World) -> None:
        run = await tools_for(world, "alice", "alice")
        output = await run("get_observation_context", {"message_ids": [world.messages["m_deadline"]]})
        # seq 1 of s2 pulls seq 2 of s2 and nothing from s1's seq 0 or 2.
        assert "April 25" in output and "rescheduled" in output
        assert "Nice, I have never been" not in output


class TestGetReasoningChain:
    async def test_deductive_observation_shows_premises_with_sources(self, world: World) -> None:
        run = await tools_for(world)
        output = await run("get_reasoning_chain", {"observation_id": world.conclusions["foodie"]})
        assert "(deductive)" in output
        assert f"[id:{world.conclusions['sushi']}] (explicit): Alice loves sushi" in output
        assert "source: https://chat.example/s1/1" in output
        assert "Premises (2)" in output

    async def test_the_id_prefix_is_tolerated(self, world: World) -> None:
        run = await tools_for(world)
        output = await run("get_reasoning_chain", {"observation_id": f"id:{world.conclusions['foodie']}"})
        assert "Alice is a foodie who relocated" in output

    async def test_conclusions_that_depend_on_an_observation(self, world: World) -> None:
        run = await tools_for(world)
        output = await run(
            "get_reasoning_chain", {"observation_id": world.conclusions["sushi"], "direction": "conclusions"}
        )
        assert "Derived Conclusions (1)" in output and "Alice is a foodie who relocated" in output
        assert "Premises" not in output

    async def test_explicit_observation_has_no_premises(self, world: World) -> None:
        run = await tools_for(world)
        output = await run(
            "get_reasoning_chain", {"observation_id": world.conclusions["sushi"], "direction": "premises"}
        )
        assert "N/A (explicit observations have no premises)" in output

    async def test_nonexistent_observation_returns_error(self, world: World) -> None:
        run = await tools_for(world)
        assert (await run("get_reasoning_chain", {"observation_id": "x" * 21})).startswith("ERROR: Observation")

    async def test_missing_observation_id_returns_error(self, world: World) -> None:
        run = await tools_for(world)
        assert (await run("get_reasoning_chain", {})).startswith("ERROR: 'observation_id' is required")

    async def test_invalid_direction_returns_error(self, world: World) -> None:
        run = await tools_for(world)
        output = await run(
            "get_reasoning_chain", {"observation_id": world.conclusions["sushi"], "direction": "sideways"}
        )
        assert output.startswith("ERROR: Invalid direction")

    async def test_a_session_allowlist_fails_closed(self, world: World) -> None:
        run = await tools_for(world, allowlist=["s1"])
        output = await run("get_reasoning_chain", {"observation_id": world.conclusions["foodie"]})
        assert output.startswith("Reasoning-chain traversal is unavailable")


class TestMessageTools:
    async def test_search_messages_ranks_by_relevance_and_cites(self, world: World) -> None:
        evidence = EvidenceAccumulator()
        run = await tools_for(world, evidence=evidence)
        output = await run("search_messages", {"query": "deadline rescheduled"})
        assert "The deadline was rescheduled to April 22" in output
        assert "source: https://chat.example/s2/2" in output
        assert f"said_at: {_at(20).isoformat()}" in output
        assert "April 25" in output  # the earlier message comes along as context
        assert world.messages["m_moved"] in evidence.messages

    async def test_search_messages_with_no_words_finds_nothing(self, world: World) -> None:
        run = await tools_for(world)
        assert (await run("search_messages", {"query": "?!"})).startswith("No messages found")

    async def test_search_messages_is_scoped_to_the_observers_sessions(self, world: World) -> None:
        run = await tools_for(world)
        assert (await run("search_messages", {"query": "Zebra"})).startswith("No messages found")
        bob = await tools_for(world, "bob", "bob")
        assert "Zebra" in await bob("search_messages", {"query": "Zebra"})

    async def test_grep_finds_exact_text_case_insensitively(self, world: World) -> None:
        run = await tools_for(world)
        output = await run("grep_messages", {"text": "BERLIN"})
        assert output.startswith("Found 1 messages containing 'BERLIN' in 1 conversation snippets")
        assert "I moved to Berlin last week" in output

    async def test_grep_escapes_wildcards(self, world: World) -> None:
        run = await tools_for(world)
        assert (await run("grep_messages", {"text": "%"})).startswith("No messages found containing")
        assert (await run("grep_messages", {"text": "_"})).startswith("No messages found containing")

    async def test_grep_requires_text(self, world: World) -> None:
        run = await tools_for(world)
        assert (await run("grep_messages", {})).startswith("ERROR: 'text' parameter is required")

    async def test_grep_merges_nearby_matches_into_one_snippet(self, world: World) -> None:
        run = await tools_for(world, "bob", "bob")
        output = await run("grep_messages", {"text": "e", "limit": 30, "context_window": 1})
        assert "conversation snippets" in output
        assert output.count("--- Snippet") <= 3

    async def test_date_range(self, world: World) -> None:
        run = await tools_for(world)
        output = await run(
            "get_messages_by_date_range",
            {"after_date": "2026-03-04", "before_date": "2026-03-15", "order": "asc"},
        )
        assert output.startswith("Found 3 messages (after 2026-03-04 and before 2026-03-15, oldest first)")
        assert output.index("moved to Berlin") < output.index("deadline is April 25")

    async def test_date_range_uses_when_it_was_said(self, world: World) -> None:
        # said_at is the 20th even though the row was stored today.
        run = await tools_for(world)
        output = await run("get_messages_by_date_range", {"after_date": "2026-03-19"})
        assert "rescheduled to April 22" in output and "Zebra" not in output

    async def test_invalid_date_is_reported(self, world: World) -> None:
        run = await tools_for(world)
        assert (await run("get_messages_by_date_range", {"after_date": "yesterday"})).startswith(
            "ERROR: Invalid after_date format"
        )

    async def test_temporal_search_applies_the_dates(self, world: World) -> None:
        run = await tools_for(world)
        output = await run("search_messages_temporal", {"query": "deadline", "after_date": "2026-03-15"})
        assert "Found 1 matching messages" in output and "rescheduled to April 22" in output
        assert "(after 2026-03-15)" in output

    async def test_the_session_allowlist_scopes_every_message_tool(self, world: World) -> None:
        run = await tools_for(world, allowlist=["s1"])
        assert (await run("search_messages", {"query": "deadline"})).startswith("No messages found")
        assert (await run("grep_messages", {"text": "deadline"})).startswith("No messages found")
        by_date = await run("get_messages_by_date_range", {})
        assert by_date.count("Berlin") == 1
        assert "April" not in by_date and "Zebra" not in by_date

    async def test_a_pinned_session_outside_the_allowlist_is_denied(self, world: World) -> None:
        run = await tools_for(world, session="s2", allowlist=["s1"])
        assert (await run("grep_messages", {"text": "deadline"})).startswith("No messages found")

    async def test_the_session_pins_message_tools(self, world: World) -> None:
        run = await tools_for(world, session="s1")
        assert (await run("grep_messages", {"text": "deadline"})).startswith("No messages found")
        assert "Berlin" in await run("grep_messages", {"text": "berlin"})

    async def test_long_output_is_cut_at_whole_snippets(self) -> None:
        def row(i: int) -> message_store.MessageRow:
            return message_store.MessageRow(
                id=i,
                public_id=f"{i:021d}",
                session_name=f"s{i}",
                peer_name="alice",
                content="word " * 600,
                seq_in_session=1,
                created_at=_at(1),
                said_at=_at(1),
                source_url=None,
                channel=None,
                token_count=600,
            )

        snippets = [([row(i)], [row(i)]) for i in range(1, 6)]
        text, truncated, original = message_store.format_message_snippets(snippets, "for query 'x'", max_chars=7000)
        assert truncated and original > 7000
        assert "snippets shown" in text and text.count("--- Snippet") < 5


class TestRecallFacts:
    def _recall(self, results: list[Any]) -> Any:
        async def recall_async(**kwargs: Any) -> Any:
            self.kwargs = kwargs
            return SimpleNamespace(results=results)

        return recall_async

    def _fact(self, **kwargs: Any) -> Any:
        base = {
            "id": "f1",
            "text": "Alice runs marathons",
            "fact_type": "world",
            "document_id": "d1",
            "metadata": {"source_url": "https://chat.example/s9/9"},
            "mentioned_at": "2026-02-01T00:00:00Z",
            "occurred_start": None,
        }
        return SimpleNamespace(**{**base, **kwargs})

    async def test_facts_come_with_their_source_and_time(self, world: World, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(world.ctx.engine, "recall_async", self._recall([self._fact()]))
        evidence = EvidenceAccumulator()
        run = await tools_for(world, evidence=evidence)
        output = await run("recall_facts", {"query": "Alice marathons"})
        assert "Found 1 retained facts" in output
        assert (
            "(world) Alice runs marathons [id: f1, source: https://chat.example/s9/9, said_at: 2026-02-01T00:00:00Z]"
            in output
        )
        assert self.kwargs["bank_id"] == world.bank_id and self.kwargs["query"] == "Alice marathons"
        assert evidence.facts["f1"].source_url == "https://chat.example/s9/9"

    async def test_a_said_at_in_the_metadata_wins(self, world: World, monkeypatch: pytest.MonkeyPatch) -> None:
        fact = self._fact(metadata={"said_at": "2026-01-01T09:00:00Z"})
        monkeypatch.setattr(world.ctx.engine, "recall_async", self._recall([fact]))
        run = await tools_for(world)
        assert "said_at: 2026-01-01T09:00:00Z" in await run("recall_facts", {"query": "x"})

    async def test_no_facts(self, world: World, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(world.ctx.engine, "recall_async", self._recall([]))
        run = await tools_for(world)
        assert await run("recall_facts", {"query": "x"}) == "No retained facts found for query 'x'"

    async def test_limit_caps_the_facts(self, world: World, monkeypatch: pytest.MonkeyPatch) -> None:
        facts = [self._fact(id=f"f{i}", text=f"fact {i}") for i in range(8)]
        monkeypatch.setattr(world.ctx.engine, "recall_async", self._recall(facts))
        run = await tools_for(world)
        assert "Found 3 retained facts" in await run("recall_facts", {"query": "x", "limit": 3})

    async def test_a_session_allowlist_fails_closed(self, world: World, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(world.ctx.engine, "recall_async", self._recall([self._fact()]))
        run = await tools_for(world, allowlist=["s1"])
        assert "unavailable for session-scoped queries" in await run("recall_facts", {"query": "x"})


class TestExecutor:
    async def test_unknown_tool(self, world: World) -> None:
        run = await tools_for(world)
        assert await run("delete_observations", {}) == "ERROR: Unknown tool 'delete_observations'"

    async def test_a_failing_handler_answers_the_model_instead_of_raising(
        self, world: World, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("database fell over")

        monkeypatch.setattr("hindsight_api.people.dialectic.tools.query_documents", boom)
        run = await tools_for(world)
        output = await run("search_memory", {"query": "x"})
        assert output == "ERROR: Tool search_memory failed: RuntimeError: database fell over"


class TestBankOverview:
    async def test_stats_and_active_people(self, world: World) -> None:
        stats = await get_workspace_stats(world.ctx, world.bank_id)
        assert (stats.peer_count, stats.session_count, stats.message_count) == (2, 3, 7)
        assert stats.oldest_message_at == _at(1) and stats.newest_message_at == _at(25)
        peers = await get_active_peers(world.ctx, world.bank_id, limit=5)
        assert [(p.name, p.message_count) for p in peers] == [("alice", 4), ("bob", 3)]

        text = format_workspace_stats(stats, peers, {"alice": ["IDENTITY: Alice"]})
        assert "People: 2" in text and "Date range: 2026-03-01 to 2026-03-25" in text
        assert "- alice (4 messages, last active 2026-03-20)\n    - IDENTITY: Alice" in text

    async def test_the_overview_honors_a_session_allowlist(self, world: World) -> None:
        stats = await get_workspace_stats(world.ctx, world.bank_id, session_names=["s3"])
        assert (stats.session_count, stats.message_count) == (1, 1)
        assert [p.name for p in await get_active_peers(world.ctx, world.bank_id, session_names=["s3"])] == ["bob"]
        empty = await get_workspace_stats(world.ctx, world.bank_id, session_names=[])
        assert empty.message_count == 0
