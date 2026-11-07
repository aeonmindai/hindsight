# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: tests/test_dialectic_prompts.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""Contracts for dialectic system prompts vs the tool loadouts they describe."""

import re

from hindsight_api.people.dialectic.core import DialecticAgent
from hindsight_api.people.dialectic.prompts import (
    PAIR_PROMPT_TOOLS,
    WORKSPACE_PROMPT_TOOLS,
    agent_system_prompt,
    workspace_agent_system_prompt,
)
from hindsight_api.people.dialectic.tools import (
    _TOOLS,
    ANSWER_READY_TOOL,
    ANSWER_READY_TOOL_NAME,
    DIALECTIC_TOOLS,
    DIALECTIC_TOOLS_MINIMAL,
    WORKSPACE_DIALECTIC_TOOLS,
    WORKSPACE_TOOLS_MINIMAL,
    tool_name,
)
from hindsight_api.people.dialectic.workspace import WorkspaceDialecticAgent
from tests.people.dialectic_support import bare_context

_ALL_TOOL_NAMES = {tool_name(spec) for spec in _TOOLS.values()} | {tool_name(ANSWER_READY_TOOL)}
# The model ends its research by calling this; every prompt lists it.
_FINISH = {ANSWER_READY_TOOL_NAME}


def _loadout_names(tools: list[dict[str, object]]) -> set[str]:
    return {tool_name(tool) for tool in tools}


def _mentioned_tools(text: str) -> set[str]:
    return {match for match in re.findall(r"`([a-z_][a-z0-9_]*)`", text) if match in _ALL_TOOL_NAMES}


def _tools_catalog(prompt: str) -> str:
    start = prompt.index("## TOOLS")
    rest = prompt[start:]
    next_heading = rest.find("\n## ", 1)
    return rest if next_heading == -1 else rest[:next_heading]


class TestPromptLoadouts:
    def test_pair_docs_match_dialectic_tools(self) -> None:
        assert _loadout_names(DIALECTIC_TOOLS) | _FINISH == PAIR_PROMPT_TOOLS

    def test_workspace_docs_match_workspace_tools(self) -> None:
        assert _loadout_names(WORKSPACE_DIALECTIC_TOOLS) | _FINISH == WORKSPACE_PROMPT_TOOLS

    def test_catalog_lists_only_offered_workspace_tools(self) -> None:
        offered = _loadout_names(WORKSPACE_TOOLS_MINIMAL) | _FINISH
        catalog = _tools_catalog(workspace_agent_system_prompt(offered))
        assert _mentioned_tools(catalog) == offered


class TestPairAgentPrompt:
    def test_teaches_world_without_workspace_sibling(self) -> None:
        prompt = agent_system_prompt("alice", "alice", None, None).lower()
        assert "workspace dialectic" not in prompt
        assert "bank dialectic" not in prompt
        assert "peer-level" not in prompt
        for term in ("hindsight", "peer", "session", "message", "conclusion"):
            assert term in prompt

    def test_states_single_turn_contract(self) -> None:
        """The caller cannot reply, so the prompt must not invite a next turn.

        Without this, answers close with offers ("If you want, I can…") and
        questions nobody can answer, and the contradiction section asked the
        model to end with "Which statement is correct?".
        """
        prompt = agent_system_prompt("alice", "alice", None, None)
        assert "no one can reply" in prompt
        assert "Current user request" in prompt
        assert "Which statement is correct?" not in prompt
        assert "Ask the user which statement is correct" not in prompt

    def test_does_not_offer_removed_write_tools(self) -> None:
        prompt = agent_system_prompt("alice", "alice", None, None)
        assert "create_observations_deductive" not in prompt
        assert "create_observations" not in prompt

    def test_agent_lists_selected_tools(self) -> None:
        agent = DialecticAgent(bare_context(), "bank", None, "alice", "alice", reasoning_level="minimal")
        offered = _loadout_names(DIALECTIC_TOOLS_MINIMAL) | _FINISH
        catalog = _tools_catalog(agent.messages[0]["content"])
        assert _mentioned_tools(catalog) == offered
        assert agent.messages[0]["content"] == agent_system_prompt(
            "alice", "alice", None, None, available_tools=offered
        )

    def test_a_session_allowlist_drops_the_tools_that_cannot_be_scoped(self) -> None:
        agent = DialecticAgent(
            bare_context(), "bank", None, "alice", "alice", reasoning_level="low", session_allowlist=["s1"]
        )
        catalog = _tools_catalog(agent.messages[0]["content"])
        assert "get_reasoning_chain" not in catalog
        assert "recall_facts" not in catalog
        assert "search_memory" in catalog


class TestSourceCitations:
    """The fork's addition: every answer cites where a claim was said."""

    def test_pair_prompt_asks_for_source_and_time(self) -> None:
        prompt = agent_system_prompt("alice", "alice", None, None)
        assert "CITE YOUR SOURCES" in prompt
        assert "[source: <url>, said_at: <time>]" in prompt

    def test_workspace_prompt_asks_for_source_and_time(self) -> None:
        prompt = workspace_agent_system_prompt(None)
        assert "CITE YOUR SOURCES" in prompt
        assert "[source: <url>, said_at: <time>]" in prompt


class TestWorkspaceAgentPrompt:
    def test_minimal_agent_matches_filtered_prompt(self) -> None:
        agent = WorkspaceDialecticAgent(bare_context(), "bank", reasoning_level="minimal")
        offered = _loadout_names(WORKSPACE_TOOLS_MINIMAL) | _FINISH
        prompt = agent.messages[0]["content"]
        assert prompt == workspace_agent_system_prompt(offered)
        catalog = _tools_catalog(prompt)
        assert _mentioned_tools(catalog) == offered
        assert "get_peer_card" not in catalog
        assert "get_reasoning_chain" not in catalog
