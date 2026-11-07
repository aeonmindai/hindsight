# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/dialectic/workspace.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""Bank-level dialectic agent (Honcho's workspace dialectic).

Answers queries across ALL people in a bank. Where DialecticAgent is
bound to a single (observer, observed) pair, this agent routes first --
bank stats, active people, and peer cards are prefetched for
orientation; message search is bank-flat and reveals which people
discussed a topic -- and then recalls through the same pair-scoped
conclusion machinery, supplying the pair as tool arguments.

Conclusion search deliberately stays pair-scoped: it matches the
(observer, observed) ownership and avoids retrieval dilution from a
bank-flat top-k. One call may name several people, which keeps a
cross-person question to one tool round without giving up that scoping.

Design carried over from plastic-labs/honcho#373 (Dan), re-grown on the
current DialecticAgent seams instead of a base-class extraction.
"""

from __future__ import annotations

import logging

from ..context import PeopleContext
from ..peer_card import get_peer_card
from ..schemas import ResolvedDialecticConfiguration
from . import prompts, settings
from .core import DialecticAgent, ToolExecutor
from .evidence import EvidenceAccumulator
from .settings import DialecticLevelSettings, ReasoningLevel
from .tools import (
    WORKSPACE_DIALECTIC_TOOLS,
    WORKSPACE_RECALL_TOOLS,
    WORKSPACE_TOOLS_MINIMAL,
    create_workspace_tool_executor,
    format_workspace_stats,
    get_active_peers,
    get_workspace_stats,
    tool_name,
)

logger = logging.getLogger(__name__)

# How many active people (with their self peer cards) to inject at prefetch.
# Routing-obvious queries should resolve without a discovery tool round --
# each avoided tool round is a full model turn.
_PREFETCH_ACTIVE_PEERS = 5


class WorkspaceDialecticAgent(DialecticAgent):
    """Dialectic agent scoped to a whole bank instead of a pair of people."""

    def __init__(
        self,
        ctx: PeopleContext,
        bank_id: str,
        session_name: str | None = None,
        reasoning_level: ReasoningLevel = "low",
        session_allowlist: list[str] | None = None,
        evidence: EvidenceAccumulator | None = None,
        instructions: ResolvedDialecticConfiguration | None = None,
    ) -> None:
        super().__init__(
            ctx,
            bank_id,
            session_name=session_name,
            observer="",
            observed="",
            reasoning_level=reasoning_level,
            session_allowlist=session_allowlist,
            evidence=evidence,
            instructions=instructions,
        )
        # Replace the pair-oriented system prompt with the bank one.
        self.messages[0] = {
            "role": "system",
            "content": prompts.workspace_agent_system_prompt(
                self._available_tool_names(),
                custom_instructions=self.instructions.custom_instructions,
            ),
        }

    # ------------------------------------------------------------------
    # DialecticAgent seams
    # ------------------------------------------------------------------

    def _layout_enabled(self) -> bool:
        # The pair layout doesn't describe a bank, and __init__ replaces the system prompt anyway.
        return False

    def _pair_brief_enabled(self) -> bool:
        # Counts are per pair; there is no pair here.
        return False

    async def _prefetch_relevant_observations(self, query: str) -> str | None:
        """Orientation + routing prefetch: stats, active people, peer cards.

        No semantic retrieval here -- a bank-flat conclusion top-k would be dominated by the most verbose
        people. Instead give the agent what it needs to ROUTE: who is here, who is active, and what is known
        about them at a glance.
        """
        _ = query
        # Like the base agent, prefetch failure degrades to no prefetched block rather than failing the whole
        # request (the caller in _prepare_messages does not guard this).
        try:
            stats = await get_workspace_stats(self.ctx, self.bank_id, session_names=self.session_allowlist)
            if stats.peer_count == 0:
                return None
            peers = await get_active_peers(
                self.ctx, self.bank_id, limit=_PREFETCH_ACTIVE_PEERS, session_names=self.session_allowlist
            )
            # `peers` is already allowlist-filtered, but a peer card is a single cross-session aggregate: an
            # in-scope person's card can still carry facts derived from sessions outside the scope. Drop cards
            # entirely under an allowlist -- same rule the get_peer_card tool enforces -- and route on stats alone.
            cards: dict[str, list[str]] = {}
            if self.session_allowlist is None:
                for peer in peers:
                    card = await get_peer_card(self.ctx, self.bank_id, peer.name, peer.name)
                    if card:
                        cards[peer.name] = card
        except Exception:
            logger.warning("Failed to prefetch bank overview for bank=%s", self.bank_id, exc_info=True)
            return None

        return format_workspace_stats(stats, peers, cards)

    def _prefetch_heading(self) -> str:
        return "Bank overview (prefetched)"

    def _prefetch_intro(self) -> str:
        return (
            "Bank scale, the most active people, and any known "
            "biographical facts about them. Use this to decide who is "
            "relevant, then read their representations with search_memory "
            "(pass every relevant person in one call), or search_messages to "
            "find people not listed here."
        )

    def _select_tools(self) -> list[dict]:
        tools = WORKSPACE_TOOLS_MINIMAL if self.reasoning_level == "minimal" else WORKSPACE_DIALECTIC_TOOLS
        # Mirror the base agent's allowlist rule, for the tools that cannot honor an allowlist: reasoning chains
        # traverse provenance across sessions, a peer card is one cross-session aggregate with no per-session
        # attribution, and retained facts carry no session. All fail closed in their handlers too; dropping them
        # here avoids paying the schema tokens and a wasted turn on a tool that can only refuse.
        if self.session_allowlist is not None:
            unscopable = {"get_reasoning_chain", "get_peer_card", "recall_facts"}
            tools = [t for t in tools if tool_name(t) not in unscopable]
        return tools

    def _tool_choice(self, level_settings: DialecticLevelSettings) -> str | None:
        """Require a tool call on the first turn.

        The pair agent prefetches the conclusions relevant to its query, so it can legitimately answer from
        context alone. This agent's prefetch is an orientation overview -- scale, active people, their cards --
        not the corpus. Left free to skip tools, the model treats that overview as everything it has: it answers
        when the overview happens to carry the fact, and otherwise writes out the search it should have run and
        asks the caller which option to take. Bank chat has no caller to answer, so that response is dead on
        arrival.

        Recall is the job, so make the first search mandatory and let the loop relax to "auto" once a recall tool
        has run (see `_force_tools_until`). Any other value a level configures is passed through untouched, so
        this only overrides the two cases that let the model opt out entirely.
        """
        choice = level_settings.TOOL_CHOICE
        if choice is None or choice == "auto":
            return "required"
        return choice

    def _force_tools_until(self) -> frozenset[str] | None:
        """Only a recall tool satisfies the forced turn.

        "required" alone is met by any tool, and an orientation call such as `get_peer_card` returns what the
        prefetch already supplied. Gating on recall keeps the model searching until it has read the corpus.
        """
        return WORKSPACE_RECALL_TOOLS

    def _max_tool_iterations(self, level_settings: DialecticLevelSettings) -> int:
        """Add the bank bonus to the level's limit.

        A pair query reads one representation; a bank query routes first and then recalls, often over several
        people. `minimal` is excluded because its single round is the level's whole contract.
        """
        if self.reasoning_level == "minimal":
            return level_settings.MAX_TOOL_ITERATIONS
        return level_settings.MAX_TOOL_ITERATIONS + settings.WORKSPACE_EXTRA_TOOL_ITERATIONS

    async def _create_tool_executor(self) -> ToolExecutor:
        return await create_workspace_tool_executor(
            self.ctx,
            self.bank_id,
            session_name=self.session_name,
            session_allowlist=self.session_allowlist,
            agent_type="workspace_dialectic",
            evidence=self.evidence,
        )
