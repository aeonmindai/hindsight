# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/dialectic/chat.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""
Chat functionality for the Dialectic API.

Provides the agentic_chat function for answering queries about people
using the DialecticAgent, and workspace_chat for answering across a bank.

Honcho's scope-peer refusal is not ported: scopes are not part of the fork.
"""

import logging
from collections.abc import AsyncIterator

from pydantic import BaseModel

from ..context import PeopleContext
from ..schemas import get_configuration
from .core import prepare_pair_agent
from .evidence import EvidenceAccumulator
from .settings import ReasoningLevel
from .workspace import WorkspaceDialecticAgent

logger = logging.getLogger(__name__)


async def agentic_chat(
    ctx: PeopleContext,
    bank_id: str,
    session_name: str | None,
    query: str,
    observer: str,
    observed: str,
    reasoning_level: ReasoningLevel = "low",
    session_allowlist: list[str] | None = None,
    response_model: type[BaseModel] | None = None,
    evidence: EvidenceAccumulator | None = None,
) -> str:
    """
    Answer a query about a person using the agentic dialectic.

    Args:
        ctx: People context
        bank_id: Bank identifier
        session_name: Session identifier (may be None for global queries)
        query: The question to answer about the person
        observer: The person making the query
        observed: The person being queried about
        reasoning_level: Level of reasoning to apply
        session_allowlist: Optional session allowlist restricting all recall
        response_model: Optional Pydantic model the answer must conform to.
            When set, the returned string is JSON matching the model's schema.
        evidence: Optional accumulator collecting what the agent read

    Returns:
        The synthesized answer string
    """
    agent = await prepare_pair_agent(
        ctx,
        bank_id,
        session_name,
        observer,
        observed,
        reasoning_level=reasoning_level,
        session_allowlist=session_allowlist,
        evidence=evidence,
    )
    return await agent.answer(query, response_model=response_model)


async def agentic_chat_stream(
    ctx: PeopleContext,
    bank_id: str,
    session_name: str | None,
    query: str,
    observer: str,
    observed: str,
    reasoning_level: ReasoningLevel = "low",
    session_allowlist: list[str] | None = None,
    response_model: type[BaseModel] | None = None,
    evidence: EvidenceAccumulator | None = None,
) -> AsyncIterator[str]:
    """
    Stream an answer to a query about a person using the agentic dialectic.

    Args:
        ctx: People context
        bank_id: Bank identifier
        session_name: Session identifier (may be None for global queries)
        query: The question to answer about the person
        observer: The person making the query
        observed: The person being queried about
        reasoning_level: Level of reasoning to apply
        session_allowlist: Optional session allowlist restricting all recall
        response_model: Optional Pydantic model the answer must conform to.
            When set, the streamed text accumulates to JSON matching the
            model's schema.
        evidence: Optional accumulator collecting what the agent read

    Yields:
        Chunks of the response text, as described in
        `DialecticAgent.answer_stream`
    """
    agent = await prepare_pair_agent(
        ctx,
        bank_id,
        session_name,
        observer,
        observed,
        reasoning_level=reasoning_level,
        session_allowlist=session_allowlist,
        evidence=evidence,
    )
    async for chunk in agent.answer_stream(query, response_model=response_model):
        yield chunk


def _workspace_agent(
    ctx: PeopleContext,
    bank_id: str,
    session_name: str | None,
    reasoning_level: ReasoningLevel,
    evidence: EvidenceAccumulator | None,
    session_allowlist: list[str] | None,
) -> WorkspaceDialecticAgent:
    configuration = get_configuration(None, None, None)
    return WorkspaceDialecticAgent(
        ctx,
        bank_id,
        session_name=session_name,
        reasoning_level=reasoning_level,
        session_allowlist=session_allowlist,
        evidence=evidence,
        instructions=configuration.dialectic,
    )


async def workspace_chat(
    ctx: PeopleContext,
    bank_id: str,
    session_name: str | None,
    query: str,
    reasoning_level: ReasoningLevel = "low",
    response_model: type[BaseModel] | None = None,
    evidence: EvidenceAccumulator | None = None,
    session_allowlist: list[str] | None = None,
) -> str:
    """Answer a query across all people in a bank."""
    agent = _workspace_agent(ctx, bank_id, session_name, reasoning_level, evidence, session_allowlist)
    return await agent.answer(query, response_model=response_model)


async def workspace_chat_stream(
    ctx: PeopleContext,
    bank_id: str,
    session_name: str | None,
    query: str,
    reasoning_level: ReasoningLevel = "low",
    response_model: type[BaseModel] | None = None,
    evidence: EvidenceAccumulator | None = None,
    session_allowlist: list[str] | None = None,
) -> AsyncIterator[str]:
    """Streaming variant of :func:`workspace_chat`."""
    agent = _workspace_agent(ctx, bank_id, session_name, reasoning_level, evidence, session_allowlist)
    async for chunk in agent.answer_stream(query, response_model=response_model):
        yield chunk
