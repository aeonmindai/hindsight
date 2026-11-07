# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/utils/agent_tools.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""
The read-only tools the dialectic agent calls, and the executor that runs them.

Ported from the dialectic half of Honcho's ``utils/agent_tools.py``: the tool definitions
(``DIALECTIC_TOOLS``, ``WORKSPACE_DIALECTIC_TOOLS``...), the handlers for memory search, message
search and reasoning chains, and ``create_tool_executor``. The write tools (``create_observations_*``,
``update_peer_card``, ``delete_observations``) belong to the dreamer and are not here.

Differences from Honcho:

* Tools are OpenAI-style function tools, which is what the engine's ``call_with_tools`` takes.
* Every conclusion, message and fact a tool returns is printed with its source (``source:`` URL and
  ``said_at`` time) so the answer can cite it.
* ``recall_facts`` is new: it searches the bank's retained memories through the engine's recall.
* ``answer_ready`` is new: the engine's tool turns always call one function, so the model ends its
  research by calling it (see ``core.py``).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, cast

from ..conclusions import Conclusion, get_child_observations, get_documents_by_ids, query_documents
from ..context import PeopleContext
from ..exceptions import ResourceNotFoundException
from ..peer_card import get_peer_card
from ..representation_utils import ALLOWLIST_SAFE_LEVELS, allowlist_safe_levels
from . import messages as message_store
from .evidence import EvidenceAccumulator, EvidenceFact

logger = logging.getLogger(__name__)

ANSWER_READY_TOOL_NAME = "answer_ready"

# Most peers one workspace search may name, and the floor of each peer's share of ``top_k``.
_WORKSPACE_SEARCH_FANOUT_LIMIT = 5
_WORKSPACE_SEARCH_MIN_TOP_K = 5
# Largest slice of the engine's recall a ``recall_facts`` call reads, in tokens.
_RECALL_FACTS_MAX_TOKENS = 2048
_RECALL_FACTS_MAX_RESULTS = 15


def _safe_int(value: Any, default: int) -> int:
    """Coerce a tool input value to int, returning default on failure.

    LLMs sometimes pass non-numeric strings (e.g. 'Infinity') for integer
    parameters which would crash ``min()`` comparisons.
    """
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _bounded_int(value: Any, default: int, *, lo: int = 1, hi: int) -> int:
    """Coerce a tool int into ``[lo, hi]``, falling back to ``default`` on bad input."""
    return max(lo, min(_safe_int(value, default), hi))


def _describe_tool_error(error: Exception) -> str:
    """Bounded description of `error`, safe to hand back to the model."""
    message = str(error).strip()
    max_chars = message_store.MAX_TOOL_OUTPUT_CHARS
    if len(message) > max_chars:
        message = message[:max_chars] + f"… [truncated {len(message) - max_chars:,} chars]"
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


def _function_tool(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def tool_name(tool: dict[str, Any]) -> str:
    """The name of a function tool."""
    return str(tool["function"]["name"])


_TOOLS: dict[str, dict[str, Any]] = {
    "search_memory": _function_tool(
        "search_memory",
        "Search conclusions about the person in memory using semantic similarity. Each result shows its id, "
        "level, and where and when it was said.",
        {
            "query": {"type": "string", "description": "Search query text"},
            "top_k": {
                "type": "integer",
                "description": "(Optional) number of results to return (default: 20, max: 40)",
            },
        },
        ["query"],
    ),
    "get_observation_context": _function_tool(
        "get_observation_context",
        "Retrieve messages for given message IDs along with surrounding context. Takes message IDs (from a "
        "conclusion's message_ids) and retrieves those messages plus the messages immediately before and after "
        "each one to provide conversation context.",
        {
            "message_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of message IDs to retrieve (get these from a conclusion's message_ids)",
            },
        },
        ["message_ids"],
    ),
    "search_messages": _function_tool(
        "search_messages",
        "Search for messages by ranked text relevance (any word of the query may match) and retrieve "
        "conversation snippets. Returns matching messages with surrounding context (2 messages before and "
        "after). Nearby matches within the same session are merged into a single snippet to avoid repetition.",
        {
            "query": {"type": "string", "description": "Search query text to find relevant messages"},
            "limit": {
                "type": "integer",
                "description": "Maximum number of matching messages to return (default: 10, max: 20)",
            },
        },
        ["query"],
    ),
    "grep_messages": _function_tool(
        "grep_messages",
        "Search for messages containing specific text (case-insensitive). Unlike ranked search, this finds EXACT "
        "text matches. Use for finding specific names, dates, phrases, or keywords mentioned in conversations. "
        "Returns messages with surrounding context.",
        {
            "text": {"type": "string", "description": "Text to search for (case-insensitive substring match)"},
            "limit": {"type": "integer", "description": "Maximum messages to return (default: 10, max: 30)"},
            "context_window": {
                "type": "integer",
                "description": "Number of messages before/after each match to include (default: 2)",
            },
        },
        ["text"],
    ),
    "get_messages_by_date_range": _function_tool(
        "get_messages_by_date_range",
        "Get messages from a specific date range. Use this to find what was discussed during a particular time "
        "period, or to compare information before vs after a date. Essential for knowledge update questions.",
        {
            "after_date": {
                "type": "string",
                "description": "Start date (ISO format, e.g., '2024-01-15'). Returns messages after this date.",
            },
            "before_date": {
                "type": "string",
                "description": "End date (ISO format). Returns messages before this date.",
            },
            "limit": {"type": "integer", "description": "Maximum messages to return (default: 20, max: 50)"},
            "order": {
                "type": "string",
                "enum": ["asc", "desc"],
                "description": "Sort order: 'asc' for oldest first, 'desc' for newest first (default: desc)",
            },
        },
        [],
    ),
    "search_messages_temporal": _function_tool(
        "search_messages_temporal",
        "Ranked text search for messages with optional date filtering. Combines text relevance with time "
        "constraints. Use after_date to find recent mentions of a topic, or before_date to find what was said about "
        "something before a certain point. Best for knowledge update questions where you need to find the MOST "
        "RECENT discussion of a topic.",
        {
            "query": {"type": "string", "description": "Search query"},
            "after_date": {
                "type": "string",
                "description": "Only return messages after this date (ISO format, e.g., '2024-01-15')",
            },
            "before_date": {"type": "string", "description": "Only return messages before this date (ISO format)"},
            "limit": {"type": "integer", "description": "Maximum messages to return (default: 10, max: 20)"},
            "context_window": {"type": "integer", "description": "Messages before/after each match (default: 2)"},
        },
        ["query"],
    ),
    "get_reasoning_chain": _function_tool(
        "get_reasoning_chain",
        "Get the reasoning chain for an observation - traverse the tree to find premises (for deductive) or sources "
        "(for inductive), and/or find conclusions derived from this observation. Use this to understand how an "
        "observation was derived or what conclusions depend on it.",
        {
            "observation_id": {
                "type": "string",
                "description": "The document ID of the observation to get the reasoning chain for",
            },
            "direction": {
                "type": "string",
                "enum": ["premises", "conclusions", "both"],
                "description": (
                    "'premises' to get what this observation is based on, 'conclusions' to get what depends on it, "
                    "'both' for full context"
                ),
            },
        },
        ["observation_id"],
    ),
    "recall_facts": _function_tool(
        "recall_facts",
        "Search the bank's retained memories (facts kept from everything that was recorded) with the engine's recall. "
        "Name the person in the query. Each fact shows where and when it was said, when that is known.",
        {
            "query": {"type": "string", "description": "What to look for, naming the person"},
            "limit": {
                "type": "integer",
                "description": f"Maximum facts to return (default: 10, max: {_RECALL_FACTS_MAX_RESULTS})",
            },
        },
        ["query"],
    ),
    "search_memory_workspace": _function_tool(
        "search_memory",
        "Search person conclusions in memory using semantic similarity. Name every person the question covers in "
        "ONE call: 'observed' takes a list, and each person's own global representation is searched (this is where "
        "most information lives). Only set 'observer' when you want one person's specific understanding of another.",
        {
            "observed": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Handles of the people to search, one entry per person",
            },
            "observer": {
                "type": "string",
                "description": "(Optional) handle of the observer; defaults to each observed person itself",
            },
            "query": {"type": "string", "description": "Search query text"},
            "top_k": {
                "type": "integer",
                "description": "(Optional) total results across all people named (default: 20, max: 40)",
            },
        },
        ["observed", "query"],
    ),
    "get_peer_card_by_name": _function_tool(
        "get_peer_card",
        "Get the peer card for a specific person relationship. Specify the observer and observed handles.",
        {
            "observer": {"type": "string", "description": "Handle of the observer"},
            "observed": {"type": "string", "description": "Handle of the observed person"},
        },
        ["observer", "observed"],
    ),
}

ANSWER_READY_TOOL: dict[str, Any] = _function_tool(
    ANSWER_READY_TOOL_NAME,
    "Call this when you have gathered what you need. Your answer is written on the next turn.",
    {},
    [],
)

# Tools for the dialectic agent (analysis)
DIALECTIC_TOOLS: list[dict[str, Any]] = [
    _TOOLS["search_memory"],
    _TOOLS["search_messages"],
    _TOOLS["get_observation_context"],
    _TOOLS["grep_messages"],  # For exact text search (names, dates, keywords)
    _TOOLS["get_messages_by_date_range"],  # For temporal/date-based queries
    _TOOLS["search_messages_temporal"],  # Ranked text search + date filtering
    _TOOLS["get_reasoning_chain"],  # Traverse reasoning trees
    _TOOLS["recall_facts"],  # The bank's retained memories
]

# Minimal tools for dialectic agent at "minimal" reasoning level
# Reduces cost by limiting tool definitions in context
DIALECTIC_TOOLS_MINIMAL: list[dict[str, Any]] = [
    _TOOLS["search_memory"],
    _TOOLS["search_messages"],
]

# Tools for the bank-level dialectic agent. Conclusion search stays pair-scoped (observer/observed are TOOL
# ARGUMENTS the agent must supply after routing). Message tools are bank-flat and double as the routing signal
# (results carry the speaker).
WORKSPACE_DIALECTIC_TOOLS: list[dict[str, Any]] = [
    _TOOLS["search_memory_workspace"],
    _TOOLS["search_messages"],
    _TOOLS["get_observation_context"],
    _TOOLS["grep_messages"],
    _TOOLS["get_peer_card_by_name"],
    _TOOLS["get_messages_by_date_range"],
    _TOOLS["search_messages_temporal"],
    _TOOLS["get_reasoning_chain"],
    _TOOLS["recall_facts"],
]

# Reduced bank loadout for reasoning_level="minimal", mirroring DIALECTIC_TOOLS_MINIMAL.
WORKSPACE_TOOLS_MINIMAL: list[dict[str, Any]] = [
    _TOOLS["search_memory_workspace"],
    _TOOLS["search_messages"],
]

# Bank tools that read the corpus. The bank agent's forced first turn is satisfied only by one of these, not by an
# orientation tool.
WORKSPACE_RECALL_TOOLS: frozenset[str] = frozenset(
    {
        "search_memory",
        "search_messages",
        "grep_messages",
        "get_observation_context",
        "get_messages_by_date_range",
        "search_messages_temporal",
        "recall_facts",
    }
)


# ---------------------------------------------------------------------------
# Formatting conclusions for the model
# ---------------------------------------------------------------------------


def conclusion_source_line(conclusion: Conclusion) -> str:
    """Where and when a conclusion was said, as the bracket printed after it."""
    parts = []
    if conclusion.source_url:
        parts.append(f"source: {conclusion.source_url}")
    when = conclusion.said_at or conclusion.created_at
    parts.append(f"said_at: {when.isoformat()}")
    if conclusion.source_message_ids:
        parts.append("message_ids: " + ", ".join(conclusion.source_message_ids))
    return "[" + ", ".join(parts) + "]"


def format_conclusions(conclusions: Sequence[Conclusion]) -> str:
    """Conclusions grouped by level, one line each: id, text, then where it was said."""
    sections: list[str] = []
    for level in ("explicit", "deductive", "inductive", "contradiction"):
        at_level = [c for c in conclusions if (c.level or "explicit") == level]
        if not at_level:
            continue
        lines = [f"[id:{c.id}] {c.content} {conclusion_source_line(c)}" for c in at_level]
        sections.append(f"{level.upper()}:\n" + "\n".join(f"- {line}" for line in lines))
    return "\n\n".join(sections)


async def search_memory(
    ctx: PeopleContext,
    bank_id: str,
    observer: str,
    observed: str,
    query: str,
    limit: int,
    levels: list[str] | None = None,
    embedding: list[float] | None = None,
    session_allowlist: list[str] | None = None,
    documents_out: list[Conclusion] | None = None,
) -> list[Conclusion]:
    """
    Search for conclusions in memory using semantic similarity.

    Args:
        ctx: People context
        bank_id: Bank identifier
        observer: The person who holds the conclusions
        observed: The person the conclusions are about
        query: Search query text
        limit: Maximum number of results
        levels: Optional list of conclusion levels to filter by (e.g., ["explicit"])
        embedding: Optional pre-computed embedding to avoid redundant embedder calls
        session_allowlist: Optional session allowlist; an empty list fails closed
        documents_out: Optional list the matched conclusions are appended to

    Returns:
        The matching conclusions, best first
    """
    # Fail closed on an empty allowlist.
    if session_allowlist is not None and not session_allowlist:
        return []

    if session_allowlist is not None:
        levels = allowlist_safe_levels(levels)
        if not levels:
            return []

    filters: dict[str, Any] = {}
    if levels:
        filters["level"] = {"in": levels}
    if session_allowlist is not None:
        filters["session_name"] = {"in": session_allowlist}

    documents = await query_documents(
        ctx,
        bank_id,
        observer=observer,
        observed=observed,
        query=query,
        top_k=limit,
        filters=filters or None,
        embedding=embedding,
    )

    if documents_out is not None:
        documents_out.extend(documents)
    return list(documents)


# ---------------------------------------------------------------------------
# Tool context and handlers
# ---------------------------------------------------------------------------


@dataclass
class ToolContext:
    """Context object passed to tool handlers."""

    ctx: PeopleContext
    bank_id: str
    observer: str
    observed: str
    session_name: str | None
    # Optional session allowlist (dialectic filters). When set, message and conclusion recall is restricted to
    # these sessions (intersected with observer membership); empty list fails closed.
    session_allowlist: list[str] | None = None
    agent_type: str | None = None  # "dialectic" or "workspace_dialectic"
    # Set only when the caller asked for evidence. Read handlers append the rows they loaded;
    # `dataclasses.replace` copies of this context share the same accumulator.
    evidence: EvidenceAccumulator | None = None


Handler = Callable[[ToolContext, dict[str, Any]], Awaitable[str]]


def _record_conclusion_evidence(tctx: ToolContext, documents: Sequence[Conclusion]) -> None:
    if tctx.evidence is not None:
        tctx.evidence.add_documents(documents)


def _record_message_evidence(tctx: ToolContext, rows: Sequence[message_store.MessageRow]) -> None:
    if tctx.evidence is not None:
        tctx.evidence.add_messages(rows)


def _record_snippet_evidence(tctx: ToolContext, snippets: Sequence[message_store.Snippet]) -> None:
    """Record every message a snippet search surfaced; both halves of a snippet count as read."""
    for matches, context in snippets:
        _record_message_evidence(tctx, matches)
        _record_message_evidence(tctx, context)


def _normalize_observation_id(obs_id: str) -> str:
    """Strip the display-format ``id:`` prefix from a model-supplied observation ID.

    Observations are presented to agents as ``[id:xxx]``, and despite tool-schema instructions to pass the bare ID,
    models sometimes copy the prefix verbatim. Since ids are nanoids whose alphabet includes ``-`` and ``_``, only
    the ``id:`` prefix and surrounding whitespace are stripped.
    """
    obs_id = obs_id.strip()
    if obs_id.lower().startswith("id:"):
        obs_id = obs_id[3:]
    return obs_id.strip()


def _parse_date(date_str: str | None, param_name: str) -> datetime | None | str:
    """Parse a date string, returning datetime, None, or error string."""
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except ValueError:
        return f"ERROR: Invalid {param_name} format '{date_str}'. Use ISO format (e.g., '2024-01-15')"


async def _handle_search_memory(tctx: ToolContext, tool_input: dict[str, Any]) -> str:
    """Handle search_memory tool."""
    top_k = _bounded_int(tool_input.get("top_k"), 20, hi=40)
    query = tool_input.get("query") or ""
    if not query:
        return "ERROR: 'query' parameter is required"
    try:
        query_embedding = await tctx.ctx.embed_query(query)
    except ValueError as e:
        # Provider/config failure, not an oversized query. Keep returning a string so the tool loop can continue.
        return f"ERROR: Embedding the query failed: {e}"

    # Restrict conclusion recall to the session allowlist when one is set. Empty allowlist fails closed,
    # and only levels with a trustworthy session stamp are served.
    documents = await search_memory(
        tctx.ctx,
        tctx.bank_id,
        tctx.observer,
        tctx.observed,
        query,
        top_k,
        levels=list(ALLOWLIST_SAFE_LEVELS) if tctx.session_allowlist is not None else None,
        embedding=query_embedding,
        session_allowlist=tctx.session_allowlist,
    )
    _record_conclusion_evidence(tctx, documents)
    if not documents:
        # Empty-memory fallback: if the memory is *empty*, we're quite early in a bank/person/session. To give good
        # answers at this stage, and make sure the model doesn't short-circuit and think there's nothing here, we
        # automatically search the message history for relevant information.
        if tctx.agent_type in ("dialectic", "workspace_dialectic"):
            limit = _bounded_int(tool_input.get("top_k"), 20, hi=20)
            snippets = await message_store.search_messages(
                tctx.ctx,
                tctx.bank_id,
                tctx.session_name,
                query,
                limit=limit,
                context_window=0,
                observer=tctx.observer or None,
                session_allowlist=tctx.session_allowlist,
            )
            _record_snippet_evidence(tctx, snippets)
            if snippets:
                message_output, _, _ = message_store.format_message_snippets(snippets, f"for query '{query}'")
                return f"No observations yet. Message search results:\n\n{message_output}"
            return (
                f"No observations found for query '{query}', and no messages found in "
                "history. Try a different phrasing or use grep_messages for exact text."
            )
        return f"No observations found for query '{query}'"
    return f"Found {len(documents)} observations for query '{query}':\n\n{format_conclusions(documents)}"


async def _handle_get_observation_context(tctx: ToolContext, tool_input: dict[str, Any]) -> str:
    """Handle get_observation_context tool."""
    message_ids = tool_input.get("message_ids") or []
    rows = await message_store.get_observation_context(
        tctx.ctx,
        tctx.bank_id,
        tctx.session_name,
        message_ids,
        observer=tctx.observer or None,
        session_allowlist=tctx.session_allowlist,
    )
    _record_message_evidence(tctx, rows)
    if not rows:
        return f"No messages found for IDs {message_ids}"
    messages_text = "\n".join(message_store.format_message(row) for row in rows)
    output = f"Retrieved {len(rows)} messages with context:\n{messages_text}"
    return message_store.truncate_output(output)[0]


async def _handle_search_messages(tctx: ToolContext, tool_input: dict[str, Any]) -> str:
    """Handle search_messages tool."""
    query = tool_input.get("query") or ""
    if not query:
        return "ERROR: 'query' parameter is required"
    limit = _bounded_int(tool_input.get("limit"), 10, hi=20)
    snippets = await message_store.search_messages(
        tctx.ctx,
        tctx.bank_id,
        tctx.session_name,
        query,
        limit=limit,
        context_window=2,
        observer=tctx.observer or None,
        session_allowlist=tctx.session_allowlist,
    )
    _record_snippet_evidence(tctx, snippets)
    if not snippets:
        return f"No messages found for query '{query}'"
    return message_store.format_message_snippets(snippets, f"for query '{query}'")[0]


async def _handle_grep_messages(tctx: ToolContext, tool_input: dict[str, Any]) -> str:
    """Handle grep_messages tool."""
    text = tool_input.get("text") or ""
    if not text:
        return "ERROR: 'text' parameter is required"
    limit = _bounded_int(tool_input.get("limit"), 10, hi=30)
    context_window = _bounded_int(tool_input.get("context_window"), 2, lo=0, hi=2)
    snippets = await message_store.grep_messages(
        tctx.ctx,
        tctx.bank_id,
        tctx.session_name,
        text,
        limit=limit,
        context_window=context_window,
        observer=tctx.observer or None,
        session_allowlist=tctx.session_allowlist,
    )
    _record_snippet_evidence(tctx, snippets)
    if not snippets:
        return f"No messages found containing '{text}'"
    total_matches = sum(len(matches) for matches, _ in snippets)
    formatted, _, _ = message_store.format_message_snippets(snippets, f"containing '{text}'", pattern=text)
    # Name the exact-match count the way Honcho's grep output does.
    return f"Found {total_matches} messages containing '{text}' in {len(snippets)} conversation snippets.\n" + formatted


async def _handle_get_messages_by_date_range(tctx: ToolContext, tool_input: dict[str, Any]) -> str:
    """Handle get_messages_by_date_range tool."""
    after_date_str = tool_input.get("after_date")
    before_date_str = tool_input.get("before_date")
    limit = _bounded_int(tool_input.get("limit"), 20, hi=20)
    order = tool_input.get("order") or "desc"

    after_date = _parse_date(after_date_str, "after_date")
    if isinstance(after_date, str):
        return after_date  # Error message
    before_date = _parse_date(before_date_str, "before_date")
    if isinstance(before_date, str):
        return before_date  # Error message

    rows = await message_store.get_messages_by_date_range(
        tctx.ctx,
        tctx.bank_id,
        tctx.session_name,
        after_date=after_date,
        before_date=before_date,
        limit=limit,
        order=order,
        observer=tctx.observer or None,
        session_allowlist=tctx.session_allowlist,
    )
    _record_message_evidence(tctx, rows)

    date_range: list[str] = []
    if after_date_str:
        date_range.append(f"after {after_date_str}")
    if before_date_str:
        date_range.append(f"before {before_date_str}")

    if not rows:
        range_desc = " and ".join(date_range) if date_range else "specified range"
        return f"No messages found {range_desc}"

    range_desc = " and ".join(date_range) if date_range else "all time"
    order_desc = "oldest first" if order == "asc" else "newest first"
    messages_text = "\n".join(message_store.format_message(row) for row in rows)
    output = f"Found {len(rows)} messages ({range_desc}, {order_desc}):\n\n{messages_text}"
    return message_store.truncate_output(output)[0]


async def _handle_search_messages_temporal(tctx: ToolContext, tool_input: dict[str, Any]) -> str:
    """Handle search_messages_temporal tool."""
    query = tool_input.get("query") or ""
    if not query:
        return "ERROR: 'query' parameter is required"

    after_date_str = tool_input.get("after_date")
    before_date_str = tool_input.get("before_date")
    limit = _bounded_int(tool_input.get("limit"), 10, hi=10)
    context_window = _bounded_int(tool_input.get("context_window"), 2, lo=0, hi=2)

    after_date = _parse_date(after_date_str, "after_date")
    if isinstance(after_date, str):
        return after_date
    before_date = _parse_date(before_date_str, "before_date")
    if isinstance(before_date, str):
        return before_date

    snippets = await message_store.search_messages_temporal(
        tctx.ctx,
        tctx.bank_id,
        tctx.session_name,
        query,
        after_date=after_date,
        before_date=before_date,
        limit=limit,
        context_window=context_window,
        observer=tctx.observer or None,
        session_allowlist=tctx.session_allowlist,
    )
    _record_snippet_evidence(tctx, snippets)
    date_filter: list[str] = []
    if after_date_str:
        date_filter.append(f"after {after_date_str}")
    if before_date_str:
        date_filter.append(f"before {before_date_str}")
    filter_desc = f" ({' and '.join(date_filter)})" if date_filter else ""

    if not snippets:
        return f"No messages found for query '{query}'{filter_desc}"
    return message_store.format_message_snippets(snippets, f"for query '{query}'{filter_desc}")[0]


def _chain_lines(documents: Sequence[Conclusion]) -> str:
    return "\n".join(
        f" - [id:{d.id}] ({d.level or 'explicit'}): {d.content} {conclusion_source_line(d)}" for d in documents
    )


async def _handle_get_reasoning_chain(tctx: ToolContext, tool_input: dict[str, Any]) -> str:
    """Handle get_reasoning_chain tool."""
    # Reasoning chains traverse provenance across sessions by design, so a session allowlist cannot be enforced on
    # the traversal without exposing out-of-scope premises/conclusions. Fail closed rather than leak.
    if tctx.session_allowlist is not None:
        return "Reasoning-chain traversal is unavailable for session-scoped queries. Use search_memory and message tools instead."
    observation_id = tool_input.get("observation_id")
    if not observation_id:
        return "ERROR: 'observation_id' is required"
    observation_id = _normalize_observation_id(observation_id)

    direction = tool_input.get("direction") or "both"
    if direction not in ("premises", "conclusions", "both"):
        return f"ERROR: Invalid direction '{direction}'. Must be 'premises', 'conclusions', or 'both'"

    docs = await get_documents_by_ids(tctx.ctx, tctx.bank_id, document_ids=[observation_id])
    if not docs or not docs[0]:
        return f"ERROR: Observation '{observation_id}' not found"

    doc: Conclusion = docs[0]
    _record_conclusion_evidence(tctx, [doc])

    output_parts: list[str] = []
    level = doc.level or "explicit"
    output_parts.append(f"**Observation [id:{doc.id}] ({level}):**\n{doc.content} {conclusion_source_line(doc)}")

    if direction in ("premises", "both"):
        if level in ("deductive", "inductive") and doc.source_ids:
            noun, heading = ("premise", "Premises") if level == "deductive" else ("source", "Sources")
            premises = await get_documents_by_ids(tctx.ctx, tctx.bank_id, document_ids=list(doc.source_ids))
            _record_conclusion_evidence(tctx, premises)
            if premises:
                output_parts.append(f"\n**{heading} ({len(premises)}):**\n" + _chain_lines(premises))
            else:
                output_parts.append(
                    f"\n**{heading}:** Referenced {len(doc.source_ids)} {noun} IDs but none found in database"
                )
        elif level == "explicit":
            output_parts.append("\n**Premises/Sources:** N/A (explicit observations have no premises)")
        else:
            output_parts.append("\n**Premises/Sources:** None recorded")

    if direction in ("conclusions", "both"):
        children = await get_child_observations(
            tctx.ctx, tctx.bank_id, observation_id, observer=tctx.observer, observed=tctx.observed
        )
        _record_conclusion_evidence(tctx, children)
        if children:
            output_parts.append(f"\n**Derived Conclusions ({len(children)}):**\n" + _chain_lines(children))
        else:
            output_parts.append("\n**Derived Conclusions:** None found")

    return "\n".join(output_parts)


def _fact_source(metadata: dict[str, Any] | None, key_options: tuple[str, ...]) -> str | None:
    for key in key_options:
        value = (metadata or {}).get(key)
        if value:
            return str(value)
    return None


async def _handle_recall_facts(tctx: ToolContext, tool_input: dict[str, Any]) -> str:
    """Handle recall_facts tool: the engine's recall over the bank's retained memories."""
    query = tool_input.get("query") or ""
    if not query:
        return "ERROR: 'query' parameter is required"
    # Retained facts carry no per-session scope, so they cannot be restricted to an allowlist: fail closed.
    if tctx.session_allowlist is not None:
        return "Retained facts are unavailable for session-scoped queries. Use search_memory and message tools instead."
    limit = _bounded_int(tool_input.get("limit"), 10, hi=_RECALL_FACTS_MAX_RESULTS)
    result = await tctx.ctx.engine.recall_async(
        bank_id=tctx.bank_id,
        query=query,
        max_tokens=_RECALL_FACTS_MAX_TOKENS,
        request_context=tctx.ctx.request_context,
    )
    facts = list(result.results)[:limit]
    if not facts:
        return f"No retained facts found for query '{query}'"
    recorded: list[EvidenceFact] = []
    lines: list[str] = []
    for fact in facts:
        source_url = _fact_source(fact.metadata, ("source_url", "url"))
        said_at = _fact_source(fact.metadata, ("said_at",)) or fact.mentioned_at or fact.occurred_start
        recorded.append(
            EvidenceFact(
                id=fact.id,
                text=fact.text,
                fact_type=fact.fact_type,
                document_id=fact.document_id,
                source_url=source_url,
                said_at=said_at,
            )
        )
        parts = [f"id: {fact.id}"]
        if source_url:
            parts.append(f"source: {source_url}")
        if said_at:
            parts.append(f"said_at: {said_at}")
        lines.append(f"- ({fact.fact_type}) {fact.text} [{', '.join(parts)}]")
    if tctx.evidence is not None:
        tctx.evidence.add_facts(recorded)
    output = f"Found {len(facts)} retained facts for query '{query}':\n\n" + "\n".join(lines)
    return message_store.truncate_output(output)[0]


async def _handle_get_peer_card_by_name(tctx: ToolContext, tool_input: dict[str, Any]) -> str:
    """get_peer_card with the pair taken from tool arguments."""
    observer = tool_input.get("observer", "")
    observed = tool_input.get("observed", "")
    if not observer or not observed:
        return "ERROR: 'observer' and 'observed' are required parameters"
    # A peer card is one cross-session aggregate with no per-session attribution, so it cannot be filtered to an
    # allowlist. Fail closed rather than leak facts derived from out-of-scope sessions.
    if tctx.session_allowlist is not None:
        return "Peer cards are unavailable for session-scoped queries. Use search_memory instead."
    try:
        peer_card = await get_peer_card(tctx.ctx, tctx.bank_id, observer, observed)
    except ResourceNotFoundException:
        # The bank agent names people from its own routing, so guessing a person that doesn't exist is an expected
        # turn, not a fault.
        return f"No peer named '{observer}' exists in this bank"
    if not peer_card:
        return f"No peer card available for {observed}"
    return f"Peer card for {observed}:\n" + "\n".join(f"- {fact}" for fact in peer_card)


def _workspace_search_pairs(tool_input: dict[str, Any]) -> list[tuple[str, str]] | str:
    """Resolve the (observer, observed) pairs one search call covers.

    `observed` names one or more people; a bare string is accepted alongside the documented list. `observer` is
    optional and defaults per pair to the observed person itself, which is the global representation the bank agent
    wants nearly every time. Returns an error string for the model when the arguments don't name anything searchable.
    """
    raw_observed = tool_input.get("observed")
    if isinstance(raw_observed, str):
        raw_observed = [raw_observed]
    if not isinstance(raw_observed, list):
        return "ERROR: 'observed' is required and takes a list of handles, one entry per person you want searched."
    observed_peers: list[str] = []
    for entry in cast("list[Any]", raw_observed):
        if isinstance(entry, str) and entry and entry not in observed_peers:
            observed_peers.append(entry)
    if not observed_peers:
        return "ERROR: 'observed' must name at least one person"
    if len(observed_peers) > _WORKSPACE_SEARCH_FANOUT_LIMIT:
        return (
            f"ERROR: at most {_WORKSPACE_SEARCH_FANOUT_LIMIT} people per search. "
            "Search the most relevant ones first, then search again."
        )
    observer = tool_input.get("observer")
    if not isinstance(observer, str) or not observer:
        return [(peer, peer) for peer in observed_peers]
    return [(observer, peer) for peer in observed_peers]


async def _handle_search_memory_workspace(tctx: ToolContext, tool_input: dict[str, Any]) -> str:
    """Pair-scoped conclusion search over one or more people named in the call.

    Searches run concurrently, one per pair, and `top_k` is the budget across all of them so a wide search costs the
    reader no more context than a narrow one.
    """
    pairs = _workspace_search_pairs(tool_input)
    if isinstance(pairs, str):
        return pairs
    top_k = _bounded_int(tool_input.get("top_k"), 20, hi=40)
    per_pair_input = {**tool_input, "top_k": max(_WORKSPACE_SEARCH_MIN_TOP_K, top_k // len(pairs))}
    results = await asyncio.gather(
        *(
            _handle_search_memory(replace(tctx, observer=observer, observed=observed), per_pair_input)
            for observer, observed in pairs
        ),
        return_exceptions=True,
    )
    # Attribute the pair in the output: the bank agent may query several pairs in one turn and must not conflate
    # their results.
    sections: list[str] = []
    failures: list[str] = []
    for (observer, observed), result in zip(pairs, results, strict=True):
        if isinstance(result, BaseException) and not isinstance(result, Exception):
            raise result  # a cancellation is not a failure of one person's search
        content = f"ERROR: {_describe_tool_error(result)}" if isinstance(result, Exception) else result
        sections.append(f"[{observer}->{observed}]\n{content}")
        if content.startswith("ERROR:"):
            failures.append(content)
    # A per-pair failure is invisible once it is nested under a pair heading, and the caller only reads the leading
    # "ERROR:". Surface the whole call as failed when no pair succeeded; a partial failure still returns what the
    # other pairs found.
    if len(failures) == len(pairs):
        return failures[0]
    return "\n\n".join(sections)


# Handlers by tool name. The bank executor consults ``_WORKSPACE_TOOL_HANDLERS`` first.
_TOOL_HANDLERS: dict[str, Handler] = {
    "search_memory": _handle_search_memory,
    "get_observation_context": _handle_get_observation_context,
    "search_messages": _handle_search_messages,
    "grep_messages": _handle_grep_messages,
    "get_messages_by_date_range": _handle_get_messages_by_date_range,
    "search_messages_temporal": _handle_search_messages_temporal,
    "get_reasoning_chain": _handle_get_reasoning_chain,
    "recall_facts": _handle_recall_facts,
}

_WORKSPACE_TOOL_HANDLERS: dict[str, Handler] = {
    "search_memory": _handle_search_memory_workspace,
    "get_peer_card": _handle_get_peer_card_by_name,
    "get_reasoning_chain": _handle_get_reasoning_chain,  # already bank-scoped
}

# Standard handlers that are safe with an empty observer/observed sentinel (they only read messages or the bank's
# retained facts, treating observer="" as unscoped visibility).
_WORKSPACE_SAFE_FALLTHROUGH_TOOLS: frozenset[str] = frozenset(
    {
        "get_observation_context",
        "search_messages",
        "grep_messages",
        "get_messages_by_date_range",
        "search_messages_temporal",
        "recall_facts",
    }
)


def get_tool_handler(tool_name_: str) -> Handler | None:
    """Default handler lookup, for `handler_resolver`s that wrap it."""
    return _TOOL_HANDLERS.get(tool_name_)


def _workspace_handler_resolver(tool_name_: str) -> Handler | None:
    handler = _WORKSPACE_TOOL_HANDLERS.get(tool_name_)
    if handler is not None:
        return handler
    if tool_name_ in _WORKSPACE_SAFE_FALLTHROUGH_TOOLS:
        return _TOOL_HANDLERS.get(tool_name_)
    return None


async def create_tool_executor(
    ctx: PeopleContext,
    bank_id: str,
    observer: str,
    observed: str,
    session_name: str | None = None,
    session_allowlist: list[str] | None = None,
    agent_type: str | None = None,
    handler_resolver: Callable[[str], Handler | None] | None = None,
    evidence: EvidenceAccumulator | None = None,
) -> Callable[[str, dict[str, Any]], Awaitable[str]]:
    """
    Create a tool executor function for the dialectic's read-only tools.

    Returns an async callable that runs one tool call and hands back the text the model reads. Unknown
    tools and failing handlers answer with an ``ERROR:`` string rather than raising, so the loop can go on.
    """
    tctx = ToolContext(
        ctx=ctx,
        bank_id=bank_id,
        observer=observer,
        observed=observed,
        session_name=session_name,
        session_allowlist=session_allowlist,
        agent_type=agent_type,
        evidence=evidence,
    )
    resolve = handler_resolver or get_tool_handler

    async def execute_tool(name: str, tool_input: dict[str, Any]) -> str:
        handler = resolve(name)
        if handler is None:
            return f"ERROR: Unknown tool '{name}'"
        try:
            return await handler(tctx, tool_input)
        except Exception as e:
            logger.warning("Dialectic tool %s failed: %s", name, e, exc_info=True)
            return f"ERROR: Tool {name} failed: {_describe_tool_error(e)}"

    return execute_tool


async def create_workspace_tool_executor(
    ctx: PeopleContext,
    bank_id: str,
    session_name: str | None = None,
    session_allowlist: list[str] | None = None,
    agent_type: str | None = None,
    evidence: EvidenceAccumulator | None = None,
) -> Callable[[str, dict[str, Any]], Awaitable[str]]:
    """Tool executor for bank-level operations (no bound pair).

    observer/observed are empty-string sentinels only ever seen by handlers in
    ``_WORKSPACE_SAFE_FALLTHROUGH_TOOLS``, which normalize them to None before reading messages (None means "no
    perspective scoping"; an empty string would read as a real person with no sessions and deny everything).
    """
    return await create_tool_executor(
        ctx,
        bank_id,
        observer="",
        observed="",
        session_name=session_name,
        session_allowlist=session_allowlist,
        agent_type=agent_type,
        handler_resolver=_workspace_handler_resolver,
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Bank overview for the bank agent's prefetch
# ---------------------------------------------------------------------------

# Peer-card facts listed per person when cards are supplied.
_STATS_CARD_FACTS = 8


@dataclass(frozen=True)
class WorkspaceStats:
    """Counts over a bank's people, sessions and messages."""

    peer_count: int
    session_count: int
    message_count: int
    oldest_message_at: datetime | None
    newest_message_at: datetime | None


@dataclass(frozen=True)
class ActivePeer:
    """A person and how much they said."""

    name: str
    message_count: int
    last_message_at: datetime | None


async def get_workspace_stats(
    ctx: PeopleContext, bank_id: str, session_names: list[str] | None = None
) -> WorkspaceStats:
    """Counts for the bank, optionally restricted to the given sessions (an empty list counts nothing)."""
    scope = "" if session_names is None else " AND m.session_name = ANY($2::text[])"
    args: list[Any] = [bank_id] + ([] if session_names is None else [session_names])
    async with ctx.conn() as conn:
        peers = await conn.fetchval(
            f"SELECT count(*) FROM {ctx.table('peers')} WHERE bank_id = $1 AND deleted_at IS NULL", bank_id
        )
        row = await conn.fetchrow(
            f"""
            SELECT count(DISTINCT m.session_name) AS sessions, count(*) AS messages,
                   min(COALESCE(m.said_at, m.created_at)) AS oldest, max(COALESCE(m.said_at, m.created_at)) AS newest
            FROM {ctx.table("messages")} m
            WHERE m.bank_id = $1 AND m.deleted_at IS NULL{scope}
            """,
            *args,
        )
    return WorkspaceStats(
        peer_count=int(peers or 0),
        session_count=int(row["sessions"]),
        message_count=int(row["messages"]),
        oldest_message_at=row["oldest"],
        newest_message_at=row["newest"],
    )


async def get_active_peers(
    ctx: PeopleContext, bank_id: str, limit: int = 5, session_names: list[str] | None = None
) -> list[ActivePeer]:
    """The people with the most messages, most active first."""
    scope = "" if session_names is None else " AND m.session_name = ANY($3::text[])"
    args: list[Any] = [bank_id, limit] + ([] if session_names is None else [session_names])
    async with ctx.conn() as conn:
        rows = await conn.fetch(
            f"""
            SELECT m.peer_name AS name, count(*) AS messages, max(COALESCE(m.said_at, m.created_at)) AS last_at
            FROM {ctx.table("messages")} m
            WHERE m.bank_id = $1 AND m.deleted_at IS NULL{scope}
            GROUP BY m.peer_name
            ORDER BY count(*) DESC, m.peer_name ASC
            LIMIT $2
            """,
            *args,
        )
    return [ActivePeer(name=r["name"], message_count=int(r["messages"]), last_message_at=r["last_at"]) for r in rows]


def format_workspace_stats(
    stats: WorkspaceStats,
    peers: Sequence[ActivePeer],
    cards: dict[str, list[str]] | None = None,
) -> str:
    """Render bank counts and most-active people as prompt-ready lines.

    Used by WorkspaceDialecticAgent's routing prefetch; ``cards`` nests each person's known biographical facts
    under it.
    """
    lines = [
        f"People: {stats.peer_count}",
        f"Sessions: {stats.session_count}",
        f"Messages: {stats.message_count}",
    ]
    if stats.oldest_message_at and stats.newest_message_at:
        lines.append(f"Date range: {stats.oldest_message_at:%Y-%m-%d} to {stats.newest_message_at:%Y-%m-%d}")
    if peers:
        lines.append("")
        lines.append(f"Most active people (top {len(peers)}):")
        for peer in peers:
            last_active = f", last active {peer.last_message_at:%Y-%m-%d}" if peer.last_message_at else ""
            lines.append(f"- {peer.name} ({peer.message_count} messages{last_active})")
            for fact in (cards or {}).get(peer.name, [])[:_STATS_CARD_FACTS]:
                lines.append(f"    - {fact}")
    return "\n".join(lines)


__all__ = [
    "ANSWER_READY_TOOL",
    "ANSWER_READY_TOOL_NAME",
    "DIALECTIC_TOOLS",
    "DIALECTIC_TOOLS_MINIMAL",
    "WORKSPACE_DIALECTIC_TOOLS",
    "WORKSPACE_RECALL_TOOLS",
    "WORKSPACE_TOOLS_MINIMAL",
    "ToolContext",
    "create_tool_executor",
    "create_workspace_tool_executor",
    "format_workspace_stats",
    "get_active_peers",
    "get_workspace_stats",
    "search_memory",
    "tool_name",
]
