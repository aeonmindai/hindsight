# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/dialectic/core.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""
Core Dialectic Agent implementation.

This agent uses tools to gather context from the memory system
and synthesize responses to queries about a peer.

Differences from Honcho:

* The model is ``ctx.llm_interactive`` (the engine's Claude-configured provider). Its tool turns always
  call exactly one function, so the research loop offers an ``answer_ready`` function the model calls when it
  has what it needs; the answer is then written by a final call that offers no tools. Honcho's loop ends when
  the model answers without a tool call; a reply with no call is still taken as the answer here.
* The provider has no token stream. ``answer_stream`` runs the research, then yields the final answer in
  pieces (the same "final only" streaming Honcho's chat route uses).
* No telemetry, metrics or run spans.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from ...engine.llm_interface import LLM_TOOL_CHOICE_REQUIRED
from ..conclusions import Conclusion
from ..context import PeopleContext
from ..exceptions import PeopleError, ResourceNotFoundException
from ..ids import generate_nanoid
from ..peer_card import get_peer_card
from ..peers import get_peer
from ..schemas import ResolvedDialecticConfiguration, get_configuration
from . import messages as message_store
from . import prompts, settings
from .evidence import Evidence, EvidenceAccumulator, Source
from .formatting import format_new_turn_with_timestamp
from .settings import DialecticLevelSettings, ReasoningLevel
from .tools import (
    ANSWER_READY_TOOL,
    ANSWER_READY_TOOL_NAME,
    DIALECTIC_TOOLS,
    DIALECTIC_TOOLS_MINIMAL,
    create_tool_executor,
    format_conclusions,
    search_memory,
    tool_name,
)

logger = logging.getLogger(__name__)

# Scopes the engine's LLM layer labels these calls with.
TOOL_SCOPE = "people_dialectic"
ANSWER_SCOPE = "people_dialectic_answer"

# Below this many tool iterations a budget line is noise.
MIN_ITERATIONS_FOR_BUDGET_LINE = 3

# Longest query the prefetch embeds, in characters; a longer query is embedded by its leading part.
_MAX_EMBED_QUERY_CHARS = 8_000

_FINAL_ANSWER_REQUEST = (
    "Based on all the information you have gathered, provide your final response now. "
    "Do not attempt to call any more tools."
)

BudgetLine = Callable[[int, int], str | None]
ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[str]]


class StructuredResponseError(PeopleError):
    """The final answer did not conform to the requested response model (the API answers 422)."""

    status_code = 422

    def __init__(self, message: str = "The model's answer did not match the requested response schema") -> None:
        super().__init__(message)


def dialectic_budget_line(used: int, limit: int) -> str | None:
    """Budget line for the dialectic: how many calls are used, and when to answer."""
    if used >= limit:
        # The loop is about to make its no-tools final call, which says so.
        return None
    line = f"Tool calls used: {used} of {limit}."
    if limit - used == 1:
        line += " On the next call, answer from what you have."
    return line


@dataclass
class ToolLoopResult:
    """What the research loop did, and the answer when the model gave one without the final call."""

    content: str | None = None
    tool_calls_made: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    hit_input_token_cap: bool = False
    hit_iteration_limit: bool = False
    conversation: list[dict[str, Any]] = field(default_factory=list)


def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
    """A rough token count of a conversation: four characters to a token."""
    return sum(len(json.dumps(m, ensure_ascii=False, default=str)) for m in messages) // 4


def _assistant_turn(content: str | None, calls: list[Any]) -> dict[str, Any]:
    turn: dict[str, Any] = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
            }
            for tc in calls
        ],
    }
    if content:
        turn["content"] = content
    return turn


def _as_plain_turns(conversation: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The conversation as role/content turns, for the final call that offers no tools.

    A tool call becomes an assistant line naming it; its result becomes a user turn.
    """
    names: dict[str, str] = {}
    out: list[dict[str, Any]] = []
    for message in conversation:
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            lines = []
            for call in message["tool_calls"]:
                function = call.get("function", call)
                names[call.get("id", "")] = function["name"]
                lines.append(f"Called `{function['name']}` with {function.get('arguments') or '{}'}")
            text = "\n".join(lines)
            if message.get("content"):
                text = f"{message['content']}\n{text}"
            out.append({"role": "assistant", "content": text})
        elif role == "tool":
            name = names.get(message.get("tool_call_id", ""), "the tool")
            out.append({"role": "user", "content": f"Result of `{name}`:\n{message.get('content') or ''}"})
        else:
            out.append({"role": role, "content": message.get("content") or ""})
    return out


async def execute_tool_loop(
    llm: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_executor: ToolExecutor,
    *,
    max_tool_iterations: int,
    max_tokens: int,
    max_input_tokens: int,
    force_tools: bool,
    force_tools_until: frozenset[str] | None,
    budget_line: BudgetLine | None,
) -> ToolLoopResult:
    """
    Run the research loop: the model calls one tool per turn, each result goes back as a ``tool`` message.

    The loop ends when the model calls ``answer_ready``, answers without a call, runs out of tool
    iterations or fills its input budget. ``force_tools`` keeps ``answer_ready`` out of the offered tools
    until a tool in ``force_tools_until`` (any tool when None) has run; the loop never pins the model past that
    first satisfying call.

    ``messages`` is not modified. The result carries the conversation to answer from.
    """
    conversation = list(messages)
    declared = [*tools, ANSWER_READY_TOOL]
    result = ToolLoopResult(conversation=conversation)
    satisfied = not force_tools

    for iteration in range(max_tool_iterations):
        if _estimate_tokens(conversation) >= max_input_tokens:
            result.hit_input_token_cap = True
            return result

        if satisfied:
            offered = declared
        elif force_tools_until is not None:
            offered = [t for t in tools if tool_name(t) in force_tools_until] or tools
        else:
            offered = tools

        turn = await llm.call_with_tools(
            messages=conversation,
            tools=offered,
            declared_tools=declared,
            max_completion_tokens=max_tokens,
            scope=TOOL_SCOPE,
            tool_choice=LLM_TOOL_CHOICE_REQUIRED,
        )
        result.input_tokens += getattr(turn, "input_tokens", 0) or 0
        result.output_tokens += getattr(turn, "output_tokens", 0) or 0
        result.iterations = iteration + 1

        if not turn.tool_calls:
            # A reply with no call is the answer.
            result.content = turn.content or ""
            return result

        if any(tc.name == ANSWER_READY_TOOL_NAME for tc in turn.tool_calls):
            return result

        conversation.append(_assistant_turn(turn.content, turn.tool_calls))
        for tc in turn.tool_calls:
            output = await tool_executor(tc.name, tc.arguments)
            failed = output.startswith("ERROR:")
            conversation.append({"role": "tool", "tool_call_id": tc.id, "content": output})
            if not failed:
                result.tool_calls_made.append({"tool_name": tc.name, "tool_input": tc.arguments, "tool_id": tc.id})
            if not failed and (force_tools_until is None or tc.name in force_tools_until):
                satisfied = True
        if budget_line is not None:
            line = budget_line(iteration + 1, max_tool_iterations)
            if line:
                conversation.append({"role": "user", "content": line})

    if max_tool_iterations > 0:
        result.hit_iteration_limit = True
        logger.debug("Dialectic tool loop reached max iterations (%d)", max_tool_iterations)
    return result


def _final_content(content: Any, response_model: type[BaseModel] | None) -> str:
    """The text a final call answered, as the API returns it: a JSON string when a response model was given."""
    if isinstance(content, BaseModel):
        return content.model_dump_json(by_alias=True)
    if response_model is not None:
        # A provider that degrades to a plain string instead of raising (an empty answer or a refusal) still
        # fails the response model.
        raise StructuredResponseError()
    return content if isinstance(content, str) else ""


def chunk_text(text: str) -> list[str]:
    """The answer in pieces for streaming: one per line break, newline kept, so the pieces rejoin to the text."""
    return [piece for piece in re.split(r"(?<=\n)", text) if piece]


class DialecticAgent:
    """
    An agentic dialectic that iteratively gathers context to answer queries.

    Unlike the standard dialectic which pre-gathers all context before a single
    LLM call, this agent uses tools to strategically gather only the context
    needed to answer the specific query.
    """

    def __init__(
        self,
        ctx: PeopleContext,
        bank_id: str,
        session_name: str | None,
        observer: str,
        observed: str,
        observer_peer_card: list[str] | None = None,
        observed_peer_card: list[str] | None = None,
        reasoning_level: ReasoningLevel = "low",
        session_allowlist: list[str] | None = None,
        evidence: EvidenceAccumulator | None = None,
        instructions: ResolvedDialecticConfiguration | None = None,
    ):
        """
        Initialize the dialectic agent.

        Args:
            ctx: People context (database, embeddings, the interactive model)
            bank_id: Bank identifier
            session_name: Session identifier (may be None for global queries)
            observer: The person making the query
            observed: The person being queried about
            observer_peer_card: Biographical information about the observer
            observed_peer_card: Biographical information about the observed person
            reasoning_level: Level of reasoning to apply
            session_allowlist: Optional session allowlist restricting all recall
                (conclusions and messages) to these sessions; empty list
                fails closed
            evidence: Optional accumulator collecting the conclusions and
                messages this run reads, for callers that asked for evidence.
                Passing None collects nothing.
            instructions: Resolved dialectic configuration carrying the custom
                instructions and where they were resolved from
        """
        self.ctx: PeopleContext = ctx
        self.bank_id: str = bank_id
        self.session_name: str | None = session_name
        self.session_allowlist: list[str] | None = session_allowlist
        self.observer: str = observer
        self.observed: str = observed
        self.observer_peer_card: list[str] | None = observer_peer_card
        self.observed_peer_card: list[str] | None = observed_peer_card
        self.reasoning_level: ReasoningLevel = reasoning_level
        self.instructions: ResolvedDialecticConfiguration = instructions or ResolvedDialecticConfiguration()

        # Initialize conversation history with system prompt
        self.messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": prompts.agent_system_prompt(
                    observer,
                    observed,
                    observer_peer_card,
                    observed_peer_card,
                    available_tools=self._available_tool_names(),
                    custom_instructions=self.instructions.custom_instructions,
                ),
            }
        ]
        self._session_history_initialized: bool = False
        self._prefetched_conclusion_count: int = 0
        self.evidence: EvidenceAccumulator | None = evidence
        self._run_id: str = generate_nanoid()  # Always generate for correlation in logs
        if self._layout_enabled():
            from ..memory_brief import memory_layout_section

            self.messages[0]["content"] += "\n\n" + memory_layout_section(
                available_tools=self._available_tool_names(),
                cite_ids=False,
            )

    def _available_tool_names(self) -> set[str]:
        """Names of the tools this query offers, plus the one that ends the research."""
        return {tool_name(tool) for tool in self._select_tools()} | {ANSWER_READY_TOOL_NAME}

    def _select_tools(self) -> list[dict[str, Any]]:
        """Pick the toolset for this query.

        Minimal reasoning uses a reduced set to reduce cost. Under a session
        allowlist `get_reasoning_chain` and `recall_facts` are dropped entirely rather than left in
        to fail at call time: chains traverse provenance across sessions and retained facts carry
        no session, so neither can be scoped, and offering them costs both the schema in context and a
        wasted turn when the model tries them.
        """
        tools = DIALECTIC_TOOLS_MINIMAL if self.reasoning_level == "minimal" else DIALECTIC_TOOLS
        if self.session_allowlist is not None:
            unscopable = {"get_reasoning_chain", "recall_facts"}
            tools = [t for t in tools if tool_name(t) not in unscopable]
        return tools

    def _tool_choice(self, level_settings: DialecticLevelSettings) -> str | None:
        """Pick the tool_choice for this query.

        Defaults to whatever the reasoning level configures. Subclasses override
        when the agent has no prefetched corpus to fall back on and so must
        search before it can answer. Forcing "required"/"any" here costs exactly
        one tool round rather than pinning the loop: `execute_tool_loop` offers
        `answer_ready` again after the first satisfying call so the model can still stop and
        synthesize.
        """
        return level_settings.TOOL_CHOICE

    def _force_tools_until(self) -> frozenset[str] | None:
        """Tools that satisfy a forced tool_choice; None means any tool does."""
        return None

    def _layout_enabled(self) -> bool:
        """Whether the memory-layout section goes in the system prompt.

        Pair-scoped only: the bank agent replaces the system prompt.
        """
        return settings.BRIEF_ENABLED

    def _pair_brief_enabled(self) -> bool:
        """Whether counts go in the first message.

        They are drawn from every session of the pair, so a session allowlist
        rules them out: they would reach past the allowed sessions.
        """
        return settings.BRIEF_ENABLED and self.session_allowlist is None

    async def _pair_brief(self) -> str | None:
        """Counts for the first message, if enabled."""
        if not self._pair_brief_enabled():
            return None
        from ..memory_brief import load_memory_counts

        try:
            return (
                await load_memory_counts(self.ctx, self.bank_id, observer=self.observer, observed=self.observed) or None
            )
        except Exception:
            logger.warning("Failed to load the memory brief", exc_info=True)
            return None

    def _budget_line(self, level_settings: DialecticLevelSettings) -> BudgetLine | None:
        """Budget line for this query, or None when off or the level is too small."""
        if not settings.BUDGET_ENABLED:
            return None
        if self._max_tool_iterations(level_settings) < MIN_ITERATIONS_FOR_BUDGET_LINE:
            return None
        return dialectic_budget_line

    def _max_tool_iterations(self, level_settings: DialecticLevelSettings) -> int:
        """Tool rounds this query gets; defaults to the reasoning level's limit."""
        return level_settings.MAX_TOOL_ITERATIONS

    async def _initialize_session_history(self) -> None:
        """Fetch and inject session history into the system prompt if configured."""
        if self._session_history_initialized:
            return
        self._session_history_initialized = True

        max_tokens = settings.SESSION_HISTORY_MAX_TOKENS
        if max_tokens == 0 or not self.session_name:
            return

        # Recent messages up to the token limit, in chronological order
        rows = await message_store.get_messages_id_range(
            self.ctx, self.bank_id, self.session_name, token_limit=max_tokens
        )
        if not rows:
            return
        if self.evidence is not None:
            self.evidence.add_messages(rows)

        formatted_messages = [
            format_new_turn_with_timestamp(msg.content, msg.when, msg.peer_name)
            + f"\n{message_store.message_source_line(msg)}"
            for msg in rows
        ]
        session_history_section = (
            "\n\n## SESSION HISTORY\n\n"
            "The following is the recent conversation history from this session. "
            "Use this as immediate context when answering the query.\n\n"
            "<session_history>\n"
            f"{chr(10).join(formatted_messages)}\n"
            "</session_history>"
        )

        # Append session history to the system prompt
        self.messages[0]["content"] += session_history_section

    async def _prefetch_relevant_observations(self, query: str) -> str | None:
        """
        Prefetch semantically relevant observations for the query.

        This provides immediate context to the agent without requiring
        tool calls, improving response quality and speed.

        Performs two separate searches to prevent retrieval dilution:
        - Explicit observations (produced by deriver)
        - Higher-level observations (produced in dreaming/background/chat)

        The number of observations fetched depends on reasoning level:
        - minimal: 10 of each type (reduced context for cost savings)
        - all others: 25 of each type

        Args:
            query: The user's query

        Returns:
            Formatted observations string or None if no observations found
        """
        # Use reduced prefetch for minimal reasoning to save tokens
        prefetch_limit = 10 if self.reasoning_level == "minimal" else 25

        try:
            # Pre-compute embedding once for both searches. A long query (e.g. one quoting recent turns) can
            # exceed the embedding cap; embed its leading part rather than skip prefetch.
            query_embedding = await self.ctx.embed_query(query[:_MAX_EMBED_QUERY_CHARS])

            # Prefetched conclusions never pass through the tool executor, so they are recorded here or not at
            # all -- and on a query that answers without a tool call they are the whole of what was read.
            prefetched: list[Conclusion] | None = [] if self.evidence is not None else None

            explicit = await search_memory(
                self.ctx,
                self.bank_id,
                self.observer,
                self.observed,
                query,
                prefetch_limit,
                levels=["explicit"],
                embedding=query_embedding,
                session_allowlist=self.session_allowlist,
                documents_out=prefetched,
            )
            derived = await search_memory(
                self.ctx,
                self.bank_id,
                self.observer,
                self.observed,
                query,
                prefetch_limit,
                levels=["deductive", "inductive", "contradiction"],
                embedding=query_embedding,
                session_allowlist=self.session_allowlist,
                documents_out=prefetched,
            )

            if not explicit and not derived:
                return None

            self._prefetched_conclusion_count = len(explicit) + len(derived)

            # Two sections. Every conclusion carries its id, so the agent can use get_reasoning_chain and cite.
            parts: list[str] = []
            if explicit:
                parts.append(format_conclusions(explicit))
            if derived:
                parts.append(format_conclusions(derived))

            # Recorded last: everything above can still fail into the handler below, which drops the whole block
            # from the prompt. Evidence should name what the agent saw, not what was fetched for it.
            if self.evidence is not None and prefetched:
                self.evidence.add_documents(prefetched)

            return "\n\n".join(parts)

        except Exception as e:
            logger.warning(f"Failed to prefetch observations: {e}")
            return None

    async def _prepare_query(self, query: str) -> ToolExecutor:
        """
        Prepare common state for answering a query.

        Handles session history initialization, observation prefetching, user message construction, and tool
        executor creation.

        Args:
            query: The question to answer about the person

        Returns:
            The tool executor
        """
        await self._initialize_session_history()
        return await self._prepare_messages(query)

    async def _prepare_messages(self, query: str) -> ToolExecutor:
        """Prefetch observations, append the user message, and build the tool executor."""
        prefetched_observations = await self._prefetch_relevant_observations(query)

        if prefetched_observations:
            user_content = (
                f"Query: {query}\n\n"
                f"## {self._prefetch_heading()}\n"
                f"{self._prefetch_intro()}\n\n"
                f"{prefetched_observations}"
            )
        else:
            user_content = f"Query: {query}"

        brief = await self._pair_brief()
        if brief:
            user_content = f"{user_content}\n\n{brief}"

        self.messages.append({"role": "user", "content": user_content})

        return await self._create_tool_executor()

    async def _create_tool_executor(self) -> ToolExecutor:
        """Build the tool executor. Subclasses override to change tool scoping
        (e.g. WorkspaceDialecticAgent uses the bank executor)."""
        return await create_tool_executor(
            self.ctx,
            self.bank_id,
            observer=self.observer,
            observed=self.observed,
            session_name=self.session_name,
            session_allowlist=self.session_allowlist,
            agent_type="dialectic",
            evidence=self.evidence,
        )

    def _prefetch_heading(self) -> str:
        """Heading for the prefetched block in the user message."""
        return "Relevant Observations (prefetched)"

    def _prefetch_intro(self) -> str:
        """Sentence introducing the prefetched block in the user message."""
        return (
            "The following observations were found to be semantically relevant to your query. "
            "Use these as primary context. You may still use tools to find additional information if needed."
        )

    def _log_response_metrics(
        self,
        loop: ToolLoopResult,
        answer_input_tokens: int,
        answer_output_tokens: int,
        iteration_limit: int,
    ) -> None:
        """Log one line about the run: tokens, tool calls and why the research stopped."""
        logger.info(
            "dialectic run=%s level=%s iterations=%d/%d tool_calls=%d prefetched=%d input_tokens=%d output_tokens=%d "
            "hit_input_token_cap=%s hit_iteration_limit=%s",
            self._run_id,
            self.reasoning_level,
            loop.iterations,
            iteration_limit,
            len(loop.tool_calls_made),
            self._prefetched_conclusion_count,
            loop.input_tokens + answer_input_tokens,
            loop.output_tokens + answer_output_tokens,
            loop.hit_input_token_cap,
            loop.hit_iteration_limit,
        )

    async def _research_and_answer(self, query: str, response_model: type[BaseModel] | None) -> str:
        """Prepare, run the research loop, and write the final answer."""
        tool_executor = await self._prepare_query(query)

        level_settings = settings.LEVELS[self.reasoning_level]
        # Use level-specific max_output_tokens if set, otherwise global default
        max_tokens = (
            level_settings.MAX_OUTPUT_TOKENS
            if level_settings.MAX_OUTPUT_TOKENS is not None
            else settings.MAX_OUTPUT_TOKENS
        )
        llm = self.ctx.llm_interactive
        tool_choice = self._tool_choice(level_settings)

        loop = await execute_tool_loop(
            llm,
            self.messages,
            self._select_tools(),
            tool_executor,
            max_tool_iterations=self._max_tool_iterations(level_settings),
            max_tokens=max_tokens,
            max_input_tokens=settings.MAX_INPUT_TOKENS,
            force_tools=tool_choice in ("required", "any"),
            force_tools_until=self._force_tools_until(),
            budget_line=self._budget_line(level_settings),
        )

        answer_input_tokens = answer_output_tokens = 0
        if loop.content is not None and response_model is None:
            content = loop.content
        else:
            final_messages = [*_as_plain_turns(loop.conversation), {"role": "user", "content": _FINAL_ANSWER_REQUEST}]
            try:
                final = await llm.call(
                    messages=final_messages,
                    response_format=response_model,
                    max_completion_tokens=max_tokens,
                    scope=ANSWER_SCOPE,
                )
            except ValidationError as exc:
                if response_model is not None:
                    raise StructuredResponseError() from exc
                raise
            answer_input_tokens = getattr(final, "input_tokens", 0) or 0
            answer_output_tokens = getattr(final, "output_tokens", 0) or 0
            content = _final_content(final.content, response_model)

        if self.evidence is not None:
            self.evidence.record_tool_calls(loop.tool_calls_made)
        self._log_response_metrics(
            loop, answer_input_tokens, answer_output_tokens, self._max_tool_iterations(level_settings)
        )
        return content

    async def answer(self, query: str, response_model: type[BaseModel] | None = None) -> str:
        """
        Answer a query about the person using agentic tool calling.

        The agent will:
        1. Receive the query
        2. Use tools to gather relevant context
        3. Synthesize a response grounded in the gathered context

        Args:
            query: The question to answer about the person
            response_model: Optional Pydantic model the final synthesis must
                conform to. When set, the returned string is JSON.

        Returns:
            The synthesized answer string
        """
        return await self._research_and_answer(query, response_model)

    async def answer_stream(self, query: str, response_model: type[BaseModel] | None = None) -> AsyncIterator[str]:
        """
        Answer a query about the person, streaming the final answer.

        The research runs first; the provider has no token stream, so the finished answer is then yielded in
        pieces that rejoin to the text (one per line). With a response model the pieces rejoin to the JSON.

        Args:
            query: The question to answer about the person
            response_model: Optional Pydantic model the answer must conform to

        Yields:
            Chunks of the answer text
        """
        content = await self._research_and_answer(query, response_model)
        for piece in chunk_text(content):
            yield piece


@dataclass(frozen=True)
class DialecticResult:
    """An answer about a person and everything it can cite.

    ``sources`` lists every conclusion, message and retained fact the agent read, each with the URL it was
    said at and the time it was said, so a claim in ``text`` can be traced. It over-reports: it holds what the
    agent read, not only what the answer leaned on.
    """

    text: str
    sources: list[Source]
    evidence: Evidence


async def prepare_pair_agent(
    ctx: PeopleContext,
    bank_id: str,
    session_name: str | None,
    observer: str,
    observed: str,
    reasoning_level: ReasoningLevel = "low",
    session_allowlist: list[str] | None = None,
    evidence: EvidenceAccumulator | None = None,
) -> DialecticAgent:
    """Validate the pair and build the agent for it (Honcho's preflight in ``agentic_chat``).

    Raises:
        ResourceNotFoundException: If the observer, the observed person or the session does not exist
    """
    await get_peer(ctx, bank_id, observer)
    if observer != observed:
        await get_peer(ctx, bank_id, observed)
    if session_name:
        async with ctx.conn() as conn:
            found = await conn.fetchval(
                f"SELECT 1 FROM {ctx.table('sessions')} WHERE bank_id = $1 AND name = $2 AND deleted_at IS NULL",
                bank_id,
                session_name,
            )
        if not found:
            raise ResourceNotFoundException(f"Session {session_name} not found in bank {bank_id}")

    configuration = get_configuration(None, None, None)
    observer_peer_card = None
    observed_peer_card = None
    if configuration.peer_card.use:
        observer_peer_card = await get_peer_card(ctx, bank_id, observer, observer)
        if observer != observed:
            observed_peer_card = await get_peer_card(ctx, bank_id, observer, observed)

    return DialecticAgent(
        ctx,
        bank_id,
        session_name=session_name,
        observer=observer,
        observed=observed,
        observer_peer_card=observer_peer_card,
        observed_peer_card=observed_peer_card,
        reasoning_level=reasoning_level,
        session_allowlist=session_allowlist,
        evidence=evidence,
        instructions=configuration.dialectic,
    )


async def dialectic_chat(
    ctx: PeopleContext,
    bank_id: str,
    observer: str,
    observed: str,
    query: str,
    *,
    session_name: str | None = None,
    reasoning_level: ReasoningLevel = "low",
    session_allowlist: list[str] | None = None,
    response_model: type[BaseModel] | None = None,
) -> DialecticResult:
    """
    Answer a query about ``observed`` from ``observer``'s memory, with the sources the answer can cite.

    The agent prefetches the conclusions closest to the query, then researches with tools over the conclusions,
    the messages and the bank's retained facts, and answers citing each claim as ``[source: <url>, said_at:
    <time>]``. Everything it read is returned in ``sources``.

    Args:
        ctx: People context
        bank_id: Bank identifier
        observer: Handle of the person whose memory is asked (use the same handle as ``observed`` for a
            person's own representation)
        observed: Handle of the person asked about
        query: The question
        session_name: Session to ground the query in (its recent history goes in the prompt)
        reasoning_level: minimal, low, medium, high or max
        session_allowlist: Restrict all recall to these sessions; an empty list fails closed
        response_model: Optional Pydantic model the answer must conform to; ``text`` is then JSON

    Returns:
        The answer text and the sources it can cite
    """
    evidence = EvidenceAccumulator()
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
    text = await agent.answer(query, response_model=response_model)
    built = evidence.build()
    return DialecticResult(text=text, sources=built.sources(), evidence=built)
