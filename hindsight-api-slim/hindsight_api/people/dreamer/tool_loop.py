# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/llm/tool_loop.py
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/utils/types.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""The tool loop a dream specialist runs on the engine's Claude-configured provider.

Honcho drives its specialists through ``honcho_llm_call`` (its own LLM layer, which is not copied).
Here the same loop runs on ``llm.call_with_tools``: one tool call per turn, each result sent back as a
``tool`` message, an optional budget line after the results, and a final call at the iteration limit
that offers only the tools that may still write.

Also holds the small per-loop state Honcho keeps in ``utils/types.py``: ``ToolResult`` and the
iteration / last-tool context variables.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Context variable for tracking the current iteration in the tool execution loop.
_current_iteration: ContextVar[int] = ContextVar("current_iteration", default=0)


def set_current_iteration(iteration: int) -> None:
    """Set the current iteration number."""
    _current_iteration.set(iteration)


def get_current_iteration() -> int:
    """Get the current iteration number."""
    return _current_iteration.get()


# After a tool finishes, the metadata dict from the handler's ToolResult is published here so the loop
# can stash it on the tool-call record (specialist rollups read created/deleted counts from it).
_last_tool_metadata: ContextVar[dict[str, Any] | None] = ContextVar("last_tool_metadata", default=None)


def set_last_tool_metadata(metadata: dict[str, Any]) -> None:
    """Publish the just-finished tool call's ToolResult metadata."""
    _last_tool_metadata.set(metadata)


def get_last_tool_metadata() -> dict[str, Any]:
    """Read the last tool call's metadata. Returns {} when no ToolResult was returned."""
    return _last_tool_metadata.get() or {}


# Whether the just-finished tool call failed. The executor returns errors to the model as strings
# rather than raising, so the loop reads this to decide what counts as a successful call.
_last_tool_error: ContextVar[bool] = ContextVar("last_tool_error", default=False)


def set_last_tool_error(is_error: bool) -> None:
    _last_tool_error.set(is_error)


def get_last_tool_error() -> bool:
    return _last_tool_error.get()


@contextmanager
def iteration_scope() -> Generator[None]:
    """Reset per-tool-loop context variables on exit."""
    iter_token = _current_iteration.set(0)
    meta_token = _last_tool_metadata.set(None)
    error_token = _last_tool_error.set(False)
    try:
        yield
    finally:
        _current_iteration.reset(iter_token)
        _last_tool_metadata.reset(meta_token)
        _last_tool_error.reset(error_token)


@dataclass
class ToolResult:
    """Internal return shape used by tool handlers.

    Handlers may return a plain ``str``, or ``ToolResult(content=..., metadata={...})`` when they
    carry structured metadata for the specialist rollups (``created_count``, ``deleted_count``...).
    The executor unwraps it before returning the string to the loop.
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __contains__(self, item: str) -> bool:
        return item in self.content

    def __str__(self) -> str:
        return self.content


BudgetLine = Callable[[int, int], str | None]


@dataclass
class ToolLoopResult:
    """What a tool loop returns: the model's last text and what the loop did."""

    content: str
    tool_calls_made: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    hit_iteration_limit: bool = False


DONE_TOOL_NAME = "done"
DONE_TOOL: dict[str, Any] = {
    "name": DONE_TOOL_NAME,
    "description": (
        "Finish the pass. Call it when everything worth writing is written, or when nothing new is worth "
        "writing. This is the DONE the instructions mention."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"summary": {"type": "string", "description": "One or two sentences on what you did."}},
    },
}


def function_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tools in Honcho's ``{name, description, input_schema}`` shape as OpenAI-style function tools."""
    return [
        t
        if "function" in t
        else {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for t in tools
    ]


def _tool_name(tool: dict[str, Any]) -> str:
    return str(tool["function"]["name"] if "function" in tool else tool["name"])


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


async def _execute_tool_calls(
    calls: list[Any],
    tool_executor: Callable[[str, dict[str, Any]], Any],
    all_tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run one response's tool calls in order; returns the ``tool`` messages to send back."""
    results: list[dict[str, Any]] = []
    for tc in calls:
        set_last_tool_metadata({})
        set_last_tool_error(False)
        try:
            tool_result = await tool_executor(tc.name, tc.arguments)
            all_tool_calls.append(
                {
                    "tool_name": tc.name,
                    "tool_input": tc.arguments,
                    "tool_result": tool_result,
                    "tool_result_metadata": get_last_tool_metadata(),
                    "error": get_last_tool_error(),
                }
            )
            results.append({"role": "tool", "tool_call_id": tc.id, "content": str(tool_result)})
        except Exception as e:
            logger.error("Tool execution failed for %s: %s", tc.name, e)
            results.append({"role": "tool", "tool_call_id": tc.id, "content": f"Error: {e}"})
    return results


async def execute_tool_loop(
    llm: Any,
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_executor: Callable[[str, dict[str, Any]], Any],
    max_tool_iterations: int,
    max_tokens: int,
    scope: str,
    budget_line: BudgetLine | None = None,
    final_call_tools: list[dict[str, Any]] | None = None,
) -> ToolLoopResult:
    """Run the loop until the model answers without a tool call or the iteration limit is reached.

    At the limit one final call is made: with ``final_call_tools`` it may still call those tools (to
    save what it decided on), without them it must answer in text.
    """
    from ...engine.llm_interface import LLM_TOOL_CHOICE_AUTO

    conversation = list(messages)
    tools = [*tools, DONE_TOOL]
    declared = function_tools(tools)
    all_tool_calls: list[dict[str, Any]] = []
    input_tokens = output_tokens = 0
    with iteration_scope():
        for iteration in range(max_tool_iterations):
            result = await llm.call_with_tools(
                messages=conversation,
                tools=declared,
                max_completion_tokens=max_tokens,
                scope=scope,
                tool_choice=LLM_TOOL_CHOICE_AUTO,
            )
            input_tokens += result.input_tokens
            output_tokens += result.output_tokens
            finished = any(tc.name == DONE_TOOL_NAME for tc in result.tool_calls)
            work = [tc for tc in result.tool_calls if tc.name != DONE_TOOL_NAME]
            if finished or not result.tool_calls:
                set_current_iteration(iteration + 1)
                await _execute_tool_calls(work, tool_executor, all_tool_calls)
                done_text = next(
                    (str(tc.arguments.get("summary") or "") for tc in result.tool_calls if tc.name == DONE_TOOL_NAME),
                    "",
                )
                return ToolLoopResult(
                    content=result.content or done_text,
                    tool_calls_made=all_tool_calls,
                    iterations=iteration + 1,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            conversation.append(_assistant_turn(result.content, result.tool_calls))
            set_current_iteration(iteration + 1)
            conversation.extend(await _execute_tool_calls(result.tool_calls, tool_executor, all_tool_calls))
            if budget_line is not None:
                line = budget_line(iteration + 1, max_tool_iterations)
                if line:
                    conversation.append({"role": "user", "content": line})

        logger.warning("Tool execution loop reached max iterations (%d)", max_tool_iterations)
        if final_call_tools:
            names = ", ".join(f"`{_tool_name(t)}`" for t in final_call_tools)
            conversation.append(
                {
                    "role": "user",
                    "content": (
                        "You have reached the maximum number of tool calls. "
                        f"You may make one final call to {names} to save what you "
                        "have decided on; no other tools are available. If there is nothing "
                        "to save, respond without calling a tool."
                    ),
                }
            )
            final = await llm.call_with_tools(
                messages=conversation,
                tools=function_tools([*final_call_tools, DONE_TOOL]),
                declared_tools=declared,
                max_completion_tokens=max_tokens,
                scope=scope,
                tool_choice=LLM_TOOL_CHOICE_AUTO,
            )
            input_tokens += final.input_tokens
            output_tokens += final.output_tokens
            allowed = {_tool_name(t) for t in final_call_tools}
            set_current_iteration(max_tool_iterations + 1)
            await _execute_tool_calls(
                [tc for tc in final.tool_calls if tc.name in allowed], tool_executor, all_tool_calls
            )
            content = final.content or ""
        else:
            conversation.append(
                {
                    "role": "user",
                    "content": (
                        "You have reached the maximum number of tool calls. "
                        "Based on all the information you have gathered, provide your final response now. "
                        "Do not attempt to call any more tools."
                    ),
                }
            )
            final_text = await llm.call(messages=conversation, max_completion_tokens=max_tokens, scope=scope)
            input_tokens += getattr(final_text, "input_tokens", 0) or 0
            output_tokens += getattr(final_text, "output_tokens", 0) or 0
            content = final_text.content if isinstance(final_text.content, str) else ""
        return ToolLoopResult(
            content=content,
            tool_calls_made=all_tool_calls,
            iterations=max_tool_iterations,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            hit_iteration_limit=True,
        )
