# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: src/utils/memory_brief.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""Shared memory-orientation text for agents that search a pair's memory.

The dreamer specialists and the dialectic each start a run knowing nothing
about how memory is laid out, and spend early tool calls finding out. This
module renders what they share: a fixed system-prompt section describing the
layout, and a counts line for the first user message. Keeping both here means
the two agents can't drift apart.
"""

from collections.abc import Iterable

from .conclusions import DocumentLevelSummary, summarize_documents_by_level
from .context import PeopleContext

LEVEL_ORDER: tuple[str, ...] = ("explicit", "deductive", "inductive", "contradiction")

_LEVELS_TEXT = """Memory for this pair is a set of conclusions at four levels:
- **explicit**: facts the peer stated directly, extracted from messages. The base level; cites nothing.
- **deductive**: conclusions that must follow from other conclusions. Cites the explicit or deductive conclusions it follows from.
- **inductive**: patterns across several conclusions. Cites two or more explicit or deductive conclusions.
- **contradiction**: conclusions that cannot all be true. Cites the two or more conclusions that conflict."""

_IDS_TEXT = """Conclusions appear in tool results as `[id:abc123] [timestamp] content`. To cite one, copy its id exactly, without the `id:` prefix. Only ids from conclusion results can be cited: message results have no ids, and invented ids are discarded."""

# Ordered as the section renders them; tools not offered to the agent are skipped.
_TOOL_COVERAGE: tuple[tuple[str, str], ...] = (
    ("search_memory", "conclusions, ranked by meaning"),
    ("get_recent_observations", "the newest conclusions, most recent first"),
    (
        "search_messages",
        "raw conversation messages, ranked by meaning; results carry no ids",
    ),
    ("grep_messages", "raw conversation messages, by exact text"),
    (
        "search_messages_temporal",
        "raw conversation messages by meaning, within a date range",
    ),
    ("get_messages_by_date_range", "raw conversation messages within a date range"),
)


def memory_layout_section(*, available_tools: Iterable[str], cite_ids: bool) -> str:
    """System-prompt section describing how a pair's memory is organized.

    The text depends only on its arguments, so it is identical across runs of
    the same agent and stays inside the cached part of the prompt.

    Args:
        available_tools: Names of the tools the agent is offered. Only those
            are described.
        cite_ids: Whether the agent sees conclusion ids and may cite them.
    """
    offered = set(available_tools)
    parts = ["## HOW MEMORY IS ORGANIZED", _LEVELS_TEXT]
    if cite_ids:
        parts.append(_IDS_TEXT)
    coverage = [f"- `{name}`: {covers}" for name, covers in _TOOL_COVERAGE if name in offered]
    if coverage:
        parts.append("What each search tool covers:\n" + "\n".join(coverage))
    return "\n\n".join(parts)


def format_memory_counts(summary: DocumentLevelSummary) -> str:
    """One line giving conclusion counts by level and the dates they span."""
    total = sum(summary.counts_by_level.values())
    if total == 0:
        return "Memory counts: no conclusions yet."
    counts = ", ".join(f"{summary.counts_by_level.get(level, 0)} {level}" for level in LEVEL_ORDER)
    span = ""
    if summary.earliest is not None and summary.latest is not None:
        span = f", created {summary.earliest.date().isoformat()}" + f" to {summary.latest.date().isoformat()}"
    return f"Memory counts: {counts} ({total} conclusions{span})."


async def load_memory_counts(ctx: PeopleContext, bank_id: str, *, observer: str, observed: str) -> str:
    """Query and format the counts line for one observer/observed pair."""
    summary = await summarize_documents_by_level(ctx, bank_id, observer=observer, observed=observed)
    return format_memory_counts(summary)


async def build_memory_brief(
    ctx: PeopleContext,
    bank_id: str,
    *,
    observer: str,
    observed: str,
    available_tools: Iterable[str],
    cite_ids: bool,
) -> tuple[str, str]:
    """The two pieces an agent starts a run with: the layout section for its system prompt and the counts
    line for its first user message.
    """
    layout = memory_layout_section(available_tools=available_tools, cite_ids=cite_ids)
    counts = await load_memory_counts(ctx, bank_id, observer=observer, observed=observed)
    return layout, counts
