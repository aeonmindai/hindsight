# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: tests/utils/test_memory_brief.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""Memory brief tests, ported from Honcho's tests/utils/test_memory_brief.py."""

import datetime

import pytest

from hindsight_api.people.conclusions import DocumentLevelSummary, delete_documents, summarize_documents_by_level
from hindsight_api.people.context import PeopleContext
from hindsight_api.people.memory_brief import (
    build_memory_brief,
    format_memory_counts,
    load_memory_counts,
    memory_layout_section,
)
from tests.people.people_helpers import make_doc, save


def test_layout_describes_only_offered_tools():
    section = memory_layout_section(available_tools=["search_memory", "search_messages"], cite_ids=False)
    assert "**contradiction**" in section
    assert "`search_memory`" in section
    assert "`search_messages`" in section
    assert "`get_recent_observations`" not in section
    assert "[id:" not in section


def test_layout_explains_ids_when_the_agent_can_cite():
    section = memory_layout_section(available_tools=[], cite_ids=True)
    assert "[id:abc123]" in section
    assert "What each search tool covers" not in section


def test_layout_is_stable_for_the_same_arguments():
    tools = ["get_recent_observations", "search_memory"]
    assert memory_layout_section(available_tools=tools, cite_ids=True) == memory_layout_section(
        available_tools=reversed(tools), cite_ids=True
    )


def test_counts_line_lists_every_level_in_order():
    summary = DocumentLevelSummary(
        counts_by_level={"deductive": 3, "explicit": 10},
        earliest=datetime.datetime(2026, 3, 2, 8, tzinfo=datetime.UTC),
        latest=datetime.datetime(2026, 10, 13, 22, tzinfo=datetime.UTC),
    )
    assert format_memory_counts(summary) == (
        "Memory counts: 10 explicit, 3 deductive, 0 inductive, 0 contradiction "
        + "(13 conclusions, created 2026-03-02 to 2026-10-13)."
    )


def test_counts_line_for_empty_memory():
    summary = DocumentLevelSummary(counts_by_level={}, earliest=None, latest=None)
    assert format_memory_counts(summary) == "Memory counts: no conclusions yet."


@pytest.mark.asyncio
async def test_summary_counts_live_conclusions_per_pair(ctx: PeopleContext, bank_id: str):
    first = await save(
        ctx,
        bank_id,
        [
            await make_doc(ctx, "likes tea"),
            await make_doc(ctx, "drinks tea daily"),
            await make_doc(ctx, "hates tea"),
            await make_doc(ctx, "is a tea drinker", level="deductive", source_ids=["a" * 21]),
        ],
    )
    hates = next(c for c in first.created_conclusions if c.content == "hates tea")
    await delete_documents(ctx, bank_id, [hates.id], observer="alice", observed="alice")
    # a different pair is not counted
    await save(ctx, bank_id, [await make_doc(ctx, "likes bob")], observer="bob", observed="alice")

    summary = await summarize_documents_by_level(ctx, bank_id, observer="alice", observed="alice")
    assert summary.counts_by_level == {"explicit": 2, "deductive": 1}
    assert summary.earliest is not None and summary.latest is not None

    line = await load_memory_counts(ctx, bank_id, observer="alice", observed="alice")
    assert line.startswith("Memory counts: 2 explicit, 1 deductive,")


@pytest.mark.asyncio
async def test_build_memory_brief_gives_layout_and_counts(ctx: PeopleContext, bank_id: str):
    layout, counts = await build_memory_brief(
        ctx, bank_id, observer="alice", observed="alice", available_tools=["search_memory"], cite_ids=True
    )
    assert "## HOW MEMORY IS ORGANIZED" in layout
    assert "`search_memory`" in layout
    assert counts == "Memory counts: no conclusions yet."
