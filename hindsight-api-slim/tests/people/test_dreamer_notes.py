# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: tests/dreamer/test_notes.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""Working notes: the update_notes tool, expiry, caps and loading (ported from Honcho's tests/dreamer/test_notes.py)."""

import datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api.people.context import PeopleContext
from hindsight_api.people.dreamer import collection_state
from hindsight_api.people.dreamer.settings import DREAM
from hindsight_api.people.dreamer.specialists import DeductionSpecialist
from hindsight_api.people.dreamer.tool_loop import ToolLoopResult, ToolResult
from hindsight_api.people.dreamer.working_memory import (
    EPOCH_KEY,
    NOTES_KEY,
    DreamState,
    LoadedNotes,
    NoteEntry,
    NotesCursor,
    _store_entries,
    apply_note_operations,
    entry_expired,
    format_notes,
    load_map_entries,
    load_notes_for_pass,
    make_update_notes_handler,
    parse_entries,
)
from tests.people.dialectic_support import bare_context
from tests.people.dreamer_support import padded_id, seed_rows, set_state, stored_state

NOW = datetime.datetime(2026, 10, 14, tzinfo=datetime.UTC)


def _entry(entry_id: str, kind: str, created_dream: int = 1) -> NoteEntry:
    return NoteEntry.model_validate(
        {
            "id": entry_id,
            "kind": kind,
            "specialist": "deduction",
            "text": f"text {entry_id}",
            "created_dream": created_dream,
            "updated_at": NOW,
        }
    )


def _apply(entries: list[NoteEntry], operations: Any, *, specialist: str = "deduction"):
    return apply_note_operations(
        entries,
        operations,
        specialist=specialist,
        run_id="r9",
        dream_number=4,
        now=NOW,
    )


def test_operations_add_replace_remove_one_entry_each():
    entries = [_entry("n1", "open"), _entry("n2", "map")]
    updated, edit = _apply(
        entries,
        [
            {"op": "add", "kind": "pending", "text": "write X", "ids": ["id:a1", "b2"]},
            {"op": "replace", "entry_id": "n1", "text": "still open"},
            {"op": "remove", "entry_id": "n2"},
        ],
    )
    assert [e.id for e in updated] == ["n1", "n3"]
    added = updated[1]
    assert (added.kind, added.ids, added.created_dream, added.specialist) == ("pending", ["a1", "b2"], 4, "deduction")
    assert updated[0].text == "still open"
    assert (edit.added, edit.replaced, edit.removed) == (["n3"], ["n1"], ["n2"])
    # The input is not mutated.
    assert entries[0].text == "text n1"


@pytest.mark.parametrize(
    "operations",
    [
        [],
        [{"op": "rewrite_all"}],
        [{"op": "add", "kind": "fact", "text": "x"}],
        [{"op": "add", "kind": "open", "text": "  "}],
        [{"op": "replace", "entry_id": "n9", "text": "x"}],
        [{"op": "remove", "entry_id": "n9"}],
        # A bad operation after a good one rejects the whole call.
        [{"op": "add", "kind": "open", "text": "ok"}, {"op": "remove"}],
    ],
)
def test_malformed_operations_change_nothing(operations: Any):
    with pytest.raises(ValueError):
        _apply([_entry("n1", "open")], operations)


@pytest.mark.parametrize(
    ("kind", "created", "dream", "expired"),
    [
        ("pending", 1, 3, False),
        ("pending", 1, 4, True),
        ("covered", 1, 5, False),
        ("covered", 1, 6, True),
        ("open", 1, 100, False),
        ("map", 1, 100, False),
    ],
)
def test_expiry_by_kind(kind: str, created: int, dream: int, expired: bool):
    assert entry_expired(_entry("n1", kind, created), dream) is expired


def _tool_ctx(ctx: PeopleContext, bank_id: str) -> SimpleNamespace:
    """What a tool handler is given: the people context, the bank and the pair."""
    return SimpleNamespace(pctx=ctx, bank_id=bank_id, observer="ann", observed="ann")


async def _seed_notes(ctx: PeopleContext, bank_id: str, notes: dict[str, Any]) -> SimpleNamespace:
    await set_state(ctx, bank_id, "ann", "ann", {NOTES_KEY: notes})
    return _tool_ctx(ctx, bank_id)


async def _stored(ctx: PeopleContext, bank_id: str) -> dict[str, Any]:
    return dict((await stored_state(ctx, bank_id))[NOTES_KEY])


async def test_handler_persists_drops_expired_and_keeps_the_cursor(ctx: PeopleContext, bank_id: str):
    cursor = {"dreams": 3, "through_id": None}
    tool_ctx = await _seed_notes(
        ctx,
        bank_id,
        {
            "cursor": cursor,
            "entries": [
                _entry("n1", "pending", created_dream=1).model_dump(mode="json"),
                _entry("n2", "map", created_dream=1).model_dump(mode="json"),
            ],
        },
    )
    handle = make_update_notes_handler(specialist="deduction", run_id="r1", dream_number=4)
    result = await handle(tool_ctx, {"operations": [{"op": "add", "kind": "open", "text": "check employer"}]})
    assert isinstance(result, ToolResult)
    assert result.metadata == {
        "notes_added": 1,
        "notes_replaced": 0,
        "notes_removed": 0,
        "pending_resolved": 0,
    }

    stored = await _stored(ctx, bank_id)
    # n1 (pending, created in dream 1) expired by dream 4; the cursor survives.
    assert [e.id for e in parse_entries(stored)] == ["n2", "n3"]
    assert stored["cursor"] == cursor


async def test_handler_rejects_writes_over_the_cap(ctx: PeopleContext, bank_id: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(DREAM, "NOTES_MAX_TOKENS", 100)
    tool_ctx = await _seed_notes(ctx, bank_id, {})
    handle = make_update_notes_handler(specialist="deduction", run_id="r1", dream_number=1)
    result = await handle(tool_ctx, {"operations": [{"op": "add", "kind": "map", "text": "word " * 200}]})
    assert isinstance(result, str) and result.startswith("ERROR:")
    assert "over the cap of 50" in result
    assert parse_entries(await _stored(ctx, bank_id)) == []


def _sized(entry_id: str, specialist: str, words: int, minute: int) -> dict[str, Any]:
    return NoteEntry(
        id=entry_id,
        kind="open",
        specialist=specialist,
        text="word " * words,
        created_dream=1,
        updated_at=NOW + datetime.timedelta(minutes=minute),
    ).model_dump(mode="json")


async def test_cap_is_shared_per_specialist_and_lets_edits_shrink(
    ctx: PeopleContext, bank_id: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(DREAM, "NOTES_MAX_TOKENS", 100)
    # Deduction is over its share, as after the cap is lowered.
    tool_ctx = await _seed_notes(
        ctx, bank_id, {"entries": [_sized("n1", "deduction", 40, 0), _sized("n2", "deduction", 40, 1)]}
    )
    deduction = make_update_notes_handler(specialist="deduction", run_id="r1", dream_number=1)
    induction = make_update_notes_handler(specialist="induction", run_id="r1", dream_number=1)

    grow = await deduction(tool_ctx, {"operations": [{"op": "add", "kind": "open", "text": "one more"}]})
    assert isinstance(grow, str) and grow.startswith("ERROR:")
    # Induction still has its own share.
    added = await induction(tool_ctx, {"operations": [{"op": "add", "kind": "open", "text": "a pattern"}]})
    assert isinstance(added, ToolResult)
    # Shrinking is allowed even while still over the share.
    shrink = await deduction(tool_ctx, {"operations": [{"op": "replace", "entry_id": "n1", "text": "short"}]})
    assert isinstance(shrink, ToolResult)
    assert [e.id for e in parse_entries(await _stored(ctx, bank_id))] == ["n1", "n2", "n3"]


async def test_shadow_mode_evicts_the_oldest_own_entries(
    ctx: PeopleContext, bank_id: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(DREAM, "NOTES_MAX_TOKENS", 100)
    monkeypatch.setattr(DREAM, "NOTES_MODE", "shadow")
    tool_ctx = await _seed_notes(
        ctx,
        bank_id,
        {
            "entries": [
                _sized("n1", "deduction", 12, 0),
                _sized("n2", "induction", 40, 1),
                _sized("n3", "deduction", 12, 2),
            ]
        },
    )
    handle = make_update_notes_handler(specialist="deduction", run_id="r1", dream_number=1)
    result = await handle(tool_ctx, {"operations": [{"op": "add", "kind": "open", "text": "word " * 12}]})
    assert isinstance(result, ToolResult)
    # n1 is deduction's oldest; induction's entry is never evicted.
    assert [e.id for e in parse_entries(await _stored(ctx, bank_id))] == ["n2", "n3", "n4"]


async def test_handler_reports_malformed_calls(ctx: PeopleContext, bank_id: str):
    tool_ctx = await _seed_notes(ctx, bank_id, {})
    handle = make_update_notes_handler(specialist="deduction", run_id="r1", dream_number=1)
    result = await handle(tool_ctx, {"operations": [{"op": "remove", "entry_id": "n5"}]})
    assert isinstance(result, str)
    assert result.startswith("ERROR: operation 1: no entry 'n5'")


async def _run(state: DreamState | None) -> tuple[Any, Any, Any]:
    loop = AsyncMock(return_value=ToolLoopResult(content="DONE", iterations=1, input_tokens=1, output_tokens=1))
    make_executor = AsyncMock(return_value=AsyncMock())
    with (
        patch("hindsight_api.people.dreamer.specialists.get_peer", new=AsyncMock()),
        patch("hindsight_api.people.dreamer.specialists.peer_card_crud.get_peer_card", new=AsyncMock()),
        patch("hindsight_api.people.dreamer.specialists.create_tool_executor", new=make_executor),
        patch("hindsight_api.people.dreamer.specialists.execute_tool_loop", new=loop),
    ):
        result = await DeductionSpecialist().run(
            bare_context(), bank_id="w", observer="alice", observed="alice", session_name=None, state=state
        )
    return loop.call_args.kwargs, make_executor.call_args.kwargs["handler_resolver"], result


@pytest.mark.parametrize("mode", ["shadow", "on"])
async def test_notes_mode_offers_the_tool(mode: str, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(DREAM, "NOTES_MODE", mode)
    kwargs, resolver, _ = await _run(DreamState(cursor=NotesCursor(dreams=2)))
    assert "update_notes" in [t["name"] for t in kwargs["tools"]]
    assert "## WORKING NOTES" in kwargs["messages"][0]["content"]
    assert resolver is not None and resolver("update_notes") is not None


async def test_notes_need_the_loaded_state(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(DREAM, "NOTES_MODE", "on")
    kwargs, resolver, _ = await _run(None)
    assert "update_notes" not in [t["name"] for t in kwargs["tools"]]
    assert resolver is None


async def test_notes_off_by_default():
    kwargs, _, _ = await _run(DreamState(cursor=NotesCursor()))
    assert "update_notes" not in [t["name"] for t in kwargs["tools"]]
    assert "WORKING NOTES" not in kwargs["messages"][0]["content"]


async def test_loading_prunes_expired_and_dangling_entries(ctx: PeopleContext, bank_id: str):
    live_id = padded_id("live")
    await seed_rows(ctx, bank_id, [("live", "explicit", NOW)])
    await set_state(
        ctx,
        bank_id,
        "ann",
        "ann",
        {
            NOTES_KEY: {
                "entries": [
                    _entry("n1", "pending", created_dream=1).model_dump(mode="json"),
                    {**_entry("n2", "open").model_dump(mode="json"), "ids": [padded_id("gone")]},
                    {**_entry("n3", "map").model_dump(mode="json"), "ids": [live_id]},
                    _entry("n4", "pending", created_dream=3).model_dump(mode="json"),
                ]
            }
        },
    )

    loaded = await load_notes_for_pass(ctx, bank_id, observer="ann", observed="ann", dream_number=4)
    assert [e.id for e in loaded.entries] == ["n3", "n4"]
    assert (loaded.pending_expired, loaded.dropped_for_missing_ids) == (1, 1)
    assert loaded.pending == 1 and loaded.tokens > 0
    # The pruned list is written back.
    assert [e.id for e in parse_entries(await _stored(ctx, bank_id))] == ["n3", "n4"]


def test_notes_section_puts_pending_first_and_asks_to_resolve():
    entries = [_entry("n1", "map"), _entry("n2", "pending")]
    text = format_notes(entries, specialist="deduction")
    assert text.index("[n2] pending (deduction)") < text.index("[n1] map (deduction)")
    assert "hints to check, not facts" in text
    assert "Resolve every pending entry marked (deduction)" in text
    # Another specialist's pending entries are shown but not its to resolve.
    assert "Resolve every pending" not in format_notes(entries, specialist="induction")
    assert "Resolve every pending" not in format_notes([_entry("n1", "map")], specialist="deduction")


async def test_dialectic_reads_only_map_entries(ctx: PeopleContext, bank_id: str):
    dead_map = {**_entry("n9", "map").model_dump(mode="json"), "ids": [padded_id("gone")]}
    await _seed_notes(
        ctx,
        bank_id,
        {
            "entries": [
                *(
                    _entry(f"n{i}", kind).model_dump(mode="json")
                    for i, kind in enumerate(["pending", "open", "covered", "map"])
                ),
                # Every conclusion it cited is gone, and no dream has pruned it.
                dead_map,
            ]
        },
    )
    entries = await load_map_entries(ctx, bank_id, observer="ann", observed="ann")
    assert [(e.id, e.kind) for e in entries] == [("n3", "map")]


@pytest.mark.parametrize(("mode", "loaded"), [("on", True), ("shadow", False)])
async def test_notes_are_loaded_only_when_on(mode: str, loaded: bool, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(DREAM, "NOTES_MODE", mode)
    load = AsyncMock(return_value=LoadedNotes(entries=[_entry("n1", "pending")], pending_expired=2))
    with patch("hindsight_api.people.dreamer.specialists.load_notes_for_pass", new=load):
        kwargs, _, result = await _run(DreamState(cursor=NotesCursor(dreams=6)))
    user = kwargs["messages"][1]["content"]
    assert ("## YOUR NOTES FROM EARLIER DREAMS" in user) is loaded
    assert load.await_count == (1 if loaded else 0)
    if loaded:
        assert load.await_args is not None
        assert load.await_args.kwargs["dream_number"] == 7
        assert (result.pending_loaded, result.pending_expired) == (1, 2)


async def test_a_failed_notes_load_runs_the_pass_without_them(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(DREAM, "NOTES_MODE", "on")
    load = AsyncMock(side_effect=RuntimeError("db down"))
    with patch("hindsight_api.people.dreamer.specialists.load_notes_for_pass", new=load):
        kwargs, _, _ = await _run(DreamState(cursor=NotesCursor(dreams=1)))
    assert load.await_count == 1
    assert "## YOUR NOTES FROM EARLIER DREAMS" not in kwargs["messages"][1]["content"]
    # The tool is still offered, so the pass can write notes.
    assert "update_notes" in [t["name"] for t in kwargs["tools"]]


@pytest.mark.parametrize("reset_after_read", [False, True])
async def test_notes_reset_mid_dream_refuses_writes(ctx: PeopleContext, bank_id: str, reset_after_read: bool):
    """A reset (scope removal) after the dream started, or between the handler's read and its write,
    must not be written over with the old notes."""
    tool_ctx = await _seed_notes(ctx, bank_id, {"entries": [_entry("n1", "map").model_dump(mode="json")]})
    handle = make_update_notes_handler(specialist="deduction", run_id="r1", dream_number=1, epoch=None)
    operations = {"operations": [{"op": "add", "kind": "open", "text": "x"}]}
    if reset_after_read:
        # The conditional write finds the epoch moved after the read.
        store = AsyncMock(return_value=False)
        with patch("hindsight_api.people.dreamer.working_memory._store_entries", new=store):
            result = await handle(tool_ctx, operations)
        assert store.await_count == 1
    else:
        await collection_state.merge_collection_internal_metadata_value(
            ctx, bank_id, "ann", "ann", NOTES_KEY, {"entries": [], EPOCH_KEY: "e2"}
        )
        result = await handle(tool_ctx, operations)

    assert isinstance(result, str) and "were reset during this dream" in result
    if not reset_after_read:
        assert parse_entries(await _stored(ctx, bank_id)) == []


async def test_store_is_refused_once_the_epoch_moved(ctx: PeopleContext, bank_id: str):
    await _seed_notes(ctx, bank_id, {"entries": [], EPOCH_KEY: "e2"})
    old = [_entry("n1", "map")]
    assert not await _store_entries(ctx, bank_id, "ann", "ann", old, epoch=None)
    assert parse_entries(await _stored(ctx, bank_id)) == []
    assert await _store_entries(ctx, bank_id, "ann", "ann", old, epoch="e2")
    assert [e.id for e in parse_entries(await _stored(ctx, bank_id))] == ["n1"]


async def test_store_with_no_epoch_creates_the_pair_state(ctx: PeopleContext, bank_id: str):
    """A pair that never dreamed has no state row yet: the first note write creates it."""
    assert await _store_entries(ctx, bank_id, "ann", "ann", [_entry("n1", "map")], epoch=None)
    assert [e.id for e in parse_entries(await _stored(ctx, bank_id))] == ["n1"]


def test_replace_updates_ids_only_when_given():
    entries = [_entry("n1", "open").model_copy(update={"ids": ["a1"]})]
    kept, _ = _apply(entries, [{"op": "replace", "entry_id": "n1", "text": "t"}])
    assert kept[0].ids == ["a1"]
    swapped, _ = _apply(entries, [{"op": "replace", "entry_id": "n1", "text": "t", "ids": ["id:b2"]}])
    assert swapped[0].ids == ["b2"]


@pytest.mark.parametrize(
    "operation",
    [
        {"op": "replace", "entry_id": "n1", "text": "mine now"},
        {"op": "remove", "entry_id": "n1"},
    ],
)
def test_specialists_cannot_edit_each_others_entries(operation: dict[str, Any]):
    with pytest.raises(ValueError, match="belongs to deduction"):
        _apply([_entry("n1", "covered")], [operation], specialist="induction")


async def test_loading_strips_dead_ids_but_keeps_the_entry(ctx: PeopleContext, bank_id: str):
    live_id, gone_id = padded_id("live"), padded_id("gone")
    await seed_rows(ctx, bank_id, [("live", "explicit", NOW)])
    await _seed_notes(
        ctx, bank_id, {"entries": [{**_entry("n1", "map").model_dump(mode="json"), "ids": [live_id, gone_id]}]}
    )

    loaded = await load_notes_for_pass(ctx, bank_id, observer="ann", observed="ann", dream_number=2)
    assert [(e.id, e.ids) for e in loaded.entries] == [("n1", [live_id])]
    assert (loaded.ids_stripped, loaded.dropped_for_missing_ids) == (1, 0)
    assert parse_entries(await _stored(ctx, bank_id))[0].ids == [live_id]
