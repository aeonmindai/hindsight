# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: tests/dreamer/test_work_list.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""The since-last-dream work list and the cursor that pages through it (ported from Honcho's tests/dreamer/test_work_list.py)."""

import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from hindsight_api.people.context import PeopleContext
from hindsight_api.people.dreamer import collection_state
from hindsight_api.people.dreamer.settings import DREAM
from hindsight_api.people.dreamer.working_memory import (
    MAX_CURSOR_HOLDS,
    NOTES_KEY,
    DreamState,
    NotesCursor,
    WorkItem,
    WorkList,
    advance_cursor,
    clear_position,
    format_work_list,
    load_dream_state,
    plan_cursor_move,
)
from tests.people.dialectic_support import bare_context
from tests.people.dreamer_support import padded_id as _id
from tests.people.dreamer_support import seed_rows, stored_state

T0 = datetime.datetime(2026, 10, 1, 12, tzinfo=datetime.UTC)


async def load_work_list(ctx: PeopleContext, bank: str, *, observer: str, observed: str):
    return await load_dream_state(ctx, bank, observer=observer, observed=observed, include_work_list=True)


async def _advance(ctx: PeopleContext, bank: str, state: DreamState | None, run_id: str, *, any_ok: bool, all_ok: bool):
    await advance_cursor(
        ctx,
        bank,
        observer="ann",
        observed="ann",
        state=state,
        run_id=run_id,
        any_specialist_succeeded=any_ok,
        all_specialists_succeeded=all_ok,
    )


async def test_backlog_is_worked_through_over_consecutive_dreams(
    ctx: PeopleContext, bank_id: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(DREAM, "BRIEF_MAX_NEW_CONCLUSIONS", 2)
    # b and c share a timestamp: paging must neither skip nor repeat either.
    await seed_rows(
        ctx,
        bank_id,
        [
            ("a", "explicit", T0),
            ("b", "explicit", T0 + datetime.timedelta(minutes=1)),
            ("c", "explicit", T0 + datetime.timedelta(minutes=1)),
            ("d", "deductive", T0 + datetime.timedelta(minutes=2)),
        ],
        {"dream": {"last_dream_at": "2026-09-01T00:00:00+00:00"}},
    )

    first_state = await load_work_list(ctx, bank_id, observer="ann", observed="ann")
    first = first_state.work_list
    assert first is not None
    assert [i.id for i in first.items] == [_id("a"), _id("b")]
    assert first.remaining == 1
    await _advance(ctx, bank_id, first_state, "r1", any_ok=True, all_ok=True)

    second_state = await load_work_list(ctx, bank_id, observer="ann", observed="ann")
    second = second_state.work_list
    assert second is not None
    assert [i.id for i in second.items] == [_id("c")]
    assert second.remaining == 0
    await _advance(ctx, bank_id, second_state, "r2", any_ok=True, all_ok=True)

    metadata = await stored_state(ctx, bank_id)
    cursor = NotesCursor.model_validate(metadata[NOTES_KEY]["cursor"])
    assert (cursor.through_id, cursor.dreams, cursor.last_run_id) == (_id("c"), 2, "r2")
    # The scheduler's key is untouched.
    assert metadata["dream"] == {"last_dream_at": "2026-09-01T00:00:00+00:00"}

    third = (await load_work_list(ctx, bank_id, observer="ann", observed="ann")).work_list
    assert third is not None and third.items == [] and third.remaining == 0


async def test_page_stops_at_the_token_cap_but_shows_one(
    ctx: PeopleContext, bank_id: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(DREAM, "BRIEF_MAX_NEW_CONCLUSION_TOKENS", 1)
    await seed_rows(
        ctx,
        bank_id,
        [
            ("a", "explicit", T0),
            ("b", "explicit", T0 + datetime.timedelta(minutes=1)),
            ("c", "explicit", T0 + datetime.timedelta(minutes=2)),
        ],
    )
    work_list = (await load_work_list(ctx, bank_id, observer="ann", observed="ann")).work_list
    assert work_list is not None
    # The first is over the cap on its own, and is still shown.
    assert [i.id for i in work_list.items] == [_id("a")]
    assert work_list.remaining == 2


async def test_remaining_never_goes_negative():
    """A commit between the count and the page can make the page the longer."""
    docs = [SimpleNamespace(id=f"x{i}", content="c", created_at=T0, internal_metadata={}) for i in range(2)]
    with (
        patch(
            "hindsight_api.people.dreamer.working_memory.collection_state.get_collection_internal_metadata_value",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "hindsight_api.people.dreamer.working_memory.conclusions_crud.query_explicit_documents_after",
            new=AsyncMock(return_value=(docs, 1)),
        ),
    ):
        state = await load_work_list(bare_context(), "w", observer="p", observed="p")
    assert state.work_list is not None
    assert len(state.work_list.items) == 2 and state.work_list.remaining == 0


@pytest.mark.parametrize("any_succeeded", [True, False])
async def test_failed_specialist_is_shown_the_same_list_again(ctx: PeopleContext, bank_id: str, any_succeeded: bool):
    await seed_rows(
        ctx,
        bank_id,
        [("a", "explicit", T0), ("b", "explicit", T0 + datetime.timedelta(minutes=1))],
        {"dream": {"last_dream_at": "2026-09-01T00:00:00+00:00"}},
    )
    state = await load_work_list(ctx, bank_id, observer="ann", observed="ann")
    await _advance(ctx, bank_id, state, "r1", any_ok=any_succeeded, all_ok=False)
    # The first dream had no position to hold; the scheduler's bound moving past the list must not lose
    # it. This is what `process_dream` does to the scheduler's key at the end of every dream.
    await collection_state.update_collection_internal_metadata(
        ctx, bank_id, "ann", "ann", {"dream": {"last_dream_at": (T0 + datetime.timedelta(hours=1)).isoformat()}}
    )

    again = await load_work_list(ctx, bank_id, observer="ann", observed="ann")
    assert again.work_list is not None
    assert [i.id for i in again.work_list.items] == [_id("a"), _id("b")]
    # A dream where a specialist ran still counts, so note expiry keeps moving.
    assert again.cursor.dreams == (1 if any_succeeded else 0)
    assert again.cursor.holds == (1 if any_succeeded else 0)


async def test_position_moves_after_repeated_holds(ctx: PeopleContext, bank_id: str):
    await seed_rows(ctx, bank_id, [("a", "explicit", T0), ("b", "explicit", T0 + datetime.timedelta(minutes=1))])
    for dream in range(1, MAX_CURSOR_HOLDS + 1):
        state = await load_work_list(ctx, bank_id, observer="ann", observed="ann")
        assert state.work_list is not None
        assert [i.id for i in state.work_list.items] == [_id("a"), _id("b")]
        await _advance(ctx, bank_id, state, f"r{dream}", any_ok=True, all_ok=False)

    after = await load_work_list(ctx, bank_id, observer="ann", observed="ann")
    assert after.work_list is not None and after.work_list.items == []
    assert (after.cursor.through_id, after.cursor.holds) == (_id("b"), 0)


def _state(items: int, remaining: int, *, holds: int = 0) -> DreamState:
    return DreamState(
        cursor=NotesCursor(holds=holds),
        work_list=WorkList(
            items=[WorkItem(id=f"x{i}", content="c", created_at=T0) for i in range(items)], remaining=remaining
        ),
    )


@pytest.mark.parametrize(
    ("all_succeeded", "holds", "backlog"),
    [(True, 0, 5), (False, 0, 7), (False, MAX_CURSOR_HOLDS - 1, 5)],
)
def test_backlog_counts_what_the_next_dream_must_show(all_succeeded: bool, holds: int, backlog: int):
    _, due = plan_cursor_move(
        _state(2, 5, holds=holds),
        any_specialist_succeeded=True,
        all_specialists_succeeded=all_succeeded,
    )
    assert due == backlog


def test_clear_position_keeps_entries_and_count():
    stored = {
        "entries": [{"id": "n1"}],
        "cursor": {"through_created_at": T0.isoformat(), "through_id": "x", "dreams": 3},
    }
    cleared = clear_position(stored)
    assert cleared is not None
    assert cleared["entries"] == [{"id": "n1"}]
    cursor = NotesCursor.model_validate(cleared["cursor"])
    assert (cursor.position(), cursor.dreams) == (None, 3)
    assert clear_position(cleared) is None
    assert clear_position(None) is None


async def test_brief_off_clears_a_stale_position(ctx: PeopleContext, bank_id: str):
    last_dream_at = T0 - datetime.timedelta(days=1)
    await seed_rows(
        ctx,
        bank_id,
        [("old", "explicit", T0 - datetime.timedelta(days=2)), ("new", "explicit", T0)],
        {
            "dream": {"last_dream_at": last_dream_at.isoformat()},
            NOTES_KEY: {
                "cursor": {
                    "through_created_at": (T0 - datetime.timedelta(days=3)).isoformat(),
                    "through_id": _id("older"),
                    "dreams": 4,
                }
            },
        },
    )
    state = await load_dream_state(ctx, bank_id, observer="ann", observed="ann", include_work_list=False)
    await _advance(ctx, bank_id, state, "r1", any_ok=True, all_ok=True)

    # Turning the brief back on starts from the last dream, not the old position.
    back_on = await load_work_list(ctx, bank_id, observer="ann", observed="ann")
    assert back_on.cursor.position() is None
    assert back_on.cursor.dreams == 5
    assert back_on.work_list is not None
    assert [i.id for i in back_on.work_list.items] == [_id("new")]


async def test_first_list_starts_at_the_last_dream(ctx: PeopleContext, bank_id: str):
    await seed_rows(
        ctx,
        bank_id,
        [("old", "explicit", T0 - datetime.timedelta(days=2)), ("new", "explicit", T0)],
        {"dream": {"last_dream_at": (T0 - datetime.timedelta(days=1)).isoformat()}},
    )
    work_list = (await load_work_list(ctx, bank_id, observer="ann", observed="ann")).work_list
    assert work_list is not None
    assert [i.id for i in work_list.items] == [_id("new")]


async def test_cursor_merge_keeps_other_note_keys(ctx: PeopleContext, bank_id: str):
    await seed_rows(ctx, bank_id, [], {NOTES_KEY: {"entries": [{"id": "n1"}]}})
    await _advance(ctx, bank_id, None, "r1", any_ok=True, all_ok=True)
    notes = (await stored_state(ctx, bank_id))[NOTES_KEY]
    assert notes["entries"] == [{"id": "n1"}]
    assert notes["cursor"]["dreams"] == 1
    assert notes["cursor"]["through_id"] is None


def test_format_lists_items_and_the_remainder():
    text = format_work_list(WorkList(items=[WorkItem(id="a1", content="likes tea", created_at=T0)], remaining=3))
    assert "- [id:a1] [2026-10-01] likes tea" in text
    assert "3 more are not shown" in text


def test_format_shows_when_the_message_was_sent():
    text = format_work_list(
        WorkList(
            items=[
                WorkItem(
                    id="a1",
                    content="likes tea",
                    created_at=T0,
                    stated_at=datetime.datetime(2024, 3, 5, tzinfo=datetime.UTC),
                )
            ],
            remaining=0,
        )
    )
    assert "- [id:a1] [2024-03-05] likes tea" in text


def test_format_empty_list():
    text = format_work_list(WorkList(items=[], remaining=0))
    assert "No explicit conclusions have been created" in text


async def test_query_ignores_deleted_and_other_levels(ctx: PeopleContext, bank_id: str):
    from hindsight_api.people import conclusions as conclusions_crud

    await seed_rows(ctx, bank_id, [("x", "explicit", T0), ("y", "inductive", T0)])
    async with ctx.conn() as conn:
        await conn.execute(f"UPDATE {ctx.table('conclusions')} SET deleted_at = $2 WHERE id = $1", _id("x"), T0)
    docs, total = await conclusions_crud.query_explicit_documents_after(
        ctx, bank_id, observer="ann", observed="ann", after=None, since=None, limit=10
    )
    assert list(docs) == [] and total == 0
