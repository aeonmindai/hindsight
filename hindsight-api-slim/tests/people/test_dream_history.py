# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: tests/crud/test_dream_history.py
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: tests/crud/test_dream_history_lock_order.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""Representation History: recording a dream's changes and reading them back (ported from Honcho's tests/crud/test_dream_history.py)."""

import asyncio
import datetime

import pytest

from hindsight_api.people import conclusions as conclusions_crud
from hindsight_api.people import peer_card as peer_card_crud
from hindsight_api.people.context import PeopleContext
from hindsight_api.people.dream_history import (
    DreamHistoryTarget,
    close_run,
    count_runs,
    delete_bank_history,
    delete_session_history,
    erase_conclusion_history,
    get_run_detail,
    list_runs,
    open_run,
    prune_dream_history,
)
from hindsight_api.people.dreamer import settings as dream_settings
from hindsight_api.people.dreamer.agent_tools import delete_observations_with_history, save_documents_with_history
from hindsight_api.people.dreamer.schemas import DreamPeerCardChange
from hindsight_api.people.exceptions import ResourceNotFoundException
from hindsight_api.people.ids import generate_nanoid
from hindsight_api.people.peers import get_or_create_peers
from tests.people.dreamer_support import (
    PARALLEL,
    age_run,
    deductive,
    open_target,
    scalar,
    seed,
    tombstone,
)
from tests.people.people_helpers import add_session, make_doc


async def _hard_delete(ctx: PeopleContext, doc_id: str) -> None:
    """What the reconciler does to a soft-deleted conclusion a few minutes later."""
    async with ctx.conn() as conn:
        await conn.execute(f"DELETE FROM {ctx.table('conclusions')} WHERE id = $1", doc_id)


async def _remove(ctx: PeopleContext, bank_id: str, ids: list[str], history: DreamHistoryTarget | None) -> None:
    await delete_observations_with_history(ctx, bank_id, ids, observer="ann", observed="ann", history=history)


async def _create(ctx: PeopleContext, bank_id: str, doc, history: DreamHistoryTarget | None, dedup: bool = False):
    return await save_documents_with_history(
        ctx, bank_id, [doc], observer="ann", observed="ann", deduplicate=dedup, history=history
    )


async def test_run_lifecycle(ctx: PeopleContext, bank_id: str):
    target = await open_target(ctx, bank_id)

    detail = await get_run_detail(ctx, bank_id, target.run_id)
    assert detail.status == "running"
    assert detail.finished_at is None

    await close_run(ctx, bank_id, target.run_id, succeeded=False)
    detail = await get_run_detail(ctx, bank_id, target.run_id)
    assert detail.status == "failed"
    assert detail.finished_at is not None

    # Closing is one-way: a second close doesn't rewrite the outcome.
    await close_run(ctx, bank_id, target.run_id, succeeded=True)
    detail = await get_run_detail(ctx, bank_id, target.run_id)
    assert detail.status == "failed"


async def test_history_disabled_opens_nothing(ctx: PeopleContext, bank_id: str, monkeypatch: pytest.MonkeyPatch):
    await get_or_create_peers(ctx, bank_id, ["ann"])
    monkeypatch.setattr(dream_settings.DREAM, "HISTORY_ENABLED", False)

    target = await open_run(
        ctx,
        run_id=generate_nanoid(),
        bank_id=bank_id,
        observer="ann",
        observed="ann",
        dream_type=None,
        trigger_reason=None,
    )

    assert target is None
    assert await scalar(ctx, f"SELECT count(*) FROM {ctx.table('dream_runs')} WHERE bank_id = $1", bank_id) == 0


async def test_open_run_failure_never_stops_the_dream(ctx: PeopleContext):
    # No such bank: the insert fails its foreign key, and the dream runs unrecorded.
    target = await open_run(
        ctx,
        run_id=generate_nanoid(),
        bank_id="no-such-bank",
        observer="ann",
        observed="ann",
        dream_type="omni",
        trigger_reason="manual",
    )
    assert target is None


async def test_created_conclusion_keeps_sources_readable_after_they_are_removed(ctx: PeopleContext, bank_id: str):
    _session_name, (coffee_id, nights_id) = await seed(ctx, bank_id)
    first = await open_target(ctx, bank_id)

    await _create(
        ctx,
        bank_id,
        await deductive(ctx, "Ann relies on caffeine to work nights", [coffee_id, nights_id]),
        first,
    )

    # A later dream removes one of the sources, and the reconciler hard-deletes it.
    second = await open_target(ctx, bank_id)
    await _remove(ctx, bank_id, [coffee_id], second)
    await _hard_delete(ctx, coffee_id)

    detail = await get_run_detail(ctx, bank_id, first.run_id)
    assert detail.created_count == 1
    [created] = detail.created
    assert created.conclusion.content == "Ann relies on caffeine to work nights"
    assert created.conclusion.status == "current"
    assert created.specialist == "deduction"
    coffee, nights = created.sources
    assert (coffee.id, coffee.status, coffee.removed_in_run) == (coffee_id, "removed", second.run_id)
    assert coffee.content == "Ann drinks coffee daily"
    assert (nights.status, nights.content) == ("current", "Ann works nights")

    removal = await get_run_detail(ctx, bank_id, second.run_id)
    assert removal.removed_count == 1
    [removed] = removal.removed
    assert (removed.id, removed.content, removed.level) == (coffee_id, "Ann drinks coffee daily", "explicit")


async def test_semantic_replacement_is_recorded_as_a_removal(ctx: PeopleContext, bank_id: str):
    # No seeded conclusions: every test embedding here is parallel, so any of them would be an equally
    # close duplicate candidate.
    session_name = generate_nanoid()
    await add_session(ctx, bank_id, session_name, ["ann"])
    first = await _create(
        ctx,
        bank_id,
        await make_doc(ctx, "Ann likes coffee", session=session_name, embedding=PARALLEL, message_ids=[1]),
        None,
    )
    [old] = first.created_conclusions
    target = await open_target(ctx, bank_id)

    result = await _create(
        ctx,
        bank_id,
        await make_doc(ctx, "Ann likes coffee and tea", session=session_name, embedding=PARALLEL, message_ids=[1]),
        target,
        dedup=True,
    )

    assert result.semantic_dup_replaced_count == 1
    [new] = result.created_conclusions
    detail = await get_run_detail(ctx, bank_id, target.run_id)
    assert [c.conclusion.id for c in detail.created] == [new.id]
    [removed] = detail.removed
    assert (removed.id, removed.content, removed.replaced_by) == (old.id, "Ann likes coffee", new.id)
    assert (detail.created_count, detail.removed_count) == (1, 1)


async def test_peer_card_change_records_before_and_after(ctx: PeopleContext, bank_id: str):
    target = await open_target(ctx, bank_id)

    await peer_card_crud.set_peer_card(ctx, bank_id, ["IDENTITY: Name: Ann"], "ann", "ann", history=target)
    await peer_card_crud.set_peer_card(
        ctx, bank_id, ["IDENTITY: Name: Ann", "ATTRIBUTE: Works nights"], "ann", "ann", history=target
    )

    detail = await get_run_detail(ctx, bank_id, target.run_id)
    assert detail.peer_card_updated
    assert detail.peer_card == DreamPeerCardChange(
        before=None, after=["IDENTITY: Name: Ann", "ATTRIBUTE: Works nights"]
    )


async def test_writes_without_history_record_nothing(ctx: PeopleContext, bank_id: str):
    _session_name, (coffee_id, _nights_id) = await seed(ctx, bank_id)

    await _remove(ctx, bank_id, [coffee_id], None)
    await peer_card_crud.set_peer_card(ctx, bank_id, ["IDENTITY: Name: Ann"], "ann", "ann")

    assert await scalar(ctx, f"SELECT count(*) FROM {ctx.table('dream_run_changes')} WHERE bank_id = $1", bank_id) == 0


async def test_deleting_the_session_deletes_its_removed_copies(ctx: PeopleContext, bank_id: str):
    session_name, (coffee_id, _nights_id) = await seed(ctx, bank_id)
    target = await open_target(ctx, bank_id)
    await _remove(ctx, bank_id, [coffee_id], target)

    async with ctx.conn() as conn, conn.transaction():
        await delete_session_history(conn, bank_id, session_name)

    detail = await get_run_detail(ctx, bank_id, target.run_id)
    assert detail.removed == []


async def test_a_conclusion_deleted_through_the_api_is_unavailable(ctx: PeopleContext, bank_id: str):
    _session_name, (coffee_id, nights_id) = await seed(ctx, bank_id)
    target = await open_target(ctx, bank_id)
    result = await _create(
        ctx, bank_id, await deductive(ctx, "Ann relies on caffeine to work nights", [coffee_id, nights_id]), target
    )
    [created_row] = result.created_conclusions

    await conclusions_crud.delete_document_by_id(ctx, bank_id, created_row.id)

    detail = await get_run_detail(ctx, bank_id, target.run_id)
    [created] = detail.created
    assert created.conclusion.status == "unavailable"
    assert created.conclusion.content is None


async def test_deleting_a_restored_conclusion_erases_the_copy_a_dream_kept(ctx: PeopleContext, bank_id: str):
    _session_name, (coffee_id, nights_id) = await seed(ctx, bank_id)
    first = await open_target(ctx, bank_id)
    await _create(
        ctx, bank_id, await deductive(ctx, "Ann relies on caffeine to work nights", [coffee_id, nights_id]), first
    )
    second = await open_target(ctx, bank_id)
    await _remove(ctx, bank_id, [coffee_id], second)
    # Something restores the same conclusion id.
    async with ctx.conn() as conn:
        await conn.execute(f"UPDATE {ctx.table('conclusions')} SET deleted_at = NULL WHERE id = $1", coffee_id)

    await conclusions_crud.delete_document_by_id(ctx, bank_id, coffee_id)
    async with ctx.conn() as conn, conn.transaction():
        assert await erase_conclusion_history(conn, bank_id, coffee_id) == 1

    removal = await get_run_detail(ctx, bank_id, second.run_id)
    assert removal.removed == []
    detail = await get_run_detail(ctx, bank_id, first.run_id)
    coffee = next(s for s in detail.created[0].sources if s.id == coffee_id)
    assert (coffee.status, coffee.content) == ("unavailable", None)


async def test_deleting_the_bank_history_deletes_runs_and_changes(ctx: PeopleContext, bank_id: str):
    _session_name, (coffee_id, _nights_id) = await seed(ctx, bank_id)
    target = await open_target(ctx, bank_id)
    await _remove(ctx, bank_id, [coffee_id], target)

    async with ctx.conn() as conn, conn.transaction():
        await delete_bank_history(conn, bank_id)

    assert await scalar(ctx, f"SELECT count(*) FROM {ctx.table('dream_runs')} WHERE bank_id = $1", bank_id) == 0
    assert await scalar(ctx, f"SELECT count(*) FROM {ctx.table('dream_run_changes')} WHERE bank_id = $1", bank_id) == 0


async def test_prune_deletes_old_runs_and_interrupts_stale_ones(ctx: PeopleContext, bank_id: str):
    old = await open_target(ctx, bank_id)
    stale = await open_target(ctx, bank_id)
    fresh = await open_target(ctx, bank_id)
    retention = datetime.timedelta(days=dream_settings.DREAM.HISTORY_RETENTION_DAYS)
    await age_run(ctx, bank_id, old.run_id, retention + datetime.timedelta(hours=1))
    await age_run(ctx, bank_id, stale.run_id, datetime.timedelta(days=2))

    # The prune is global: a test running beside this one may delete `old` first, so the count is not asserted.
    await prune_dream_history(ctx)

    with pytest.raises(ResourceNotFoundException):
        await get_run_detail(ctx, bank_id, old.run_id)
    assert (await get_run_detail(ctx, bank_id, stale.run_id)).status == "interrupted"
    assert (await get_run_detail(ctx, bank_id, fresh.run_id)).status == "running"


async def _age_past_retention(ctx: PeopleContext, bank_id: str, run_id: str) -> None:
    retention = datetime.timedelta(days=dream_settings.DREAM.HISTORY_RETENTION_DAYS)
    await age_run(ctx, bank_id, run_id, retention + datetime.timedelta(hours=1))


async def test_prune_keeps_what_a_dream_created_while_it_exists(ctx: PeopleContext, bank_id: str):
    _session_name, (coffee_id, nights_id) = await seed(ctx, bank_id)
    target = await open_target(ctx, bank_id)
    await _create(
        ctx, bank_id, await deductive(ctx, "Ann relies on caffeine to work nights", [coffee_id, nights_id]), target
    )
    await _remove(ctx, bank_id, [coffee_id], target)
    await peer_card_crud.set_peer_card(ctx, bank_id, ["IDENTITY: Name: Ann"], "ann", "ann", history=target)
    await _age_past_retention(ctx, bank_id, target.run_id)

    await prune_dream_history(ctx)

    detail = await get_run_detail(ctx, bank_id, target.run_id)
    assert detail.pruned_at is not None
    [created] = detail.created
    assert created.conclusion.content == "Ann relies on caffeine to work nights"
    assert created.conclusion.status == "current"
    assert detail.removed == []
    assert detail.peer_card is None
    # The counts are what the dream did, not what is left.
    assert (detail.created_count, detail.removed_count) == (1, 1)
    assert detail.peer_card_updated

    # Once that conclusion is gone, the next prune deletes the run.
    await _remove(ctx, bank_id, [created.conclusion.id], None)
    await prune_dream_history(ctx)
    with pytest.raises(ResourceNotFoundException):
        await get_run_detail(ctx, bank_id, target.run_id)


async def test_prune_leaves_a_run_with_nothing_to_drop_unmarked(ctx: PeopleContext, bank_id: str):
    _session_name, (coffee_id, nights_id) = await seed(ctx, bank_id)
    target = await open_target(ctx, bank_id)
    await _create(
        ctx, bank_id, await deductive(ctx, "Ann relies on caffeine to work nights", [coffee_id, nights_id]), target
    )
    await _age_past_retention(ctx, bank_id, target.run_id)

    await prune_dream_history(ctx)

    detail = await get_run_detail(ctx, bank_id, target.run_id)
    assert detail.pruned_at is None
    assert len(detail.created) == 1


async def test_prune_walks_every_batch(ctx: PeopleContext, bank_id: str, monkeypatch: pytest.MonkeyPatch):
    _session_name, (coffee_id, nights_id) = await seed(ctx, bank_id)
    kept = await open_target(ctx, bank_id)
    await _create(
        ctx, bank_id, await deductive(ctx, "Ann relies on caffeine to work nights", [coffee_id, nights_id]), kept
    )
    empty = [await open_target(ctx, bank_id) for _ in range(3)]
    for run in [kept, *empty]:
        await _age_past_retention(ctx, bank_id, run.run_id)
    monkeypatch.setattr("hindsight_api.people.dream_history._PRUNE_BATCH_SIZE", 1)

    await prune_dream_history(ctx)

    remaining = await list_runs(ctx, bank_id)
    assert [run.id for run in remaining] == [kept.run_id]


async def test_get_run_detail_is_scoped_to_its_bank(ctx: PeopleContext, bank_id: str, people_engine):
    target = await open_target(ctx, bank_id)
    other = f"{bank_id}-other"
    await people_engine.ensure_bank_profile(bank_id=other, request_context=ctx.request_context)

    with pytest.raises(ResourceNotFoundException):
        await get_run_detail(ctx, other, target.run_id)


async def test_list_runs_filters_by_pair_and_pages_newest_first(ctx: PeopleContext, bank_id: str):
    ann_run = await open_target(ctx, bank_id, "ann", "ann")
    bob_run = await open_target(ctx, bank_id, "ann", "bob")
    later = await open_target(ctx, bank_id, "ann", "ann")

    assert [r.id for r in await list_runs(ctx, bank_id)] == [later.run_id, bob_run.run_id, ann_run.run_id]
    assert [r.id for r in await list_runs(ctx, bank_id, observed="bob")] == [bob_run.run_id]
    assert [r.id for r in await list_runs(ctx, bank_id, observer="ann", observed="ann")] == [
        later.run_id,
        ann_run.run_id,
    ]
    assert [r.id for r in await list_runs(ctx, bank_id, limit=1, offset=1)] == [bob_run.run_id]
    assert await count_runs(ctx, bank_id) == 3
    assert await count_runs(ctx, bank_id, observed="ann") == 2


# Nothing hard-deletes a peer or a session here; these tombstone directly. History must read as it
# would after a hard deletion.


async def test_a_tombstoned_session_hides_its_conclusions_and_copies(ctx: PeopleContext, bank_id: str):
    session_name, (coffee_id, nights_id) = await seed(ctx, bank_id)
    first = await open_target(ctx, bank_id)
    await _create(
        ctx, bank_id, await deductive(ctx, "Ann relies on caffeine to work nights", [coffee_id, nights_id]), first
    )
    second = await open_target(ctx, bank_id)
    await _remove(ctx, bank_id, [nights_id], second)

    detail = await get_run_detail(ctx, bank_id, first.run_id)
    assert [s.status for s in detail.created[0].sources] == ["current", "removed"]

    await tombstone(ctx, "sessions", bank_id, session_name)

    detail = await get_run_detail(ctx, bank_id, first.run_id)
    [created] = detail.created
    # The deductive conclusion has no session, so the session doesn't hide it.
    assert created.conclusion.status == "current"
    assert [(s.id, s.status, s.content) for s in created.sources] == [
        (coffee_id, "unavailable", None),
        (nights_id, "unavailable", None),
    ]
    removal = await get_run_detail(ctx, bank_id, second.run_id)
    assert removal.removed == []


@pytest.mark.parametrize("deleted", ["observer", "observed"])
async def test_a_tombstoned_peer_hides_the_runs_about_it(ctx: PeopleContext, bank_id: str, deleted: str):
    target = await open_target(ctx, bank_id, "carol", "ann")
    unrelated = await open_target(ctx, bank_id, "ann", "ann")

    await tombstone(ctx, "peers", bank_id, "carol" if deleted == "observer" else "ann")

    listed = {run.id for run in await list_runs(ctx, bank_id)}
    assert target.run_id not in listed
    if deleted == "observer":
        assert unrelated.run_id in listed
    with pytest.raises(ResourceNotFoundException):
        await get_run_detail(ctx, bank_id, target.run_id)


async def test_a_removed_copy_from_a_hidden_run_is_unavailable(ctx: PeopleContext, bank_id: str):
    await get_or_create_peers(ctx, bank_id, ["ann", "assistant"])
    session_name = generate_nanoid()
    await add_session(ctx, bank_id, session_name, ["ann", "assistant"])
    source = await save_documents_with_history(
        ctx,
        bank_id,
        [await make_doc(ctx, "Ann mentioned a night shift", session=session_name)],
        observer="assistant",
        observed="ann",
        deduplicate=False,
        history=None,
    )
    [source_row] = source.created_conclusions

    # A run on ann's own representation cites a source from another pair.
    cites = await open_target(ctx, bank_id, "ann", "ann")
    await _create(ctx, bank_id, await deductive(ctx, "Ann works nights", [source_row.id]), cites)
    removal = await open_target(ctx, bank_id, "assistant", "ann")
    await delete_observations_with_history(
        ctx, bank_id, [source_row.id], observer="assistant", observed="ann", history=removal
    )

    detail = await get_run_detail(ctx, bank_id, cites.run_id)
    [source_ref] = detail.created[0].sources
    assert (source_ref.status, source_ref.content) == ("removed", "Ann mentioned a night shift")

    await tombstone(ctx, "peers", bank_id, "assistant")

    detail = await get_run_detail(ctx, bank_id, cites.run_id)
    [source_ref] = detail.created[0].sources
    assert (source_ref.status, source_ref.content) == ("unavailable", None)


async def test_session_and_bank_history_deletion_do_not_deadlock(ctx: PeopleContext, bank_id: str):
    """Session deletion and bank deletion take dream history in the same lock order (ported lock-order test)."""
    session_name, (coffee_id, _nights_id) = await seed(ctx, bank_id)
    target = await open_target(ctx, bank_id)
    await _remove(ctx, bank_id, [coffee_id], target)

    async def delete_session() -> None:
        async with ctx.conn() as conn, conn.transaction():
            await delete_session_history(conn, bank_id, session_name)

    async def delete_bank() -> None:
        async with ctx.conn() as conn, conn.transaction():
            await delete_bank_history(conn, bank_id)

    await asyncio.wait_for(asyncio.gather(delete_session(), delete_bank()), timeout=20)
    assert await scalar(ctx, f"SELECT count(*) FROM {ctx.table('dream_runs')} WHERE bank_id = $1", bank_id) == 0
