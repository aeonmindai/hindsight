# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: tests/crud/test_peer_card.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""Peer card tests, ported from Honcho's tests/crud/test_peer_card.py."""

import sys
import types
from typing import Any

import pytest

from hindsight_api.people import schemas
from hindsight_api.people.context import PeopleContext
from hindsight_api.people.exceptions import ResourceNotFoundException
from hindsight_api.people.peer_card import construct_peer_card_label, get_peer_card, set_peer_card
from hindsight_api.people.peers import get_or_create_peers


async def _peers(ctx: PeopleContext, bank_id: str, *names: str) -> None:
    await get_or_create_peers(ctx, bank_id, [schemas.PeerSpec(name=n) for n in names])


@pytest.mark.asyncio
async def test_peer_card_get_set_roundtrip(ctx: PeopleContext, bank_id: str):
    """Roundtrip set/get for peer card on a valid peer."""
    await _peers(ctx, bank_id, "alice")

    # Initially absent
    assert await get_peer_card(ctx, bank_id, observer="alice", observed="alice") is None

    # Set and read back
    value_1 = ["Initial peer card text"]
    await set_peer_card(ctx, bank_id, value_1, observer="alice", observed="alice")
    assert await get_peer_card(ctx, bank_id, observer="alice", observed="alice") == value_1

    # Update and read back
    value_2 = ["Updated peer card text", "Another line"]
    await set_peer_card(ctx, bank_id, value_2, observer="alice", observed="alice")
    assert await get_peer_card(ctx, bank_id, observer="alice", observed="alice") == value_2


@pytest.mark.asyncio
async def test_peer_card_defaults_to_the_observers_own(ctx: PeopleContext, bank_id: str):
    await _peers(ctx, bank_id, "alice")
    await set_peer_card(ctx, bank_id, ["own card"], "alice")
    assert await get_peer_card(ctx, bank_id, "alice") == ["own card"]


@pytest.mark.asyncio
async def test_get_peer_card_missing_peer_raises(ctx: PeopleContext, bank_id: str):
    """Getting a peer card for a non-existent peer should raise ResourceNotFoundException."""
    await _peers(ctx, bank_id, "alice")
    with pytest.raises(ResourceNotFoundException):
        await get_peer_card(ctx, bank_id, observer="missing-peer", observed="missing-peer")


@pytest.mark.asyncio
async def test_get_peer_card_missing_bank_raises(ctx: PeopleContext, bank_id: str):
    """Getting a peer card for a bank with no such peer should raise ResourceNotFoundException."""
    await _peers(ctx, bank_id, "alice")
    with pytest.raises(ResourceNotFoundException):
        await get_peer_card(ctx, "missing-bank", observer="alice", observed="alice")


@pytest.mark.asyncio
async def test_set_peer_card_creates_the_observer(ctx: PeopleContext, bank_id: str):
    """Setting a card for a peer that does not exist yet creates the peer (get-or-create)."""
    await set_peer_card(ctx, bank_id, ["new here"], observer="newcomer", observed="newcomer")
    assert await get_peer_card(ctx, bank_id, observer="newcomer", observed="newcomer") == ["new here"]


@pytest.mark.asyncio
async def test_peer_card_empty_list(ctx: PeopleContext, bank_id: str):
    """Setting and getting an empty peer card should work."""
    await _peers(ctx, bank_id, "alice")
    await set_peer_card(ctx, bank_id, [], observer="alice", observed="alice")
    assert await get_peer_card(ctx, bank_id, observer="alice", observed="alice") == []


@pytest.mark.asyncio
async def test_peer_card_strips_nul(ctx: PeopleContext, bank_id: str):
    """NUL is stripped before the jsonb write; free-form lines are kept as-is."""
    await _peers(ctx, bank_id, "alice")
    await set_peer_card(
        ctx, bank_id, ["Free-form\x00 line", "\x00", "Literal \\u0000 text"], observer="alice", observed="alice"
    )
    assert await get_peer_card(ctx, bank_id, observer="alice", observed="alice") == [
        "Free-form line",
        "",
        "Literal \\u0000 text",
    ]


@pytest.mark.asyncio
async def test_peer_card_multiple_lines(ctx: PeopleContext, bank_id: str):
    """Peer card should support multiple lines."""
    await _peers(ctx, bank_id, "alice")
    multi_line_card = [
        "First observation about the peer",
        "Second observation about the peer",
        "Third observation about the peer",
        "Fourth observation about the peer",
    ]
    await set_peer_card(ctx, bank_id, multi_line_card, observer="alice", observed="alice")
    result = await get_peer_card(ctx, bank_id, observer="alice", observed="alice")
    assert result == multi_line_card
    assert len(result) == 4


@pytest.mark.asyncio
async def test_peer_card_different_observer_observed(ctx: PeopleContext, bank_id: str):
    """Test peer card with different observer and observed peers."""
    await _peers(ctx, bank_id, "alice", "bob")

    card_1_to_2 = ["Peer 1's observation of peer 2"]
    await set_peer_card(ctx, bank_id, card_1_to_2, observer="alice", observed="bob")
    card_2_to_1 = ["Peer 2's observation of peer 1"]
    await set_peer_card(ctx, bank_id, card_2_to_1, observer="bob", observed="alice")

    # Verify each card is independent
    assert await get_peer_card(ctx, bank_id, observer="alice", observed="bob") == card_1_to_2
    assert await get_peer_card(ctx, bank_id, observer="bob", observed="alice") == card_2_to_1


@pytest.mark.asyncio
async def test_peer_card_self_and_other_observations(ctx: PeopleContext, bank_id: str):
    """A peer can have both a self-observation and observations of others."""
    await _peers(ctx, bank_id, "alice", "bob")
    self_card = ["Self observation"]
    other_card = ["Observation of other peer"]
    await set_peer_card(ctx, bank_id, self_card, observer="alice", observed="alice")
    await set_peer_card(ctx, bank_id, other_card, observer="alice", observed="bob")

    assert await get_peer_card(ctx, bank_id, observer="alice", observed="alice") == self_card
    assert await get_peer_card(ctx, bank_id, observer="alice", observed="bob") == other_card


@pytest.mark.asyncio
async def test_peer_card_multiple_observers_same_observed(ctx: PeopleContext, bank_id: str):
    """Multiple peers can observe the same peer with different cards."""
    await _peers(ctx, bank_id, "alice", "bob", "carol")
    card_2_to_1 = ["Peer 2's view of peer 1"]
    card_3_to_1 = ["Peer 3's view of peer 1"]
    await set_peer_card(ctx, bank_id, card_2_to_1, observer="bob", observed="alice")
    await set_peer_card(ctx, bank_id, card_3_to_1, observer="carol", observed="alice")

    assert await get_peer_card(ctx, bank_id, observer="bob", observed="alice") == card_2_to_1
    assert await get_peer_card(ctx, bank_id, observer="carol", observed="alice") == card_3_to_1


@pytest.mark.asyncio
async def test_peer_card_update_does_not_affect_others(ctx: PeopleContext, bank_id: str):
    """Updating one peer card doesn't affect other peer cards."""
    await _peers(ctx, bank_id, "alice", "bob")
    card_2 = ["Peer 1's observation of peer 2"]
    await set_peer_card(ctx, bank_id, ["Peer 1 self observation"], observer="alice", observed="alice")
    await set_peer_card(ctx, bank_id, card_2, observer="alice", observed="bob")

    new_card_1 = ["Updated self observation"]
    await set_peer_card(ctx, bank_id, new_card_1, observer="alice", observed="alice")

    assert await get_peer_card(ctx, bank_id, observer="alice", observed="alice") == new_card_1
    assert await get_peer_card(ctx, bank_id, observer="alice", observed="bob") == card_2


@pytest.mark.asyncio
async def test_peer_card_is_kept_per_bank(ctx: PeopleContext, bank_id: str, people_engine):
    """The same handle in two banks holds two unrelated cards."""
    other_bank = bank_id + "-other"
    await people_engine.ensure_bank_profile(bank_id=other_bank, request_context=ctx.request_context)
    await set_peer_card(ctx, bank_id, ["one"], observer="alice", observed="alice")
    await set_peer_card(ctx, other_bank, ["two"], observer="alice", observed="alice")
    assert await get_peer_card(ctx, bank_id, observer="alice", observed="alice") == ["one"]
    assert await get_peer_card(ctx, other_bank, observer="alice", observed="alice") == ["two"]


@pytest.mark.asyncio
async def test_peer_card_special_characters(ctx: PeopleContext, bank_id: str):
    """Peer cards handle special characters correctly."""
    await _peers(ctx, bank_id, "alice")
    special_card = [
        "Line with special chars: @#$%^&*()",
        "Line with unicode: 你好世界 🌍",
        "Line with quotes: \"double\" and 'single'",
        "Line with newlines embedded\\n",
    ]
    await set_peer_card(ctx, bank_id, special_card, observer="alice", observed="alice")
    assert await get_peer_card(ctx, bank_id, observer="alice", observed="alice") == special_card


@pytest.mark.asyncio
async def test_peer_card_large_content(ctx: PeopleContext, bank_id: str):
    """Peer cards can handle large amounts of content."""
    await _peers(ctx, bank_id, "alice")
    large_card = [f"Observation number {i}" for i in range(100)]
    await set_peer_card(ctx, bank_id, large_card, observer="alice", observed="alice")
    result = await get_peer_card(ctx, bank_id, observer="alice", observed="alice")
    assert result == large_card
    assert len(result) == 100


def test_construct_helpers_labels():
    """Helper label constructors return expected values."""
    assert construct_peer_card_label(observer="a", observed="a") == "peer_card"
    assert construct_peer_card_label(observer="a", observed="b") == "b_peer_card"


def test_construct_peer_card_label_with_special_chars():
    """Label construction with special characters in peer names."""
    assert construct_peer_card_label(observer="peer-1", observed="peer-2") == "peer-2_peer_card"
    assert construct_peer_card_label(observer="peer_1", observed="peer_2") == "peer_2_peer_card"
    assert construct_peer_card_label(observer="peer-1", observed="peer-1") == "peer_card"


def test_construct_peer_card_label_edge_cases():
    """Label construction edge cases."""
    assert construct_peer_card_label(observer="", observed="") == "peer_card"
    assert construct_peer_card_label(observer="a", observed="") == "_peer_card"
    long_name = "a" * 100
    assert construct_peer_card_label(observer=long_name, observed=long_name) == "peer_card"
    assert construct_peer_card_label(observer="a", observed=long_name) == f"{long_name}_peer_card"


def _fake_history(monkeypatch: pytest.MonkeyPatch, calls: list[dict[str, Any]], fail: bool = False) -> None:
    async def record_peer_card(conn: Any, target: Any, bank_id: str, *, before: Any, after: Any) -> None:
        calls.append({"target": target, "bank_id": bank_id, "before": before, "after": after})
        if fail:
            raise RuntimeError("history write failed")

    fake = types.ModuleType("hindsight_api.people.dream_history")
    fake.record_peer_card = record_peer_card  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hindsight_api.people.dream_history", fake)


@pytest.mark.asyncio
async def test_set_peer_card_records_the_card_before_and_after(
    ctx: PeopleContext, bank_id: str, monkeypatch: pytest.MonkeyPatch
):
    """With a dream history target, the card before and after goes to the dream run."""
    calls: list[dict[str, Any]] = []
    _fake_history(monkeypatch, calls)
    await set_peer_card(ctx, bank_id, ["first"], observer="alice", observed="alice")
    assert calls == []  # no history asked for

    await set_peer_card(ctx, bank_id, ["second"], observer="alice", observed="alice", history="run-1")
    assert calls == [{"target": "run-1", "bank_id": bank_id, "before": ["first"], "after": ["second"]}]
    assert await get_peer_card(ctx, bank_id, observer="alice", observed="alice") == ["second"]


@pytest.mark.asyncio
async def test_a_failed_history_write_leaves_the_card_alone(
    ctx: PeopleContext, bank_id: str, monkeypatch: pytest.MonkeyPatch
):
    """The history row and the card change commit together or not at all."""
    await set_peer_card(ctx, bank_id, ["first"], observer="alice", observed="alice")
    _fake_history(monkeypatch, [], fail=True)
    with pytest.raises(RuntimeError):
        await set_peer_card(ctx, bank_id, ["second"], observer="alice", observed="alice", history="run-1")
    assert await get_peer_card(ctx, bank_id, observer="alice", observed="alice") == ["first"]


@pytest.mark.asyncio
async def test_set_peer_card_can_join_a_callers_transaction(ctx: PeopleContext, bank_id: str):
    await set_peer_card(ctx, bank_id, ["first"], observer="alice", observed="alice")
    async with ctx.conn() as db:
        with pytest.raises(RuntimeError):
            async with db.transaction():
                await set_peer_card(ctx, bank_id, ["second"], observer="alice", observed="alice", conn=db)
                assert await get_peer_card(ctx, bank_id, observer="alice", observed="alice", conn=db) == ["second"]
                raise RuntimeError("roll back")
    assert await get_peer_card(ctx, bank_id, observer="alice", observed="alice") == ["first"]
