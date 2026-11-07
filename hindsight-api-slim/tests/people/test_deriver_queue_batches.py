# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: tests/deriver/test_queue_processing.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""Draining a claimed representation work unit: which messages and queue items make one batch."""

from typing import Any
from unittest.mock import patch

import pytest

from hindsight_api.people.context import PeopleContext
from hindsight_api.people.deriver import queue_manager as qm_module
from hindsight_api.people.deriver.queue_manager import QueueManager
from hindsight_api.people.deriver.settings import DERIVER
from hindsight_api.people.schemas import get_configuration
from tests.people.deriver_support import claim, items_of, seed_messages, seed_queue, seed_unit

BATCH = "hindsight_api.people.deriver.queue_manager.process_representation_batch"


class _Batches:
    """A stand-in for ``process_representation_batch`` that records what each batch held."""

    def __init__(self) -> None:
        self.batches: list[dict[str, Any]] = []

    async def __call__(self, _ctx: Any, _bank_id: str, messages: list[Any], configuration: Any, **kwargs: Any) -> None:
        self.batches.append(
            {
                "messages": [m.id for m in messages],
                "queued": kwargs["queue_item_message_ids"],
                "observed": kwargs["observed"],
                "observers": kwargs["observers"],
                "configuration": configuration,
                **{k: kwargs[k] for k in ("hit_batch_token_cap", "was_flush_enabled", "batch_max_tokens")},
            }
        )


async def _drain(ctx: PeopleContext, bank_id: str, key: str) -> _Batches:
    qm, worker_id, _ = await claim(ctx, bank_id, key)
    recorder = _Batches()
    with patch(BATCH, recorder):
        await qm.process_work_unit(key, worker_id)
    return recorder


@pytest.fixture(autouse=True)
def _default_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DERIVER, "REPRESENTATION_BATCH_TARGET_INPUT_TOKENS", 1024)


@pytest.mark.asyncio
class TestRepresentationBatching:
    async def test_batching_respects_token_limits(self, ctx: PeopleContext, bank_id: str) -> None:
        limit = DERIVER.REPRESENTATION_BATCH_TARGET_INPUT_TOKENS
        key, items, ids = await seed_unit(ctx, bank_id, [limit // 2, limit // 2, limit // 2])

        recorder = await _drain(ctx, bank_id, key)

        # Two batches because of the token limit.
        assert [len(b["messages"]) for b in recorder.batches] == [2, 1]
        assert recorder.batches[0]["queued"] == ids[:2] and recorder.batches[1]["queued"] == ids[2:]
        assert all(item.processed for item in await items_of(ctx, bank_id, key))

    async def test_first_message_exceeding_the_limit_is_included_alone(self, ctx: PeopleContext, bank_id: str) -> None:
        limit = DERIVER.REPRESENTATION_BATCH_TARGET_INPUT_TOKENS
        key, _, _ = await seed_unit(ctx, bank_id, [limit + 1000, 100, 200])

        recorder = await _drain(ctx, bank_id, key)

        assert [len(b["messages"]) for b in recorder.batches] == [1, 2]

    async def test_a_batch_at_exactly_the_limit_is_kept_whole(self, ctx: PeopleContext, bank_id: str) -> None:
        limit = DERIVER.REPRESENTATION_BATCH_TARGET_INPUT_TOKENS
        key, _, _ = await seed_unit(ctx, bank_id, [limit // 2, limit // 2, 1])

        recorder = await _drain(ctx, bank_id, key)

        assert [len(b["messages"]) for b in recorder.batches] == [2, 1]

    async def test_batches_carry_what_the_deriver_needs(self, ctx: PeopleContext, bank_id: str) -> None:
        key, items, ids = await seed_unit(ctx, bank_id, [50, 50], peer="alice")

        recorder = await _drain(ctx, bank_id, key)

        (batch,) = recorder.batches
        assert batch["observed"] == "alice" and batch["observers"] == ["alice"]
        assert batch["configuration"] == get_configuration(None, None, None)
        assert batch["was_flush_enabled"] is DERIVER.FLUSH_ENABLED
        assert batch["batch_max_tokens"] == 1024
        assert batch["hit_batch_token_cap"] is False

    async def test_messages_are_batched_across_peers_then_filtered_by_work_unit(
        self, ctx: PeopleContext, bank_id: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Messages are windowed chronologically across all peers; each work unit keeps only its own queue
        items, with the other peers' messages as context."""
        monkeypatch.setattr(DERIVER, "REPRESENTATION_BATCH_TARGET_INPUT_TOKENS", 2000)
        # alice(250), bob(400), steve(300), alice(500), bob(500), alice(40), steve(100)
        specs = [
            ("alice", 250),
            ("bob", 400),
            ("steve", 300),
            ("alice", 500),
            ("bob", 500),
            ("alice", 40),
            ("steve", 100),
        ]
        ids = await seed_messages(ctx, bank_id, specs)
        alice_key, alice_items = await seed_queue(ctx, bank_id, [ids[0], ids[3], ids[5]], observed="alice")
        bob_key, bob_items = await seed_queue(ctx, bank_id, [ids[1], ids[4]], observed="bob")
        steve_key, steve_items = await seed_queue(ctx, bank_id, [ids[2], ids[6]], observed="steve")

        qm = QueueManager(lambda: ctx)
        async with ctx.conn() as conn:
            claimed = {
                key: (await qm.claim_work_units(conn, bank_id, [key]))[key] for key in (alice_key, bob_key, steve_key)
            }

        # Bob: the preceding alice message (context), then everything through bob's last item.
        bob = await qm.get_queue_item_batch(bank_id, "representation", bob_key, claimed[bob_key])
        assert {m.id for m in bob.messages_context} == set(ids[0:5])
        assert [i.id for i in bob.items_to_process] == [i.id for i in bob_items]
        assert all(i.payload["observed"] == "bob" for i in bob.items_to_process)

        # Steve: the preceding bob message, then through steve's last item (message 7).
        steve = await qm.get_queue_item_batch(bank_id, "representation", steve_key, claimed[steve_key])
        assert {m.id for m in steve.messages_context} == set(ids[1:7])
        assert all(i.payload["observed"] == "steve" for i in steve.items_to_process)

        # Alice starts at message 1 (nothing precedes it).
        alice = await qm.get_queue_item_batch(bank_id, "representation", alice_key, claimed[alice_key])
        assert alice.messages_context[0].id == ids[0]
        assert [i.message_id for i in alice.items_to_process] == [ids[0], ids[3], ids[5]]

    async def test_each_work_unit_anchors_at_its_own_first_unprocessed_message(
        self, ctx: PeopleContext, bank_id: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(DERIVER, "REPRESENTATION_BATCH_TARGET_INPUT_TOKENS", 1500)
        # bob(800), steve(800), alice(100), alice(200)
        ids = await seed_messages(ctx, bank_id, [("bob", 800), ("steve", 800), ("alice", 100), ("alice", 200)])
        bob_key, _ = await seed_queue(ctx, bank_id, [ids[0]], observed="bob")
        steve_key, _ = await seed_queue(ctx, bank_id, [ids[1]], observed="steve")
        alice_key, _ = await seed_queue(ctx, bank_id, [ids[2], ids[3]], observed="alice")
        qm = QueueManager(lambda: ctx)
        async with ctx.conn() as conn:
            claimed = {
                key: (await qm.claim_work_units(conn, bank_id, [key]))[key] for key in (bob_key, steve_key, alice_key)
            }

        alice = await qm.get_queue_item_batch(bank_id, "representation", alice_key, claimed[alice_key])
        # steve is the preceding message of a different peer, then alice's two
        assert [m.id for m in alice.messages_context] == [ids[1], ids[2], ids[3]]
        bob = await qm.get_queue_item_batch(bank_id, "representation", bob_key, claimed[bob_key])
        assert [m.id for m in bob.messages_context] == [ids[0]]
        # bob(800) + steve(800) is over the cap, but the first unprocessed message (steve's) is always kept
        # together with the preceding context message.
        steve = await qm.get_queue_item_batch(bank_id, "representation", steve_key, claimed[steve_key])
        assert [m.id for m in steve.messages_context] == [ids[0], ids[1]]

    async def test_the_preceding_message_of_the_same_peer_is_not_context(
        self, ctx: PeopleContext, bank_id: str
    ) -> None:
        ids = await seed_messages(ctx, bank_id, [("alice", 10), ("alice", 10), ("alice", 10)])
        key, items = await seed_queue(ctx, bank_id, ids, observed="alice")
        qm, _, aqs_id = await claim(ctx, bank_id, key)
        await qm.mark_queue_items_as_processed(items[:1], key)

        batch = await qm.get_queue_item_batch(bank_id, "representation", key, aqs_id)

        assert [m.id for m in batch.messages_context] == ids[1:]
        assert [i.id for i in batch.items_to_process] == [i.id for i in items[1:]]

    async def test_batches_follow_the_conversation_order_of_one_session_only(
        self, ctx: PeopleContext, bank_id: str
    ) -> None:
        a = await seed_messages(ctx, bank_id, [("alice", 10), ("alice", 10)], session="s1")
        b = await seed_messages(ctx, bank_id, [("alice", 10)], session="s2")
        key, _ = await seed_queue(ctx, bank_id, a, observed="alice", session="s1")
        qm, _, aqs_id = await claim(ctx, bank_id, key)

        batch = await qm.get_queue_item_batch(bank_id, "representation", key, aqs_id)

        assert [m.id for m in batch.messages_context] == a
        assert b[0] not in {m.id for m in batch.messages_context}

    async def test_a_batch_stops_at_a_configuration_change(self, ctx: PeopleContext, bank_id: str) -> None:
        ids = await seed_messages(ctx, bank_id, [("alice", 10)] * 3)
        custom = get_configuration(None, None, None)
        custom.reasoning.custom_instructions = "Prefer dates."
        key, first = await seed_queue(ctx, bank_id, ids[:2], observed="alice")
        _, last = await seed_queue(ctx, bank_id, ids[2:], observed="alice", configuration=custom)
        qm, _, aqs_id = await claim(ctx, bank_id, key)

        batch = await qm.get_queue_item_batch(bank_id, "representation", key, aqs_id)

        assert [i.id for i in batch.items_to_process] == [i.id for i in first]
        assert [m.id for m in batch.messages_context] == ids[:2]
        assert batch.configuration == get_configuration(None, None, None)

    async def test_an_unowned_unit_gets_an_empty_batch(self, ctx: PeopleContext, bank_id: str) -> None:
        key, _, _ = await seed_unit(ctx, bank_id, [10])
        qm, _, _ = await claim(ctx, bank_id, key)
        batch = await qm.get_queue_item_batch(bank_id, "representation", key, "somebody-elses-claim")
        assert batch.items_to_process == [] and batch.messages_context == []
        assert batch.batch_max_tokens == 1024

    async def test_a_drained_unit_gets_an_empty_batch(self, ctx: PeopleContext, bank_id: str) -> None:
        key, items, _ = await seed_unit(ctx, bank_id, [10])
        qm, _, aqs_id = await claim(ctx, bank_id, key)
        await qm.mark_queue_items_as_processed(items, key)
        batch = await qm.get_queue_item_batch(bank_id, "representation", key, aqs_id)
        assert batch.items_to_process == [] and batch.messages_context == []


@pytest.mark.asyncio
class TestHitBatchTokenCap:
    async def test_cap_is_reported_when_it_clamps_the_batch(self, ctx: PeopleContext, bank_id: str) -> None:
        cap = DERIVER.REPRESENTATION_BATCH_TARGET_INPUT_TOKENS
        key, _, _ = await seed_unit(ctx, bank_id, [cap // 2, cap // 2, cap // 2])
        qm, _, aqs_id = await claim(ctx, bank_id, key)
        batch = await qm.get_queue_item_batch(bank_id, "representation", key, aqs_id)
        assert len(batch.items_to_process) == 2
        assert batch.hit_batch_token_cap is True

    async def test_cap_is_not_reported_when_everything_fits(self, ctx: PeopleContext, bank_id: str) -> None:
        key, _, _ = await seed_unit(ctx, bank_id, [100, 100])
        qm, _, aqs_id = await claim(ctx, bank_id, key)
        batch = await qm.get_queue_item_batch(bank_id, "representation", key, aqs_id)
        assert batch.hit_batch_token_cap is False

    async def test_cap_reflects_the_post_filter_batch(
        self, ctx: PeopleContext, bank_id: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`hit_batch_token_cap` must reflect the actually-returned batch (after the configuration filter),
        not the pre-filter superset."""
        cap = DERIVER.REPRESENTATION_BATCH_TARGET_INPUT_TOKENS
        # M1 + M2 sum to exactly the cap; M3 pushes over it. The cap is binding on the pre-filter batch
        # [M1, M2]. A configuration trim that keeps only QI(M1) leaves [M1] alone: 400 < cap.
        key, _, ids = await seed_unit(ctx, bank_id, [400, cap - 400, 300])
        real_resolve = qm_module._resolve_batch_configuration

        def fake_resolve(items: list[Any]) -> tuple[list[Any], Any]:
            _kept, cfg = real_resolve(items)
            return (items[:1] if items else []), cfg

        monkeypatch.setattr(qm_module, "_resolve_batch_configuration", fake_resolve)
        qm, _, aqs_id = await claim(ctx, bank_id, key)

        result = await qm.get_queue_item_batch(bank_id, "representation", key, aqs_id)

        assert [m.id for m in result.messages_context] == ids[:1]
        assert result.hit_batch_token_cap is False

    async def test_cap_fires_when_trailing_context_is_trimmed(self, ctx: PeopleContext, bank_id: str) -> None:
        """SQL kept trailing NON-QUEUE context past the last queued item, the configuration filter trims that
        context away, and the cap check must still see that the SQL cap clamped queue work."""
        cap = DERIVER.REPRESENTATION_BATCH_TARGET_INPUT_TOKENS
        # M1 (alice, queued, 200), M2 (alice, queued, 200), M3 (bob, context, cap - 700), M4 (alice, queued, 400):
        # cumulative M3 = cap - 300, M4 = cap + 100 > cap, so M4 is excluded by the cap.
        ids = await seed_messages(ctx, bank_id, [("alice", 200), ("alice", 200), ("bob", cap - 700), ("alice", 400)])
        key, _ = await seed_queue(ctx, bank_id, [ids[0], ids[1], ids[3]], observed="alice")
        qm, _, aqs_id = await claim(ctx, bank_id, key)

        result = await qm.get_queue_item_batch(bank_id, "representation", key, aqs_id)

        assert [i.message_id for i in result.items_to_process] == [ids[0], ids[1]]
        assert result.hit_batch_token_cap is True
