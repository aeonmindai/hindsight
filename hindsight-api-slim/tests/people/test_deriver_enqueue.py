# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: tests/deriver/test_enqueue_dream.py
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: tests/deriver/test_queue_operations.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""Enqueueing retained messages and dreams."""

import asyncio
from datetime import UTC, datetime
from typing import Any

import asyncpg
import pytest

from hindsight_api.people.context import PeopleContext
from hindsight_api.people.deriver.enqueue import (
    IncomingMessage,
    create_dream_record,
    create_representation_record,
    enqueue_dream,
    enqueue_messages,
    get_effective_observe_me,
    split_content,
)
from hindsight_api.people.dreamer.schemas import DreamType
from hindsight_api.people.jsonb import from_jsonb, to_jsonb
from hindsight_api.people.schemas import get_configuration

SAID_AT = datetime(2026, 5, 1, 9, 30, tzinfo=UTC)


async def _fetch(ctx: PeopleContext, sql: str, *args: Any) -> list[Any]:
    async with ctx.conn() as conn:
        return await conn.fetch(sql, *args)


async def _queue(ctx: PeopleContext, bank_id: str) -> list[Any]:
    return await _fetch(ctx, f"SELECT * FROM {PeopleContext.table('queue')} WHERE bank_id = $1 ORDER BY id", bank_id)


async def _messages(ctx: PeopleContext, bank_id: str) -> list[Any]:
    return await _fetch(ctx, f"SELECT * FROM {PeopleContext.table('messages')} WHERE bank_id = $1 ORDER BY id", bank_id)


def _incoming(peer: str = "alice", content: str = "I moved to Lisbon", **kwargs: Any) -> IncomingMessage:
    kwargs.setdefault("session", "slack:C1")
    kwargs.setdefault("said_at", SAID_AT)
    return IncomingMessage(peer=peer, content=content, **kwargs)


@pytest.mark.asyncio
class TestEnqueueMessages:
    async def test_stores_the_people_rows_and_queues_one_representation_item(
        self, ctx: PeopleContext, bank_id: str
    ) -> None:
        ids = await enqueue_messages(
            ctx,
            bank_id,
            [_incoming(channel="#general", source_url="https://slack.example/p1", document_id="doc-1")],
        )

        messages = await _messages(ctx, bank_id)
        assert [m["id"] for m in messages] == ids
        message = messages[0]
        assert (message["peer_name"], message["said_by"], message["session_name"]) == ("alice", "alice", "slack:C1")
        assert (message["source_url"], message["said_at"], message["channel"]) == (
            "https://slack.example/p1",
            SAID_AT,
            "#general",
        )
        assert message["token_count"] > 0
        assert message["seq_in_session"] == 1
        assert from_jsonb(message["internal_metadata"]) == {"document_id": "doc-1", "part": "0"}

        sessions = await _fetch(ctx, f"SELECT name FROM {PeopleContext.table('sessions')} WHERE bank_id = $1", bank_id)
        assert [s["name"] for s in sessions] == ["slack:C1"]
        session_peers = await _fetch(
            ctx, f"SELECT peer_name, left_at FROM {PeopleContext.table('session_peers')} WHERE bank_id = $1", bank_id
        )
        assert [(s["peer_name"], s["left_at"]) for s in session_peers] == [("alice", None)]

        queue = await _queue(ctx, bank_id)
        assert len(queue) == 1
        item = queue[0]
        assert item["task_type"] == "representation"
        assert item["message_id"] == ids[0]
        assert item["work_unit_key"] == f"representation:{bank_id}:slack%3AC1:alice"
        assert item["processed"] is False
        payload = from_jsonb(item["payload"])
        assert payload["observers"] == ["alice"]
        assert payload["observed"] == "alice"
        assert payload["session_name"] == "slack:C1"
        assert payload["content"] == "I moved to Lisbon"
        assert "bank_id" not in payload and "message_id" not in payload

    async def test_sequence_numbers_follow_the_session(self, ctx: PeopleContext, bank_id: str) -> None:
        await enqueue_messages(ctx, bank_id, [_incoming(content="one"), _incoming("bob", "two")])
        await enqueue_messages(ctx, bank_id, [_incoming(content="three"), _incoming(content="elsewhere", session="s2")])

        by_session = {(m["session_name"], m["content"]): m["seq_in_session"] for m in await _messages(ctx, bank_id)}
        assert by_session == {
            ("slack:C1", "one"): 1,
            ("slack:C1", "two"): 2,
            ("slack:C1", "three"): 3,
            ("s2", "elsewhere"): 1,
        }
        # Different speakers of one session are different work units (the observed person is in the key).
        keys = {q["work_unit_key"] for q in await _queue(ctx, bank_id)}
        assert len(keys) == 3

    async def test_the_same_document_text_is_stored_and_queued_once(self, ctx: PeopleContext, bank_id: str) -> None:
        first = await enqueue_messages(ctx, bank_id, [_incoming(document_id="doc-1")])
        again = await enqueue_messages(ctx, bank_id, [_incoming(document_id="doc-1")])

        assert again == first
        assert len(await _messages(ctx, bank_id)) == 1
        assert len(await _queue(ctx, bank_id)) == 1

        changed = await enqueue_messages(ctx, bank_id, [_incoming(content="I moved to Porto", document_id="doc-1")])
        assert changed != first
        assert len(await _messages(ctx, bank_id)) == 2
        assert len(await _queue(ctx, bank_id)) == 2

    async def test_long_content_is_split_into_messages_of_one_session(self, ctx: PeopleContext, bank_id: str) -> None:
        paragraph = "word " * 5000  # 25000 characters
        text = "\n\n".join([paragraph] * 5)
        ids = await enqueue_messages(ctx, bank_id, [_incoming(content=text, document_id="long")])

        messages = await _messages(ctx, bank_id)
        assert len(ids) == len(messages) >= 2
        assert all(len(m["content"]) <= 65535 for m in messages)
        assert "".join(m["content"] for m in messages).replace("\n", "").replace(" ", "") == text.replace(
            "\n", ""
        ).replace(" ", "")
        assert [from_jsonb(m["internal_metadata"])["part"] for m in messages] == [str(i) for i in range(len(messages))]

    async def test_an_existing_message_is_queued_without_being_stored_again(
        self, ctx: PeopleContext, bank_id: str
    ) -> None:
        (message_id,) = await enqueue_messages(ctx, bank_id, [_incoming()])
        async with ctx.conn() as conn:
            await conn.execute(f"DELETE FROM {PeopleContext.table('queue')} WHERE bank_id = $1", bank_id)

        ids = await enqueue_messages(ctx, bank_id, [_incoming(message_id=message_id)])

        assert ids == [message_id]
        assert len(await _messages(ctx, bank_id)) == 1
        assert [q["message_id"] for q in await _queue(ctx, bank_id)] == [message_id]

    async def test_a_message_that_does_not_exist_is_refused(self, ctx: PeopleContext, bank_id: str) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            await enqueue_messages(ctx, bank_id, [_incoming(message_id=999_999_999)])
        assert await _queue(ctx, bank_id) == []

    async def test_a_failed_enqueue_stores_nothing(self, ctx: PeopleContext, bank_id: str) -> None:
        """Message rows and queue rows are written in one transaction."""
        with pytest.raises(ValueError):
            await enqueue_messages(ctx, bank_id, [_incoming(), _incoming(message_id=999_999_999)])
        assert await _messages(ctx, bank_id) == []
        assert await _queue(ctx, bank_id) == []

    async def test_nul_bytes_are_stripped(self, ctx: PeopleContext, bank_id: str) -> None:
        await enqueue_messages(ctx, bank_id, [_incoming(content="I like\x00 tea")])
        assert [m["content"] for m in await _messages(ctx, bank_id)] == ["I like tea"]

    async def test_blank_messages_are_not_stored(self, ctx: PeopleContext, bank_id: str) -> None:
        ids = await enqueue_messages(ctx, bank_id, [_incoming(content="  \n "), _incoming(content="\x00")])
        assert ids == []
        assert await _messages(ctx, bank_id) == []
        assert await _queue(ctx, bank_id) == []

    async def test_an_empty_list_does_nothing(self, ctx: PeopleContext, bank_id: str) -> None:
        assert await enqueue_messages(ctx, bank_id, []) == []

    async def test_a_person_who_left_the_session_is_back_when_they_speak(
        self, ctx: PeopleContext, bank_id: str
    ) -> None:
        await enqueue_messages(ctx, bank_id, [_incoming()])
        async with ctx.conn() as conn:
            await conn.execute(
                f"UPDATE {PeopleContext.table('session_peers')} SET left_at = now() WHERE bank_id = $1", bank_id
            )
        await enqueue_messages(ctx, bank_id, [_incoming(content="again")])
        rows = await _fetch(
            ctx, f"SELECT left_at FROM {PeopleContext.table('session_peers')} WHERE bank_id = $1", bank_id
        )
        assert [r["left_at"] for r in rows] == [None]

    async def test_observers_follow_the_peer_and_session_configuration(self, ctx: PeopleContext, bank_id: str) -> None:
        await enqueue_messages(ctx, bank_id, [_incoming("alice"), _incoming("bob")])
        async with ctx.conn() as conn:
            # bob watches the others; carol has left; dave does not want to be observed
            await conn.execute(
                f"UPDATE {PeopleContext.table('session_peers')} SET configuration = $2::jsonb "
                "WHERE bank_id = $1 AND peer_name = 'bob'",
                bank_id,
                to_jsonb({"observe_others": True}),
            )
        await enqueue_messages(ctx, bank_id, [_incoming("carol"), _incoming("dave")])
        async with ctx.conn() as conn:
            await conn.execute(
                f"UPDATE {PeopleContext.table('session_peers')} SET configuration = $2::jsonb, left_at = now() "
                "WHERE bank_id = $1 AND peer_name = 'carol'",
                bank_id,
                to_jsonb({"observe_others": True}),
            )
            await conn.execute(
                f"UPDATE {PeopleContext.table('peers')} SET configuration = $2::jsonb "
                "WHERE bank_id = $1 AND name = 'dave'",
                bank_id,
                to_jsonb({"observe_me": False}),
            )
        await enqueue_messages(ctx, bank_id, [_incoming("alice", "second"), _incoming("dave", "mine")])

        queue = [q for q in await _queue(ctx, bank_id)]
        payloads = {(q["message_id"], from_jsonb(q["payload"])["content"]): from_jsonb(q["payload"]) for q in queue}
        # "second" by alice: alice observes herself and bob observes alice; carol left, so she does not.
        second = next(p for (_, content), p in payloads.items() if content == "second")
        assert second["observers"] == ["alice", "bob"]
        # dave is not observed: nothing is queued for what he said.
        assert not any(content == "mine" for (_, content) in payloads)

    async def test_reasoning_disabled_queues_nothing(self, ctx: PeopleContext, bank_id: str) -> None:
        from hindsight_api.people.schemas import MessageConfiguration

        configuration = MessageConfiguration.model_validate({"reasoning": {"enabled": False}})
        await enqueue_messages(ctx, bank_id, [_incoming(configuration=configuration)])
        assert len(await _messages(ctx, bank_id)) == 1
        assert await _queue(ctx, bank_id) == []


class TestEnqueueHelpers:
    def test_split_content_keeps_short_text_whole(self) -> None:
        assert split_content("short", 100) == ["short"]

    def test_split_content_breaks_at_paragraphs_then_lines_then_spaces_then_anywhere(self) -> None:
        assert split_content("aaa\n\nbbb\n\nccc", 8) == ["aaa\n\nbbb", "ccc"]
        assert split_content("aaa\nbbb\nccc", 8) == ["aaa\nbbb", "ccc"]
        assert split_content("aaa bbb ccc", 8) == ["aaa bbb", "ccc"]
        assert split_content("x" * 25, 10) == ["x" * 10, "x" * 10, "x" * 5]

    def test_create_representation_record_needs_a_bank_and_a_message_id(self) -> None:
        conf = get_configuration(None, None, None)
        message = {
            "bank_id": "b",
            "session_name": "s",
            "message_id": 5,
            "content": "hi",
            "created_at": SAID_AT,
        }
        record = create_representation_record(message, conf, "sid", observers=["a"], observed="a")
        assert (record["task_type"], record["bank_id"], record["message_id"], record["session_id"]) == (
            "representation",
            "b",
            5,
            "sid",
        )
        assert record["work_unit_key"] == "representation:b:s:a"
        with pytest.raises(TypeError):
            create_representation_record({**message, "message_id": None}, conf, observers=["a"], observed="a")

    def test_effective_observe_me_prefers_the_session_setting(self) -> None:
        assert get_effective_observe_me("a", {}) is True
        assert get_effective_observe_me("a", {"a": [{"observe_me": False}, {}, True]}) is False
        assert get_effective_observe_me("a", {"a": [{"observe_me": False}, {"observe_me": True}, True]}) is True
        assert get_effective_observe_me("a", {"a": [{"observe_me": True}, {"observe_me": False}, True]}) is False


@pytest.mark.asyncio
class TestEnqueueDream:
    async def test_inserts_one_dream_item(self, ctx: PeopleContext, bank_id: str) -> None:
        await enqueue_dream(ctx, bank_id, "alice", "bob", DreamType.OMNI, trigger_reason="threshold")

        (item,) = await _queue(ctx, bank_id)
        assert item["task_type"] == "dream"
        assert item["message_id"] is None
        assert item["work_unit_key"] == f"dream:omni:{bank_id}:alice:bob"
        payload = from_jsonb(item["payload"])
        assert (payload["observer"], payload["observed"], payload["trigger_reason"]) == ("alice", "bob", "threshold")

    async def test_a_pending_dream_is_not_enqueued_twice(self, ctx: PeopleContext, bank_id: str) -> None:
        await enqueue_dream(ctx, bank_id, "alice", "bob", DreamType.OMNI)
        await enqueue_dream(ctx, bank_id, "alice", "bob", DreamType.OMNI)
        assert len(await _queue(ctx, bank_id)) == 1

        # The dream type is part of the key: a card refresh does not collide with a pending omni dream.
        await enqueue_dream(ctx, bank_id, "alice", "bob", DreamType.CARD_REFRESH)
        assert len(await _queue(ctx, bank_id)) == 2

    async def test_a_dream_in_progress_is_not_enqueued(self, ctx: PeopleContext, bank_id: str) -> None:
        record = create_dream_record(bank_id, observer="alice", observed="bob", dream_type=DreamType.OMNI)
        async with ctx.conn() as conn:
            await conn.execute(
                f"INSERT INTO {PeopleContext.table('active_queue_sessions')} (bank_id, id, work_unit_key) "
                "VALUES ($1, 'claim', $2)",
                bank_id,
                record["work_unit_key"],
            )
        await enqueue_dream(ctx, bank_id, "alice", "bob", DreamType.OMNI)
        assert await _queue(ctx, bank_id) == []

    async def test_a_dream_enqueued_again_after_it_ran_is_queued(self, ctx: PeopleContext, bank_id: str) -> None:
        await enqueue_dream(ctx, bank_id, "alice", "bob", DreamType.OMNI)
        async with ctx.conn() as conn:
            await conn.execute(
                f"UPDATE {PeopleContext.table('queue')} SET processed = true WHERE bank_id = $1", bank_id
            )
        await enqueue_dream(ctx, bank_id, "alice", "bob", DreamType.OMNI)
        assert [q["processed"] for q in await _queue(ctx, bank_id)] == [True, False]

    async def test_a_second_concurrent_insert_of_the_same_dream_is_rejected(
        self, ctx: PeopleContext, bank_id: str
    ) -> None:
        """Two replicas listing the same due dream both try to enqueue it. enqueue_dream checks for a
        pending row and then inserts, so two concurrent calls can both pass the check;
        uq_people_queue_dream_pending is what keeps the second insert out."""
        record = create_dream_record(bank_id, observer="alice", observed="bob", dream_type=DreamType.OMNI)
        insert = (
            f"INSERT INTO {PeopleContext.table('queue')} (bank_id, session_id, work_unit_key, task_type, payload) "
            "VALUES ($1, NULL, $2, 'dream', $3::jsonb)"
        )
        inserted = asyncio.Event()
        release = asyncio.Event()

        async def first_insert() -> None:
            async with ctx.conn() as first, first.transaction():
                await first.execute(insert, bank_id, record["work_unit_key"], to_jsonb(record["payload"]))
                inserted.set()
                await release.wait()

        async def second_insert() -> None:
            await inserted.wait()
            async with ctx.conn() as second:
                await second.execute(insert, bank_id, record["work_unit_key"], to_jsonb(record["payload"]))

        committing = asyncio.create_task(first_insert())
        pending = asyncio.create_task(second_insert())
        await inserted.wait()
        await asyncio.sleep(0.2)  # the second insert is now waiting on the first's uncommitted row
        release.set()
        await committing
        with pytest.raises(asyncpg.UniqueViolationError, match="uq_people_queue_dream_pending"):
            await pending

    async def test_a_losing_concurrent_enqueue_is_swallowed(self, ctx: PeopleContext, bank_id: str) -> None:
        await asyncio.gather(*(enqueue_dream(ctx, bank_id, "alice", "bob", DreamType.OMNI) for _ in range(4)))
        assert len(await _queue(ctx, bank_id)) == 1
