# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: tests/deriver/test_queue_invariants.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""The deriver end to end: enqueue what people said, drain the queue, read the conclusions.

No test here calls a model: ``FakeLLM`` returns scripted answers. The invariant scenarios inject one fault
into the deriver, drain the queue the way the polling loop does, then check that nothing vanishes (every
message has a queue item), nothing gets stuck (no item left unprocessed, no work unit left claimed) and
errors stay visible (an errored item says why).
"""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest

from hindsight_api.people.context import PeopleContext
from hindsight_api.people.deriver import IncomingMessage, enqueue_messages, process_available
from hindsight_api.people.deriver.enqueue import enqueue_dream
from hindsight_api.people.deriver.queue_manager import MAX_RETRYABLE_ATTEMPTS, QueueManager
from hindsight_api.people.deriver.settings import DERIVER
from hindsight_api.people.dreamer.schemas import DreamType
from hindsight_api.people.jsonb import from_jsonb, to_jsonb
from hindsight_api.people.representation_utils import ExplicitObservationBase, PromptRepresentation
from tests.people.conftest import FakeLLM
from tests.people.deriver_support import claim_rows

BATCH = "hindsight_api.people.deriver.queue_manager.process_representation_batch"
RETRY_BACKOFF = "hindsight_api.people.deriver.queue_manager.RETRY_BACKOFF_SECONDS"
SAID_AT = datetime(2026, 5, 1, 9, 30, tzinfo=UTC)
MESSAGES_PER_SCENARIO = 3
MAX_DRAIN_ROUNDS = 50


def _answer(*contents: str, source_indices: list[int] | None = None) -> PromptRepresentation:
    return PromptRepresentation(
        explicit=[ExplicitObservationBase(content=c, source_indices=source_indices or [0]) for c in contents]
    )


async def _rows(ctx: PeopleContext, sql: str, *args: Any) -> list[Any]:
    async with ctx.conn() as conn:
        return await conn.fetch(sql, *args)


async def _queue(ctx: PeopleContext, bank_id: str) -> list[Any]:
    return await _rows(ctx, f"SELECT * FROM {PeopleContext.table('queue')} WHERE bank_id = $1 ORDER BY id", bank_id)


async def _conclusions(ctx: PeopleContext, bank_id: str) -> list[Any]:
    return await _rows(
        ctx,
        f"SELECT * FROM {PeopleContext.table('conclusions')} WHERE bank_id = $1 AND deleted_at IS NULL ORDER BY created_at, id",
        bank_id,
    )


@pytest.fixture(autouse=True)
def _fast_deriver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(DERIVER, "FLUSH_ENABLED", False)
    monkeypatch.setattr(DERIVER, "REPRESENTATION_BATCH_WORK_UNIT_TARGET_TOKENS", 512)
    monkeypatch.setattr(DERIVER, "REPRESENTATION_BATCH_TARGET_INPUT_TOKENS", 1024)
    monkeypatch.setattr(DERIVER, "REPRESENTATION_BATCH_MAX_AGE_SECONDS", 1800)
    monkeypatch.setattr(DERIVER, "WORKERS", 4)
    monkeypatch.setattr(RETRY_BACKOFF, 0.0)


@pytest.mark.asyncio
class TestDeriveWhatPeopleSaid:
    async def test_enqueue_then_drain_saves_conclusions_that_cite_their_source(
        self, ctx: PeopleContext, bank_id: str, fake_llm: FakeLLM
    ) -> None:
        fake_llm.script = [_answer("alice moved to Lisbon in May 2026", source_indices=[0])]
        (message_id,) = await enqueue_messages(
            ctx,
            bank_id,
            [
                IncomingMessage(
                    peer="alice",
                    content="I moved to Lisbon last week",
                    session="slack:C1",
                    said_at=SAID_AT,
                    channel="#general",
                    source_url="https://slack.example/archives/C1/p1",
                    document_id="doc-1",
                )
            ],
        )

        processed = await process_available(ctx, bank_id)

        assert processed == 1
        assert len(fake_llm.calls) == 1
        prompt = fake_llm.calls[0]["messages"][0]["content"]
        # The deriver reads the message as the person said it, dated by when they said it.
        assert (
            '<message idx="0" peer="alice" target="true" time="2026-05-01 09:30:00">I moved to Lisbon last week</message>'
            in prompt
        )
        (conclusion,) = await _conclusions(ctx, bank_id)
        assert (conclusion["observer"], conclusion["observed"], conclusion["level"]) == ("alice", "alice", "explicit")
        assert conclusion["content"] == "alice moved to Lisbon in May 2026"
        # Every conclusion keeps where it came from.
        assert conclusion["source_url"] == "https://slack.example/archives/C1/p1"
        assert conclusion["said_at"] == SAID_AT
        (public_id,) = [
            m["public_id"]
            for m in await _rows(
                ctx,
                f"SELECT public_id FROM {PeopleContext.table('messages')} WHERE bank_id = $1 AND id = $2",
                bank_id,
                message_id,
            )
        ]
        sources = await _rows(
            ctx,
            f"SELECT source_id, kind FROM {PeopleContext.table('conclusion_sources')} WHERE bank_id = $1 AND conclusion_id = $2",
            bank_id,
            conclusion["id"],
        )
        assert [(s["source_id"], s["kind"]) for s in sources] == [(public_id, "message")]
        assert all(q["processed"] and q["error"] is None for q in await _queue(ctx, bank_id))
        assert await claim_rows(ctx, bank_id) == 0

    async def test_a_second_drain_finds_nothing(self, ctx: PeopleContext, bank_id: str, fake_llm: FakeLLM) -> None:
        fake_llm.script = [_answer("alice likes tea")]
        await enqueue_messages(ctx, bank_id, [IncomingMessage("alice", "I like tea", "s1", said_at=SAID_AT)])
        assert await process_available(ctx, bank_id) == 1
        assert await process_available(ctx, bank_id) == 0
        assert len(fake_llm.calls) == 1

    async def test_a_conversation_is_derived_once_per_speaker_with_the_other_side_as_context(
        self, ctx: PeopleContext, bank_id: str, fake_llm: FakeLLM
    ) -> None:
        def script(_index: int, messages: list[dict[str, Any]], _fmt: Any) -> PromptRepresentation:
            prompt = messages[0]["content"]
            who = "alice" if "Write `alice`" in prompt else "bob"
            return _answer(f"{who} said something worth keeping")

        fake_llm.script = script
        await enqueue_messages(
            ctx,
            bank_id,
            [
                IncomingMessage("bob", "Which database should we use?", "s1", said_at=SAID_AT),
                IncomingMessage("alice", "SQLite, the first one", "s1", said_at=SAID_AT),
                IncomingMessage("bob", "Great, thanks", "s1", said_at=SAID_AT),
            ],
        )

        processed = await process_available(ctx, bank_id)

        assert processed == 3
        prompts = {
            ("alice" if "Write `alice`" in c["messages"][0]["content"] else "bob"): c["messages"][0]["content"]
            for c in fake_llm.calls
        }
        assert len(fake_llm.calls) == 2
        # alice's unit: bob's question precedes her answer as context and is not a target message
        assert 'peer="bob" target="false"' in prompts["alice"] and 'peer="alice" target="true"' in prompts["alice"]
        observed = sorted((c["observer"], c["observed"]) for c in await _conclusions(ctx, bank_id))
        assert observed == [("alice", "alice"), ("bob", "bob")]

    async def test_a_person_who_watches_the_others_gets_the_same_conclusions(
        self, ctx: PeopleContext, bank_id: str, fake_llm: FakeLLM
    ) -> None:
        fake_llm.script = [_answer("alice runs the Lisbon office")]
        await enqueue_messages(ctx, bank_id, [IncomingMessage("bob", "hello", "s1", said_at=SAID_AT)])
        async with ctx.conn() as conn:
            await conn.execute(
                f"UPDATE {PeopleContext.table('session_peers')} SET configuration = $2::jsonb "
                "WHERE bank_id = $1 AND peer_name = 'bob'",
                bank_id,
                to_jsonb({"observe_others": True}),
            )
        await enqueue_messages(
            ctx, bank_id, [IncomingMessage("alice", "I run the Lisbon office", "s1", said_at=SAID_AT)]
        )
        fake_llm.calls.clear()

        await process_available(ctx, bank_id)

        pairs = {(c["observer"], c["observed"]) for c in await _conclusions(ctx, bank_id)}
        assert ("alice", "alice") in pairs and ("bob", "alice") in pairs

    async def test_the_batch_gate_holds_a_small_unit_back_until_flushed(
        self, ctx: PeopleContext, bank_id: str, fake_llm: FakeLLM
    ) -> None:
        fake_llm.script = [_answer("alice likes tea")]
        await enqueue_messages(ctx, bank_id, [IncomingMessage("alice", "I like tea", "s1", said_at=SAID_AT)])

        assert await process_available(ctx, bank_id, flush=False) == 0
        assert fake_llm.calls == []
        assert await process_available(ctx, bank_id) == 1
        assert len(fake_llm.calls) == 1

    async def test_only_the_asked_bank_is_drained(self, ctx: PeopleContext, bank_id: str, fake_llm: FakeLLM) -> None:
        other = f"test-people-{uuid.uuid4().hex[:10]}"
        await ctx.engine.ensure_bank_profile(bank_id=other, request_context=ctx.request_context)
        fake_llm.script = [_answer("alice likes tea")]
        await enqueue_messages(ctx, bank_id, [IncomingMessage("alice", "I like tea", "s1", said_at=SAID_AT)])
        await enqueue_messages(ctx, other, [IncomingMessage("alice", "I like tea", "s1", said_at=SAID_AT)])

        assert await process_available(ctx, bank_id) == 1

        assert all(q["processed"] for q in await _queue(ctx, bank_id))
        assert not any(q["processed"] for q in await _queue(ctx, other))

    async def test_reasoning_disabled_for_a_message_makes_no_model_call(
        self, ctx: PeopleContext, bank_id: str, fake_llm: FakeLLM
    ) -> None:
        from hindsight_api.people.schemas import MessageConfiguration

        configuration = MessageConfiguration.model_validate({"reasoning": {"enabled": False}})
        await enqueue_messages(
            ctx, bank_id, [IncomingMessage("alice", "I like tea", "s1", said_at=SAID_AT, configuration=configuration)]
        )
        assert await process_available(ctx, bank_id) == 0
        assert fake_llm.calls == []

    async def test_a_dream_in_the_queue_is_handed_to_the_dreamer(self, ctx: PeopleContext, bank_id: str) -> None:
        await enqueue_dream(ctx, bank_id, "alice", "bob", DreamType.OMNI, trigger_reason="threshold")

        with patch("hindsight_api.people.dreamer.orchestrator.process_dream", new_callable=AsyncMock) as process:
            assert await process_available(ctx, bank_id) == 1

        process.assert_awaited_once()
        _ctx, payload, bank = process.await_args.args
        assert (payload.observer, payload.observed, payload.trigger_reason, bank) == (
            "alice",
            "bob",
            "threshold",
            bank_id,
        )
        assert process.await_args.kwargs["queue_item_id"] is not None
        assert all(q["processed"] and q["error"] is None for q in await _queue(ctx, bank_id))

    async def test_concurrent_drains_derive_each_unit_once(
        self, ctx: PeopleContext, bank_id: str, fake_llm: FakeLLM
    ) -> None:
        import asyncio

        fake_llm.script = lambda _i, messages, _f: _answer(
            "fact about " + ("alice" if "`alice`" in messages[0]["content"] else "bob")
        )
        await enqueue_messages(
            ctx,
            bank_id,
            [
                IncomingMessage(f"p{i}", f"I am person {i} and I like tea very much", "s1", said_at=SAID_AT)
                for i in range(6)
            ],
        )

        totals = await asyncio.gather(*(process_available(ctx, bank_id) for _ in range(3)))

        assert sum(totals) == 6
        assert len(fake_llm.calls) == 6
        assert all(q["processed"] for q in await _queue(ctx, bank_id))


@dataclass(frozen=True)
class Scenario:
    # Raised in order by the deriver call, which then succeeds
    deriver_errors: tuple[Exception, ...] = ()
    # Raised by every deriver call
    deriver_error_forever: Exception | None = None
    # A queue row whose key parse_work_unit_key rejects
    unparsable_work_unit_key: bool = False


_OUTAGE = aiohttp.ClientConnectionError("upstream unreachable")

SCENARIOS = [
    pytest.param(Scenario(), id="healthy"),
    pytest.param(Scenario(deriver_errors=(_OUTAGE,) * (MAX_RETRYABLE_ATTEMPTS - 1)), id="provider-outage-recovers"),
    pytest.param(Scenario(deriver_error_forever=_OUTAGE), id="provider-outage-outlasts-retries"),
    pytest.param(Scenario(deriver_error_forever=ValueError("malformed output")), id="deriver-fails-terminally"),
    pytest.param(Scenario(unparsable_work_unit_key=True), id="unparsable-work-unit-key"),
]


@pytest.mark.asyncio
class TestQueueInvariants:
    """Nothing vanishes, nothing gets stuck, errors stay visible."""

    @pytest.mark.parametrize("scenario", SCENARIOS)
    async def test_invariants_hold(self, ctx: PeopleContext, bank_id: str, scenario: Scenario) -> None:
        message_ids = await enqueue_messages(
            ctx,
            bank_id,
            [IncomingMessage("alice", f"fact number {i}", "s1", said_at=SAID_AT) for i in range(MESSAGES_PER_SCENARIO)],
        )
        if scenario.unparsable_work_unit_key:
            async with ctx.conn() as conn:
                await conn.execute(
                    f"UPDATE {PeopleContext.table('queue')} SET work_unit_key = 'representation:not-a-valid-key' "
                    "WHERE bank_id = $1",
                    bank_id,
                )

        deriver = AsyncMock(
            side_effect=scenario.deriver_error_forever or [*scenario.deriver_errors, *([None] * MAX_DRAIN_ROUNDS)]
        )
        with patch(BATCH, deriver):
            # Claim and process work units until none are claimable, like the polling loop.
            qm = QueueManager(lambda: ctx, bank_id=bank_id)
            for _ in range(MAX_DRAIN_ROUNDS):
                if not await qm.process_available(bank_id, flush=True):
                    unfinished = [q for q in await _queue(ctx, bank_id) if not q["processed"]]
                    if not unfinished:
                        break

        queue = await _queue(ctx, bank_id)
        # nothing vanishes: every message has a representation queue item
        assert sorted(q["message_id"] for q in queue if q["task_type"] == "representation") == sorted(message_ids)
        # nothing gets stuck: no item left unprocessed and no work unit stays claimed
        assert all(q["processed"] for q in queue), [(q["id"], q["error"]) for q in queue if not q["processed"]]
        assert await claim_rows(ctx, bank_id) == 0
        # errors stay visible: an item that did not get derived says why
        errored = [q for q in queue if q["error"] is not None]
        if scenario.deriver_error_forever is not None or scenario.unparsable_work_unit_key:
            assert len(errored) == len(queue)
        else:
            assert errored == []

    async def test_a_failed_enqueue_leaves_no_message_without_its_queue_item(
        self, ctx: PeopleContext, bank_id: str
    ) -> None:
        """A message is never stored with its derivation lost: the queue insert failing rolls the message
        back and the caller sees the error."""
        with patch(
            "hindsight_api.people.deriver.enqueue._insert_queue_records",
            new=AsyncMock(side_effect=RuntimeError("insert failed")),
        ):
            with pytest.raises(RuntimeError, match="insert failed"):
                await enqueue_messages(ctx, bank_id, [IncomingMessage("alice", "hello", "s1", said_at=SAID_AT)])
        messages = await _rows(ctx, f"SELECT id FROM {PeopleContext.table('messages')} WHERE bank_id = $1", bank_id)
        assert messages == []
        assert await _queue(ctx, bank_id) == []

    async def test_a_transient_enqueue_failure_is_retried(self, ctx: PeopleContext, bank_id: str) -> None:
        import asyncpg

        from hindsight_api.people.deriver import enqueue as enqueue_module

        real = enqueue_module._insert_queue_records
        calls = 0

        async def flaky(*args: Any, **kwargs: Any) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise asyncpg.exceptions.DeadlockDetectedError("deadlock detected")
            await real(*args, **kwargs)

        with patch.object(enqueue_module, "_insert_queue_records", flaky):
            await enqueue_messages(ctx, bank_id, [IncomingMessage("alice", "hello", "s1", said_at=SAID_AT)])

        assert calls == 2
        assert len(await _queue(ctx, bank_id)) == 1
        assert (
            len(await _rows(ctx, f"SELECT id FROM {PeopleContext.table('messages')} WHERE bank_id = $1", bank_id)) == 1
        )

    async def test_queue_payload_round_trips_as_json(self, ctx: PeopleContext, bank_id: str) -> None:
        await enqueue_messages(ctx, bank_id, [IncomingMessage("alice", "hello", "s1", said_at=SAID_AT)])
        (item,) = await _queue(ctx, bank_id)
        payload = from_jsonb(item["payload"])
        assert payload["configuration"]["reasoning"]["enabled"] is True
