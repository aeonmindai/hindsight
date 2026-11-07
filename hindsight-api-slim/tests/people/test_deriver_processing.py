# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: tests/deriver/test_deriver_processing.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""The deriver's batch processing: one model call, a save for every observer.

No test here calls a model: ``FakeLLM`` returns scripted answers and records the calls.
"""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import asyncpg
import pytest

from hindsight_api.engine.response_models import LLMCallResult, TokenUsage
from hindsight_api.people.conclusions import CreateDocumentsResult
from hindsight_api.people.context import PeopleContext
from hindsight_api.people.deriver.deriver import (
    EmptyRepresentationError,
    RepresentationSaveError,
    process_representation_tasks_batch,
)
from hindsight_api.people.deriver.models import Message
from hindsight_api.people.deriver.settings import DERIVER
from hindsight_api.people.deriver.work_unit import construct_work_unit_key, parse_work_unit_key
from hindsight_api.people.representation import RepresentationManager
from hindsight_api.people.representation_utils import (
    ExplicitObservationBase,
    PromptRepresentation,
    Representation,
)
from hindsight_api.people.schemas import get_configuration
from tests.people.conftest import FakeLLM

BANK = "bank-1"


def _message(row_id: int, public_id: str, peer: str, content: str = "...", tokens: int = 5) -> Message:
    return Message(
        id=row_id,
        bank_id=BANK,
        public_id=public_id,
        session_name="session-1",
        peer_name=peer,
        content=content,
        token_count=tokens,
        created_at=datetime.now(UTC),
    )


def _answer(*contents: str, source_indices: list[int] | None = None) -> PromptRepresentation:
    return PromptRepresentation(
        explicit=[ExplicitObservationBase(content=c, source_indices=source_indices or []) for c in contents]
    )


class _UsageLLM(FakeLLM):
    """A FakeLLM whose answers report ``output_tokens`` (the deriver reads it to spot truncation)."""

    def __init__(self, script: Any, output_tokens: int) -> None:
        super().__init__(script)
        self.output_tokens = output_tokens

    async def call(self, messages: list[dict[str, Any]], response_format: Any = None, **kwargs: Any) -> LLMCallResult:
        result = await super().call(messages, response_format, **kwargs)
        return LLMCallResult(content=result.content, usage=TokenUsage(output_tokens=self.output_tokens))


async def _run(ctx: PeopleContext, messages: list[Message], **overrides: Any) -> None:
    kwargs: dict[str, Any] = {
        "observers": ["bob"],
        "observed": "alice",
        "queue_item_message_ids": [m.id for m in messages],
        "session_id": "canonical-session-1",
    }
    kwargs.update(overrides)
    configuration = kwargs.pop("configuration", get_configuration(None, None, None))
    await process_representation_tasks_batch(ctx, BANK, messages, configuration, **kwargs)


@pytest.mark.asyncio
class TestDeriverProcessing:
    async def test_cited_indices_resolve_to_batch_public_ids(self, ctx: PeopleContext, fake_llm: FakeLLM) -> None:
        """source_indices are batch positions; the saved observation carries the public ids of those
        messages, including a non-target context message."""
        messages = [
            _message(10, "bob_asks_______000010", "bob"),
            _message(20, "alice_picks____000020", "alice"),
            _message(30, "bob_acks_______000030", "bob"),
        ]
        fake_llm.script = [_answer("alice chose SQLite", source_indices=[1, 0, 7])]
        saved: list[Representation] = []

        async def capture_save(
            _self: RepresentationManager, observations: Representation, *_args: Any, **_kwargs: Any
        ) -> CreateDocumentsResult:
            saved.append(observations)
            return CreateDocumentsResult()

        with patch.object(RepresentationManager, "save_representation", capture_save):
            await _run(ctx, messages, queue_item_message_ids=[20])

        assert len(saved) == 1
        observation = saved[0].explicit[0]
        assert observation.message_ids == [20]
        # Citation order is preserved; the out-of-range index is dropped.
        assert observation.source_message_ids == ["alice_picks____000020", "bob_asks_______000010"]

    async def test_one_call_goes_to_the_background_provider_with_the_prompt(
        self, ctx: PeopleContext, fake_llm: FakeLLM
    ) -> None:
        said_at = datetime(2026, 3, 4, 12, 30, tzinfo=UTC)
        message = _message(1, "msg_1__________000001", "alice", content="I moved to Lisbon")
        message.said_at = said_at
        fake_llm.script = [_answer("alice moved to Lisbon")]

        with patch.object(
            RepresentationManager, "save_representation", AsyncMock(return_value=CreateDocumentsResult())
        ) as save:
            await _run(ctx, [message])

        assert len(fake_llm.calls) == 1
        call = fake_llm.calls[0]
        assert call["response_format"] is PromptRepresentation
        assert call["max_completion_tokens"] == DERIVER.MAX_OUTPUT_TOKENS
        assert call["scope"] == "people_deriver"
        prompt = call["messages"][0]["content"]
        # The prompt carries what was said and when the person said it (not when it was stored).
        assert (
            '<message idx="0" peer="alice" target="true" time="2026-03-04 12:30:00">I moved to Lisbon</message>'
            in prompt
        )
        assert "Write `alice` as the subject of every observation" in prompt
        save.assert_awaited_once()
        assert save.await_args.args[2] == "session-1"
        assert save.await_args.args[3] == said_at

    async def test_reasoning_disabled_skips_the_call(self, ctx: PeopleContext, fake_llm: FakeLLM) -> None:
        configuration = get_configuration(None, None, None)
        configuration.reasoning.enabled = False
        await _run(ctx, [_message(1, "msg_1__________000001", "alice")], configuration=configuration)
        assert fake_llm.calls == []

    async def test_custom_instructions_reach_the_prompt(self, ctx: PeopleContext, fake_llm: FakeLLM) -> None:
        configuration = get_configuration(None, None, None)
        configuration.reasoning.custom_instructions = "Prefer concrete timeline facts."
        fake_llm.script = [_answer()]

        await _run(ctx, [_message(1, "msg_1__________000001", "alice")], configuration=configuration)

        prompt = fake_llm.calls[0]["messages"][0]["content"]
        assert "CUSTOM INSTRUCTIONS:" in prompt
        assert "Prefer concrete timeline facts." in prompt

    @pytest.mark.parametrize(
        ("answer", "output_tokens", "parse_class"),
        [
            # A refusal comes back as its text, not the model.
            ("I can't help with that.", 10, "content_filtered"),
            # The output budget ran out right after the opening of the list.
            (PromptRepresentation(explicit=[]), DERIVER.MAX_OUTPUT_TOKENS, "truncated"),
        ],
    )
    async def test_degraded_empty_representation_is_refused(
        self, ctx: PeopleContext, people_engine: Any, answer: Any, output_tokens: int, parse_class: str
    ) -> None:
        """A truncated, filtered, or refused empty is retried, then refused, with the prompt's size but none
        of its text."""
        secret = "my-sentinel-secret-4417"
        llm = _UsageLLM([answer], output_tokens)
        people_engine._consolidation_llm_config = llm

        with pytest.raises(EmptyRepresentationError) as exc_info:
            await _run(ctx, [_message(1, "msg_1__________000001", "alice", content=f"my password is {secret}")])

        assert exc_info.value.parse_class == parse_class
        assert secret not in str(exc_info.value)
        assert len(llm.calls) == 3

    async def test_a_degraded_answer_is_retried_and_the_second_one_is_used(
        self, ctx: PeopleContext, fake_llm: FakeLLM
    ) -> None:
        fake_llm.script = ["I can't help with that.", _answer("alice has a dog")]
        with patch.object(
            RepresentationManager, "save_representation", AsyncMock(return_value=CreateDocumentsResult())
        ) as save:
            await _run(ctx, [_message(1, "msg_1__________000001", "alice")])
        assert len(fake_llm.calls) == 2
        save.assert_awaited_once()

    @pytest.mark.parametrize(
        ("answer", "output_tokens"),
        [
            # Zero output tokens is not a degradation signal: backends report 0 when the provider omits usage.
            (PromptRepresentation(explicit=[]), 0),
            # A truncated response that still yielded observations keeps them.
            (PromptRepresentation(explicit=[ExplicitObservationBase(content="Has a dog")]), DERIVER.MAX_OUTPUT_TOKENS),
        ],
    )
    async def test_clean_or_salvaged_representation_is_accepted(
        self, ctx: PeopleContext, people_engine: Any, answer: Any, output_tokens: int
    ) -> None:
        llm = _UsageLLM([answer], output_tokens)
        people_engine._consolidation_llm_config = llm
        with patch.object(
            RepresentationManager, "save_representation", AsyncMock(return_value=CreateDocumentsResult())
        ):
            await _run(ctx, [_message(1, "msg_1__________000001", "alice")])
        assert len(llm.calls) == 1

    async def test_anything_is_accepted_with_no_observed_messages(self, ctx: PeopleContext, people_engine: Any) -> None:
        """With no messages from the observed person there is nothing to lose."""
        llm = _UsageLLM([PromptRepresentation(explicit=[])], DERIVER.MAX_OUTPUT_TOKENS)
        people_engine._consolidation_llm_config = llm
        await _run(ctx, [_message(1, "msg_1__________000001", "bob")])
        assert len(llm.calls) == 1

    async def test_all_observer_saves_failing_surfaces_failure(self, ctx: PeopleContext, fake_llm: FakeLLM) -> None:
        """When every observer's save_representation fails, the batch must raise."""
        fake_llm.script = [_answer("The user has a dog named Rover")]
        failing_save = AsyncMock(side_effect=RuntimeError("429 RESOURCE_EXHAUSTED"))
        with (
            patch.object(RepresentationManager, "save_representation", failing_save),
            pytest.raises(RepresentationSaveError, match="save_representation failed"),
        ):
            await _run(ctx, [_message(1, "msg_1__________000001", "alice")])

    async def test_partial_observer_failure_is_processed(self, ctx: PeopleContext, fake_llm: FakeLLM) -> None:
        """When some observers save and one fails, the batch does NOT raise (saved observers are kept)."""
        fake_llm.script = [_answer("The user has a dog named Rover")]
        observers_saved: list[str] = []

        async def save(self: RepresentationManager, *_args: Any, **_kwargs: Any) -> CreateDocumentsResult:
            if self.observer == "carol":
                raise RuntimeError("boom")
            observers_saved.append(self.observer)
            return CreateDocumentsResult()

        with patch.object(RepresentationManager, "save_representation", save):
            await _run(ctx, [_message(1, "msg_1__________000001", "alice")], observers=["alice", "carol", "bob"])

        assert observers_saved == ["alice", "bob"]

    async def test_retryable_observer_save_reraises(self, ctx: PeopleContext, fake_llm: FakeLLM) -> None:
        """A transient save failure is re-raised so the queue manager retries the batch."""
        fake_llm.script = [_answer("The user has a dog named Rover")]
        transient = asyncpg.exceptions.DeadlockDetectedError("deadlock detected")

        async def save(self: RepresentationManager, *_args: Any, **_kwargs: Any) -> CreateDocumentsResult:
            if self.observer == "carol":
                raise transient
            return CreateDocumentsResult()

        with (
            patch.object(RepresentationManager, "save_representation", save),
            pytest.raises(asyncpg.exceptions.DeadlockDetectedError),
        ):
            await _run(ctx, [_message(1, "msg_1__________000001", "alice")], observers=["alice", "carol"])

    async def test_one_call_is_saved_once_per_observer(self, ctx: PeopleContext, fake_llm: FakeLLM) -> None:
        fake_llm.script = [_answer("The user has a dog named Rover")]
        observers: list[str] = []

        async def save(self: RepresentationManager, *_args: Any, **_kwargs: Any) -> CreateDocumentsResult:
            observers.append(self.observer)
            assert self.observed == "alice"
            return CreateDocumentsResult()

        with patch.object(RepresentationManager, "save_representation", save):
            await _run(ctx, [_message(1, "msg_1__________000001", "alice")], observers=["alice", "bob"])
        assert len(fake_llm.calls) == 1
        assert observers == ["alice", "bob"]

    async def test_session_id_is_resolved_when_missing(self, ctx: PeopleContext, fake_llm: FakeLLM) -> None:
        """A direct caller without the queue's session id and configuration still gets a derivation."""
        fake_llm.script = [_answer()]
        await process_representation_tasks_batch(
            ctx,
            BANK,
            [_message(1, "msg_1__________000001", "alice")],
            None,
            observers=["alice"],
            observed="alice",
            queue_item_message_ids=[1],
        )
        assert len(fake_llm.calls) == 1


class TestWorkUnitKeys:
    def test_work_unit_key_generation(self) -> None:
        key = construct_work_unit_key(
            "bank-1", {"task_type": "representation", "session_name": "s1", "observers": ["bob"], "observed": "alice"}
        )
        # Representation tasks don't include the observer in the key: one unit saves to every observer.
        assert key == "representation:bank-1:s1:alice"
        parsed = parse_work_unit_key(key)
        assert (parsed.task_type, parsed.bank_id, parsed.session_name, parsed.observed) == (
            "representation",
            "bank-1",
            "s1",
            "alice",
        )
        assert parsed.observer is None

    def test_names_with_colons_and_percent_signs_round_trip(self) -> None:
        key = construct_work_unit_key(
            "bank:1", {"task_type": "representation", "session_name": "slack:C0%1", "observed": "a:b"}
        )
        assert key.count(":") == 3
        parsed = parse_work_unit_key(key)
        assert (parsed.bank_id, parsed.session_name, parsed.observed) == ("bank:1", "slack:C0%1", "a:b")

    def test_dream_work_unit_key(self) -> None:
        key = construct_work_unit_key(
            "bank-1", {"task_type": "dream", "dream_type": "omni", "observer": "alice", "observed": "bob"}
        )
        assert key == "dream:omni:bank-1:alice:bob"
        parsed = parse_work_unit_key(key)
        assert (parsed.task_type, parsed.dream_type, parsed.bank_id, parsed.observer, parsed.observed) == (
            "dream",
            "omni",
            "bank-1",
            "alice",
            "bob",
        )

    def test_a_dream_without_a_type_is_refused(self) -> None:
        with pytest.raises(ValueError, match="dream_type is required"):
            construct_work_unit_key("bank-1", {"task_type": "dream", "observer": "a", "observed": "b"})

    @pytest.mark.parametrize(
        "key", ["representation:bank-1:s1", "representation:a:b:c:d", "dream:omni:bank-1:a", "summary:x:y:z"]
    )
    def test_invalid_keys_raise(self, key: str) -> None:
        with pytest.raises(ValueError):
            parse_work_unit_key(key)
