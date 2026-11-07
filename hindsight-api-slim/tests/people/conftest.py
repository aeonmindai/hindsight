"""Fixtures for the people tests (ported from Honcho's tests for peers, representations, the deriver
and the dreamer). No test here calls a model: `FakeLLM` returns scripted answers and records the calls.
"""

import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest
import pytest_asyncio
from pydantic import BaseModel

from hindsight_api.engine.cross_encoder import RRFPassthroughCrossEncoder
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.response_models import LLMCallResult, LLMToolCallResult
from hindsight_api.people.context import PeopleContext
from tests.test_retain_same_document_concurrency import _StubEmbeddings


class FakeLLM:
    """A scripted stand-in for the engine's Claude-configured provider.

    `script` is a list of answers returned in order (the last repeats), or a callable
    `(call_index, messages, response_format) -> answer`. An answer for `call` is a pydantic
    instance, a dict (validated into `response_format` when one is given) or a string. An answer for
    `call_with_tools` is an `LLMToolCallResult`. Every call is recorded in `calls`.
    """

    provider = "mock"
    model = "mock"

    def __init__(self, script: "list[Any] | Callable[..., Any] | None" = None) -> None:
        self.script = script if script is not None else [""]
        self.calls: list[dict[str, Any]] = []

    def _answer(self, messages: list[dict[str, Any]], response_format: Any) -> Any:
        index = len(self.calls) - 1
        if callable(self.script):
            return self.script(index, messages, response_format)
        return self.script[min(index, len(self.script) - 1)]

    async def call(self, messages: list[dict[str, Any]], response_format: Any = None, **kwargs: Any) -> LLMCallResult:
        self.calls.append({"kind": "call", "messages": messages, "response_format": response_format, **kwargs})
        answer = self._answer(messages, response_format)
        if isinstance(answer, dict) and isinstance(response_format, type) and issubclass(response_format, BaseModel):
            answer = response_format.model_validate(answer)
        return LLMCallResult(content=answer)

    async def call_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], **kwargs: Any
    ) -> LLMToolCallResult:
        self.calls.append({"kind": "tools", "messages": messages, "tools": tools, **kwargs})
        answer = self._answer(messages, None)
        return (
            answer
            if isinstance(answer, LLMToolCallResult)
            else LLMToolCallResult(content=str(answer), finish_reason="stop")
        )


@pytest_asyncio.fixture
async def people_engine(pg0_db_url: str) -> AsyncIterator[MemoryEngine]:
    """A memory engine on the test database with stub embeddings and no model."""
    engine = MemoryEngine(
        db_url=pg0_db_url,
        memory_llm_provider="mock",
        memory_llm_api_key="",
        memory_llm_model="mock",
        embeddings=_StubEmbeddings(),
        cross_encoder=RRFPassthroughCrossEncoder(),
        pool_min_size=1,
        pool_max_size=4,
        run_migrations=False,
    )
    await engine.initialize()
    try:
        yield engine
    finally:
        await engine.close()


@pytest.fixture
def fake_llm(people_engine: MemoryEngine) -> FakeLLM:
    """A `FakeLLM` installed as both the background and the interactive provider of the engine."""
    llm = FakeLLM()
    people_engine._consolidation_llm_config = llm  # type: ignore[assignment]
    people_engine._reflect_llm_config = llm  # type: ignore[assignment]
    return llm


@pytest.fixture
def ctx(people_engine: MemoryEngine) -> PeopleContext:
    return PeopleContext.internal(people_engine)


@pytest_asyncio.fixture
async def bank_id(people_engine: MemoryEngine, ctx: PeopleContext) -> str:
    """A fresh bank per test."""
    bank = f"test-people-{uuid.uuid4().hex[:10]}"
    await people_engine.ensure_bank_profile(bank_id=bank, request_context=ctx.request_context)
    return bank
