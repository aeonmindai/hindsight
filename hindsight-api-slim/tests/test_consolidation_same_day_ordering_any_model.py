"""#5938 on any model: on the same day, the later statement is current whatever the model answers.

Upstream's fix (PR 5966) was checked on one small Gemini model. Its ordering is decided in code
(`_FactBatchResponse.to_actions` compares the times), so it must hold when the model answers badly. The
scripted model here does the wrong thing in both arrival orders of a same-day pair, 10:48 (the amendment)
and 09:48 (the rule it amends): it always rewrites the observation to the fact it was just shown.
No model is called.
"""

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio

from hindsight_api import RequestContext
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.consolidation.consolidator import _CreateAction, _UpdateAction, run_consolidation_job
from hindsight_api.engine.cross_encoder import RRFPassthroughCrossEncoder
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.providers.mock_llm import MockLLM
from tests.consolidation_answers import batch_fact_ids, fact_answer
from tests.test_retain_same_document_concurrency import _StubEmbeddings

pytestmark = pytest.mark.memory_backend_incompatible

AMENDMENT = "Overage billing includes a 25-unit monthly grace allowance before block rounding."
OLDER_RULE = "Overage billing uses 50-unit blocks rounded up, with no waivers for small overages."
AMENDMENT_AT = datetime(2026, 10, 21, 10, 48, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def enable_observations():
    config = _get_raw_config()
    original = config.enable_observations
    config.enable_observations = True
    yield
    config.enable_observations = original


@pytest_asyncio.fixture
async def engine(pg0_db_url: str):
    memory = MemoryEngine(
        db_url=pg0_db_url,
        memory_llm_provider="mock",
        memory_llm_api_key="",
        memory_llm_model="mock",
        embeddings=_StubEmbeddings(),
        cross_encoder=RRFPassthroughCrossEncoder(),
        pool_min_size=1,
        pool_max_size=5,
        run_migrations=False,
    )
    await memory.initialize()
    try:
        yield memory
    finally:
        await memory.close()


def _llm(callback):
    mock_llm = MockLLM(provider="mock", api_key="", base_url="", model="mock-model")
    mock_llm.set_response_callback(callback)
    wrapper = MagicMock()
    wrapper.with_config.return_value = mock_llm
    return wrapper


async def _insert_fact(engine: MemoryEngine, bank_id: str, text: str, said_at: datetime) -> uuid.UUID:
    unit_id = uuid.uuid4()
    embedding = (await engine.embeddings.encode([text]))[0]
    from hindsight_api.engine.retain.types import embedding_to_pgvector

    async with engine._pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO memory_units (id, bank_id, text, fact_type, event_date, mentioned_at, embedding, metadata) "
            "VALUES ($1, $2, $3, 'world', $4, $4, $5::vector, $6::jsonb)",
            unit_id,
            bank_id,
            text,
            said_at,
            embedding_to_pgvector(embedding),
            json.dumps({"said_by": "asha", "channel": "slack"}),
        )
    return unit_id


async def _consolidate(engine: MemoryEngine, bank_id: str, callback) -> None:
    original = engine._consolidation_llm_config
    engine._consolidation_llm_config = _llm(callback)
    raw = _get_raw_config()
    fake = type(raw)(**{**{f: getattr(raw, f) for f in raw.__dataclass_fields__}, "consolidation_llm_batch_size": 4})
    try:
        with (
            patch.object(engine._config_resolver, "resolve_full_config", return_value=fake),
            patch.object(engine, "submit_async_consolidation"),
        ):
            await run_consolidation_job(memory_engine=engine, bank_id=bank_id, request_context=RequestContext())
    finally:
        engine._consolidation_llm_config = original


async def _observations(engine: MemoryEngine, bank_id: str) -> list[dict]:
    async with engine._pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, text, source_memory_ids FROM memory_units WHERE bank_id = $1 AND fact_type = 'observation'",
            bank_id,
        )
    return [dict(r) for r in rows]


async def _two_facts(engine: MemoryEngine, bank_id: str, first: tuple[str, datetime], second: tuple[str, datetime]):
    """Consolidate `first`, then `second` with a model that rewrites the observation to `second`'s text."""
    await engine.ensure_bank_profile(bank_id=bank_id, request_context=RequestContext())
    first_text, first_at = first
    second_text, second_at = second
    first_id = await _insert_fact(engine, bank_id, first_text, first_at)

    def create_from_first(messages, scope):
        if scope != "consolidation":
            return fact_answer()
        prompt = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
        return fact_answer(
            creates=[_CreateAction(text=first_text, source_fact_ids=[fid]) for fid in batch_fact_ids(prompt)]
        )

    await _consolidate(engine, bank_id, create_from_first)
    assert [o["text"] for o in await _observations(engine, bank_id)] == [first_text]

    second_id = await _insert_fact(engine, bank_id, second_text, second_at)

    def rewrite_from_second(messages, scope):
        if scope != "consolidation":
            return fact_answer()
        prompt = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
        return fact_answer(
            updates=[_UpdateAction(observation_id="O1", text=second_text, source_fact_ids=batch_fact_ids(prompt))]
        )

    await _consolidate(engine, bank_id, rewrite_from_second)
    return first_id, second_id


@pytest.mark.asyncio
async def test_an_older_same_day_fact_arriving_later_does_not_replace_the_amendment(engine: MemoryEngine):
    bank_id = f"test-5938-older-late-{uuid.uuid4().hex[:8]}"
    amendment, older = await _two_facts(
        engine, bank_id, (AMENDMENT, AMENDMENT_AT), (OLDER_RULE, AMENDMENT_AT.replace(hour=9))
    )

    (observation,) = await _observations(engine, bank_id)
    # The current text is still the amendment; the older rule is only an earlier state, dated.
    assert observation["text"].startswith(AMENDMENT)
    assert OLDER_RULE in observation["text"]
    assert observation["text"].index(AMENDMENT) < observation["text"].index(OLDER_RULE)
    assert "2026-10-21 09:48" in observation["text"]
    assert {str(s) for s in observation["source_memory_ids"]} == {str(amendment), str(older)}


@pytest.mark.asyncio
async def test_the_amendment_arriving_later_replaces_the_older_rule_and_the_old_text_stays_in_history(
    engine: MemoryEngine,
):
    bank_id = f"test-5938-newer-late-{uuid.uuid4().hex[:8]}"
    older, amendment = await _two_facts(
        engine, bank_id, (OLDER_RULE, AMENDMENT_AT.replace(hour=9)), (AMENDMENT, AMENDMENT_AT)
    )

    (observation,) = await _observations(engine, bank_id)
    assert observation["text"] == AMENDMENT
    assert {str(s) for s in observation["source_memory_ids"]} == {str(amendment), str(older)}
    history = await engine.get_observation_history(
        bank_id=bank_id, memory_id=str(observation["id"]), request_context=RequestContext()
    )
    assert history, "the earlier text must stay in the observation's history"
    assert any(OLDER_RULE in str(entry) for entry in history)
