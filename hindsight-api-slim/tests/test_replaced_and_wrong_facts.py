"""Replaced and wrong facts are never current, and a pointer names a record without copying it (RUN-2355 5, 6).

The rules live in the database (migration a5f1c3e7d902), so the tests go through the engine's own curation
path and then try to break the rule with plain SQL.
"""

import json
import uuid
from collections.abc import AsyncIterator
from datetime import datetime

import asyncpg
import pytest
import pytest_asyncio

from hindsight_api import RequestContext
from hindsight_api.engine.cross_encoder import RRFPassthroughCrossEncoder
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.retain.types import embedding_to_pgvector
from tests.test_retain_same_document_concurrency import _StubEmbeddings

pytestmark = pytest.mark.memory_backend_incompatible


@pytest_asyncio.fixture
async def engine(pg0_db_url: str) -> AsyncIterator[MemoryEngine]:
    memory = MemoryEngine(
        db_url=pg0_db_url,
        memory_llm_provider="mock",
        memory_llm_api_key="",
        memory_llm_model="mock",
        embeddings=_StubEmbeddings(),
        cross_encoder=RRFPassthroughCrossEncoder(),
        pool_min_size=1,
        pool_max_size=3,
        run_migrations=False,
    )
    await memory.initialize()
    try:
        yield memory
    finally:
        await memory.close()


async def _seed(engine: MemoryEngine, bank_id: str, text: str, said_at: str, source_url: str) -> uuid.UUID:
    unit_id = uuid.uuid4()
    said = datetime.fromisoformat(said_at)
    async with engine._pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO memory_units (id, bank_id, text, fact_type, event_date, mentioned_at, metadata) "
            "VALUES ($1, $2, $3, 'world', $4, $4, $5::jsonb)",
            unit_id,
            bank_id,
            text,
            said,
            json.dumps({"said_by": "asha", "said_at": said_at, "source_url": source_url, "channel": "slack"}),
        )
        embedding = (await engine.embeddings.encode([text]))[0]
        await conn.execute(
            "UPDATE memory_units SET embedding = $2::vector WHERE id = $1", unit_id, embedding_to_pgvector(embedding)
        )
    return unit_id


async def _live_ids(engine: MemoryEngine, bank_id: str) -> set[str]:
    listed = await engine.list_memory_units(bank_id=bank_id, request_context=RequestContext())
    return {item["id"] for item in listed["items"]}


@pytest.mark.asyncio
async def test_replaced_fact_leaves_the_current_set_and_stays_in_history_with_date_and_source(engine: MemoryEngine):
    context = RequestContext()
    bank_id = f"test-replaced-{uuid.uuid4().hex[:8]}"
    await engine.ensure_bank_profile(bank_id=bank_id, request_context=context)
    old = await _seed(
        engine, bank_id, "The Pro plan costs 40 dollars.", "2026-11-03T09:10:00+00:00", "https://x.test/m/1"
    )
    new = await _seed(
        engine, bank_id, "The Pro plan costs 50 dollars.", "2026-11-03T09:10:30+00:00", "https://x.test/m/2"
    )

    await engine.update_memory_unit(
        bank_id=bank_id,
        memory_id=str(old),
        state="invalidated",
        reason="price changed",
        invalid_kind="replaced",
        replaced_by=str(new),
        request_context=context,
    )

    assert await _live_ids(engine, bank_id) == {str(new)}
    recalled = await engine.recall_async(
        bank_id=bank_id, query="What does the Pro plan cost?", fact_type=["world"], request_context=context
    )
    assert {r.id for r in recalled.results} == {str(new)}
    history = await engine.get_replaced_history(bank_id=bank_id, memory_id=str(new), request_context=context)
    assert [h["id"] for h in history] == [str(old)]
    entry = history[0]
    assert entry["text"] == "The Pro plan costs 40 dollars."
    assert entry["said_at"].startswith("2026-11-03T09:10:00")
    assert entry["said_by"] == "asha"
    assert entry["source_url"] == "https://x.test/m/1"
    assert entry["replaced_by"] == str(new)
    assert entry["replaced_at"] is not None


@pytest.mark.asyncio
async def test_a_replacement_chain_lists_every_earlier_fact_newest_first(engine: MemoryEngine):
    context = RequestContext()
    bank_id = f"test-chain-{uuid.uuid4().hex[:8]}"
    await engine.ensure_bank_profile(bank_id=bank_id, request_context=context)
    first = await _seed(engine, bank_id, "Launch is on the 1st.", "2026-11-01T08:00:00+00:00", "https://x.test/1")
    second = await _seed(engine, bank_id, "Launch is on the 8th.", "2026-11-02T08:00:00+00:00", "https://x.test/2")
    third = await _seed(engine, bank_id, "Launch is on the 15th.", "2026-11-03T08:00:00+00:00", "https://x.test/3")
    for old, new in ((first, second), (second, third)):
        await engine.update_memory_unit(
            bank_id=bank_id,
            memory_id=str(old),
            state="invalidated",
            invalid_kind="replaced",
            replaced_by=str(new),
            request_context=context,
        )

    history = await engine.get_replaced_history(bank_id=bank_id, memory_id=str(third), request_context=context)

    assert [h["text"] for h in history] == ["Launch is on the 8th.", "Launch is on the 1st."]
    assert await _live_ids(engine, bank_id) == {str(third)}


@pytest.mark.asyncio
async def test_a_wrong_fact_cannot_become_current_by_revert_or_by_insert(engine: MemoryEngine):
    context = RequestContext()
    bank_id = f"test-wrong-{uuid.uuid4().hex[:8]}"
    await engine.ensure_bank_profile(bank_id=bank_id, request_context=context)
    wrong = await _seed(engine, bank_id, "Asha is the CTO.", "2026-11-03T09:00:00+00:00", "https://x.test/w")
    await engine.update_memory_unit(
        bank_id=bank_id,
        memory_id=str(wrong),
        state="invalidated",
        invalid_kind="wrong",
        reason="never true",
        request_context=context,
    )
    assert await _live_ids(engine, bank_id) == set()

    # A plain revert is refused.
    with pytest.raises(ValueError, match="marked wrong"):
        await engine.update_memory_unit(bank_id=bank_id, memory_id=str(wrong), state="valid", request_context=context)
    assert await _live_ids(engine, bank_id) == set()

    # So is putting the same id back with SQL: the database refuses it, whatever path the caller took.
    async with engine._pool.acquire() as conn:
        with pytest.raises(asyncpg.exceptions.CheckViolationError, match="marked wrong"):
            await conn.execute(
                "INSERT INTO memory_units (id, bank_id, text, fact_type) VALUES ($1, $2, 'Asha is the CTO.', 'world')",
                wrong,
                bank_id,
            )

    # Saying so on purpose brings it back.
    await engine.update_memory_unit(
        bank_id=bank_id, memory_id=str(wrong), state="valid", revive_dead=True, request_context=context
    )
    assert await _live_ids(engine, bank_id) == {str(wrong)}


@pytest.mark.asyncio
async def test_invalid_kind_is_checked_before_anything_moves(engine: MemoryEngine):
    context = RequestContext()
    bank_id = f"test-kind-{uuid.uuid4().hex[:8]}"
    await engine.ensure_bank_profile(bank_id=bank_id, request_context=context)
    unit = await _seed(engine, bank_id, "A fact.", "2026-11-03T09:00:00+00:00", "https://x.test/k")

    with pytest.raises(ValueError, match="replaced_by"):
        await engine.update_memory_unit(
            bank_id=bank_id, memory_id=str(unit), state="invalidated", invalid_kind="replaced", request_context=context
        )
    with pytest.raises(ValueError, match="state='invalidated'"):
        await engine.update_memory_unit(
            bank_id=bank_id, memory_id=str(unit), reason="x", invalid_kind="wrong", request_context=context
        )
    assert await _live_ids(engine, bank_id) == {str(unit)}


@pytest.mark.asyncio
async def test_the_archive_refuses_a_replaced_row_that_replaces_nothing(engine: MemoryEngine):
    context = RequestContext()
    bank_id = f"test-archive-{uuid.uuid4().hex[:8]}"
    await engine.ensure_bank_profile(bank_id=bank_id, request_context=context)
    unit = await _seed(engine, bank_id, "A fact.", "2026-11-03T09:00:00+00:00", "https://x.test/a")
    await engine.update_memory_unit(bank_id=bank_id, memory_id=str(unit), state="invalidated", request_context=context)

    async with engine._pool.acquire() as conn:
        with pytest.raises(asyncpg.exceptions.CheckViolationError):
            await conn.execute(
                "UPDATE invalidated_memory_units SET invalid_kind = 'replaced' WHERE id = $1 AND bank_id = $2",
                unit,
                bank_id,
            )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metadata", "accepted"),
    [
        ({"pointer_kind": "crm", "pointer_id": "deal_42"}, True),
        ({"pointer_kind": "linear", "pointer_id": "RUN-2355"}, True),
        ({"pointer_kind": "stripe", "pointer_id": "pi_3Q"}, True),
        ({"pointer_kind": "jira", "pointer_id": "X-1"}, False),
        ({"pointer_kind": "crm"}, False),
        ({"pointer_id": "deal_42"}, False),
        ({"pointer_kind": "crm", "pointer_id": ""}, False),
        ({"source_url": "https://x.test"}, True),
    ],
)
async def test_the_database_accepts_only_well_formed_pointers(engine: MemoryEngine, metadata: dict, accepted: bool):
    bank_id = f"test-pointer-{uuid.uuid4().hex[:8]}"
    await engine.ensure_bank_profile(bank_id=bank_id, request_context=RequestContext())
    insert = engine._pool.acquire()
    async with insert as conn:
        statement = conn.execute(
            "INSERT INTO memory_units (id, bank_id, text, fact_type, metadata) VALUES ($1, $2, 'x', 'world', $3::jsonb)",
            uuid.uuid4(),
            bank_id,
            json.dumps(metadata),
        )
        if accepted:
            await statement
        else:
            with pytest.raises(asyncpg.exceptions.CheckViolationError):
                await statement
