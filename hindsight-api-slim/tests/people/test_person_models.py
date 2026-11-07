"""Person models and the retain hook (RUN-2355 items 1, 2 and 3): every new person gets `person:<handle>`
after the next consolidation with nobody asking, and each consolidation refreshes it. No model is called."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hindsight_api.engine.consolidation import consolidator
from hindsight_api.people.context import PeopleContext
from hindsight_api.people.peers import get_or_create_peers
from hindsight_api.people.person_models import (
    CONFIDENCE_TAG,
    PERSON_SOURCE_QUERY,
    ensure_person_models,
    handle_of,
    leaves_out_confidences,
    person_model_content,
    person_model_id,
)
from hindsight_api.people.runtime import after_retain, incoming_messages
from tests.people.people_helpers import add_message, add_session, make_doc

pytestmark = pytest.mark.memory_backend_incompatible


async def _models(engine, bank_id):
    async with engine._pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, name, source_query, content, trigger FROM mental_models WHERE bank_id = $1 ORDER BY id",
            bank_id,
        )
    return [dict(r) for r in rows]


@pytest.mark.asyncio
async def test_every_new_person_gets_a_person_model_with_nobody_asking(people_engine, ctx: PeopleContext, bank_id):
    await get_or_create_peers(ctx, bank_id, ["asha", "dev"])

    created = await ensure_person_models(people_engine, bank_id)

    assert sorted(created) == ["asha", "dev"]
    models = await _models(people_engine, bank_id)
    assert [m["name"] for m in models] == ["person:asha", "person:dev"]
    assert [handle_of(m["id"]) for m in models] == ["asha", "dev"]
    for model in models:
        assert "refresh_after_consolidation" in str(model["trigger"])
        assert "true" in str(model["trigger"]).lower()
        # A person model is shown to every agent: it never reads a confidence, nor another model that might carry one.
        assert leaves_out_confidences(model["trigger"])
        assert CONFIDENCE_TAG in str(model["trigger"])


@pytest.mark.asyncio
async def test_a_person_model_covers_who_they_are_and_cites_sources(people_engine, ctx, bank_id):
    await get_or_create_peers(ctx, bank_id, ["asha"])
    await ensure_person_models(people_engine, bank_id)
    (model,) = await _models(people_engine, bank_id)

    for section in (
        "Who they are",
        "Role and company",
        "How they work and talk",
        "Likes and dislikes",
        "What they are working on",
        "Open promises",
        "Who they work with",
    ):
        assert section in model["source_query"]
    assert "asha" in model["source_query"]
    assert "source_url" in model["source_query"] and "said_at" in model["source_query"]
    assert model["source_query"] == PERSON_SOURCE_QUERY.format(handle="asha")


@pytest.mark.asyncio
async def test_a_second_run_creates_nothing_and_keeps_the_model(people_engine, ctx, bank_id):
    await get_or_create_peers(ctx, bank_id, ["asha"])
    await ensure_person_models(people_engine, bank_id)

    assert await ensure_person_models(people_engine, bank_id) == []
    assert len(await _models(people_engine, bank_id)) == 1


@pytest.mark.asyncio
async def test_a_person_who_arrives_later_gets_a_model_at_the_next_run(people_engine, ctx, bank_id):
    await get_or_create_peers(ctx, bank_id, ["asha"])
    await ensure_person_models(people_engine, bank_id)
    await get_or_create_peers(ctx, bank_id, ["meera"])

    assert await ensure_person_models(people_engine, bank_id) == ["meera"]


@pytest.mark.asyncio
async def test_the_first_content_lists_conclusions_with_their_sources(people_engine, ctx, bank_id):
    await add_session(ctx, bank_id, "s1", ["asha"])
    from hindsight_api.people.conclusions import create_documents

    said_at = datetime(2026, 11, 3, 9, 10, tzinfo=timezone.utc)
    message = await add_message(
        ctx,
        bank_id,
        "s1",
        "asha",
        "I lead the billing rewrite",
        seq=1,
        source_url="https://slack.test/m/1",
        said_at=said_at,
    )
    doc = await make_doc(ctx, "asha leads the billing rewrite", session="s1", source_message_ids=[message])
    await create_documents(ctx, bank_id, [doc], observer="asha", observed="asha")

    content = await person_model_content(ctx, bank_id, "asha")

    assert content.startswith("# asha")
    assert "- asha leads the billing rewrite [https://slack.test/m/1, 2026-11-03]" in content


def test_handles_with_a_slash_make_a_safe_model_id():
    assert person_model_id("a/b") == "person:a_b"
    assert handle_of("person:asha") == "asha"
    assert handle_of("other-model") is None


@pytest.mark.asyncio
async def test_consolidation_creates_the_models_when_people_are_enabled(people_engine, ctx, bank_id):
    await get_or_create_peers(ctx, bank_id, ["asha"])
    enabled = SimpleNamespace(people_enabled=True)

    with patch.object(consolidator, "get_config", return_value=enabled):
        await consolidator._ensure_person_models(people_engine, bank_id, ctx.request_context)

    assert [m["name"] for m in await _models(people_engine, bank_id)] == ["person:asha"]


@pytest.mark.asyncio
async def test_consolidation_leaves_models_alone_when_people_are_off(people_engine, ctx, bank_id):
    await get_or_create_peers(ctx, bank_id, ["asha"])

    with patch.object(consolidator, "get_config", return_value=SimpleNamespace(people_enabled=False)):
        await consolidator._ensure_person_models(people_engine, bank_id, ctx.request_context)

    assert await _models(people_engine, bank_id) == []


@pytest.mark.asyncio
async def test_a_failing_person_model_never_fails_the_consolidation(people_engine, ctx, bank_id):
    with (
        patch.object(consolidator, "get_config", return_value=SimpleNamespace(people_enabled=True)),
        patch("hindsight_api.people.person_models.ensure_person_models", side_effect=RuntimeError("boom")),
    ):
        await consolidator._ensure_person_models(people_engine, bank_id, ctx.request_context)


# ----------------------------------------------------------------------------- the retain hook


def test_only_items_that_say_who_said_them_reach_the_deriver():
    contents = [
        {
            "content": "I will send the contract tomorrow",
            "document_id": "m1",
            "metadata": {
                "said_by": "asha",
                "said_at": "2026-11-03T09:10:00+00:00",
                "channel": "discord",
                "source_url": "https://discord.test/1",
            },
        },
        {"content": "an item with no speaker", "metadata": {"channel": "discord"}},
        {"content": "   ", "metadata": {"said_by": "dev"}},
    ]

    (message,) = incoming_messages(contents)

    assert message.peer == "asha"
    assert message.session == "discord:m1"
    assert message.channel == "discord"
    assert message.source_url == "https://discord.test/1"
    assert message.said_at == datetime(2026, 11, 3, 9, 10, tzinfo=timezone.utc)
    assert message.document_id == "m1"


@pytest.mark.asyncio
async def test_a_retained_message_makes_its_speaker_a_peer_and_queues_the_message(people_engine, ctx, bank_id):
    contents = [
        {
            "content": "I prefer short messages and I am working on the billing rewrite",
            "document_id": "m2",
            "metadata": {
                "said_by": "meera",
                "said_at": "2026-11-03T09:10:00+00:00",
                "channel": "slack",
                "source_url": "https://slack.test/m/2",
            },
        }
    ]

    await after_retain(people_engine, bank_id, contents)

    async with people_engine._pool.acquire() as conn:
        peers = [r["name"] for r in await conn.fetch("SELECT name FROM people_peers WHERE bank_id = $1", bank_id)]
        message = await conn.fetchrow(
            "SELECT peer_name, source_url, said_by, channel, said_at FROM people_messages WHERE bank_id = $1", bank_id
        )
        queued = await conn.fetchval("SELECT count(*) FROM people_queue WHERE bank_id = $1", bank_id)
    assert peers == ["meera"]
    assert message["source_url"] == "https://slack.test/m/2"
    assert message["said_by"] == "meera"
    assert message["channel"] == "slack"
    assert message["said_at"] == datetime(2026, 11, 3, 9, 10, tzinfo=timezone.utc)
    assert queued >= 1


@pytest.mark.asyncio
async def test_the_retain_hook_never_fails_the_retain(people_engine, bank_id):
    with patch("hindsight_api.people.deriver.enqueue.enqueue_messages", side_effect=RuntimeError("down")):
        await after_retain(people_engine, bank_id, [{"content": "hello", "metadata": {"said_by": "asha"}}])


@pytest.mark.asyncio
async def test_a_model_made_before_the_exclusion_gets_its_trigger_updated(people_engine, ctx, bank_id):
    await get_or_create_peers(ctx, bank_id, ["asha"])
    await ensure_person_models(people_engine, bank_id)
    async with people_engine._pool.acquire() as conn:
        await conn.execute(
            "UPDATE mental_models SET trigger = '{\"refresh_after_consolidation\": true}'::jsonb WHERE bank_id = $1",
            bank_id,
        )
    (before,) = await _models(people_engine, bank_id)
    assert not leaves_out_confidences(before["trigger"])

    created = await ensure_person_models(people_engine, bank_id)

    assert created == []
    (after,) = await _models(people_engine, bank_id)
    assert leaves_out_confidences(after["trigger"])


def test_only_a_trigger_that_excludes_confidences_counts():
    assert not leaves_out_confidences(None)
    assert not leaves_out_confidences('{"refresh_after_consolidation": true}')
    assert not leaves_out_confidences({"tag_groups": [{"tags": ["x"]}]})
    assert leaves_out_confidences(
        {"exclude_mental_models": True, "tag_groups": [{"not": {"tags": ["confidence"], "match": "any_strict"}}]}
    )
