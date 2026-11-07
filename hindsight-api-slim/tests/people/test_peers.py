# Copyright (C) Plastic Labs <hello@plasticlabs.ai>
# Adapted from plastic-labs/honcho @ 3326373c35f6538fb3b7af97803abe328c5aafb4: tests/routes/test_peers.py
# Licensed under the GNU Affero General Public License v3.0 (see LICENSE and NOTICE at the repository root).
# Modified by Aeonmind AI for Hindsight's store, banks and Claude configuration.
"""Peer tests, ported from Honcho's tests/routes/test_peers.py (routes) and its peer CRUD helpers.

The routes run through the people router mounted on a FastAPI app over the test engine; the requests are
made in-process on the test's own event loop.
"""

from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.people import schemas
from hindsight_api.people.api import create_people_router
from hindsight_api.people.context import PeopleContext
from hindsight_api.people.exceptions import ConflictException, ResourceNotFoundException, ValidationException
from hindsight_api.people.peers import (
    get_messages,
    get_or_create_peers,
    get_peer,
    get_peers,
    get_sessions_for_peer,
    list_person_handles,
    search_messages,
    update_peer,
)
from tests.people.people_helpers import add_message, add_session


@pytest_asyncio.fixture
async def client(people_engine: MemoryEngine) -> AsyncIterator[httpx.AsyncClient]:
    app = FastAPI()
    app.include_router(create_people_router(lambda: people_engine))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c


def base(bank_id: str) -> str:
    return f"/v1/default/banks/{bank_id}"


# --- CRUD ---------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_or_create_peers_creates_then_gets(ctx: PeopleContext, bank_id: str):
    first = await get_or_create_peers(ctx, bank_id, [schemas.PeerSpec(name="alice", metadata={"k": "v"})])
    assert first.created is True
    assert first.resource[0].name == "alice"
    assert first.resource[0].h_metadata == {"k": "v"}
    assert len(first.resource[0].id) == 21

    second = await get_or_create_peers(ctx, bank_id, ["alice"])
    assert second.created is False
    assert second.resource[0].id == first.resource[0].id
    # metadata was not asked to change, so it stays
    assert second.resource[0].h_metadata == {"k": "v"}


@pytest.mark.asyncio
async def test_get_or_create_peers_updates_provided_metadata_and_configuration(ctx: PeopleContext, bank_id: str):
    await get_or_create_peers(ctx, bank_id, [schemas.PeerSpec(name="alice", metadata={"a": 1})])
    result = await get_or_create_peers(
        ctx, bank_id, [schemas.PeerSpec(name="alice", metadata={"a": 2}, configuration={"observe_me": False})]
    )
    assert result.created is False
    peer = await get_peer(ctx, bank_id, "alice")
    assert peer.h_metadata == {"a": 2}
    assert peer.configuration == {"observe_me": False}


@pytest.mark.asyncio
async def test_get_or_create_peers_keeps_the_order_asked_for(ctx: PeopleContext, bank_id: str):
    await get_or_create_peers(ctx, bank_id, ["bob"])
    result = await get_or_create_peers(ctx, bank_id, ["carol", "bob", "alice"])
    assert [p.name for p in result.resource] == ["carol", "bob", "alice"]
    assert result.created is True


@pytest.mark.asyncio
async def test_get_or_create_peers_collapses_repeated_names(ctx: PeopleContext, bank_id: str):
    result = await get_or_create_peers(ctx, bank_id, ["bob", "bob"])
    assert [p.name for p in result.resource] == ["bob"]


@pytest.mark.asyncio
async def test_get_or_create_peers_rejects_impossible_and_bad_names(ctx: PeopleContext, bank_id: str):
    with pytest.raises(ValidationException, match="NUL"):
        await get_or_create_peers(ctx, bank_id, ["bad\x00name"])
    with pytest.raises(ValidationException, match="at most 512"):
        await get_or_create_peers(ctx, bank_id, ["a" * 513])
    with pytest.raises(ValidationException, match="must match pattern"):
        await get_or_create_peers(ctx, bank_id, ["has space"])
    with pytest.raises(ValidationException, match="must match pattern"):
        await get_or_create_peers(ctx, bank_id, [""])
    # nothing was created by the failed calls
    assert await list_person_handles(ctx, bank_id) == []


@pytest.mark.asyncio
async def test_handles_may_carry_dots_and_at_signs(ctx: PeopleContext, bank_id: str):
    result = await get_or_create_peers(ctx, bank_id, ["jane.doe@example.com"])
    assert result.resource[0].name == "jane.doe@example.com"


@pytest.mark.asyncio
async def test_get_peer_missing_raises(ctx: PeopleContext, bank_id: str):
    with pytest.raises(ResourceNotFoundException):
        await get_peer(ctx, bank_id, "nobody")


@pytest.mark.asyncio
async def test_a_deleted_peer_comes_back_as_a_new_peer(ctx: PeopleContext, bank_id: str):
    first = (await get_or_create_peers(ctx, bank_id, ["alice"])).resource[0]
    async with ctx.conn() as db:
        await db.execute(
            f'UPDATE {ctx.table("peers")} SET deleted_at = now(), internal_metadata = \'{{"peer_card": ["old"]}}\'::jsonb '
            "WHERE bank_id = $1 AND name = 'alice'",
            bank_id,
        )
    with pytest.raises(ResourceNotFoundException):
        await get_peer(ctx, bank_id, "alice")
    result = await get_or_create_peers(ctx, bank_id, ["alice"])
    assert result.created is True
    assert result.resource[0].id == first.id
    assert result.resource[0].internal_metadata == {}


@pytest.mark.asyncio
async def test_concurrent_creation_of_the_same_peer_yields_one_peer(ctx: PeopleContext, bank_id: str):
    import asyncio

    results = await asyncio.gather(*[get_or_create_peers(ctx, bank_id, ["racer"]) for _ in range(6)])
    assert len({r.resource[0].id for r in results}) == 1
    assert sum(1 for r in results if r.created) >= 1
    assert await list_person_handles(ctx, bank_id) == ["racer"]


@pytest.mark.asyncio
async def test_get_peers_orders_filters_and_pages(ctx: PeopleContext, bank_id: str):
    for i, name in enumerate(["a", "b", "c", "d"]):
        await get_or_create_peers(ctx, bank_id, [schemas.PeerSpec(name=name, metadata={"group": i % 2})])

    peers, total = await get_peers(ctx, bank_id)
    assert [p.name for p in peers] == ["a", "b", "c", "d"]
    assert total == 4

    peers, _ = await get_peers(ctx, bank_id, reverse=True)
    assert [p.name for p in peers] == ["d", "c", "b", "a"]

    peers, total = await get_peers(ctx, bank_id, filters={"group": 1})
    assert [p.name for p in peers] == ["b", "d"]
    assert total == 2

    peers, total = await get_peers(ctx, bank_id, reverse=True, limit=2, offset=1)
    assert [p.name for p in peers] == ["c", "b"]
    assert total == 4


@pytest.mark.asyncio
async def test_get_peers_refuses_operator_filters(ctx: PeopleContext, bank_id: str):
    with pytest.raises(ValidationException, match="Unsupported filter"):
        await get_peers(ctx, bank_id, filters={"group": {"$gt": 0}})


@pytest.mark.asyncio
async def test_update_peer(ctx: PeopleContext, bank_id: str):
    updated = await update_peer(
        ctx, bank_id, "alice", schemas.PeerUpdate(metadata={"m": 1}, configuration={"observe_me": True})
    )
    assert updated.h_metadata == {"m": 1}
    assert updated.configuration == {"observe_me": True}
    # a null field leaves the stored value alone
    again = await update_peer(ctx, bank_id, "alice", schemas.PeerUpdate(metadata=None, configuration={"x": 2}))
    assert again.h_metadata == {"m": 1}
    assert again.configuration == {"x": 2}
    assert (await get_peer(ctx, bank_id, "alice")).configuration == {"x": 2}


@pytest.mark.asyncio
async def test_get_sessions_for_peer(ctx: PeopleContext, bank_id: str):
    await add_session(ctx, bank_id, "s1", ["alice", "bob"], {"topic": "x"})
    await add_session(ctx, bank_id, "s2", ["alice"], {"topic": "y"})
    await add_session(ctx, bank_id, "s3", ["bob"])

    sessions, total = await get_sessions_for_peer(ctx, bank_id, "alice")
    assert [s.name for s in sessions] == ["s1", "s2"]
    assert total == 2

    sessions, _ = await get_sessions_for_peer(ctx, bank_id, "alice", reverse=True)
    assert [s.name for s in sessions] == ["s2", "s1"]

    sessions, total = await get_sessions_for_peer(ctx, bank_id, "alice", filters={"topic": "y"})
    assert [s.name for s in sessions] == ["s2"]
    assert total == 1

    sessions, total = await get_sessions_for_peer(ctx, bank_id, "nobody")
    assert sessions == [] and total == 0


@pytest.mark.asyncio
async def test_messages_keep_where_they_came_from(ctx: PeopleContext, bank_id: str):
    await add_session(ctx, bank_id, "s1", ["alice"])
    await add_message(ctx, bank_id, "s1", "alice", "I moved to Berlin", seq=1, source_url="https://slack/x/1")
    await add_message(ctx, bank_id, "s1", "alice", "I like green tea", seq=2)

    messages, total = await get_messages(ctx, bank_id, "s1")
    assert total == 2
    assert [m.content for m in messages] == ["I moved to Berlin", "I like green tea"]
    assert messages[0].source_url == "https://slack/x/1"
    assert messages[0].said_by == "alice"
    assert messages[0].channel == "slack"
    with pytest.raises(ResourceNotFoundException):
        await get_messages(ctx, bank_id, "missing")


@pytest.mark.asyncio
async def test_search_messages(ctx: PeopleContext, bank_id: str):
    await add_session(ctx, bank_id, "s1", ["alice", "bob"])
    await add_session(ctx, bank_id, "s2", ["alice"])
    await add_message(ctx, bank_id, "s1", "alice", "I moved to Berlin last year", seq=1)
    await add_message(ctx, bank_id, "s1", "bob", "Berlin is cold", seq=2)
    await add_message(ctx, bank_id, "s2", "alice", "100% sure about the green tea", seq=1)

    found = await search_messages(ctx, bank_id, "alice", "Berlin")
    assert [m.content for m in found] == ["I moved to Berlin last year"]
    # substring match, with LIKE wildcards taken literally
    assert [m.content for m in await search_messages(ctx, bank_id, "alice", "100%")] == [
        "100% sure about the green tea"
    ]
    assert await search_messages(ctx, bank_id, "alice", "50%") == []
    # filter by session
    assert await search_messages(ctx, bank_id, "alice", "tea", filters={"session_id": "s1"}) == []
    assert len(await search_messages(ctx, bank_id, "alice", "tea", filters={"session_id": "s2"})) == 1
    with pytest.raises(ValidationException):
        await search_messages(ctx, bank_id, "alice", "tea", filters={"channel": "x"})


@pytest.mark.asyncio
async def test_conflict_exception_is_a_people_error():
    assert issubclass(ConflictException, Exception)


# --- routes -------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_route_get_or_create_peer(client: httpx.AsyncClient, bank_id: str):
    response = await client.post(
        f"{base(bank_id)}/people", json={"id": "alice", "metadata": {"peer_key": "peer_value"}}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["id"] == "alice"
    assert data["metadata"] == {"peer_key": "peer_value"}
    assert len(data["uid"]) == 21
    assert "internal_metadata" not in data

    again = await client.post(f"{base(bank_id)}/people", json={"id": "alice"})
    assert again.status_code == 200
    assert again.json()["uid"] == data["uid"]


@pytest.mark.asyncio
async def test_route_get_or_create_peer_with_all_optional_params(client: httpx.AsyncClient, bank_id: str):
    metadata = {"key": "value", "number": 42}
    configuration = {"feature1": True, "feature2": False}
    response = await client.post(
        f"{base(bank_id)}/people", json={"id": "bob", "metadata": metadata, "configuration": configuration}
    )
    assert response.status_code in [200, 201]
    assert response.json()["metadata"] == metadata
    assert response.json()["configuration"] == configuration


@pytest.mark.asyncio
async def test_route_get_or_create_creates_the_bank_when_missing(
    client: httpx.AsyncClient, people_engine: MemoryEngine
):
    response = await client.post("/v1/default/banks/people-new-bank/people", json={"id": "alice"})
    assert response.status_code == 201


@pytest.mark.asyncio
async def test_route_rejects_bad_handles(client: httpx.AsyncClient, bank_id: str):
    response = await client.post(f"{base(bank_id)}/people", json={"id": "has space"})
    assert response.status_code == 422
    response = await client.post(f"{base(bank_id)}/people", json={"id": "a" * 513})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_route_list_peers(client: httpx.AsyncClient, bank_id: str):
    for name in ["a", "b", "c"]:
        await client.post(f"{base(bank_id)}/people", json={"id": name, "metadata": {"n": name}})

    response = await client.post(f"{base(bank_id)}/people/list")
    assert response.status_code == 200
    page = response.json()
    assert [p["id"] for p in page["items"]] == ["a", "b", "c"]
    assert page["total"] == 3 and page["page"] == 1 and page["pages"] == 1

    response = await client.post(f"{base(bank_id)}/people/list", json={"filters": {}})
    assert response.status_code == 200
    assert response.json()["total"] == 3

    response = await client.post(f"{base(bank_id)}/people/list", json={"filters": {"n": "b"}})
    assert [p["id"] for p in response.json()["items"]] == ["b"]

    response = await client.post(f"{base(bank_id)}/people/list?reverse=true&size=2&page=2")
    page = response.json()
    assert [p["id"] for p in page["items"]] == ["a"]
    assert page["pages"] == 2


@pytest.mark.asyncio
async def test_route_update_peer(client: httpx.AsyncClient, bank_id: str):
    await client.post(f"{base(bank_id)}/people", json={"id": "alice", "metadata": {"a": 1}})
    response = await client.put(f"{base(bank_id)}/people/alice", json={"configuration": {"observe_me": False}})
    assert response.status_code == 200
    data = response.json()
    assert data["metadata"] == {"a": 1}
    assert data["configuration"] == {"observe_me": False}


@pytest.mark.asyncio
async def test_route_sessions_for_peer(client: httpx.AsyncClient, ctx: PeopleContext, bank_id: str):
    await add_session(ctx, bank_id, "s1", ["alice"])
    await add_session(ctx, bank_id, "s2", ["alice"])
    response = await client.post(f"{base(bank_id)}/people/alice/sessions?reverse=true")
    assert response.status_code == 200
    assert [s["id"] for s in response.json()["items"]] == ["s2", "s1"]


@pytest.mark.asyncio
async def test_route_card_roundtrip(client: httpx.AsyncClient, bank_id: str):
    await client.post(f"{base(bank_id)}/people", json={"id": "alice"})
    response = await client.get(f"{base(bank_id)}/people/alice/card")
    assert response.status_code == 200
    assert response.json() == {"peer_card": None}

    response = await client.put(
        f"{base(bank_id)}/people/alice/card", json={"peer_card": ["Works at Acme", "Lives in Berlin"]}
    )
    assert response.status_code == 200
    assert response.json()["peer_card"] == ["Works at Acme", "Lives in Berlin"]

    response = await client.put(
        f"{base(bank_id)}/people/alice/card?target=bob", json={"peer_card": ["Alice's view of Bob"]}
    )
    assert response.status_code == 200
    assert (await client.get(f"{base(bank_id)}/people/alice/card?target=bob")).json()["peer_card"] == [
        "Alice's view of Bob"
    ]
    assert (await client.get(f"{base(bank_id)}/people/alice/card")).json()["peer_card"] == [
        "Works at Acme",
        "Lives in Berlin",
    ]


@pytest.mark.asyncio
async def test_route_card_of_a_missing_peer_is_404(client: httpx.AsyncClient, bank_id: str):
    response = await client.get(f"{base(bank_id)}/people/ghost/card")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_route_search_peer(client: httpx.AsyncClient, ctx: PeopleContext, bank_id: str):
    await add_session(ctx, bank_id, "s1", ["alice"])
    await add_message(ctx, bank_id, "s1", "alice", "I moved to Berlin", seq=1, source_url="https://slack/x/1")
    response = await client.post(f"{base(bank_id)}/people/alice/search", json={"query": "Berlin"})
    assert response.status_code == 200
    [message] = response.json()
    assert message["content"] == "I moved to Berlin"
    assert message["peer_id"] == "alice"
    assert message["session_id"] == "s1"
    assert message["source_url"] == "https://slack/x/1"

    response = await client.post(f"{base(bank_id)}/people/alice/search", json={"query": "nothing like this"})
    assert response.json() == []
    response = await client.post(f"{base(bank_id)}/people/alice/search", json={"query": "x", "limit": 0})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_route_session_and_message_listing(client: httpx.AsyncClient, ctx: PeopleContext, bank_id: str):
    await add_session(ctx, bank_id, "s1", ["alice"])
    await add_message(ctx, bank_id, "s1", "alice", "one", seq=1)
    await add_message(ctx, bank_id, "s1", "alice", "two", seq=2)

    response = await client.post(f"{base(bank_id)}/people-sessions/list")
    assert [s["id"] for s in response.json()["items"]] == ["s1"]
    response = await client.get(f"{base(bank_id)}/people-sessions/s1/messages?reverse=true")
    assert [m["content"] for m in response.json()["items"]] == ["two", "one"]
    assert (await client.get(f"{base(bank_id)}/people-sessions/nope/messages")).status_code == 404
