"""The dream history and schedule routes."""

import httpx
import pytest
from fastapi import FastAPI

from hindsight_api.people.context import PeopleContext
from hindsight_api.people.dream_history import close_run
from hindsight_api.people.dreamer.api import create_dream_router
from hindsight_api.people.dreamer.settings import DREAM
from tests.people.dreamer_support import open_target, scalar, seed


@pytest.fixture
async def client(people_engine):
    app = FastAPI()
    app.include_router(create_dream_router(lambda: people_engine))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
        yield http


async def test_list_pages_runs_newest_first(client: httpx.AsyncClient, ctx: PeopleContext, bank_id: str):
    first = await open_target(ctx, bank_id, "ann", "ann")
    second = await open_target(ctx, bank_id, "ann", "bob")
    await close_run(ctx, bank_id, first.run_id, succeeded=True)

    response = await client.post(f"/v1/default/banks/{bank_id}/dreams/list")
    assert response.status_code == 200
    body = response.json()
    assert [run["id"] for run in body["items"]] == [second.run_id, first.run_id]
    assert body["items"][1]["status"] == "completed"

    filtered = await client.post(f"/v1/default/banks/{bank_id}/dreams/list", json={"observed_id": "bob"})
    assert [run["id"] for run in filtered.json()["items"]] == [second.run_id]

    paged = await client.post(f"/v1/default/banks/{bank_id}/dreams/list?size=1")
    assert (paged.json()["total"], paged.json()["pages"], len(paged.json()["items"])) == (2, 2, 1)


async def test_get_one_run_with_its_changes_and_404_for_an_unknown_run(
    client: httpx.AsyncClient, ctx: PeopleContext, bank_id: str
):
    _session, (coffee_id, _nights_id) = await seed(ctx, bank_id)
    target = await open_target(ctx, bank_id)
    from hindsight_api.people.dreamer.agent_tools import delete_observations_with_history

    await delete_observations_with_history(ctx, bank_id, [coffee_id], observer="ann", observed="ann", history=target)

    response = await client.get(f"/v1/default/banks/{bank_id}/dreams/{target.run_id}")
    assert response.status_code == 200
    body = response.json()
    assert body["removed_count"] == 1
    assert [r["id"] for r in body["removed"]] == [coffee_id]

    missing = await client.get(f"/v1/default/banks/{bank_id}/dreams/no-such-run")
    assert missing.status_code == 404


async def test_schedule_enqueues_a_manual_dream(client: httpx.AsyncClient, ctx: PeopleContext, bank_id: str):
    response = await client.post(
        f"/v1/default/banks/{bank_id}/dreams/schedule", json={"observer": "ann", "dream_type": "card_refresh"}
    )
    assert response.status_code == 202
    assert response.json()["observed"] == "ann"
    payload = await scalar(
        ctx, f"SELECT payload::text FROM {ctx.table('queue')} WHERE bank_id = $1 AND task_type = 'dream'", bank_id
    )
    assert '"trigger_reason": "manual"' in payload and '"delay_reason": "immediate"' in payload


async def test_schedule_is_refused_when_dreams_are_off(
    client: httpx.AsyncClient, bank_id: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(DREAM, "ENABLED", False)
    response = await client.post(
        f"/v1/default/banks/{bank_id}/dreams/schedule", json={"observer": "ann", "dream_type": "omni"}
    )
    assert response.status_code == 400
