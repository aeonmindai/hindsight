"""The dialectic's HTTP routes: person chat and bank chat, with sources, evidence and streaming."""

import json
from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI

from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.people.dialectic.api import create_dialectic_router, extract_session_allowlist
from hindsight_api.people.exceptions import ValidationException
from tests.people.conftest import FakeLLM
from tests.people.dialectic_support import World, answer_ready, tool_turn, world  # noqa: F401  (world is a fixture)
from tests.people.dialectic_support import at as _at


@pytest_asyncio.fixture
async def client(people_engine: MemoryEngine) -> AsyncIterator[httpx.AsyncClient]:
    app = FastAPI()
    app.include_router(create_dialectic_router(lambda: people_engine))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        yield http


def _frames(body: str) -> list[dict]:
    return [json.loads(line[len("data: ") :]) for line in body.split("\n\n") if line.startswith("data: ")]


class TestPersonChat:
    async def test_answers_with_sources(self, world: World, client: httpx.AsyncClient, fake_llm: FakeLLM) -> None:  # noqa: F811
        fake_llm.script = [tool_turn("grep_messages", text="Berlin"), answer_ready(), "Alice lives in Berlin."]
        response = await client.post(
            f"/v1/default/banks/{world.bank_id}/people/alice/chat", json={"query": "Where does Alice live?"}
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["content"] == "Alice lives in Berlin."
        assert body["evidence"] is None
        sources = {s["id"]: s for s in body["sources"]}
        assert sources[world.messages["m_berlin"]]["source_url"] == "https://chat.example/s1/3"
        assert sources[world.messages["m_berlin"]]["said_at"] == _at(5).isoformat()

    async def test_include_evidence(self, world: World, client: httpx.AsyncClient, fake_llm: FakeLLM) -> None:  # noqa: F811
        fake_llm.script = [tool_turn("grep_messages", text="Berlin"), answer_ready(), "ok"]
        response = await client.post(
            f"/v1/default/banks/{world.bank_id}/people/alice/chat",
            json={"query": "q", "include_evidence": True},
        )
        evidence = response.json()["evidence"]
        assert [c["tool_name"] for c in evidence["tool_calls"]] == ["grep_messages"]
        assert world.conclusions["sushi"] in {c["id"] for c in evidence["conclusions"]}
        assert all("content" not in m for m in evidence["messages"])

    async def test_target_asks_about_another_person(
        self, world: World, client: httpx.AsyncClient, fake_llm: FakeLLM
    ) -> None:  # noqa: F811
        fake_llm.script = ["ok"]
        response = await client.post(
            f"/v1/default/banks/{world.bank_id}/people/bob/chat", json={"query": "q", "target": "alice"}
        )
        assert response.status_code == 200
        assert "perspective of bob's understanding of alice" in fake_llm.calls[0]["messages"][0]["content"]

    async def test_a_missing_target_is_a_404(self, world: World, client: httpx.AsyncClient, fake_llm: FakeLLM) -> None:  # noqa: F811
        response = await client.post(
            f"/v1/default/banks/{world.bank_id}/people/alice/chat", json={"query": "q", "target": "nobody"}
        )
        assert response.status_code == 404
        assert fake_llm.calls == []

    async def test_a_new_person_is_created_as_the_observer(
        self, world: World, client: httpx.AsyncClient, fake_llm: FakeLLM
    ) -> None:  # noqa: F811
        fake_llm.script = ["I have no information about dana."]
        response = await client.post(f"/v1/default/banks/{world.bank_id}/people/dana/chat", json={"query": "q"})
        assert response.status_code == 200
        assert response.json()["content"] == "I have no information about dana."
        assert response.json()["sources"] == []

    async def test_an_empty_answer_is_null_content(
        self, world: World, client: httpx.AsyncClient, fake_llm: FakeLLM
    ) -> None:  # noqa: F811
        fake_llm.script = [""]
        response = await client.post(f"/v1/default/banks/{world.bank_id}/people/alice/chat", json={"query": "q"})
        assert response.json()["content"] is None

    async def test_response_format_returns_json(
        self, world: World, client: httpx.AsyncClient, fake_llm: FakeLLM
    ) -> None:  # noqa: F811
        fake_llm.script = [answer_ready(), {"city": "Berlin"}]
        response = await client.post(
            f"/v1/default/banks/{world.bank_id}/people/alice/chat",
            json={
                "query": "q",
                "response_format": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        )
        assert response.status_code == 200, response.text
        assert json.loads(response.json()["content"]) == {"city": "Berlin"}

    async def test_an_unsupported_response_format_is_a_422(self, world: World, client: httpx.AsyncClient) -> None:  # noqa: F811
        response = await client.post(
            f"/v1/default/banks/{world.bank_id}/people/alice/chat",
            json={"query": "q", "response_format": {"type": "string"}},
        )
        assert response.status_code == 422
        assert "Invalid response_format" in response.text

    async def test_a_model_answer_that_misses_the_schema_is_a_422(
        self, world: World, client: httpx.AsyncClient, fake_llm: FakeLLM
    ) -> None:  # noqa: F811
        fake_llm.script = [answer_ready(), "I cannot say."]
        response = await client.post(
            f"/v1/default/banks/{world.bank_id}/people/alice/chat",
            json={
                "query": "q",
                "response_format": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            },
        )
        assert response.status_code == 422

    async def test_session_filters_scope_the_answer(
        self, world: World, client: httpx.AsyncClient, fake_llm: FakeLLM
    ) -> None:  # noqa: F811
        fake_llm.script = ["ok"]
        response = await client.post(
            f"/v1/default/banks/{world.bank_id}/people/alice/chat",
            json={"query": "q", "filters": {"session_id": ["s2"]}},
        )
        assert {s["session"] for s in response.json()["sources"]} == {"s2"}

    @pytest.mark.parametrize(
        "payload",
        [
            {"query": "q", "filters": {"peer_id": "bob"}},
            {"query": "q", "filters": {"session_id": 5}},
            {"query": "q", "session_id": "s1", "filters": {"session_id": ["s2"]}},
            {"query": ""},
            {"query": "x" * 10_001},
        ],
    )
    async def test_bad_requests_are_422(self, world: World, client: httpx.AsyncClient, payload: dict) -> None:  # noqa: F811
        response = await client.post(f"/v1/default/banks/{world.bank_id}/people/alice/chat", json=payload)
        assert response.status_code == 422

    async def test_nul_bytes_are_stripped_from_the_query(
        self, world: World, client: httpx.AsyncClient, fake_llm: FakeLLM
    ) -> None:  # noqa: F811
        fake_llm.script = ["ok"]
        await client.post(f"/v1/default/banks/{world.bank_id}/people/alice/chat", json={"query": "su\u0000shi"})
        assert "Query: sushi" in fake_llm.calls[0]["messages"][1]["content"]


class TestStreaming:
    async def test_frames_then_a_terminal_event_with_sources(
        self, world: World, client: httpx.AsyncClient, fake_llm: FakeLLM
    ) -> None:  # noqa: F811
        fake_llm.script = [tool_turn("grep_messages", text="Berlin"), answer_ready(), "line one\nline two"]
        response = await client.post(
            f"/v1/default/banks/{world.bank_id}/people/alice/chat", json={"query": "q", "stream": True}
        )
        assert response.headers["content-type"].startswith("text/event-stream")
        frames = _frames(response.text)
        assert [f["delta"]["content"] for f in frames[:-1]] == ["line one\n", "line two"]
        assert all(f["done"] is False for f in frames[:-1])
        final = frames[-1]
        assert final["done"] is True and "evidence" not in final
        assert world.messages["m_berlin"] in {s["id"] for s in final["sources"]}

    async def test_the_terminal_event_carries_evidence_when_asked(
        self, world: World, client: httpx.AsyncClient, fake_llm: FakeLLM
    ) -> None:  # noqa: F811
        fake_llm.script = ["ok"]
        response = await client.post(
            f"/v1/default/banks/{world.bank_id}/people/alice/chat",
            json={"query": "q", "stream": True, "include_evidence": True},
        )
        final = _frames(response.text)[-1]
        assert final["done"] is True and final["evidence"]["conclusions"]

    async def test_a_missing_target_is_a_404_before_the_stream_starts(
        self, world: World, client: httpx.AsyncClient
    ) -> None:  # noqa: F811
        response = await client.post(
            f"/v1/default/banks/{world.bank_id}/people/alice/chat",
            json={"query": "q", "stream": True, "target": "nobody"},
        )
        assert response.status_code == 404

    async def test_a_failure_in_the_stream_is_an_error_event(
        self, world: World, client: httpx.AsyncClient, fake_llm: FakeLLM
    ) -> None:  # noqa: F811
        fake_llm.script = [answer_ready(), "not json"]
        response = await client.post(
            f"/v1/default/banks/{world.bank_id}/people/alice/chat",
            json={
                "query": "q",
                "stream": True,
                "response_format": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            },
        )
        frames = _frames(response.text)
        assert frames[-1]["done"] is True and "error" in frames[-1]


class TestBankChat:
    async def test_answers_across_the_bank(self, world: World, client: httpx.AsyncClient, fake_llm: FakeLLM) -> None:  # noqa: F811
        fake_llm.script = [tool_turn("grep_messages", text="Zebra"), answer_ready(), "Bob works on Zebra."]
        response = await client.post(f"/v1/default/banks/{world.bank_id}/people/chat", json={"query": "Who has Zebra?"})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["content"] == "Bob works on Zebra."
        assert world.messages["m_zebra"] in {s["id"] for s in body["sources"]}
        assert "## Bank overview (prefetched)" in fake_llm.calls[0]["messages"][1]["content"]

    async def test_stream(self, world: World, client: httpx.AsyncClient, fake_llm: FakeLLM) -> None:  # noqa: F811
        fake_llm.script = [tool_turn("grep_messages", text="Zebra"), answer_ready(), "answer"]
        response = await client.post(
            f"/v1/default/banks/{world.bank_id}/people/chat", json={"query": "q", "stream": True}
        )
        frames = _frames(response.text)
        assert frames[0]["delta"]["content"] == "answer" and frames[-1]["done"] is True

    async def test_a_person_named_chat_is_still_reachable(
        self, world: World, client: httpx.AsyncClient, fake_llm: FakeLLM
    ) -> None:  # noqa: F811
        fake_llm.script = ["ok"]
        response = await client.post(f"/v1/default/banks/{world.bank_id}/people/chat/chat", json={"query": "q"})
        assert response.status_code == 200


class TestSessionAllowlist:
    def test_none_without_filters(self) -> None:
        assert extract_session_allowlist(None) is None
        assert extract_session_allowlist({}) is None

    @pytest.mark.parametrize("value", ["s1", ["s1"], {"in": ["s1"]}])
    def test_accepts_the_three_shapes(self, value: object) -> None:
        assert extract_session_allowlist({"session_id": value}, must_include="s1") == ["s1"]

    def test_rejects_other_keys_shapes_and_uncovered_sessions(self) -> None:
        for filters, must in [({"x": 1}, None), ({"session_id": 3}, None), ({"session_id": ["a"]}, "b")]:
            with pytest.raises(ValidationException):
                extract_session_allowlist(filters, must_include=must)
