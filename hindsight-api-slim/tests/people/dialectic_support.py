"""Helpers for the dialectic tests: a context without a database, scripted tool turns, and a seeded bank."""

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest_asyncio

from hindsight_api.engine.response_models import LLMToolCall, LLMToolCallResult
from hindsight_api.models import RequestContext
from hindsight_api.people.context import PeopleContext
from hindsight_api.people.dialectic.evidence import EvidenceAccumulator
from hindsight_api.people.dialectic.tools import create_tool_executor
from tests.people.conftest import FakeLLM
from tests.people.people_helpers import add_message, add_session, make_doc, save


def bare_context(llm: FakeLLM | None = None) -> PeopleContext:
    """A people context whose engine is only a stand-in for the model: for tests that touch no database."""
    engine = SimpleNamespace(_reflect_llm_config=llm or FakeLLM(), _consolidation_llm_config=llm or FakeLLM())
    return PeopleContext(engine=engine, request_context=RequestContext(internal=True))  # type: ignore[arg-type]


def tool_turn(name: str, **arguments: Any) -> LLMToolCallResult:
    """One scripted tool turn: the model calls ``name`` with ``arguments``."""
    return LLMToolCallResult(
        content=None,
        tool_calls=[LLMToolCall(id=f"call_{name}_{len(json.dumps(arguments))}", name=name, arguments=arguments)],
        finish_reason="tool_calls",
    )


def answer_ready() -> LLMToolCallResult:
    return tool_turn("answer_ready")


def at(day: int) -> datetime:
    """Noon UTC on a day of March 2026."""
    return datetime(2026, 3, day, 12, 0, tzinfo=UTC)


@dataclass
class World:
    """A seeded bank: message public ids and conclusion ids by key."""

    ctx: PeopleContext
    bank_id: str
    messages: dict[str, str] = field(default_factory=dict)
    conclusions: dict[str, str] = field(default_factory=dict)


async def tools_for(
    world: World,
    observer: str = "alice",
    observed: str = "alice",
    *,
    session: str | None = None,
    allowlist: list[str] | None = None,
    evidence: EvidenceAccumulator | None = None,
) -> Any:
    """The tool executor of the pair agent over a seeded bank."""
    return await create_tool_executor(
        world.ctx,
        world.bank_id,
        observer,
        observed,
        session_name=session,
        session_allowlist=allowlist,
        agent_type="dialectic",
        evidence=evidence,
    )


@pytest_asyncio.fixture
async def world(ctx: PeopleContext, bank_id: str) -> World:
    """Alice and Bob talked in s1, Alice alone in s2, Bob alone in s3; conclusions cite their messages."""
    w = World(ctx, bank_id)
    await add_session(ctx, bank_id, "s1", ["alice", "bob"])
    await add_session(ctx, bank_id, "s2", ["alice"])
    await add_session(ctx, bank_id, "s3", ["bob"])
    rows = [
        ("m_sushi", "s1", "alice", "I love sushi from Nobu", 1, "https://chat.example/s1/1", at(1)),
        ("m_nice", "s1", "bob", "Nice, I have never been", 2, "https://chat.example/s1/2", at(2)),
        ("m_berlin", "s1", "alice", "I moved to Berlin last week", 3, "https://chat.example/s1/3", at(5)),
        ("m_cool", "s1", "bob", "Cool, send photos", 4, "https://chat.example/s1/4", at(6)),
        ("m_deadline", "s2", "alice", "My deadline is April 25", 1, "https://chat.example/s2/1", at(10)),
        ("m_moved", "s2", "alice", "The deadline was rescheduled to April 22", 2, "https://chat.example/s2/2", at(20)),
        ("m_zebra", "s3", "bob", "My secret project is called Zebra", 1, "https://chat.example/s3/1", at(25)),
    ]
    for key, session, peer, content, seq, url, said in rows:
        w.messages[key] = await add_message(ctx, bank_id, session, peer, content, seq=seq, source_url=url, said_at=said)

    async def conclude(
        key: str, content: str, observer: str, observed: str, *, session: str | None, cites: list[str], **kw: Any
    ) -> None:
        doc = await make_doc(ctx, content, session=session, source_message_ids=[w.messages[m] for m in cites], **kw)
        result = await save(ctx, bank_id, [doc], observer=observer, observed=observed)
        w.conclusions[key] = result.created_conclusions[0].id

    await conclude("sushi", "Alice loves sushi", "alice", "alice", session="s1", cites=["m_sushi"])
    await conclude("berlin", "Alice lives in Berlin", "alice", "alice", session="s1", cites=["m_berlin"])
    await conclude("deadline", "Alice's deadline is April 22", "alice", "alice", session="s2", cites=["m_moved"])
    await conclude("zebra", "Bob works on Zebra", "bob", "bob", session="s3", cites=["m_zebra"])
    await conclude("bob_of_alice", "Bob thinks Alice is a foodie", "bob", "alice", session="s1", cites=["m_sushi"])
    deduction = await make_doc(
        ctx,
        "Alice is a foodie who relocated",
        level="deductive",
        session=None,
        source_ids=[w.conclusions["sushi"], w.conclusions["berlin"]],
        premises=["Alice loves sushi", "Alice lives in Berlin"],
    )
    result = await save(ctx, bank_id, [deduction], observer="alice", observed="alice")
    w.conclusions["foodie"] = result.created_conclusions[0].id
    return w
