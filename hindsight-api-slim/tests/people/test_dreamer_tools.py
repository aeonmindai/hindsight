"""The tool loop and the dreamer's tools (adapted for the fork: Honcho's loop and tool handlers run on its own
LLM layer and SQLAlchemy; these run on the engine's provider and asyncpg)."""

import json
from typing import Any

import pytest

from hindsight_api.people import peer_card as peer_card_crud
from hindsight_api.people.context import PeopleContext
from hindsight_api.people.dreamer.agent_tools import (
    DEDUCTION_SPECIALIST_TOOLS,
    create_tool_executor,
)
from hindsight_api.people.dreamer.tool_loop import (
    DONE_TOOL_NAME,
    ToolResult,
    execute_tool_loop,
    function_tools,
    get_current_iteration,
)
from tests.people.conftest import FakeLLM
from tests.people.dialectic_support import bare_context, tool_turn
from tests.people.dreamer_support import open_target, scalar, seed
from tests.people.people_helpers import add_message

TOOLS = [{"name": "ping", "description": "Ping", "input_schema": {"type": "object", "properties": {}}}]


class TestToolLoop:
    async def test_runs_tools_until_done(self):
        llm = FakeLLM([tool_turn("ping", n=1), tool_turn("ping", n=2), tool_turn(DONE_TOOL_NAME, summary="all done")])
        seen: list[tuple[str, dict[str, Any], int]] = []

        async def executor(name: str, tool_input: dict[str, Any]) -> str:
            seen.append((name, tool_input, get_current_iteration()))
            return f"pong {tool_input['n']}"

        result = await execute_tool_loop(
            llm,
            messages=[{"role": "user", "content": "go"}],
            tools=TOOLS,
            tool_executor=executor,
            max_tool_iterations=5,
            max_tokens=100,
            scope="test",
        )

        assert seen == [("ping", {"n": 1}, 1), ("ping", {"n": 2}, 2)]
        assert (result.content, result.iterations, result.hit_iteration_limit) == ("all done", 3, False)
        assert [c["tool_name"] for c in result.tool_calls_made] == ["ping", "ping"]
        # The model saw the done tool next to the caller's tools, and results came back as tool messages.
        first = llm.calls[0]
        assert [t["function"]["name"] for t in first["tools"]] == ["ping", DONE_TOOL_NAME]
        last_messages = llm.calls[2]["messages"]
        assert [m["role"] for m in last_messages] == ["user", "assistant", "tool", "assistant", "tool"]
        assert last_messages[2]["content"] == "pong 1"
        assert last_messages[1]["tool_calls"][0]["function"]["arguments"] == json.dumps({"n": 1})

    async def test_a_tool_error_goes_back_to_the_model(self):
        llm = FakeLLM([tool_turn("ping"), tool_turn(DONE_TOOL_NAME)])

        async def executor(name: str, tool_input: dict[str, Any]) -> str:
            raise RuntimeError("boom")

        result = await execute_tool_loop(
            llm, messages=[], tools=TOOLS, tool_executor=executor, max_tool_iterations=3, max_tokens=10, scope="t"
        )
        assert llm.calls[1]["messages"][-1] == {
            "role": "tool",
            "tool_call_id": llm.calls[1]["messages"][-2]["tool_calls"][0]["id"],
            "content": "Error: boom",
        }
        assert result.tool_calls_made == []

    async def test_budget_line_follows_the_tool_results(self):
        llm = FakeLLM([tool_turn("ping"), tool_turn(DONE_TOOL_NAME)])

        async def executor(name: str, tool_input: dict[str, Any]) -> str:
            return "pong"

        await execute_tool_loop(
            llm,
            messages=[],
            tools=TOOLS,
            tool_executor=executor,
            max_tool_iterations=4,
            max_tokens=10,
            scope="t",
            budget_line=lambda used, limit: f"used {used} of {limit}",
        )
        assert llm.calls[1]["messages"][-1] == {"role": "user", "content": "used 1 of 4"}

    async def test_the_final_call_offers_only_the_write_tools(self):
        write = {"name": "write", "description": "Write", "input_schema": {"type": "object", "properties": {}}}
        llm = FakeLLM([tool_turn("ping"), tool_turn("ping"), tool_turn("write"), tool_turn("ping")])
        ran: list[str] = []

        async def executor(name: str, tool_input: dict[str, Any]) -> str:
            ran.append(name)
            return "ok"

        result = await execute_tool_loop(
            llm,
            messages=[],
            tools=[*TOOLS, write],
            tool_executor=executor,
            max_tool_iterations=2,
            max_tokens=10,
            scope="t",
            final_call_tools=[write],
        )

        final = llm.calls[2]
        assert [t["function"]["name"] for t in final["tools"]] == ["write", DONE_TOOL_NAME]
        # The prompt still declares every tool, so the cached prefix is the same.
        assert [t["function"]["name"] for t in final["declared_tools"]] == ["ping", "write", DONE_TOOL_NAME]
        assert ran == ["ping", "ping", "write"]
        assert result.hit_iteration_limit is True

    async def test_a_tool_outside_the_final_set_is_dropped_at_the_limit(self):
        write = {"name": "write", "description": "Write", "input_schema": {"type": "object", "properties": {}}}
        llm = FakeLLM([tool_turn("ping"), tool_turn("ping")])
        ran: list[str] = []

        async def executor(name: str, tool_input: dict[str, Any]) -> str:
            ran.append(name)
            return "ok"

        await execute_tool_loop(
            llm,
            messages=[],
            tools=[*TOOLS, write],
            tool_executor=executor,
            max_tool_iterations=1,
            max_tokens=10,
            scope="t",
            final_call_tools=[write],
        )
        assert ran == ["ping"]

    async def test_without_final_tools_the_limit_asks_for_a_text_answer(self):
        llm = FakeLLM([tool_turn("ping"), "the summary"])

        async def executor(name: str, tool_input: dict[str, Any]) -> str:
            return "ok"

        result = await execute_tool_loop(
            llm, messages=[], tools=TOOLS, tool_executor=executor, max_tool_iterations=1, max_tokens=10, scope="t"
        )
        assert result.content == "the summary"
        assert llm.calls[1]["kind"] == "call"
        assert "maximum number of tool calls" in llm.calls[1]["messages"][-1]["content"]

    def test_function_tools_wraps_honcho_shaped_tools_once(self):
        [wrapped] = function_tools(TOOLS)
        assert wrapped["function"]["parameters"] == TOOLS[0]["input_schema"]
        assert function_tools([wrapped]) == [wrapped]


async def _executor(ctx: PeopleContext, bank_id: str, history=None, **kwargs: Any):
    return await create_tool_executor(
        ctx, bank_id, "ann", "ann", include_observation_ids=True, agent_type="deduction", history=history, **kwargs
    )


class TestDreamerTools:
    def test_tool_definitions_are_complete_schemas(self):
        for tool in DEDUCTION_SPECIALIST_TOOLS:
            assert tool["name"] and tool["description"]
            assert tool["input_schema"]["type"] == "object"

    async def test_create_observations_grounds_sources_and_dates_them(self, ctx: PeopleContext, bank_id: str):
        _session, (coffee_id, nights_id) = await seed(ctx, bank_id)
        execute = await _executor(ctx, bank_id)

        result = await execute(
            "create_observations_deductive",
            {
                "observations": [
                    {"content": "Ann relies on caffeine", "source_ids": [f"id:{coffee_id}", nights_id, "fabricated"]},
                    {"content": "Ann is a ghost", "source_ids": ["fabricated"]},
                ]
            },
        )

        assert result.startswith("Created 1 observations for ann by ann")
        assert "Failed 1" in result and "do not resolve to existing observations" in result
        row = await _fetch_one(
            ctx, bank_id, "SELECT id, level, internal_metadata FROM {t} WHERE content = 'Ann relies on caffeine'"
        )
        assert row["level"] == "deductive"
        # The fabricated id is gone and the "id:" prefix is stripped.
        sources = await _fetch_all(
            ctx,
            f"SELECT source_id FROM {ctx.table('conclusion_sources')} WHERE bank_id = $1 AND conclusion_id = $2 "
            "AND kind = 'conclusion' ORDER BY position",
            bank_id,
            row["id"],
        )
        assert [s["source_id"] for s in sources] == [coffee_id, nights_id]
        # Dated to the latest source, not to now.
        assert json.loads(row["internal_metadata"])["message_created_at"].endswith("Z")

    async def test_the_dreamer_cannot_mint_explicit_conclusions(self, ctx: PeopleContext, bank_id: str):
        """The tool forces its own level, so asking for an explicit conclusion yields a deductive one."""
        _session, (coffee_id, _nights_id) = await seed(ctx, bank_id)
        execute = await _executor(ctx, bank_id)

        result = await execute(
            "create_observations_deductive",
            {"observations": [{"content": "Ann likes mornings", "level": "explicit", "source_ids": [coffee_id]}]},
        )

        assert "(0 explicit, 1 deductive, 0 inductive, 0 contradiction)" in result

    async def test_validation_failures_are_reported(self, ctx: PeopleContext, bank_id: str):
        execute = await _executor(ctx, bank_id)
        assert (await execute("create_observations_deductive", {"observations": []})).startswith("ERROR:")
        assert (await execute("create_observations_deductive", {"observations": "x"})).startswith("ERROR:")
        result = await execute("create_observations_deductive", {"observations": [{"content": "x"}]})
        assert result.startswith("ERROR: All observations failed validation")

    async def test_peer_card_entries_are_validated_and_deduplicated(self, ctx: PeopleContext, bank_id: str):
        await seed(ctx, bank_id)
        target = await open_target(ctx, bank_id)
        execute = await _executor(ctx, bank_id, history=target)

        result = await execute(
            "update_peer_card",
            {
                "content": [
                    "IDENTITY: Name: Ann",
                    "IDENTITY: name: ann",  # a case-insensitive duplicate: dropped without complaint
                    "identity: name: ann",  # wrong prefix case
                    "TRAIT: shy",  # not an allowed prefix
                    "ATTRIBUTE: ",  # no body
                    "ATTRIBUTE: Works nights",
                ]
            },
        )

        assert result.startswith("Updated peer card for ann by ann with 2 entries.")
        assert "Rejected 3 of 5 entries" in result
        assert await peer_card_crud.get_peer_card(ctx, bank_id, "ann", "ann") == [
            "IDENTITY: Name: Ann",
            "ATTRIBUTE: Works nights",
        ]

    async def test_a_card_of_only_invalid_entries_keeps_the_existing_card(self, ctx: PeopleContext, bank_id: str):
        await seed(ctx, bank_id)
        await peer_card_crud.set_peer_card(ctx, bank_id, ["IDENTITY: Name: Ann"], "ann", "ann")
        execute = await _executor(ctx, bank_id)

        result = await execute("update_peer_card", {"content": ["TRAIT: shy"]})

        assert result.startswith("Rejected all 1 entries")
        assert await peer_card_crud.get_peer_card(ctx, bank_id, "ann", "ann") == ["IDENTITY: Name: Ann"]

    async def test_peer_card_creation_can_be_disabled_by_configuration(self, ctx: PeopleContext, bank_id: str):
        from hindsight_api.people.schemas import get_configuration

        config = get_configuration(None, {"peer_card": {"create": False}}, None)
        await seed(ctx, bank_id)
        execute = await _executor(ctx, bank_id, configuration=config)
        result = await execute("update_peer_card", {"content": ["IDENTITY: Name: Ann"]})
        assert "disabled" in result
        assert await peer_card_crud.get_peer_card(ctx, bank_id, "ann", "ann") is None

    async def test_search_and_recent_return_ids_the_model_can_cite(self, ctx: PeopleContext, bank_id: str):
        _session, (coffee_id, nights_id) = await seed(ctx, bank_id)
        execute = await _executor(ctx, bank_id)

        recent = await execute("get_recent_observations", {"limit": 5})
        assert f"[id:{coffee_id}]" in recent and f"[id:{nights_id}]" in recent

        found = await execute("search_memory", {"query": "Ann drinks coffee daily", "top_k": 5})
        assert found.startswith("Found 2 observations for query 'Ann drinks coffee daily'")
        assert f"[id:{coffee_id}]" in found

        empty = await (await _executor(ctx, bank_id + "-none"))("search_memory", {"query": "anything"})
        assert empty == "No observations found for query 'anything'"

    async def test_search_messages_finds_what_was_said(self, ctx: PeopleContext, bank_id: str):
        session, _ids = await seed(ctx, bank_id)
        await add_message(ctx, bank_id, session, "ann", "I switched to decaf last week", seq=1)
        execute = await _executor(ctx, bank_id)

        found = await execute("search_messages", {"query": "decaf"})
        assert "I switched to decaf last week" in found
        assert await execute("search_messages", {"query": "zeppelin"}) == "No messages found for query 'zeppelin'"

    async def test_delete_reports_what_it_removed(self, ctx: PeopleContext, bank_id: str):
        _session, (coffee_id, _nights_id) = await seed(ctx, bank_id)
        execute = await _executor(ctx, bank_id)

        assert (
            await execute("delete_observations", {"observation_ids": ["id:" + coffee_id, "nope"]})
            == "Deleted 1 observations"
        )
        assert (await execute("delete_observations", {"observation_ids": []})).startswith("ERROR:")
        assert (
            await scalar(
                ctx,
                f"SELECT count(*) FROM {ctx.table('conclusions')} WHERE bank_id = $1 AND deleted_at IS NULL",
                bank_id,
            )
            == 1
        )

    async def test_executor_answers_unknown_tools_and_missing_parameters(self, ctx: PeopleContext, bank_id: str):
        execute = await _executor(ctx, bank_id)
        assert await execute("no_such_tool", {}) == "Unknown tool: no_such_tool"
        assert (await execute("search_memory", {})).startswith("Tool search_memory missing required parameter")

    async def test_executor_publishes_metadata_for_the_loop(self, ctx: PeopleContext, bank_id: str):
        from hindsight_api.people.dreamer.tool_loop import get_last_tool_error, get_last_tool_metadata

        await seed(ctx, bank_id)
        execute = await _executor(ctx, bank_id)
        await execute("search_memory", {"query": "coffee"})
        assert get_last_tool_metadata()["results_count"] == 2
        assert get_last_tool_error() is False
        await execute("no_such_tool", {})
        assert get_last_tool_error() is True

    def test_tool_result_behaves_like_its_content(self):
        result = ToolResult(content="Created 2", metadata={"created_count": 2})
        assert "Created 2" in result and str(result) == "Created 2"

    async def test_a_ground_loop_over_the_real_executor(self, ctx: PeopleContext, bank_id: str):
        """The loop and the executor together: the model reads memory, then finishes."""
        await seed(ctx, bank_id)
        execute = await _executor(ctx, bank_id)
        llm = bare_context().llm_background
        llm.script = [tool_turn("get_recent_observations", limit=2), tool_turn(DONE_TOOL_NAME)]

        result = await execute_tool_loop(
            llm,
            messages=[],
            tools=DEDUCTION_SPECIALIST_TOOLS,
            tool_executor=execute,
            max_tool_iterations=4,
            max_tokens=100,
            scope="t",
        )
        assert [c["tool_name"] for c in result.tool_calls_made] == ["get_recent_observations"]
        assert "Found 2 recent observations" in str(result.tool_calls_made[0]["tool_result"])


async def _fetch_one(ctx: PeopleContext, bank_id: str, sql_with_t: str) -> Any:
    sql = sql_with_t.replace("{t}", ctx.table("conclusions"))
    async with ctx.conn() as conn:
        row = await conn.fetchrow(sql.replace("WHERE ", "WHERE bank_id = $1 AND ", 1), bank_id)
    assert row is not None
    return {**dict(row), "internal_metadata": row["internal_metadata"]}


async def _fetch_all(ctx: PeopleContext, sql: str, *args: Any) -> list[Any]:
    async with ctx.conn() as conn:
        return list(await conn.fetch(sql, *args))


@pytest.fixture(autouse=True)
def _reset_loop_state():
    from hindsight_api.people.dreamer.tool_loop import set_current_iteration

    yield
    set_current_iteration(0)
