# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp")

from mcp.shared.exceptions import MCPError

from ag2 import Agent, Context, Variable
from ag2.events import ToolCallEvent, ToolErrorEvent, ToolResultEvent, ToolResultsEvent
from ag2.exceptions import ToolNotFoundError
from ag2.mcp import AskContext, MCPFunctionTool, MCPServer, mcp_tool
from ag2.stream import MemoryStream
from ag2.testing import TestConfig, TrackingConfig
from ag2.tools import MCPServerConfig, MCPStdioServerConfig, MCPToolkit, Toolkit, tool
from ag2.tools.builtin.tool_search import ToolSearchTool
from ag2.tools.types import FunctionToolSchema, Tool, ToolSchema
from test._serving import serving


@mcp_tool
def echo(message: str) -> str:
    """Echo a message."""
    return f"echo: {message}"


@mcp_tool
def reverse(message: str) -> str:
    """Reverse a message."""
    return message[::-1]


class _DiscoveryCounter:
    def __init__(self) -> None:
        self.calls = 0
        self.fail_next = False

    async def __call__(self, access: object) -> AskContext:
        self.calls += 1
        if self.fail_next:
            self.fail_next = False
            raise MCPError(-32000, "test discovery failure")
        return AskContext()


@asynccontextmanager
async def _serve(
    *tools: MCPFunctionTool,
    counter: _DiscoveryCounter | None = None,
) -> AsyncGenerator[str]:
    server = MCPServer(
        Agent("served", config=TestConfig("unused")),
        tools=tools,
        context_provider=counter,
        path="/mcp",
    )
    async with serving(server) as base_url:
        yield f"{base_url}/mcp/"


def _offered(config: TrackingConfig) -> list[list[str]]:
    return [
        sorted(schema.function.name for schema in call.tools if isinstance(schema, FunctionToolSchema))
        for call in config.calls
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "values", "expected"),
    [
        ("allowed_tools", [["echo"], ["reverse"], [], ["echo"]], [["echo"], ["reverse"], [], ["echo"]]),
        (
            "blocked_tools",
            [["ask", "reverse"], ["ask", "echo"], ["ask", "echo", "reverse"], []],
            [["echo"], ["reverse"], [], ["ask", "echo", "reverse"]],
        ),
        (
            "tool_name_prefix",
            ["first_", "second_", ""],
            [
                ["first_ask", "first_echo", "first_reverse"],
                ["second_ask", "second_echo", "second_reverse"],
                ["ask", "echo", "reverse"],
            ],
        ),
    ],
)
async def test_each_conversation_uses_its_discovery_configuration(
    field: str, values: list[Any], expected: list[list[str]]
) -> None:
    tracking = TrackingConfig(TestConfig("done"))
    async with _serve(echo, reverse) as url:
        toolkit = MCPToolkit(MCPServerConfig(server_url=url, **{field: Variable("selection")}))
        agent = Agent("consumer", config=tracking, tools=[toolkit])
        for value in values:
            await agent.ask("List tools", variables={"selection": value})

    assert _offered(tracking) == expected


@pytest.mark.asyncio
async def test_a_tool_removed_by_the_next_conversation_is_not_executed() -> None:
    call = ToolCallEvent(name="echo", arguments='{"message":"hi"}')
    tracking = TrackingConfig(TestConfig(call, "done", raise_tool_errors=False))
    async with _serve(echo) as url:
        toolkit = MCPToolkit(MCPServerConfig(server_url=url, allowed_tools=Variable("allowed")))
        agent = Agent("consumer", config=tracking, tools=[toolkit])
        await agent.ask("Echo", variables={"allowed": ["echo"]})
        await agent.ask("No tools", variables={"allowed": []})

    assert _offered(tracking) == [["echo"], ["echo"], [], []]
    tracking.mock.assert_any_call(ToolResultsEvent([ToolResultEvent.from_call(call, result="echo: hi")]))
    tracking.mock.assert_called_with(
        ToolResultsEvent([ToolErrorEvent.from_call(call, error=ToolNotFoundError("echo"))])
    )


@pytest.mark.asyncio
async def test_switching_server_urls_refreshes_the_tools_and_calls_the_current_server() -> None:
    echo_call = ToolCallEvent(name="echo", arguments='{"message":"hi"}')
    reverse_call = ToolCallEvent(name="reverse", arguments='{"message":"hi"}')
    first = TrackingConfig(TestConfig(echo_call, "done"))
    second = TrackingConfig(TestConfig(reverse_call, "done"))
    toolkit = MCPToolkit(MCPServerConfig(server_url=Variable("url")))
    agent = Agent("consumer", config=TestConfig("unused"), tools=[toolkit])

    async with _serve(echo) as first_url, _serve(reverse) as second_url:
        await agent.ask("Echo", config=first, variables={"url": first_url})
        await agent.ask("Reverse", config=second, variables={"url": second_url})

    assert _offered(first) == [["ask", "echo"], ["ask", "echo"]]
    assert _offered(second) == [["ask", "reverse"], ["ask", "reverse"]]
    first.mock.assert_called_with(ToolResultsEvent([ToolResultEvent.from_call(echo_call, result="echo: hi")]))
    second.mock.assert_called_with(ToolResultsEvent([ToolResultEvent.from_call(reverse_call, result="ih")]))


@pytest.mark.asyncio
@pytest.mark.parametrize("variable", [False, True], ids=["static", "variable"])
async def test_mutating_an_allowlist_does_not_mutate_the_cached_configuration(variable: bool) -> None:
    allowed = ["echo"]
    context = Context(stream=MemoryStream(), variables={"allowed": allowed})
    async with _serve(echo, reverse) as url:
        toolkit = MCPToolkit(
            MCPServerConfig(server_url=url, allowed_tools=Variable("allowed") if variable else allowed)
        )
        first = list(await toolkit.schemas(context))
        allowed[:] = ["reverse"]
        second = list(await toolkit.schemas(context))

    assert [
        [schema.function.name for schema in schemas if isinstance(schema, FunctionToolSchema)]
        for schemas in (first, second)
    ] == [["echo"], ["reverse"]]


@pytest.mark.asyncio
async def test_unchanged_configuration_reuses_discovery_across_contexts() -> None:
    counter = _DiscoveryCounter()
    tracking = TrackingConfig(TestConfig("done"))
    async with _serve(echo, counter=counter) as url:
        toolkit = MCPToolkit(MCPServerConfig(server_url=url, allowed_tools=Variable("allowed")))
        agent = Agent("consumer", config=tracking, tools=[toolkit])
        await asyncio.gather(
            agent.ask("First", variables={"allowed": ["echo"], "unrelated": 1}),
            agent.ask("Second", variables={"allowed": ["echo"], "unrelated": 2}),
        )

    assert _offered(tracking) == [["echo"], ["echo"]]
    assert counter.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["headers", "authorization_token"])
async def test_new_credentials_trigger_discovery_even_when_the_url_is_unchanged(field: str) -> None:
    counter = _DiscoveryCounter()
    tracking = TrackingConfig(TestConfig("done"))
    async with _serve(echo, counter=counter) as url:
        toolkit = MCPToolkit(MCPServerConfig(server_url=url, **{field: Variable("credentials")}))
        agent = Agent("consumer", config=tracking, tools=[toolkit])
        values = [{"X-Tenant": "first"}, {"X-Tenant": "second"}] if field == "headers" else ["first", "second"]
        for value in values:
            await agent.ask("List tools", variables={"credentials": value})

    assert _offered(tracking) == [["ask", "echo"], ["ask", "echo"]]
    assert counter.calls == 2


@pytest.mark.asyncio
async def test_failed_rediscovery_does_not_return_stale_tools_and_can_be_retried() -> None:
    counter = _DiscoveryCounter()
    tracking = TrackingConfig(TestConfig("done"))
    async with _serve(echo, reverse, counter=counter) as url:
        agent = Agent(
            "consumer",
            config=tracking,
            tools=[MCPToolkit(MCPServerConfig(server_url=url, allowed_tools=Variable("allowed")))],
        )
        await agent.ask("First", variables={"allowed": ["echo"]})
        counter.fail_next = True
        with pytest.RaisesGroup(
            pytest.RaisesExc(MCPError, match="test discovery failure"), flatten_subgroups=True, allow_unwrapped=True
        ):
            await agent.ask("Second", variables={"allowed": ["reverse"]})
        await agent.ask("Retry", variables={"allowed": ["reverse"]})

    assert _offered(tracking) == [["echo"], ["reverse"]]
    assert counter.calls == 3


def local_echo(message: str) -> str:
    return f"local: {message}"


@pytest.mark.asyncio
@pytest.mark.parametrize("local_name", ["local_echo", "echo"])
async def test_rediscovery_preserves_tools_added_locally_to_the_toolkit(local_name: str) -> None:
    call = ToolCallEvent(name=local_name, arguments='{"message":"hi"}')
    tracking = TrackingConfig(TestConfig(call, "done"))
    async with _serve(echo, reverse) as url:
        toolkit = MCPToolkit(MCPServerConfig(server_url=url, allowed_tools=Variable("allowed")))
        toolkit.tool(local_echo, name=local_name)
        agent = Agent("consumer", config=tracking, tools=[toolkit])
        await agent.ask("First", variables={"allowed": ["reverse"]})
        await agent.ask("Second", variables={"allowed": ["echo"]})

    assert _offered(tracking) == [sorted({local_name, "reverse"})] * 2 + [sorted({local_name, "echo"})] * 2
    tracking.mock.assert_called_with(ToolResultsEvent([ToolResultEvent.from_call(call, result="local: hi")]))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prefixes", "expected"),
    [(["remote_", ""], ["echo", "remote_echo"]), (["", "remote_"], ["echo"])],
)
async def test_variable_default_factory_does_not_change_selected_tools(
    prefixes: list[str], expected: list[str]
) -> None:
    call = ToolCallEvent(name="echo", arguments='{"message":"hi"}')
    tracking = TrackingConfig(TestConfig(call, "done"))
    async with _serve(echo) as url:
        agent = Agent(
            "consumer",
            config=tracking,
            tools=[
                MCPToolkit(
                    MCPServerConfig(
                        server_url=url,
                        allowed_tools=["echo"],
                        tool_name_prefix=Variable(
                            "prefix", default_factory=partial(next, iter(prefixes), prefixes[-1])
                        ),
                    )
                ),
                tool(local_echo, name="echo"),
            ],
        )
        await agent.ask("List tools")

    assert _offered(tracking) == [expected, expected]
    tracking.mock.assert_called_with(ToolResultsEvent([ToolResultEvent.from_call(call, result="local: hi")]))


@pytest.mark.asyncio
async def test_rediscovered_remote_tools_still_lose_to_tools_declared_in_code() -> None:
    call = ToolCallEvent(name="echo", arguments='{"message":"hi"}')
    tracking = TrackingConfig(TestConfig(call, "done"))
    async with _serve(echo, reverse) as url:
        agent = Agent(
            "consumer",
            config=tracking,
            tools=[
                MCPToolkit(MCPServerConfig(server_url=url, allowed_tools=Variable("allowed"))),
                tool(local_echo, name="echo"),
            ],
        )
        await agent.ask("First", variables={"allowed": ["echo", "reverse"]})
        await agent.ask("Second", variables={"allowed": ["echo"]})

    assert _offered(tracking) == [["echo", "reverse"]] * 2 + [["echo"]] * 2
    tracking.mock.assert_called_with(ToolResultsEvent([ToolResultEvent.from_call(call, result="local: hi")]))


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["args", "env", "cwd"])
async def test_stdio_discovery_tracks_the_current_launch_configuration(field: str, tmp_path: Path) -> None:
    script = str(Path(__file__).with_name("fixtures") / "mcp_context_server.py")
    options: dict[str, Any] = {"command": sys.executable, "args": [script], field: Variable("launch")}
    if field == "args":
        values: list[Any] = [[script, "first"], [script, "second"]]
    elif field == "env":
        values = [{"AG2_MCP_TEST_TOOL": "first"}, {"AG2_MCP_TEST_TOOL": "second"}]
    else:
        values = [tmp_path / "first", tmp_path / "second"]
        for directory in values:
            directory.mkdir()
    agent = Agent("consumer", config=TestConfig("unused"), tools=[MCPToolkit(MCPStdioServerConfig(**options))])

    for name, value in zip(("first", "second"), values, strict=True):
        call = ToolCallEvent(name=name)
        tracking = TrackingConfig(TestConfig(call, "done"))
        await agent.ask("Call the available tool", config=tracking, variables={"launch": value})
        assert _offered(tracking) == [[name], [name]]
        tracking.mock.assert_called_with(ToolResultsEvent([ToolResultEvent.from_call(call, result=f"called {name}")]))


class _PauseBeforeRegistration(Tool):
    name = "pause_before_registration"

    async def schemas(self, context: Context) -> list[ToolSchema]:
        if context.variables.get("pause"):
            context.dependencies["entered"].set()
            await asyncio.wait_for(context.dependencies["release"].wait(), timeout=10)
        return []


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapper", ["direct", "toolkit", "deferred"])
async def test_concurrent_conversations_keep_their_own_tools_during_registration(wrapper: str) -> None:
    first_call = ToolCallEvent(name="first_echo", arguments='{"message":"hi"}')
    second_call = ToolCallEvent(name="second_reverse", arguments='{"message":"hi"}')
    first = TrackingConfig(TestConfig(first_call, "done"))
    second = TrackingConfig(TestConfig(second_call, "done"))
    entered = asyncio.Event()
    release = asyncio.Event()

    async with _serve(echo, reverse) as url:
        toolkit = MCPToolkit(
            MCPServerConfig(server_url=url, allowed_tools=Variable("allowed"), tool_name_prefix=Variable("prefix"))
        )
        wrapped = (
            Toolkit(toolkit) if wrapper == "toolkit" else ToolSearchTool(toolkit) if wrapper == "deferred" else toolkit
        )
        agent = Agent("consumer", config=TestConfig("unused"), tools=[wrapped, _PauseBeforeRegistration()])
        pending = asyncio.create_task(
            agent.ask(
                "First",
                config=first,
                variables={"allowed": ["echo"], "prefix": "first_", "pause": True},
                dependencies={"entered": entered, "release": release},
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), timeout=10)
            await agent.ask("Second", config=second, variables={"allowed": ["echo", "reverse"], "prefix": "second_"})
        finally:
            release.set()
            await pending

    assert _offered(first) == [["first_echo"], ["first_echo"]]
    assert _offered(second) == [["second_echo", "second_reverse"], ["second_echo", "second_reverse"]]
    first.mock.assert_called_with(ToolResultsEvent([ToolResultEvent.from_call(first_call, result="echo: hi")]))
    second.mock.assert_called_with(ToolResultsEvent([ToolResultEvent.from_call(second_call, result="ih")]))
