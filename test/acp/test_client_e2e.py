# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import Any

import pytest
from acp import schema
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from ag2 import Agent
from ag2.acp import MCPCapabilityError
from ag2.acp.testing import ACPTurn, fake_acp_config
from ag2.events import BaseEvent, ModelReasoning
from ag2.events.tool_events import BuiltinToolCallEvent, BuiltinToolResultEvent
from ag2.exceptions import UnsupportedToolError
from ag2.tools.builtin.mcp_server import MCPServerTool
from ag2.tools.builtin.web_search import WebSearchTool
from ag2.tools.final.function_tool import FunctionTool


def _text(text: str) -> schema.TextContentBlock:
    return schema.TextContentBlock(type="text", text=text)


def _text_update(text: str) -> schema.AgentMessageChunk:
    return schema.AgentMessageChunk(session_update="agent_message_chunk", content=_text(text))


@pytest.mark.asyncio
async def test_ask_streams_thoughts_tools_and_returns_text() -> None:
    cfg = fake_acp_config(
        ACPTurn(
            updates=[
                schema.AgentThoughtChunk(session_update="agent_thought_chunk", content=_text("planning")),
                schema.AgentMessageChunk(session_update="agent_message_chunk", content=_text("done")),
                schema.ToolCallStart(session_update="tool_call", tool_call_id="t1", title="Echo", status="pending"),
                schema.ToolCallProgress(
                    session_update="tool_call_update",
                    tool_call_id="t1",
                    status="completed",
                    content=[schema.ContentToolCallContent(type="content", content=_text("ok"))],
                ),
            ],
            usage=schema.Usage(input_tokens=3, output_tokens=1, total_tokens=4),
        ),
        permission_policy="auto",
    )
    agent = Agent("acp", config=cfg)

    seen: list[BaseEvent] = []

    try:
        async with agent.run("hello") as run:
            run.stream.subscribe(lambda e: seen.append(e))
            result = await run.result()
    finally:
        await cfg.aclose()

    assert result.body == "done"
    assert any(isinstance(e, ModelReasoning) and e.content == "planning" for e in seen)
    assert any(isinstance(e, BuiltinToolCallEvent) and e.name == "Echo" for e in seen)
    assert any(isinstance(e, BuiltinToolResultEvent) for e in seen)


@pytest.mark.asyncio
async def test_turn_timeout_surfaces_timeout() -> None:
    cfg = fake_acp_config(ACPTurn(hang=True), permission_policy="auto", turn_timeout=0.5)
    agent = Agent("acp", config=cfg)

    try:
        async with agent.run("hang") as run:
            result = await run.result()
    finally:
        await cfg.aclose()
    # The turn timed out; body is whatever streamed before the timeout (empty here).
    assert result.body == ""


@pytest.mark.asyncio
async def test_aclose_closes_session() -> None:
    cfg = fake_acp_config(
        ACPTurn(updates=[schema.AgentMessageChunk(session_update="agent_message_chunk", content=_text("hi"))]),
        permission_policy="auto",
    )
    agent = Agent("acp", config=cfg)

    async with agent.run("hello") as run:
        await run.result()

    assert cfg._sessions  # a live session was created
    conns = [s.conn for s in cfg._sessions.values()]
    await cfg.aclose()
    assert cfg._sessions == {}
    for conn in conns:
        assert conn is not None and conn.closed  # the connection context was exited


@pytest.mark.asyncio
async def test_function_tools_are_exposed_and_callable_over_mcp() -> None:
    observed: dict[str, Any] = {}

    def add(a: int, b: int) -> int:
        """Add two integers."""
        observed["args"] = (a, b)
        return a + b

    cfg: Any = None  # assigned below; on_prompt closure needs it

    async def drive_mcp() -> None:
        # Runs inside the fake agent's prompt turn — exactly when a real CLI
        # agent would call the gateway.
        session = next(iter(cfg._sessions.values()))
        assert session.gateway is not None and session.gateway.url is not None
        observed["mcp_servers"] = session.conn.new_session_kwargs["mcp_servers"]
        async with (
            streamable_http_client(session.gateway.url) as (read, write, _),
            ClientSession(read, write) as mcp_session,
        ):
            await mcp_session.initialize()
            listed = await mcp_session.list_tools()
            observed["tool_names"] = [t.name for t in listed.tools]
            result = await mcp_session.call_tool("add", {"a": 2, "b": 3})
            observed["call_text"] = result.content[0].text
            observed["call_is_error"] = result.isError

    cfg = fake_acp_config(
        ACPTurn(updates=[_text_update("done")], on_prompt=drive_mcp),
        permission_policy="auto",
    )
    agent = Agent("acp", config=cfg, tools=[add])
    try:
        async with agent.run("please add 2 and 3 using the add tool") as run:
            result = await run.result()
    finally:
        await cfg.aclose()

    assert result.body == "done"
    assert observed["tool_names"] == ["add"]
    assert observed["args"] == (2, 3)
    assert observed["call_is_error"] is not True
    assert observed["call_text"] == "5"
    # the gateway itself was advertised to the agent via session/new
    (gateway_server,) = [s for s in observed["mcp_servers"] if s.name == "ag2"]
    assert gateway_server.url.startswith("http://127.0.0.1:")


@pytest.mark.asyncio
async def test_expose_tools_false_disables_gateway() -> None:
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    cfg = fake_acp_config(
        ACPTurn(updates=[_text_update("done")]),
        permission_policy="auto",
        expose_tools=False,
    )
    agent = Agent("acp", config=cfg, tools=[add])
    try:
        async with agent.run("hello") as run:
            await run.result()
        session = next(iter(cfg._sessions.values()))
        assert session.gateway is None
        assert session.conn.new_session_kwargs.get("mcp_servers") is None
    finally:
        await cfg.aclose()


@pytest.mark.asyncio
async def test_provider_builtin_tool_is_hard_error() -> None:
    cfg = fake_acp_config(ACPTurn(updates=[_text_update("done")]), permission_policy="auto")
    agent = Agent("acp", config=cfg, tools=[WebSearchTool()])
    try:
        with pytest.raises(UnsupportedToolError):
            async with agent.run("hello") as run:
                await run.result()
        assert cfg._sessions == {}  # nothing leaked on the failure path
    finally:
        await cfg.aclose()


@pytest.mark.asyncio
async def test_concurrent_tool_calls_are_correlated() -> None:
    async def add(a: int, b: int) -> int:
        """Add two integers."""
        await asyncio.sleep(0.05)  # keep both calls in flight simultaneously
        return a + b

    observed: dict[str, Any] = {}
    cfg: Any = None

    async def drive_mcp() -> None:
        session = next(iter(cfg._sessions.values()))
        async with (
            streamable_http_client(session.gateway.url) as (read, write, _),
            ClientSession(read, write) as mcp_session,
        ):
            await mcp_session.initialize()
            first, second = await asyncio.gather(
                mcp_session.call_tool("add", {"a": 1, "b": 2}),
                mcp_session.call_tool("add", {"a": 3, "b": 4}),
            )
            observed["results"] = (first.content[0].text, second.content[0].text)

    cfg = fake_acp_config(ACPTurn(updates=[_text_update("done")], on_prompt=drive_mcp), permission_policy="auto")
    agent = Agent("acp", config=cfg, tools=[add])
    try:
        async with agent.run("add things") as run:
            await run.result()
    finally:
        await cfg.aclose()

    assert observed["results"] == ("3", "7")  # each call got its own result, not the other's


@pytest.mark.asyncio
async def test_unknown_tool_name_returns_error_not_hang() -> None:
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    observed: dict[str, Any] = {}
    cfg: Any = None

    async def drive_mcp() -> None:
        session = next(iter(cfg._sessions.values()))
        async with (
            streamable_http_client(session.gateway.url) as (read, write, _),
            ClientSession(read, write) as mcp_session,
        ):
            await mcp_session.initialize()
            result = await asyncio.wait_for(mcp_session.call_tool("nope", {}), timeout=5)
            observed["is_error"] = result.isError
            observed["text"] = result.content[0].text

    cfg = fake_acp_config(ACPTurn(updates=[_text_update("done")], on_prompt=drive_mcp), permission_policy="auto")
    agent = Agent("acp", config=cfg, tools=[add])
    try:
        async with agent.run("hello") as run:
            await run.result()
    finally:
        await cfg.aclose()

    assert observed["is_error"] is True
    assert "nope" in observed["text"]


@pytest.mark.asyncio
async def test_capability_error_tears_down_gateway_and_session() -> None:
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    cfg = fake_acp_config(
        ACPTurn(updates=[_text_update("done")]),
        permission_policy="auto",
        agent_capabilities=schema.AgentCapabilities(),  # no HTTP MCP support
    )
    agent = Agent("acp", config=cfg, tools=[add])
    try:
        with pytest.raises(MCPCapabilityError):
            async with agent.run("hello") as run:
                await run.result()
        assert cfg._sessions == {}  # the started gateway did not leak a session
    finally:
        await cfg.aclose()


@pytest.mark.asyncio
async def test_second_turn_hot_updates_gateway_tools() -> None:
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    def mul(a: int, b: int) -> int:
        """Multiply two integers."""
        return a * b

    observed: dict[str, Any] = {}
    cfg: Any = None

    def snapshot_tools(key: str):
        async def probe() -> None:
            session = next(iter(cfg._sessions.values()))
            async with (
                streamable_http_client(session.gateway.url) as (read, write, _),
                ClientSession(read, write) as mcp_session,
            ):
                await mcp_session.initialize()
                listed = await mcp_session.list_tools()
                observed[key] = sorted(t.name for t in listed.tools)

        return probe

    cfg = fake_acp_config(
        ACPTurn(updates=[_text_update("one")], on_prompt=snapshot_tools("turn1")),
        ACPTurn(updates=[_text_update("two")], on_prompt=snapshot_tools("turn2")),
        permission_policy="auto",
    )
    agent = Agent("acp", config=cfg, tools=[add])
    try:
        async with agent.run("first") as run:
            reply = await run.result()
        await reply.ask("second", tools=[FunctionTool.ensure_tool(mul)])
    finally:
        await cfg.aclose()

    assert observed["turn1"] == ["add"]
    assert observed["turn2"] == ["add", "mul"]  # the gateway serves the new turn's snapshot


@pytest.mark.asyncio
async def test_second_turn_external_server_drift_is_hard_error() -> None:
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    cfg = fake_acp_config(
        ACPTurn(updates=[_text_update("one")]),
        permission_policy="auto",
    )
    agent = Agent("acp", config=cfg, tools=[add])
    try:
        async with agent.run("first") as run:
            reply = await run.result()
        with pytest.raises(ValueError, match="MCPServerTool set changed"):
            await reply.ask("second", tools=[MCPServerTool(server_url="https://x/mcp", server_label="ext")])
    finally:
        await cfg.aclose()


@pytest.mark.asyncio
async def test_second_turn_function_tools_without_gateway_is_hard_error() -> None:
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    cfg = fake_acp_config(
        ACPTurn(updates=[_text_update("one")]),
        permission_policy="auto",
    )
    agent = Agent("acp", config=cfg)  # first turn exposes nothing -> no gateway
    try:
        async with agent.run("first") as run:
            reply = await run.result()
        with pytest.raises(ValueError, match="without a tool"):
            await reply.ask("second", tools=[FunctionTool.ensure_tool(add)])
    finally:
        await cfg.aclose()


@pytest.mark.asyncio
async def test_gateway_shuts_down_with_session() -> None:
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    cfg = fake_acp_config(
        ACPTurn(updates=[_text_update("done")]),
        permission_policy="auto",
    )
    agent = Agent("acp", config=cfg, tools=[add])
    async with agent.run("hello") as run:
        await run.result()
    session = next(iter(cfg._sessions.values()))
    gateway = session.gateway
    assert gateway is not None and gateway.url is not None
    await cfg.aclose()
    assert gateway.url is None  # closed together with the session
