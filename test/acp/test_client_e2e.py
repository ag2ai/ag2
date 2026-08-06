# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from typing import Any

import pytest
from acp import schema
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from ag2 import Agent
from ag2.acp import MCPCapabilityError
from ag2.acp.testing import ACPTurn, fake_acp_config
from ag2.events import BaseEvent, ModelReasoning, ModelResponse
from ag2.events.tool_events import BuiltinToolCallEvent, BuiltinToolResultEvent
from ag2.exceptions import UnsupportedToolError
from ag2.tools.builtin.mcp_server import MCPServerTool
from ag2.tools.builtin.web_search import WebSearchTool
from ag2.tools.final.function_tool import FunctionTool


def _text(text: str) -> schema.TextContentBlock:
    return schema.TextContentBlock(type="text", text=text)


def _text_update(text: str) -> schema.AgentMessageChunk:
    return schema.AgentMessageChunk(session_update="agent_message_chunk", content=_text(text))


def _model_option(current: str, *values: str) -> schema.SessionConfigOptionSelect:
    return schema.SessionConfigOptionSelect(
        id="model",
        name="Model",
        category="model",
        type="select",
        current_value=current,
        options=[schema.SessionConfigSelectOption(value=v, name=v) for v in values],
    )


def _grouped_model_option(current: str, groups: dict[str, list[str]]) -> schema.SessionConfigOptionSelect:
    """A model picker whose entries are groups rather than flat options.

    ACP allows either shape. No CLI agent tested so far emits the grouped one
    (Claude Code advertises 5 flat options, Kilo 320), so it needs covering
    here or it goes unexercised entirely.
    """
    return schema.SessionConfigOptionSelect(
        id="model",
        name="Model",
        category="model",
        type="select",
        current_value=current,
        options=[
            schema.SessionConfigSelectGroup(
                group=name,
                name=name,
                options=[schema.SessionConfigSelectOption(value=v, name=v) for v in values],
            )
            for name, values in groups.items()
        ],
    )


def _image_update(data: str = "aGVsbG8=") -> schema.AgentMessageChunk:
    return schema.AgentMessageChunk(
        session_update="agent_message_chunk",
        content=schema.ImageContentBlock(type="image", data=data, mime_type="image/png"),
    )


def _hi_turn() -> ACPTurn:
    return ACPTurn(updates=[_text_update("hi")])


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

    seen: list[BaseEvent] = []
    try:
        async with agent.run("hang") as run:
            run.stream.subscribe(lambda e: seen.append(e))
            result = await run.result()
    finally:
        await cfg.aclose()

    # The turn timed out; body is whatever streamed before the timeout (empty here).
    assert result.body == ""
    [response] = [e for e in seen if isinstance(e, ModelResponse)]
    assert response.finish_reason == "timeout"


@pytest.mark.asyncio
class TestModelSelection:
    async def test_selected_via_config_option(self) -> None:
        calls: list[tuple[str, str | bool]] = []
        cfg = fake_acp_config(
            _hi_turn(),
            config_options=[_model_option("provider/default", "provider/default", "provider/smart")],
            config_option_calls=calls,
            permission_policy="auto",
            model="provider/smart",
        )
        agent = Agent("acp", config=cfg)

        try:
            reply = await agent.ask("hello")
        finally:
            await cfg.aclose()

        assert reply.body == "hi"
        assert calls == [("model", "provider/smart")]

    async def test_matching_current_is_not_resent(self) -> None:
        calls: list[tuple[str, str | bool]] = []
        cfg = fake_acp_config(
            _hi_turn(),
            config_options=[_model_option("provider/default", "provider/default", "provider/smart")],
            config_option_calls=calls,
            permission_policy="auto",
            model="provider/default",
        )
        agent = Agent("acp", config=cfg)

        try:
            await agent.ask("hello")
        finally:
            await cfg.aclose()

        assert calls == []

    async def test_ignored_when_agent_has_no_model_option(self) -> None:
        # No configOptions advertised — `model` stays response metadata, as before.
        calls: list[tuple[str, str | bool]] = []
        cfg = fake_acp_config(_hi_turn(), config_option_calls=calls, permission_policy="auto", model="provider/smart")
        agent = Agent("acp", config=cfg)

        try:
            reply = await agent.ask("hello")
        finally:
            await cfg.aclose()

        assert reply.body == "hi"
        assert calls == []
        assert reply.response.model == "provider/smart"

    async def test_reports_agent_default_when_model_unset(self) -> None:
        """With no `model` set, report what the agent says it runs, not None.

        This is the case that matters most: an agent sitting on a default it
        cannot answer with (Kilo ships on an image model) produces an empty
        reply, and `model=None` would strip the one clue worth having.
        """
        cfg = fake_acp_config(
            _hi_turn(),
            config_options=[_model_option("provider/image-only", "provider/image-only", "provider/text")],
            permission_policy="auto",
        )
        agent = Agent("acp", config=cfg)

        try:
            reply = await agent.ask("hello")
        finally:
            await cfg.aclose()

        assert reply.response.model == "provider/image-only"

    async def test_not_offered_raises(self) -> None:
        cfg = fake_acp_config(
            _hi_turn(),
            config_options=[_model_option("provider/default", "provider/default")],
            permission_policy="auto",
            model="provider/nonexistent",
        )
        agent = Agent("acp", config=cfg)

        try:
            with pytest.raises(ValueError, match="not offered by the ACP agent"):
                await agent.ask("hello")
        finally:
            await cfg.aclose()

    async def test_not_offered_suggests_close_matches(self) -> None:
        """Agents can offer hundreds of ids, so the error points at near misses."""
        cfg = fake_acp_config(
            _hi_turn(),
            config_options=[_model_option("a/default", "a/default", "a/claude-haiku-4.5", "a/claude-sonnet-4")],
            permission_policy="auto",
            model="a/claude-haiku-4-5",  # dashes instead of the dot
        )
        agent = Agent("acp", config=cfg)

        try:
            with pytest.raises(ValueError, match=r"\(3 offered\).*a/claude-haiku-4\.5"):
                await agent.ask("hello")
        finally:
            await cfg.aclose()

    async def test_selected_from_grouped_options(self) -> None:
        """ACP allows grouped entries; no live agent emits them, so cover it here."""
        calls: list[tuple[str, str | bool]] = []
        cfg = fake_acp_config(
            _hi_turn(),
            config_options=[
                _grouped_model_option(
                    "anthropic/default",
                    {"Anthropic": ["anthropic/default", "anthropic/haiku"], "Google": ["google/flash"]},
                )
            ],
            config_option_calls=calls,
            permission_policy="auto",
            model="google/flash",
        )
        agent = Agent("acp", config=cfg)

        try:
            reply = await agent.ask("hello")
        finally:
            await cfg.aclose()

        assert reply.body == "hi"
        assert calls == [("model", "google/flash")]

    async def test_not_offered_in_any_group_raises(self) -> None:
        cfg = fake_acp_config(
            _hi_turn(),
            config_options=[_grouped_model_option("anthropic/default", {"Anthropic": ["anthropic/default"]})],
            permission_policy="auto",
            model="google/flash",
        )
        agent = Agent("acp", config=cfg)

        try:
            with pytest.raises(ValueError, match="not offered by the ACP agent"):
                await agent.ask("hello")
        finally:
            await cfg.aclose()


@pytest.mark.asyncio
class TestEmptyTurnWarning:
    """A turn that ends normally yet produced nothing is worth a word.

    Some CLI agents swallow provider-side failures (an unauthorized or
    text-incapable model) and end the turn with ``end_turn`` and no output at
    all — indistinguishable, from AG2's side, from a healthy silent turn. Warn
    so the empty reply is not the only clue.
    """

    async def test_empty_end_turn_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        cfg = fake_acp_config(ACPTurn(updates=[]), permission_policy="auto", model="provider/model")
        agent = Agent("acp", config=cfg)

        try:
            with caplog.at_level(logging.WARNING, logger="ag2.acp.client"):
                reply = await agent.ask("hello")
        finally:
            await cfg.aclose()

        assert reply.body == ""
        [record] = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert "provider/model" in record.getMessage()

    async def test_tool_only_turn_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        """The agent worked — it just had nothing to say. Not a silent failure."""
        cfg = fake_acp_config(
            ACPTurn(
                updates=[
                    schema.ToolCallStart(session_update="tool_call", tool_call_id="t1", title="Write", status="pending")
                ]
            ),
            permission_policy="auto",
        )
        agent = Agent("acp", config=cfg)

        try:
            with caplog.at_level(logging.WARNING, logger="ag2.acp.client"):
                reply = await agent.ask("hello")
        finally:
            await cfg.aclose()

        assert reply.body == ""
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    async def test_file_only_turn_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        """An image/audio reply is output too, even though it carries no text."""
        cfg = fake_acp_config(ACPTurn(updates=[_image_update()]), permission_policy="auto")
        agent = Agent("acp", config=cfg)

        try:
            with caplog.at_level(logging.WARNING, logger="ag2.acp.client"):
                reply = await agent.ask("draw something")
        finally:
            await cfg.aclose()

        assert reply.body == ""
        assert len(reply.response.files) == 1
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    async def test_text_turn_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        cfg = fake_acp_config(_hi_turn(), permission_policy="auto")
        agent = Agent("acp", config=cfg)

        try:
            with caplog.at_level(logging.WARNING, logger="ag2.acp.client"):
                reply = await agent.ask("hello")
        finally:
            await cfg.aclose()

        assert reply.body == "hi"
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []

    async def test_timed_out_turn_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        """A timeout already reports itself through ``finish_reason``."""
        cfg = fake_acp_config(ACPTurn(hang=True), permission_policy="auto", turn_timeout=0.5)
        agent = Agent("acp", config=cfg)

        try:
            with caplog.at_level(logging.WARNING, logger="ag2.acp.client"):
                reply = await agent.ask("hang")
        finally:
            await cfg.aclose()

        assert reply.body == ""
        assert [r for r in caplog.records if r.levelno == logging.WARNING] == []


@pytest.mark.asyncio
async def test_aclose_closes_session() -> None:
    cfg = fake_acp_config(
        ACPTurn(updates=[schema.AgentMessageChunk(session_update="agent_message_chunk", content=_text("hi"))]),
        permission_policy="auto",
    )
    agent = Agent("acp", config=cfg)

    async with agent.run("hello") as run:
        await run.result()

    assert cfg.sessions  # a live session was created
    conns = [s.conn for s in cfg.sessions.values()]
    await cfg.aclose()
    assert cfg.sessions == {}
    for conn in conns:
        assert conn is not None and conn.closed  # the connection context was exited


@pytest.mark.asyncio
async def test_config_as_async_context_manager_closes_sessions() -> None:
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    cfg = fake_acp_config(ACPTurn(updates=[_text_update("hi")]), permission_policy="auto")

    async with cfg as entered:
        assert entered is cfg
        agent = Agent("acp", config=cfg, tools=[add])
        async with agent.run("hello") as run:
            await run.result()
        assert cfg.sessions  # live while the config scope is open
        gateways = [s.gateway for s in cfg.sessions.values() if s.gateway is not None]
        assert gateways and all(g.url is not None for g in gateways)

    assert cfg.sessions == {}  # exiting the scope tore everything down
    assert all(g.url is None for g in gateways)  # and stopped the tool gateways


@pytest.mark.asyncio
async def test_external_server_named_like_the_gateway_is_rejected() -> None:
    """Two identically-named mcp_servers entries would silently shadow each other."""

    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    cfg = fake_acp_config(ACPTurn(updates=[_text_update("hi")]), permission_policy="auto")
    agent = Agent(
        "acp",
        config=cfg,
        tools=[add, MCPServerTool(server_url="https://example.com/mcp", server_label="ag2")],
    )
    try:
        with pytest.raises(ValueError, match="collides"):
            async with agent.run("hello") as run:
                await run.result()
        assert cfg.sessions == {}  # nothing leaked on the failure path
    finally:
        await cfg.aclose()


@pytest.mark.asyncio
async def test_external_server_named_like_the_gateway_is_fine_without_function_tools() -> None:
    """No function tools means no gateway, so there is nothing to collide with."""
    cfg = fake_acp_config(ACPTurn(updates=[_text_update("hi")]), permission_policy="auto")
    agent = Agent(
        "acp",
        config=cfg,
        tools=[MCPServerTool(server_url="https://example.com/mcp", server_label="ag2")],
    )
    try:
        async with agent.run("hello") as run:
            result = await run.result()
        assert result.body == "hi"
        session = next(iter(cfg.sessions.values()))
        assert session.gateway is None
        assert [s.name for s in session.conn.new_session_kwargs["mcp_servers"]] == ["ag2"]
    finally:
        await cfg.aclose()


@pytest.mark.asyncio
async def test_config_context_manager_closes_sessions_on_error() -> None:
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    cfg = fake_acp_config(ACPTurn(updates=[_text_update("hi")]), permission_policy="auto")

    with pytest.raises(RuntimeError, match="boom"):
        async with cfg:
            agent = Agent("acp", config=cfg, tools=[add])
            async with agent.run("hello") as run:
                await run.result()
            assert cfg.sessions
            raise RuntimeError("boom")

    assert cfg.sessions == {}  # torn down even though the block raised


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
        session = next(iter(cfg.sessions.values()))
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
        session = next(iter(cfg.sessions.values()))
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
        assert cfg.sessions == {}  # nothing leaked on the failure path
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
        session = next(iter(cfg.sessions.values()))
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
        session = next(iter(cfg.sessions.values()))
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
        assert cfg.sessions == {}  # the started gateway did not leak a session
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
            session = next(iter(cfg.sessions.values()))
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
    session = next(iter(cfg.sessions.values()))
    gateway = session.gateway
    assert gateway is not None and gateway.url is not None
    await cfg.aclose()
    assert gateway.url is None  # closed together with the session
