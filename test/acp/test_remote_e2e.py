# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""A remote ACP agent, driven through the ordinary Ask and Run surface.

Every test here scripts a turn against the same in-process agent the local tests
use and asserts on the Reply, the events, and the errors — never on which
transport is underneath. That is the design's whole claim: above the connection
hook, a remote agent is indistinguishable from a launched one. No test opens a
network connection.
"""

import pytest

pytest.importorskip("websockets")
pytest.importorskip("h2")

import asyncio
import contextlib
import socket
from pathlib import Path
from typing import Any

from acp import schema
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from ag2 import Agent
from ag2.acp import ACPRemoteConfig, ACPTransportError, MCPCapabilityError
from ag2.acp.testing import ACPTurn, fake_acp_config, fake_remote_acp_config
from ag2.events import BaseEvent, ModelReasoning
from ag2.events.tool_events import BuiltinToolCallEvent
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


def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@pytest.mark.asyncio
async def test_ask_against_a_remote_agent_returns_a_reply() -> None:
    cfg = fake_remote_acp_config(
        ACPTurn(
            updates=[
                schema.AgentThoughtChunk(session_update="agent_thought_chunk", content=_text("planning")),
                _text_update("done"),
                schema.ToolCallStart(session_update="tool_call", tool_call_id="t1", title="Echo", status="pending"),
            ],
            usage=schema.Usage(input_tokens=3, output_tokens=1, total_tokens=4),
        ),
        url="https://box.internal/acp",
        permission_policy="auto",
        expose_tools=False,
    )
    agent = Agent("acp", config=cfg)

    seen: list[BaseEvent] = []
    try:
        async with agent.run("hello") as run:
            run.stream.subscribe(lambda e: seen.append(e))
            reply = await run.result()
    finally:
        await cfg.aclose()

    assert reply.body == "done"
    assert any(isinstance(e, ModelReasoning) and e.content == "planning" for e in seen)
    assert any(isinstance(e, BuiltinToolCallEvent) and e.name == "Echo" for e in seen)


@pytest.mark.asyncio
async def test_run_against_a_websocket_agent_returns_a_reply() -> None:
    """A ws:// URL is the only difference from the HTTP case, and changes nothing."""
    cfg = fake_remote_acp_config(
        ACPTurn(updates=[_text_update("done")]),
        url="wss://box.internal/acp",
        permission_policy="auto",
        expose_tools=False,
    )
    try:
        assert (await Agent("acp", config=cfg).ask("hello")).body == "done"
    finally:
        await cfg.aclose()


@pytest.mark.asyncio
async def test_a_second_turn_reuses_the_remote_session() -> None:
    cfg = fake_remote_acp_config(
        ACPTurn(updates=[_text_update("one")]),
        ACPTurn(updates=[_text_update("two")]),
        permission_policy="auto",
        expose_tools=False,
    )
    try:
        reply = await Agent("acp", config=cfg).ask("first")
        assert reply.body == "one"
        assert (await reply.ask("second")).body == "two"
        assert len(cfg.sessions) == 1
    finally:
        await cfg.aclose()


@pytest.mark.asyncio
async def test_model_selection_behaves_as_it_does_locally() -> None:
    calls: list[tuple[str, str | bool]] = []
    cfg = fake_remote_acp_config(
        ACPTurn(updates=[_text_update("hi")]),
        config_options=[_model_option("sonnet", "sonnet", "opus")],
        config_option_calls=calls,
        model="opus",
        permission_policy="auto",
        expose_tools=False,
    )
    try:
        reply = await Agent("acp", config=cfg).ask("hello")
    finally:
        await cfg.aclose()

    assert calls == [("model", "opus")]
    assert reply.body == "hi"


@pytest.mark.asyncio
async def test_a_model_the_remote_agent_does_not_offer_is_rejected() -> None:
    cfg = fake_remote_acp_config(
        ACPTurn(updates=[_text_update("hi")]),
        config_options=[_model_option("sonnet", "sonnet", "opus")],
        model="haiku",
        permission_policy="auto",
        expose_tools=False,
    )
    try:
        with pytest.raises(ValueError, match="is not offered by the ACP agent"):
            await Agent("acp", config=cfg).ask("hello")
        assert cfg.sessions == {}
    finally:
        await cfg.aclose()


@pytest.mark.asyncio
async def test_filesystem_and_terminal_capabilities_are_advertised() -> None:
    advertised: list[schema.ClientCapabilities | None] = []
    cfg = fake_remote_acp_config(
        ACPTurn(updates=[_text_update("hi")]),
        initialize_calls=advertised,
        permission_policy="auto",
        expose_tools=False,
    )
    try:
        await Agent("acp", config=cfg).ask("hello")
    finally:
        await cfg.aclose()

    (capabilities,) = advertised
    assert capabilities is not None
    assert capabilities.fs.read_text_file is True and capabilities.fs.write_text_file is True
    assert capabilities.terminal is True


@pytest.mark.asyncio
async def test_mediated_filesystem_access_stays_inside_fs_root(tmp_path: Path) -> None:
    """A remote agent reads and writes the *local* workspace, and no further."""
    (tmp_path / "inside.txt").write_text("workspace content", encoding="utf-8")
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    observed: dict[str, Any] = {}
    cfg: Any = None

    async def agent_reads_files() -> None:
        bridge = next(iter(cfg.sessions.values())).bridge
        observed["inside"] = (await bridge.read_text_file(path="inside.txt", session_id="s")).content
        await bridge.write_text_file(content="written by the agent", path="new.txt", session_id="s")
        try:
            await bridge.read_text_file(path=str(outside), session_id="s")
        except PermissionError as e:
            observed["escape"] = str(e)

    cfg = fake_remote_acp_config(
        ACPTurn(updates=[_text_update("hi")], on_prompt=agent_reads_files),
        cwd=str(tmp_path),
        permission_policy="auto",
        expose_tools=False,
    )
    try:
        await Agent("acp", config=cfg).ask("hello")
    finally:
        await cfg.aclose()

    assert observed["inside"] == "workspace content"
    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "written by the agent"
    assert "escapes fs_root" in observed["escape"]


@pytest.mark.asyncio
class TestToolExposure:
    async def test_no_gateway_address_refuses_at_session_start(self) -> None:
        cfg = fake_remote_acp_config(
            ACPTurn(updates=[_text_update("hi")]),
            url="https://box.internal/acp",
            permission_policy="auto",
        )
        agent = Agent("acp", config=cfg, tools=[add])
        try:
            with pytest.raises(MCPCapabilityError) as raised:
                await agent.ask("hello")
            assert cfg.sessions == {}  # nothing was started behind the refusal
        finally:
            await cfg.aclose()

        message = str(raised.value)
        assert "loopback only" in message  # what is wrong
        assert "gateway_address=" in message and "expose_tools=False" in message  # what to do about it

    async def test_no_loopback_url_is_ever_handed_to_a_remote_agent(self) -> None:
        """The refusal lands before the agent is contacted, so no URL can leak."""
        reached_the_agent: list[Any] = []
        cfg = fake_remote_acp_config(
            ACPTurn(updates=[_text_update("hi")]),
            initialize_calls=reached_the_agent,
            permission_policy="auto",
        )
        agent = Agent("acp", config=cfg, tools=[add])
        try:
            with pytest.raises(MCPCapabilityError):
                await agent.ask("hello")
            assert reached_the_agent == []  # no initialize, so no session/new, so no mcp_servers
        finally:
            await cfg.aclose()

    async def test_tool_exposure_off_needs_no_extra_configuration(self) -> None:
        cfg = fake_remote_acp_config(
            ACPTurn(updates=[_text_update("hi")]),
            permission_policy="auto",
            expose_tools=False,
        )
        try:
            assert (await Agent("acp", config=cfg, tools=[add]).ask("hello")).body == "hi"
            assert next(iter(cfg.sessions.values())).gateway is None
        finally:
            await cfg.aclose()

    async def test_an_advertised_address_lets_the_agent_call_the_tools(self, routable_host: str) -> None:
        observed: dict[str, Any] = {}
        cfg: Any = None

        async def agent_calls_the_gateway() -> None:
            session = next(iter(cfg.sessions.values()))
            (advertised,) = session.conn.new_session_kwargs["mcp_servers"]
            observed["url"] = advertised.url
            async with (
                streamable_http_client(advertised.url) as (read, write),
                ClientSession(read, write) as mcp,
            ):
                await mcp.initialize()
                observed["tools"] = sorted(t.name for t in (await mcp.list_tools()).tools)
                observed["result"] = (await mcp.call_tool("add", {"a": 2, "b": 3})).content[0].text

        host = routable_host
        cfg = fake_remote_acp_config(
            ACPTurn(updates=[_text_update("hi")], on_prompt=agent_calls_the_gateway),
            gateway_address=host,
            permission_policy="auto",
        )
        try:
            await Agent("acp", config=cfg, tools=[add]).ask("hello")
        finally:
            await cfg.aclose()

        # Not a loopback URL: the agent was handed the address it was told to dial.
        assert observed["url"].startswith(f"http://{host}:")
        assert observed["tools"] == ["add"]
        assert observed["result"] == "5"

    async def test_an_advertised_port_is_the_port_the_gateway_binds(self, routable_host: str) -> None:
        host = routable_host
        with contextlib.closing(socket.socket()) as probe:
            probe.bind((host, 0))
            port = probe.getsockname()[1]

        cfg = fake_remote_acp_config(
            ACPTurn(updates=[_text_update("hi")]),
            gateway_address=f"{host}:{port}",
            permission_policy="auto",
        )
        try:
            await Agent("acp", config=cfg, tools=[add]).ask("hello")
            gateway = next(iter(cfg.sessions.values())).gateway
            assert gateway.url.startswith(f"http://{host}:{port}/")
        finally:
            await cfg.aclose()

    async def test_tools_added_on_a_later_turn_hot_update_the_gateway(self, routable_host: str) -> None:
        def mul(a: int, b: int) -> int:
            """Multiply two integers."""
            return a * b

        observed: dict[str, Any] = {}
        cfg: Any = None

        def snapshot(key: str) -> Any:
            async def probe() -> None:
                session = next(iter(cfg.sessions.values()))
                async with (
                    streamable_http_client(session.gateway.url) as (read, write),
                    ClientSession(read, write) as mcp,
                ):
                    await mcp.initialize()
                    observed[key] = sorted(t.name for t in (await mcp.list_tools()).tools)

            return probe

        cfg = fake_remote_acp_config(
            ACPTurn(updates=[_text_update("one")], on_prompt=snapshot("turn1")),
            ACPTurn(updates=[_text_update("two")], on_prompt=snapshot("turn2")),
            gateway_address=routable_host,
            permission_policy="auto",
        )
        try:
            reply = await Agent("acp", config=cfg, tools=[add]).ask("first")
            await reply.ask("second", tools=[FunctionTool.ensure_tool(mul)])
        finally:
            await cfg.aclose()

        assert observed == {"turn1": ["add"], "turn2": ["add", "mul"]}

    async def test_a_local_config_keeps_its_loopback_gateway(self) -> None:
        cfg = fake_acp_config(ACPTurn(updates=[_text_update("hi")]), permission_policy="auto")
        try:
            await Agent("acp", config=cfg, tools=[add]).ask("hello")
            gateway = next(iter(cfg.sessions.values())).gateway
            assert gateway is not None
            assert gateway.address.is_loopback
            assert gateway.url.startswith("http://127.0.0.1:")
        finally:
            await cfg.aclose()


@pytest.mark.asyncio
class TestLifecycle:
    async def test_turn_timeout_cancels_over_a_remote_connection(self) -> None:
        cfg = fake_remote_acp_config(
            ACPTurn(hang=True),
            permission_policy="auto",
            expose_tools=False,
            turn_timeout=0.2,
        )
        try:
            async with Agent("acp", config=cfg).run("hang") as run:
                reply = await run.result()
            assert reply.body == ""
            assert cfg.sessions  # the cancel was honoured, so the session survives
        finally:
            await cfg.aclose()

    async def test_an_unresponsive_cancel_closes_the_connection(self) -> None:
        """No process to kill: the hard stop is the connection going away."""
        opened: list[Any] = []
        cfg: Any = None

        async def ignore_the_cancel() -> None:
            session = next(iter(cfg.sessions.values()))
            opened.append(session.conn)
            await asyncio.Event().wait()  # never returns, and never honours session/cancel

        cfg = fake_remote_acp_config(
            ACPTurn(on_prompt=ignore_the_cancel),
            permission_policy="auto",
            expose_tools=False,
            turn_timeout=0.2,
            cancel_timeout=0.2,
        )
        try:
            reply = await Agent("acp", config=cfg).ask("hang")
            assert reply.body == ""
        finally:
            await cfg.aclose()

        (conn,) = opened
        assert conn.closed  # the connection hook was exited — no process was involved

    @pytest.mark.parametrize(
        ("url", "override", "transport"),
        [
            ("http://box.internal/acp", None, "http"),
            ("https://box.internal/acp", None, "http"),
            ("HTTPS://box.internal/acp", None, "http"),
            ("ws://box.internal/acp", None, "websocket"),
            ("wss://box.internal/acp", None, "websocket"),
            # An override is for a proxy or gateway whose scheme says nothing
            # about what it speaks; it wins over the scheme either way.
            ("wss://box.internal/acp", "http", "http"),
            ("https://box.internal/acp", "websocket", "websocket"),
        ],
        ids=["http", "https", "uppercase", "ws", "wss", "override-http", "override-websocket"],
    )
    async def test_a_transport_failure_names_the_transport_the_url_chose(
        self, url: str, override: str | None, transport: str
    ) -> None:
        # The error is where the chosen transport becomes observable, and it is
        # raised rather than answered with an empty reply: a blank body cannot be
        # told apart from an agent that genuinely had nothing to say.
        def drop_the_connection() -> Any:
            raise ConnectionError("Connection closed")

        cfg = fake_remote_acp_config(
            ACPTurn(on_prompt=drop_the_connection),  # type: ignore[arg-type]
            url=url,
            transport=override,
            permission_policy="auto",
            expose_tools=False,
        )
        try:
            with pytest.raises(ACPTransportError) as raised:
                await Agent("acp", config=cfg).ask("hello")
        finally:
            await cfg.aclose()

        assert raised.value.transport == transport
        assert transport in str(raised.value)

    async def test_a_transport_failure_takes_the_session_with_it(self) -> None:
        """No reconnect and no resume: the dead connection is not left to be reused."""
        opened: list[Any] = []
        cfg: Any = None

        async def drop_the_connection() -> None:
            opened.append(next(iter(cfg.sessions.values())).conn)
            raise ConnectionError("Connection closed")

        cfg = fake_remote_acp_config(
            ACPTurn(on_prompt=drop_the_connection),
            ACPTurn(updates=[_text_update("hi")]),
            permission_policy="auto",
            expose_tools=False,
        )
        try:
            with pytest.raises(ACPTransportError):
                await Agent("acp", config=cfg).ask("hello")
            assert cfg.sessions == {}  # nothing left holding an open client
        finally:
            await cfg.aclose()

        (conn,) = opened
        assert conn.closed

    async def test_teardown_closes_a_remote_session_with_no_process_handle(self) -> None:
        cfg = fake_remote_acp_config(
            ACPTurn(updates=[_text_update("hi")]),
            permission_policy="auto",
            expose_tools=False,
        )
        await Agent("acp", config=cfg).ask("hello")

        (session,) = cfg.sessions.values()
        assert session.proc is None  # nothing to kill
        conn = session.conn

        await cfg.aclose()

        assert cfg.sessions == {}
        assert conn.closed

    async def test_config_teardown_covers_every_session_it_started(self) -> None:
        cfg = fake_remote_acp_config(
            ACPTurn(updates=[_text_update("hi")]),
            ACPTurn(updates=[_text_update("hi")]),
            permission_policy="auto",
            expose_tools=False,
        )
        async with cfg:
            agent = Agent("acp", config=cfg)
            await agent.ask("first")
            await agent.ask("second")
            conns = [s.conn for s in cfg.sessions.values()]
            assert len(conns) == 2  # a separate ask() is a separate stream, so a separate session

        assert cfg.sessions == {}
        assert all(conn.closed for conn in conns)


@pytest.mark.asyncio
async def test_the_remote_config_is_a_drop_in_for_the_local_one() -> None:
    """Same script, same turns, same Reply — the transport is the only difference."""
    turns = (ACPTurn(updates=[_text_update("same answer")]),)
    local = fake_acp_config(*turns, permission_policy="auto", expose_tools=False)
    remote = fake_remote_acp_config(*turns, permission_policy="auto", expose_tools=False)
    try:
        local_reply = await Agent("acp", config=local).ask("hello")
        remote_reply = await Agent("acp", config=remote).ask("hello")
    finally:
        await local.aclose()
        await remote.aclose()

    assert local_reply.body == remote_reply.body == "same answer"
    assert isinstance(remote, ACPRemoteConfig)
