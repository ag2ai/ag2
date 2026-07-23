# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import signal
import socket
from contextlib import asynccontextmanager, suppress
from typing import Any

import httpx
import pytest
from acp import schema
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from ag2.acp.bridge import BridgeState
from ag2.acp.config import ACPConfig
from ag2.acp.tool_gateway import MCPCapabilityError, ToolGateway, partition_tools
from ag2.events import BinaryInput, ClientToolCallEvent, ToolErrorEvent, ToolResultEvent
from ag2.events.tool_events import ToolResult
from ag2.exceptions import UnsupportedToolError
from ag2.tools.builtin.mcp_server import MCPServerToolSchema
from ag2.tools.builtin.web_search import WebSearchToolSchema
from ag2.tools.final import FunctionToolSchema
from ag2.tools.final.function_tool import FunctionDefinition


def _fn(name: str) -> FunctionToolSchema:
    return FunctionToolSchema(
        function=FunctionDefinition(name=name, description="d", parameters={"type": "object", "properties": {}})
    )


def test_partition_keeps_function_tools() -> None:
    functions, external = partition_tools([_fn("a"), _fn("b")])
    assert [f.function.name for f in functions] == ["a", "b"]
    assert external == []


def test_partition_translates_mcp_server_tool() -> None:
    tool = MCPServerToolSchema(
        server_url="https://mcp.example.com/mcp",
        server_label="ext",
        authorization_token="tok123",
        headers={"X-Env": "prod"},
    )
    functions, external = partition_tools([tool])
    assert functions == []
    (server,) = external
    assert isinstance(server, schema.HttpMcpServer)
    assert server.name == "ext"
    assert server.url == "https://mcp.example.com/mcp"
    header_map = {h.name: h.value for h in server.headers}
    assert header_map["X-Env"] == "prod"
    assert header_map["Authorization"] == "Bearer tok123"


@pytest.mark.parametrize(
    "filters",
    [
        {"allowed_tools": ["a"]},
        {"blocked_tools": ["b"]},
        {"allowed_tools": []},  # empty list is a filter ("allow nothing"), not an absent one
    ],
)
def test_partition_rejects_mcp_server_tool_filters(filters: dict[str, Any]) -> None:
    tool = MCPServerToolSchema(server_url="https://x/mcp", server_label="ext", **filters)
    with pytest.raises(ValueError, match="allowed_tools"):
        partition_tools([tool])


def test_partition_rejects_provider_builtin() -> None:
    with pytest.raises(UnsupportedToolError, match="web_search"):
        partition_tools([_fn("a"), WebSearchToolSchema()])


def test_capability_error_message_names_agent() -> None:
    err = MCPCapabilityError("codex-acp")
    assert "codex-acp" in str(err)
    assert "expose_tools" in str(err)


def _fn_add() -> FunctionToolSchema:
    return FunctionToolSchema(
        function=FunctionDefinition(
            name="add",
            description="Add two integers",
            parameters={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
        )
    )


@pytest.mark.asyncio
async def test_gateway_serves_tools_list_over_http() -> None:
    state = BridgeState(ACPConfig())
    gateway = ToolGateway(state, [_fn_add()])
    url = await gateway.start()
    try:
        assert url.startswith("http://127.0.0.1:") and url.endswith("/mcp")
        assert gateway.as_acp_server().url == url
        async with streamable_http_client(url) as (read, write, _), ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
        (tool,) = listed.tools
        assert tool.name == "add"
        assert tool.description == "Add two integers"
        assert tool.inputSchema["required"] == ["a", "b"]
    finally:
        await gateway.close()


@pytest.mark.asyncio
async def test_gateway_close_is_idempotent_and_frees_port() -> None:
    state = BridgeState(ACPConfig())
    gateway = ToolGateway(state, [_fn_add()])
    url = await gateway.start()
    port = int(url.removeprefix("http://127.0.0.1:").split("/")[0])
    await gateway.close()
    await gateway.close()  # idempotent
    assert gateway.url is None
    # the port no longer accepts connections at all
    with pytest.raises(ConnectionRefusedError), socket.socket() as probe:
        probe.connect(("127.0.0.1", port))


class _FakeStream:
    def __init__(self) -> None:
        self.pending: asyncio.Future | None = None

    def get(self, _expr):
        stream = self

        @asynccontextmanager
        async def cm():
            fut: asyncio.Future = asyncio.get_running_loop().create_future()
            stream.pending = fut
            try:
                yield fut
            finally:
                stream.pending = None

        return cm()


class _FakeContext:
    """Stands in for ConversationContext: send() answers the pending stream.get().

    ``respond=None`` models a tool call that never completes (no subscriber
    answers the event) — used to exercise bounded shutdown.
    """

    def __init__(self, respond=None) -> None:
        self.stream = _FakeStream()
        self.sent: list = []
        self.first_send = asyncio.Event()
        self._respond = respond

    async def send(self, event) -> None:
        self.sent.append(event)
        self.first_send.set()
        assert self.stream.pending is not None
        if self._respond is not None:
            self.stream.pending.set_result(self._respond(event))


@pytest.mark.asyncio
async def test_call_tool_executes_via_event_stream() -> None:
    state = BridgeState(ACPConfig())
    state.context = _FakeContext(lambda call: ToolResultEvent.from_call(call, "sum is 5"))
    gateway = ToolGateway(state, [_fn_add()])
    url = await gateway.start()
    try:
        async with streamable_http_client(url) as (read, write, _), ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("add", {"a": 2, "b": 3})
    finally:
        await gateway.close()

    assert result.isError is not True
    assert result.content[0].text == "sum is 5"
    (call,) = state.context.sent
    assert call.name == "add"
    assert call.serialized_arguments == {"a": 2, "b": 3}


@pytest.mark.asyncio
async def test_call_tool_maps_tool_error_to_is_error() -> None:
    state = BridgeState(ACPConfig())
    state.context = _FakeContext(lambda call: ToolErrorEvent.from_call(call, RuntimeError("boom")))
    gateway = ToolGateway(state, [_fn_add()])
    url = await gateway.start()
    try:
        async with streamable_http_client(url) as (read, write, _), ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("add", {"a": 1, "b": 1})
    finally:
        await gateway.close()

    assert result.isError is True
    assert "boom" in result.content[0].text


@pytest.mark.asyncio
async def test_call_tool_without_active_run_is_error() -> None:
    state = BridgeState(ACPConfig())  # state.context is None
    gateway = ToolGateway(state, [_fn_add()])
    url = await gateway.start()
    try:
        async with streamable_http_client(url) as (read, write, _), ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("add", {"a": 1, "b": 1})
    finally:
        await gateway.close()
    assert result.isError is True  # the lowlevel server converts the raised RuntimeError
    assert "no active AG2 run" in result.content[0].text


@pytest.mark.asyncio
async def test_call_tool_rejects_client_tool() -> None:
    state = BridgeState(ACPConfig())
    state.context = _FakeContext(lambda call: ClientToolCallEvent.from_call(call))
    gateway = ToolGateway(state, [_fn_add()])
    url = await gateway.start()
    try:
        async with streamable_http_client(url) as (read, write, _), ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("add", {"a": 1, "b": 1})
    finally:
        await gateway.close()
    assert result.isError is True
    assert "client-side execution" in result.content[0].text


@pytest.mark.asyncio
async def test_call_tool_serializes_data_result_as_json() -> None:
    state = BridgeState(ACPConfig())
    state.context = _FakeContext(lambda call: ToolResultEvent.from_call(call, {"sum": 5}))
    gateway = ToolGateway(state, [_fn_add()])
    url = await gateway.start()
    try:
        async with streamable_http_client(url) as (read, write, _), ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("add", {"a": 2, "b": 3})
    finally:
        await gateway.close()
    assert result.isError is not True
    assert result.content[0].text == '{"sum": 5}'


@pytest.mark.asyncio
async def test_call_tool_maps_image_result_to_image_content() -> None:
    png = b"\x89PNG\r\n\x1a\nfake"
    state = BridgeState(ACPConfig())
    state.context = _FakeContext(
        lambda call: ToolResultEvent.from_call(call, ToolResult(BinaryInput(png, media_type="image/png")))
    )
    gateway = ToolGateway(state, [_fn_add()])
    url = await gateway.start()
    try:
        async with streamable_http_client(url) as (read, write, _), ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("add", {"a": 1, "b": 1})
    finally:
        await gateway.close()
    assert result.isError is not True
    (block,) = result.content
    assert block.type == "image"
    assert block.mimeType == "image/png"
    assert base64.b64decode(block.data) == png


@pytest.mark.asyncio
async def test_start_and_close_leave_signal_handlers_untouched() -> None:
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    state = BridgeState(ACPConfig())
    gateway = ToolGateway(state, [_fn_add()])
    await gateway.start()
    try:
        # uvicorn's stock serve() would have swapped these for its own handler
        assert {sig: signal.getsignal(sig) for sig in before} == before
    finally:
        await gateway.close()
    assert {sig: signal.getsignal(sig) for sig in before} == before


@pytest.mark.asyncio
async def test_request_with_foreign_host_header_is_rejected() -> None:
    state = BridgeState(ACPConfig())
    gateway = ToolGateway(state, [_fn_add()])
    url = await gateway.start()
    try:
        async with httpx.AsyncClient() as client:
            # url + "/" skips Starlette's /mcp -> /mcp/ redirect, which fires
            # before the transport security check.
            response = await client.post(
                f"{url}/",
                headers={
                    "Host": "evil.example.com:9999",
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
                json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
            )
    finally:
        await gateway.close()
    assert response.status_code == 421  # DNS-rebinding protection: Host not in the allowlist


@pytest.mark.asyncio
async def test_close_is_bounded_with_a_stuck_call_in_flight() -> None:
    state = BridgeState(ACPConfig())
    state.context = _FakeContext(respond=None)  # the tool call never completes
    gateway = ToolGateway(state, [_fn_add()], close_timeout=0.5)
    url = await gateway.start()

    async def stuck_call() -> None:
        async with streamable_http_client(url) as (read, write, _), ClientSession(read, write) as session:
            await session.initialize()
            await session.call_tool("add", {"a": 1, "b": 1})

    task = asyncio.ensure_future(stuck_call())
    # wait until the call is in flight inside the gateway
    await asyncio.wait_for(state.context.first_send.wait(), timeout=5)

    # Without bounded shutdown this would wait forever on the in-flight request.
    await asyncio.wait_for(gateway.close(), timeout=10)

    task.cancel()
    with suppress(BaseException):
        await task
