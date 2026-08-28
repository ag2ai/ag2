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
from mcp.server.streamable_http import CONTENT_TYPE_JSON, CONTENT_TYPE_SSE, MCP_SESSION_ID_HEADER
from mcp.shared.inbound import MCP_PROTOCOL_VERSION_HEADER
from mcp.types import CallToolResult
from mcp_types.version import LATEST_HANDSHAKE_VERSION

from ag2.acp.bridge import BridgeState
from ag2.acp.config import ACPConfig
from ag2.acp.tool_gateway import (
    HUMAN_INPUT_GATEWAY_TOOL_ERROR,
    GatewayAddress,
    MCPCapabilityError,
    ToolGateway,
    partition_tools,
)
from ag2.events import BinaryInput, ClientToolCallEvent, ToolErrorEvent, ToolResultEvent
from ag2.events.tool_events import ToolResult
from ag2.exceptions import HumanInputFailedError, HumanInputNotProvidedError, UnsupportedToolError
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


_MCP_HEADERS = {"Accept": f"{CONTENT_TYPE_JSON}, {CONTENT_TYPE_SSE}", "Content-Type": CONTENT_TYPE_JSON}
_INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": LATEST_HANDSHAKE_VERSION,
        "capabilities": {},
        "clientInfo": {"name": "t", "version": "1"},
    },
}
_CALL_ADD = {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {"name": "add", "arguments": {"a": 2, "b": 3}},
}


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
        assert url.startswith("http://127.0.0.1:") and "/mcp/" in url
        assert gateway.as_acp_server().url == url
        async with streamable_http_client(url) as (read, write), ClientSession(read, write) as session:
            await session.initialize()
            listed = await session.list_tools()
        (tool,) = listed.tools
        assert tool.name == "add"
        assert tool.description == "Add two integers"
        assert tool.input_schema["required"] == ["a", "b"]
    finally:
        await gateway.close()


@pytest.mark.asyncio
async def test_gateway_issues_no_session_and_still_serves_a_call() -> None:
    """The gateway is per-turn and per-request, so it hands out no MCP session.

    A session would be pure overhead on a server created for one turn: nothing
    outlives a request, so there is no cross-call state for a session id to name.
    The tool call in the same test keeps the header assertion honest — a server
    that answered nothing would issue no session id either.
    """
    state = BridgeState(ACPConfig())
    state.context = _FakeContext(lambda call: ToolResultEvent.from_call(call, "sum is 5"))
    gateway = ToolGateway(state, [_fn_add()])
    url = await gateway.start()
    try:
        async with httpx.AsyncClient() as client:
            initialize = await client.post(f"{url}/", headers=_MCP_HEADERS, json=_INITIALIZE)
            called = await client.post(
                f"{url}/",
                headers={**_MCP_HEADERS, MCP_PROTOCOL_VERSION_HEADER: LATEST_HANDSHAKE_VERSION},
                json=_CALL_ADD,
            )
    finally:
        await gateway.close()

    assert initialize.status_code == 200
    assert MCP_SESSION_ID_HEADER not in initialize.headers
    assert called.status_code == 200
    assert MCP_SESSION_ID_HEADER not in called.headers
    assert called.json()["result"]["content"] == [{"type": "text", "text": "sum is 5"}]


@pytest.mark.asyncio
class TestGatewayAddress:
    """The gateway's binding, and what it takes to widen it.

    The loopback default, the DNS-rebinding protection and the host allowlist
    are one deliberate arrangement; a caller who advertises an address for a
    remote agent loosens all three at once, so what exactly moves — and what
    does not — is worth pinning down.
    """

    async def test_default_binds_loopback_only(self) -> None:
        gateway = ToolGateway(BridgeState(ACPConfig()), [_fn_add()])
        url = await gateway.start()
        try:
            assert gateway.address == GatewayAddress()
            assert url.startswith("http://127.0.0.1:")
        finally:
            await gateway.close()

    async def test_default_rejects_a_forged_host_header(self) -> None:
        gateway = ToolGateway(BridgeState(ACPConfig()), [_fn_add()])
        url = await gateway.start()
        try:
            async with httpx.AsyncClient() as client:
                # Trailing slash: Starlette would otherwise answer the Mount's
                # redirect before the transport security check ever runs.
                response = await client.post(f"{url}/", json={}, headers={"Host": "attacker.example"})
        finally:
            await gateway.close()
        assert response.status_code >= 400  # DNS-rebinding protection is on

    async def test_advertised_address_binds_there_and_serves_it(self, routable_host: str) -> None:
        host = routable_host
        gateway = ToolGateway(BridgeState(ACPConfig()), [_fn_add()], address=GatewayAddress(host=host))
        url = await gateway.start()
        try:
            assert url.startswith(f"http://{host}:")
            async with streamable_http_client(url) as (read, write), ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
            assert [t.name for t in listed.tools] == ["add"]
        finally:
            await gateway.close()

    async def test_advertised_address_still_rejects_a_forged_host_header(self, routable_host: str) -> None:
        """Widened to match the advertised address, and no further."""
        host = routable_host
        gateway = ToolGateway(BridgeState(ACPConfig()), [_fn_add()], address=GatewayAddress(host=host))
        url = await gateway.start()
        try:
            async with httpx.AsyncClient() as client:
                # Trailing slash: Starlette would otherwise answer the Mount's
                # redirect before the transport security check ever runs.
                response = await client.post(f"{url}/", json={}, headers={"Host": "attacker.example"})
        finally:
            await gateway.close()
        assert response.status_code >= 400

    async def test_ipv6_loopback_serves_the_url_it_hands_out(self) -> None:
        gateway = ToolGateway(BridgeState(ACPConfig()), [_fn_add()], address=GatewayAddress(host="::1"))
        try:
            url = await gateway.start()
        except OSError:  # pragma: no cover - a host without IPv6
            pytest.skip("no IPv6 loopback on this host")
        try:
            assert url.startswith("http://[::1]:")
            async with streamable_http_client(url) as (read, write), ClientSession(read, write) as session:
                await session.initialize()
                listed = await session.list_tools()
            assert [t.name for t in listed.tools] == ["add"]
        finally:
            await gateway.close()

    async def test_advertised_port_is_the_port_it_binds(self, routable_host: str) -> None:
        host = routable_host
        with socket.socket() as probe:
            probe.bind((host, 0))
            port = probe.getsockname()[1]

        gateway = ToolGateway(BridgeState(ACPConfig()), [_fn_add()], address=GatewayAddress(host=host, port=port))
        url = await gateway.start()
        try:
            assert url.startswith(f"http://{host}:{port}/")
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
        # A turn that failed sends once more on its way out, to close the call
        # off in history; by then nothing is waiting on the stream any more.
        if self.stream.pending is None or self.stream.pending.done():
            return
        if self._respond is not None:
            self.stream.pending.set_result(self._respond(event))


@pytest.mark.asyncio
class TestCallTool:
    """``tools/call`` end to end: a real MCP HTTP client against the gateway."""

    async def _call(self, state: BridgeState, arguments: dict[str, int]) -> CallToolResult:
        gateway = ToolGateway(state, [_fn_add()])
        url = await gateway.start()
        try:
            async with streamable_http_client(url) as (read, write), ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool("add", arguments)
        finally:
            await gateway.close()

    async def test_executes_via_event_stream(self) -> None:
        state = BridgeState(ACPConfig())
        state.context = _FakeContext(lambda call: ToolResultEvent.from_call(call, "sum is 5"))

        result = await self._call(state, {"a": 2, "b": 3})

        assert result.is_error is not True
        assert result.content[0].text == "sum is 5"
        (call,) = state.context.sent
        assert call.name == "add"
        assert call.serialized_arguments == {"a": 2, "b": 3}

    async def test_maps_tool_error_to_is_error(self) -> None:
        state = BridgeState(ACPConfig())
        state.context = _FakeContext(lambda call: ToolErrorEvent.from_call(call, RuntimeError("boom")))

        result = await self._call(state, {"a": 1, "b": 1})

        assert result.is_error is True
        assert "boom" in result.content[0].text

    async def test_without_active_run_is_error(self) -> None:
        state = BridgeState(ACPConfig())  # state.context is None

        result = await self._call(state, {"a": 1, "b": 1})

        assert result.is_error is True  # the lowlevel server converts the raised RuntimeError
        assert "no active AG2 run" in result.content[0].text

    async def test_rejects_client_tool(self) -> None:
        state = BridgeState(ACPConfig())
        state.context = _FakeContext(lambda call: ClientToolCallEvent.from_call(call))

        result = await self._call(state, {"a": 1, "b": 1})

        assert result.is_error is True
        assert "client-side execution" in result.content[0].text

    async def test_serializes_data_result_as_json(self) -> None:
        state = BridgeState(ACPConfig())
        state.context = _FakeContext(lambda call: ToolResultEvent.from_call(call, {"sum": 5}))

        result = await self._call(state, {"a": 2, "b": 3})

        assert result.is_error is not True
        assert result.content[0].text == '{"sum": 5}'

    async def test_maps_image_result_to_image_content(self) -> None:
        png = b"\x89PNG\r\n\x1a\nfake"
        state = BridgeState(ACPConfig())
        state.context = _FakeContext(
            lambda call: ToolResultEvent.from_call(call, ToolResult(BinaryInput(png, media_type="image/png")))
        )

        result = await self._call(state, {"a": 1, "b": 1})

        assert result.is_error is not True
        (block,) = result.content
        assert block.type == "image"
        assert block.mime_type == "image/png"
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
async def test_guessable_path_does_not_reach_the_tools() -> None:
    """Knowing the port is not enough: the random path segment is the credential."""
    state = BridgeState(ACPConfig())
    gateway = ToolGateway(state, [_fn_add()])
    url = await gateway.start()
    try:
        port = url.removeprefix("http://127.0.0.1:").split("/")[0]
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://127.0.0.1:{port}/mcp/",  # what a port scan would try
                headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            )
    finally:
        await gateway.close()
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_each_gateway_gets_a_distinct_path() -> None:
    state = BridgeState(ACPConfig())
    first, second = ToolGateway(state, [_fn_add()]), ToolGateway(state, [_fn_add()])
    first_url = await first.start()
    try:
        second_url = await second.start()
        try:
            assert first_url != second_url
            # not merely a different port -- the secret differs too
            assert first_url.split("/mcp/")[1] != second_url.split("/mcp/")[1]
        finally:
            await second.close()
    finally:
        await first.close()


@pytest.mark.asyncio
async def test_close_is_bounded_with_a_stuck_call_in_flight() -> None:
    state = BridgeState(ACPConfig())
    state.context = _FakeContext(respond=None)  # the tool call never completes
    gateway = ToolGateway(state, [_fn_add()], close_timeout=0.5)
    url = await gateway.start()

    async def stuck_call() -> None:
        async with streamable_http_client(url) as (read, write), ClientSession(read, write) as session:
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


def _raise(error: BaseException):
    """A ``_FakeContext`` responder that fails instead of answering.

    ``send`` calls the responder, so raising here is what a tool raising out of
    the stream looks like from the gateway's side.
    """

    def respond(_call):
        raise error

    return respond


@pytest.mark.asyncio
class TestAHumanInputFailureIsNotToolOutput:
    """A tool that asked a person and got nowhere has not produced a tool result.

    Caught as an ordinary exception here it becomes the CLI agent's tool output,
    and the agent — told only that a tool broke — is free to look for another way
    to do what an approval refused it. The gateway cannot raise into the task
    awaiting ``session/prompt``, so it records the failure on the run's state
    instead; the client raises it from there.
    """

    async def _call(self, state: BridgeState) -> CallToolResult:
        gateway = ToolGateway(state, [_fn_add()])
        url = await gateway.start()
        try:
            async with streamable_http_client(url) as (read, write), ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool("add", {"a": 1, "b": 1})
        finally:
            await gateway.close()

    async def test_the_failure_is_recorded_and_the_turn_is_stopped(self) -> None:
        state = BridgeState(ACPConfig())
        error = HumanInputNotProvidedError()
        state.context = _FakeContext(_raise(error))

        result = await self._call(state)

        assert state.channel_failure is error
        assert state.channel_failed.is_set()
        assert result.is_error is True

    async def test_ag2s_own_advice_does_not_become_the_agents_tool_output(self) -> None:
        """The request still has to be answered, but not with the host's internals.

        ``HumanInputNotProvidedError`` reads "pass hitl_hook=..." — advice for
        whoever wired this agent up, landing in someone else's conversation as
        though the tool had said it.
        """
        state = BridgeState(ACPConfig())
        state.context = _FakeContext(_raise(HumanInputNotProvidedError()))

        result = await self._call(state)

        text = result.content[0].text
        assert text == HUMAN_INPUT_GATEWAY_TOOL_ERROR
        assert "hitl_hook" not in text

    async def test_the_first_failure_is_the_one_reported(self) -> None:
        """Later calls in the same turn are being cancelled, not diagnosing anything."""
        state = BridgeState(ACPConfig())
        first = HumanInputNotProvidedError()
        state.context = _FakeContext(_raise(first))
        await self._call(state)

        state.context = _FakeContext(_raise(HumanInputFailedError(RuntimeError("queue down"))))
        await self._call(state)

        assert state.channel_failure is first

    async def test_an_ordinary_tool_failure_still_reads_as_one(self) -> None:
        state = BridgeState(ACPConfig())
        state.context = _FakeContext(_raise(RuntimeError("boom")))

        result = await self._call(state)

        assert result.is_error is True
        assert "boom" in result.content[0].text
        assert state.channel_failure is None
        assert not state.channel_failed.is_set()
