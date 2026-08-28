# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""What a caller observes of a served agent's conversation continuity.

Every assertion here is on what a client sees — the turns the agent was given —
never on how the key behind them was derived.
"""

from collections.abc import Iterable
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from dirty_equals import IsPartialDict, IsStr
from mcp.server.streamable_http import CONTENT_TYPE_JSON, CONTENT_TYPE_SSE, MCP_SESSION_ID_HEADER
from mcp.shared.inbound import MCP_METHOD_HEADER, MCP_NAME_HEADER, MCP_PROTOCOL_VERSION_HEADER
from mcp.types import CallToolResult, TextContent
from mcp.types import Tool as MCPTool
from mcp_types import CLIENT_CAPABILITIES_META_KEY, PROTOCOL_VERSION_META_KEY
from mcp_types.version import LATEST_HANDSHAKE_VERSION, LATEST_MODERN_VERSION
from pydantic import BaseModel

from ag2 import Agent, Context
from ag2.context import StreamId
from ag2.events import BaseEvent, ModelRequest, TextInput
from ag2.history import MemoryStorage
from ag2.mcp import MCPFunctionTool, MCPServer, SessionConfig
from ag2.mcp.security import AccessToken, oauth2_scheme, require
from ag2.mcp.sessions import CONVERSATION_META_KEY
from ag2.mcp.testing import connect, connect_modern, serve
from ag2.mcp.tools import ToolContext
from ag2.testing import TestConfig

from ._helpers import RecordingConfig

_JSON = {"Accept": f"{CONTENT_TYPE_JSON}, {CONTENT_TYPE_SSE}", "Content-Type": CONTENT_TYPE_JSON}


class Weather(BaseModel):
    city: str
    temp_c: float


def _agent(config: RecordingConfig) -> Agent:
    return Agent("greeter", config=config)


def _echo(arguments: dict[str, Any], _ctx: ToolContext) -> str:
    return str(arguments["text"])


class _RecordingStorage(MemoryStorage):
    """A ``Storage`` that also remembers which streams were written through it."""

    def __init__(self) -> None:
        super().__init__()
        self.stream_ids: list[StreamId] = []

    async def save_event(self, event: BaseEvent, context: Context) -> None:
        if context.stream.id not in self.stream_ids:
            self.stream_ids.append(context.stream.id)
        await super().save_event(event, context)


def _texts(events: Iterable[BaseEvent]) -> list[str]:
    """The user-side text of each turn recorded in ``events``."""
    return [
        part.content for e in events if isinstance(e, ModelRequest) for part in e.parts if isinstance(part, TextInput)
    ]


async def _modern_call(
    client: httpx.AsyncClient,
    message: str,
    *,
    request_id: int,
    conversation: str | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """POST one ``tools/call`` as a modern-era client and return its result."""
    result, _response = await _modern_call_with_response(
        client, message, request_id=request_id, conversation=conversation, token=token
    )
    return result


async def _modern_call_with_response(
    client: httpx.AsyncClient,
    message: str,
    *,
    request_id: int,
    conversation: str | None = None,
    token: str | None = None,
) -> tuple[dict[str, Any], httpx.Response]:
    """POST one ``tools/call`` as a modern-era client, keeping the HTTP response.

    No handshake: the version/capabilities envelope rides in ``params._meta``,
    and the revision requires the method and tool name in headers too. The
    response itself is returned for the assertions that are about HTTP.
    """
    arguments: dict[str, Any] = {"message": message}
    if conversation is not None:
        arguments["conversation"] = conversation
    headers = {
        **_JSON,
        MCP_PROTOCOL_VERSION_HEADER: LATEST_MODERN_VERSION,
        MCP_METHOD_HEADER: "tools/call",
        MCP_NAME_HEADER: "ask",
    }
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    response = await client.post(
        "/mcp",
        headers=headers,
        json={
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {
                "name": "ask",
                "arguments": arguments,
                "_meta": {
                    PROTOCOL_VERSION_META_KEY: LATEST_MODERN_VERSION,
                    CLIENT_CAPABILITIES_META_KEY: {},
                },
            },
        },
    )
    assert response.status_code == 200
    return response.json()["result"], response


async def _open_handshake_session(
    client: httpx.AsyncClient, *, request_id: int, token: str | None = None
) -> dict[str, str]:
    """Run the ``initialize`` handshake and return the headers its session needs."""
    opening = _JSON if token is None else {**_JSON, "Authorization": f"Bearer {token}"}
    response = await client.post(
        "/mcp",
        headers=opening,
        json={
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "initialize",
            "params": {
                "protocolVersion": LATEST_HANDSHAKE_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "t", "version": "1"},
            },
        },
    )
    assert response.status_code == 200
    session_id = response.headers[MCP_SESSION_ID_HEADER]
    headers = {**opening, MCP_PROTOCOL_VERSION_HEADER: LATEST_HANDSHAKE_VERSION, MCP_SESSION_ID_HEADER: session_id}
    await client.post("/mcp", headers=headers, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
    return headers


async def _handshake_call(
    client: httpx.AsyncClient, headers: dict[str, str], message: str, *, request_id: int
) -> dict[str, Any]:
    response = await client.post(
        "/mcp",
        headers=headers,
        json={
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {"name": "ask", "arguments": {"message": message}},
        },
    )
    assert response.status_code == 200
    return response.json()["result"]


@pytest.mark.asyncio
class TestModernEraStartsFresh:
    """2026-07-28: a connection is not a conversation, and neither is a process.

    The revision says servers must not use connection or process identity to
    establish context, and it issues nothing else to key on — so an unnamed
    conversation starts empty on every transport.
    """

    async def test_stream_calls_do_not_share_a_conversation(self) -> None:
        config = RecordingConfig(TestConfig("ok", "ok"))
        server = MCPServer(_agent(config))

        async with connect_modern(server) as session:
            await session.call_tool("ask", {"message": "first"})
            await session.call_tool("ask", {"message": "second"})

        assert config.prompts == [["first"], ["second"]]

    async def test_http_calls_do_not_share_a_conversation(self) -> None:
        config = RecordingConfig(TestConfig("ok", "ok"))
        app = MCPServer(_agent(config), json_response=True)

        async with serve(app) as client:
            await _modern_call(client, "first", request_id=1)
            await _modern_call(client, "second", request_id=2)

        assert config.prompts == [["first"], ["second"]]


@pytest.mark.asyncio
class TestHandshakeEraContinuity:
    """Up to 2025-11-25 the session exists at the protocol level, so it keys history.

    Pinned against regression: withdrawing the process fallback from the modern
    era must not withdraw it from the era whose revisions prescribe it.
    """

    async def test_stdio_style_stream_accumulates(self) -> None:
        config = RecordingConfig(TestConfig("ok", "ok"))
        server = MCPServer(_agent(config))

        async with connect(server) as session:
            await session.call_tool("ask", {"message": "first"})
            await session.call_tool("ask", {"message": "second"})

        assert config.prompts == [["first"], ["first", "second"]]

    async def test_http_session_accumulates(self) -> None:
        config = RecordingConfig(TestConfig("ok", "ok"))
        app = MCPServer(_agent(config), json_response=True)

        async with serve(app) as client:
            headers = await _open_handshake_session(client, request_id=1)
            await _handshake_call(client, headers, "first", request_id=2)
            await _handshake_call(client, headers, "second", request_id=3)

        assert config.prompts == [["first"], ["first", "second"]]

    async def test_different_http_sessions_are_isolated(self) -> None:
        config = RecordingConfig(TestConfig("ok", "ok"))
        app = MCPServer(_agent(config), json_response=True)

        async with serve(app) as client:
            first = await _open_handshake_session(client, request_id=1)
            second = await _open_handshake_session(client, request_id=2)
            await _handshake_call(client, first, "first", request_id=3)
            await _handshake_call(client, second, "second", request_id=4)

        assert config.prompts == [["first"], ["second"]]


def _handle(result: CallToolResult) -> str:
    """The conversation handle a result carries, as a programmatic client reads it."""
    assert result.meta is not None
    return result.meta[CONVERSATION_META_KEY]


def _modern_handle(result: dict[str, Any]) -> str:
    """:func:`_handle` for the raw JSON a modern-era POST returns."""
    return str(result["_meta"][CONVERSATION_META_KEY])


@pytest.mark.asyncio
class TestConversationHandle:
    """Continuity the caller names, which is the only kind the modern era has."""

    async def test_modern_era_continues_by_handle(self) -> None:
        config = RecordingConfig(TestConfig("ok", "ok"))
        server = MCPServer(_agent(config))

        async with connect_modern(server) as session:
            first = await session.call_tool("ask", {"message": "first"})
            await session.call_tool("ask", {"message": "second", "conversation": _handle(first)})

        assert config.prompts == [["first"], ["first", "second"]]

    async def test_handshake_era_continues_by_handle(self) -> None:
        config = RecordingConfig(TestConfig("ok", "ok"))
        server = MCPServer(_agent(config))

        async with connect(server) as session:
            first = await session.call_tool("ask", {"message": "first"})
            await session.call_tool("ask", {"message": "second", "conversation": _handle(first)})

        assert config.prompts == [["first"], ["first", "second"]]

    async def test_different_handles_stay_isolated(self) -> None:
        config = RecordingConfig(TestConfig("ok", "ok", "ok", "ok"))
        server = MCPServer(_agent(config))

        async with connect_modern(server) as session:
            one = _handle(await session.call_tool("ask", {"message": "one"}))
            two = _handle(await session.call_tool("ask", {"message": "two"}))
            await session.call_tool("ask", {"message": "one again", "conversation": one})
            await session.call_tool("ask", {"message": "two again", "conversation": two})

        assert one != two
        assert config.prompts == [
            ["one"],
            ["two"],
            ["one", "one again"],
            ["two", "two again"],
        ]

    async def test_handle_is_readable_and_machine_readable_and_they_agree(self) -> None:
        server = MCPServer(_agent(RecordingConfig(TestConfig("hello"))))

        async with connect_modern(server) as session:
            result = await session.call_tool("ask", {"message": "hi"})

        handle = _handle(result)
        # The agent's own reply leads; the handle rides in the block after it, so
        # the model can recover from an expired one without reading `_meta`.
        reply, trailer = result.content
        assert reply == TextContent(type="text", text="hello")
        assert handle in trailer.text

    async def test_handles_are_unguessable(self) -> None:
        server = MCPServer(_agent(RecordingConfig(TestConfig("ok", "ok"))))

        async with connect_modern(server) as session:
            first = _handle(await session.call_tool("ask", {"message": "one"}))
            second = _handle(await session.call_tool("ask", {"message": "two"}))

        # Version-4 UUIDs: opaque, unguessable, and header-safe.
        assert UUID(first).version == 4
        assert UUID(second).version == 4
        assert first != second


@pytest.mark.asyncio
class TestBlankHandle:
    """A blank handle names no conversation, so it reads as none being named.

    The reader of the handle channel is the model, and a model given an optional
    string argument routinely sends ``""`` rather than omitting the key. Read as
    an unknown handle, that would make its every first call an error and leave it
    unable to start a conversation at all.
    """

    @pytest.mark.parametrize("blank", ["", "   ", "\n"])
    async def test_starts_a_new_conversation_in_the_modern_era(self, blank: str) -> None:
        config = RecordingConfig(TestConfig("ok"))
        server = MCPServer(_agent(config))

        async with connect_modern(server) as session:
            result = await session.call_tool("ask", {"message": "first", "conversation": blank})

        assert result.is_error is False
        # A handle comes back, so the caller that could not omit the key can still
        # continue what it just started.
        assert UUID(_handle(result)).version == 4
        assert config.prompts == [["first"]]

    async def test_the_conversation_it_started_continues_by_its_handle(self) -> None:
        config = RecordingConfig(TestConfig("ok", "ok"))
        server = MCPServer(_agent(config))

        async with connect_modern(server) as session:
            first = await session.call_tool("ask", {"message": "first", "conversation": ""})
            await session.call_tool("ask", {"message": "second", "conversation": _handle(first)})

        assert config.prompts == [["first"], ["first", "second"]]

    async def test_falls_back_to_the_transport_session_in_the_handshake_era(self) -> None:
        config = RecordingConfig(TestConfig("ok", "ok"))
        server = MCPServer(_agent(config))

        # Naming nothing is what a blank handle means, so the handshake era keys on
        # the session it has — as it does for a call that omits the argument.
        async with connect(server) as session:
            await session.call_tool("ask", {"message": "first", "conversation": ""})
            await session.call_tool("ask", {"message": "second", "conversation": ""})

        assert config.prompts == [["first"], ["first", "second"]]


@pytest.mark.asyncio
class TestUnknownHandle:
    """A handle the registry does not know is an error, never a fall-through.

    Falling through to the transport session would let any caller name a
    conversation with a string of their choosing and evict other callers'
    conversations out of a bounded registry.
    """

    async def test_is_an_error_flagged_result_not_a_protocol_error(self) -> None:
        server = MCPServer(_agent(RecordingConfig(TestConfig("ok"))))

        async with connect_modern(server) as session:
            result = await session.call_tool("ask", {"message": "hi", "conversation": str(uuid4())})

        assert result.is_error is True

    async def test_does_not_start_a_conversation_under_the_supplied_string(self) -> None:
        config = RecordingConfig(TestConfig("ok", "ok"))
        server = MCPServer(_agent(config))
        chosen = str(uuid4())

        async with connect_modern(server) as session:
            rejected = await session.call_tool("ask", {"message": "first", "conversation": chosen})
            retried = await session.call_tool("ask", {"message": "second", "conversation": chosen})

        assert rejected.is_error is True
        assert retried.is_error is True
        # Neither call reached the agent, so nothing was adopted under `chosen`.
        assert config.prompts == []

    async def test_does_not_fall_back_to_the_transport_session(self) -> None:
        config = RecordingConfig(TestConfig("ok", "ok"))
        server = MCPServer(_agent(config))

        async with connect(server) as session:
            await session.call_tool("ask", {"message": "first"})
            rejected = await session.call_tool("ask", {"message": "second", "conversation": str(uuid4())})

        assert rejected.is_error is True
        # The handshake-era session's own history is untouched by the rejected call.
        assert config.prompts == [["first"]]


def _conversation_argument(tool: MCPTool) -> dict[str, Any]:
    """The advertised ``conversation`` argument, or ``{}`` when it is not offered."""
    return tool.input_schema["properties"].get("conversation", {})


@pytest.mark.asyncio
class TestAdvertisedConversationArgument:
    async def test_present_when_conversations_are_enabled(self) -> None:
        server = MCPServer(_agent(RecordingConfig(TestConfig("ok"))))

        async with connect(server) as session:
            (tool,) = (await session.list_tools()).tools

        assert _conversation_argument(tool) == IsPartialDict({"type": "string"})
        assert tool.input_schema["required"] == ["message"]

    async def test_absent_when_conversations_are_disabled(self) -> None:
        server = MCPServer(_agent(RecordingConfig(TestConfig("ok"))), sessions=False)

        async with connect(server) as session:
            (tool,) = (await session.list_tools()).tools

        assert _conversation_argument(tool) == {}

    async def test_description_states_the_configured_lifetime(self) -> None:
        server = MCPServer(
            _agent(RecordingConfig(TestConfig("ok"))),
            sessions=SessionConfig(max_sessions=64, ttl=900.0),
        )

        async with connect(server) as session:
            (tool,) = (await session.list_tools()).tools

        assert _conversation_argument(tool) == IsPartialDict({"description": IsStr(regex=r".*\b900\b.*\b64\b.*")})

    async def test_presenting_one_anyway_is_refused_not_dropped(self) -> None:
        """With conversations off, a handle is answered, not quietly discarded.

        The server mints no handles, so omitting the argument would not restore
        continuity either — which is why this is refused as unsupported rather
        than reported as an unknown handle.
        """
        config = RecordingConfig(TestConfig("ok"))
        server = MCPServer(_agent(config), sessions=False)

        async with connect(server) as session:
            result = await session.call_tool("ask", {"message": "first", "conversation": str(uuid4())})

        assert result.is_error is True
        assert result.content == [
            TextContent(
                type="text",
                text=(
                    "This server does not maintain conversations, so the 'conversation' argument is "
                    "not supported; omit it. Each call is independent."
                ),
            )
        ]
        # Refused before the agent ran: a rejected call is not half a turn.
        assert config.prompts == []


@pytest.mark.asyncio
async def test_structured_content_is_exactly_the_output_schema() -> None:
    """``structuredContent`` is the agent's response schema, and nothing else.

    It is advertised verbatim as the tool's ``outputSchema``, which MCP requires
    structured content to conform to, so a server field mixed in would break the
    tool's own declared contract.
    """
    agent = Agent(
        "weather",
        config=TestConfig('{"city": "SF", "temp_c": 18.5}'),
        response_schema=Weather,
    )
    server = MCPServer(agent)

    async with connect_modern(server) as session:
        (tool,) = (await session.list_tools()).tools
        result = await session.call_tool("ask", {"message": "weather in SF?"})

    assert result.structured_content == {"city": "SF", "temp_c": 18.5}
    assert tool.output_schema is not None
    assert set(tool.output_schema["properties"]) == {"city", "temp_c"}
    # The handle still travels, just not through the declared output contract.
    assert _handle(result)


@pytest.mark.asyncio
async def test_stateless_transport_serves_conversations_by_handle() -> None:
    """``stateless=True`` with ``sessions=True`` is coherent, not contradictory.

    It was contradictory only while continuity depended on the transport issuing
    a session id; with handles it means "no transport session, conversations by
    handle", so it constructs without complaint and serves them.
    """
    config = RecordingConfig(TestConfig("ok", "ok"))
    app = MCPServer(_agent(config), stateless=True, json_response=True)

    async with serve(app) as client:
        first, response = await _modern_call_with_response(client, "first", request_id=1)
        # The one assertion that is genuinely about HTTP: no session comes back.
        assert MCP_SESSION_ID_HEADER not in response.headers
        handle = _modern_handle(first)
        await _modern_call(client, "second", request_id=2, conversation=handle)

    assert config.prompts == [["first"], ["first", "second"]]


@pytest.mark.asyncio
class TestRegistryGuaranteesApplyToHandles:
    """The registry keys on an opaque string, so a handle drops into it unchanged."""

    async def test_the_bound_evicts_a_handle_named_conversation(self) -> None:
        config = RecordingConfig(TestConfig("ok", "ok", "ok"))
        server = MCPServer(_agent(config), sessions=SessionConfig(max_sessions=1))

        async with connect_modern(server) as session:
            evicted = _handle(await session.call_tool("ask", {"message": "one"}))
            await session.call_tool("ask", {"message": "two"})
            result = await session.call_tool("ask", {"message": "one again", "conversation": evicted})

        assert result.is_error is True

    async def test_the_configured_storage_backend_holds_the_history(self) -> None:
        storage = _RecordingStorage()
        config = RecordingConfig(TestConfig("ok", "ok"))
        server = MCPServer(_agent(config), sessions=SessionConfig(storage=storage))

        async with connect_modern(server) as session:
            first = _handle(await session.call_tool("ask", {"message": "first"}))
            await session.call_tool("ask", {"message": "second", "conversation": first})

        # One conversation, its turns replayed from the backend the operator
        # configured — the same path a session-named conversation takes.
        (stream_id,) = storage.stream_ids
        assert _texts(await storage.get_history(stream_id)) == ["first", "second"]


@pytest.mark.asyncio
async def test_custom_tools_are_untouched() -> None:
    """The handle applies to the conversational tool; a custom tool's state is its own."""
    server = MCPServer(
        _agent(RecordingConfig(TestConfig("ok"))),
        tools=[MCPFunctionTool(name="echo", description="Echo", handler=_echo)],
    )

    async with connect_modern(server) as session:
        tools = {t.name: t for t in (await session.list_tools()).tools}
        result = await session.call_tool("echo", {"text": "hi"})

    assert "conversation" not in tools["echo"].input_schema.get("properties", {})
    assert result.meta is None or CONVERSATION_META_KEY not in result.meta
    # One block and no trailer: the handle never rides on a custom tool's reply.
    assert result.content == [TextContent(type="text", text="hi")]


_TOKENS = {
    "alice": AccessToken(token="alice", client_id="shared-client", scopes=[], subject="alice"),
    "bob": AccessToken(token="bob", client_id="shared-client", scopes=[], subject="bob"),
    # No subject: the identity falls back to the client id, which is always present.
    "kiosk": AccessToken(token="kiosk", client_id="kiosk-client", scopes=[]),
    "other-kiosk": AccessToken(token="other-kiosk", client_id="other-kiosk-client", scopes=[]),
}


class _TokenVerifier:
    """Accepts the fixture tokens above and nothing else."""

    async def verify_token(self, token: str) -> AccessToken | None:
        return _TOKENS.get(token)


def _authenticated(config: RecordingConfig) -> MCPServer:
    return MCPServer(
        _agent(config),
        json_response=True,
        security=require(
            oauth2_scheme(url="https://auth.example.com"),
            resource_url="http://test/mcp",
            verifier=_TokenVerifier(),
        ),
    )


@pytest.mark.asyncio
class TestPrincipalBinding:
    """A handle names a conversation; it does not on its own confer the right to read one.

    The handle comes back in readable content, so it passes through the model's
    context, the client's logs and any tracing in between — further than a
    transport header ever went.
    """

    async def test_the_creating_principal_continues_normally(self) -> None:
        config = RecordingConfig(TestConfig("ok", "ok"))

        async with serve(_authenticated(config)) as client:
            first = await _modern_call(client, "first", request_id=1, token="alice")
            handle = _modern_handle(first)
            await _modern_call(client, "second", request_id=2, conversation=handle, token="alice")

        assert config.prompts == [["first"], ["first", "second"]]

    async def test_another_principal_is_refused_indistinguishably(self) -> None:
        config = RecordingConfig(TestConfig("ok", "ok"))

        async with serve(_authenticated(config)) as client:
            first = await _modern_call(client, "first", request_id=1, token="alice")
            handle = _modern_handle(first)
            stolen = await _modern_call(client, "second", request_id=2, conversation=handle, token="bob")
            unknown = await _modern_call(client, "second", request_id=3, conversation=str(uuid4()), token="bob")

        assert stolen["isError"] is True
        # Byte-for-byte the unknown-handle answer, so it does not disclose that
        # the handle exists.
        assert stolen["content"] == unknown["content"]
        assert config.prompts == [["first"]]

    async def test_the_binding_is_checked_on_every_call_not_only_at_creation(self) -> None:
        config = RecordingConfig(TestConfig("ok", "ok", "ok"))

        async with serve(_authenticated(config)) as client:
            first = await _modern_call(client, "first", request_id=1, token="alice")
            handle = _modern_handle(first)
            continued = await _modern_call(client, "second", request_id=2, conversation=handle, token="alice")
            swapped = await _modern_call(client, "third", request_id=3, conversation=handle, token="bob")

        assert continued["isError"] is False
        # The conversation was live and reachable a call earlier: the swapped
        # credential is what stops it, not the handle having gone away.
        assert swapped["isError"] is True

    async def test_a_session_named_conversation_is_unreachable_by_another_principal(self) -> None:
        """The other name a conversation goes by is closed to a swapped credential too.

        A handshake-era conversation is keyed by the MCP session, which ag2 does
        not revalidate — it does not have to. The transport refuses a session id
        presented with a credential other than the one that opened it, answering
        as though the session did not exist, so the swapped caller never reaches
        the conversation to begin with.
        """
        config = RecordingConfig(TestConfig("ok", "ok"))

        async with serve(_authenticated(config)) as client:
            headers = await _open_handshake_session(client, request_id=1, token="alice")
            await _handshake_call(client, headers, "first", request_id=2)
            swapped = await client.post(
                "/mcp",
                headers={**headers, "Authorization": "Bearer bob"},
                json={
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "ask", "arguments": {"message": "second"}},
                },
            )

        assert swapped.status_code == 404
        # Alice's turn is the only one the agent ever saw.
        assert config.prompts == [["first"]]

    async def test_a_token_with_no_subject_binds_via_its_client_id(self) -> None:
        config = RecordingConfig(TestConfig("ok", "ok", "ok"))

        async with serve(_authenticated(config)) as client:
            first = await _modern_call(client, "first", request_id=1, token="kiosk")
            handle = _modern_handle(first)
            same = await _modern_call(client, "second", request_id=2, conversation=handle, token="kiosk")
            other = await _modern_call(client, "third", request_id=3, conversation=handle, token="other-kiosk")

        assert same["isError"] is False
        assert other["isError"] is True
