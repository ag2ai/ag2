# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Expose an AG2 agent's locally-executable tools to an ACP CLI agent over MCP.

``partition_tools`` splits the turn's tool schemas into (a) function tools the
:class:`ToolGateway` serves itself over an in-process streamable-HTTP MCP
server, and (b) external ``MCPServerTool`` servers translated directly into ACP
``mcp_servers`` entries. Provider server-side builtin tools (``web_search`` and
friends) are flags inside a provider API request — there is nothing local to
execute — so they are rejected with :class:`~ag2.exceptions.UnsupportedToolError`;
CLI agents ship their own native equivalents.

The ``mcp`` SDK — together with the ``uvicorn``/``starlette`` server stack it
depends on — ships with the ``acp`` extra. A broken install surfaces on
``import ag2.acp`` with the ``ag2[acp]`` install hint (see the package
``__init__``).
"""

import asyncio
import base64
import json
import logging
import secrets
import socket
from collections.abc import AsyncGenerator, Generator, Iterable, Sequence
from contextlib import asynccontextmanager, contextmanager, suppress
from dataclasses import dataclass
from ipaddress import ip_address
from typing import TYPE_CHECKING, Any

import uvicorn
from acp import schema
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import CallToolResult, ImageContent, TextContent
from mcp.types import Tool as MCPTool
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.types import Receive, Scope, Send

from ag2.events import (
    BinaryInput,
    ClientToolCallEvent,
    DataInput,
    FileIdInput,
    TextInput,
    ToolCallEvent,
    ToolErrorEvent,
    ToolResultEvent,
    UrlInput,
)
from ag2.exceptions import AG2Error, UnsupportedToolError
from ag2.tools.builtin.mcp_server import MCPServerToolSchema
from ag2.tools.final import FunctionToolSchema

if TYPE_CHECKING:
    from ag2.tools.schemas import ToolSchema

    from .bridge import BridgeState

logger = logging.getLogger(__name__)

GATEWAY_SERVER_NAME = "ag2"
GATEWAY_PATH = "/mcp"


class MCPCapabilityError(AG2Error):
    """AG2 cannot expose the agent's tools to it over HTTP MCP.

    One error type covers both ways that can be true — the agent does not speak
    HTTP MCP, or it cannot reach the gateway that would serve the tools — so a
    caller who asked for tool exposure and cannot have it meets a single failure
    mode rather than two.
    """

    def __init__(self, agent: str, reason: str | None = None) -> None:
        super().__init__(
            reason
            or (
                f"ACP agent {agent!r} does not support HTTP MCP servers "
                "(initialize returned mcp_capabilities.http=false), so AG2 cannot expose "
                "the agent's tools to it. Remove the tools or set expose_tools=False."
            )
        )


@dataclass(frozen=True, slots=True)
class GatewayAddress:
    """Where the tool gateway binds, and the address the ACP agent is handed.

    The default is loopback with an OS-assigned port: the gateway is reachable
    only from this host, which is all a locally-launched agent needs. A caller
    driving a *remote* agent who has arranged network reachability supplies a
    non-loopback address; that widens the bind address, the DNS-rebinding
    allowlist and the allowed origins to match it, and no further — which is why
    it is an explicit opt-in rather than something inferred from a remote URL.

    Attributes:
        host: The interface to bind, and the host the agent dials.
        port: The port to bind; ``0`` lets the OS assign one, and the assigned
            port is what the agent is told to dial.
    """

    host: str = "127.0.0.1"
    port: int = 0

    @classmethod
    def parse(cls, address: str) -> "GatewayAddress":
        """Parse a ``host``, ``host:port``, ``ipv6`` or ``[ipv6]:port`` address.

        Strict about what it accepts, because the failure it would otherwise
        cause — a URL pasted in, say — surfaces as a name-resolution error on
        the first turn that carries a tool, a long way from the typo.

        Raises:
            ValueError: when the address is empty, looks like a URL, or has a
                port that is not a number in range.
        """
        text = address.strip()
        if not text:
            raise ValueError("a tool gateway address cannot be empty")
        if any(c in text for c in "/@ \t"):
            raise ValueError(f"tool gateway address {address!r} must be a host or host:port, not a URL")
        host, port = text, 0
        if text.startswith("["):  # bracketed IPv6, optionally with a port
            closing = text.find("]")
            if closing == -1:
                raise ValueError(f"unbalanced brackets in tool gateway address {address!r}")
            host, rest = text[1:closing], text[closing + 1 :]
            if rest and not rest.startswith(":"):
                raise ValueError(f"trailing {rest!r} in tool gateway address {address!r}")
            port = _parse_port(rest[1:], address) if rest else 0
        elif text.count(":") == 1:  # host:port (a bare IPv6 literal has several colons)
            host, _, raw_port = text.partition(":")
            port = _parse_port(raw_port, address)
        elif ":" in text:  # several colons: only a bare IPv6 literal can be one
            try:
                ip_address(text)
            except ValueError:
                raise ValueError(
                    f"tool gateway address {address!r} is neither a host, a host:port, nor an IPv6 literal"
                ) from None
        if not host:
            raise ValueError(f"tool gateway address {address!r} has no host")
        return cls(host=host, port=port)

    @property
    def is_loopback(self) -> bool:
        return self.host in ("127.0.0.1", "localhost", "::1")

    @property
    def authority(self) -> str:
        """The host as it appears in a URL — bracketed when it is an IPv6 literal."""
        return f"[{self.host}]" if ":" in self.host else self.host


def _parse_port(raw: str, address: str) -> int:
    try:
        port = int(raw)
    except ValueError:
        raise ValueError(f"tool gateway address {address!r} has a non-numeric port {raw!r}") from None
    if not 0 <= port <= 65535:
        raise ValueError(f"tool gateway address {address!r} has an out-of-range port {port}")
    return port


def partition_tools(tools: "Iterable[ToolSchema]") -> tuple[list[FunctionToolSchema], list[schema.HttpMcpServer]]:
    """Split tool schemas into gateway-served function tools and pass-through MCP servers.

    Raises:
        ValueError: for ``MCPServerTool`` with ``allowed_tools``/``blocked_tools``
            (ACP has no per-server tool filter; silently dropping the filter
            would widen access).
        UnsupportedToolError: for any other schema type — provider server-side
            builtins execute only inside that provider's API call.
    """
    functions: list[FunctionToolSchema] = []
    external: list[schema.HttpMcpServer] = []
    for tool in tools:
        if isinstance(tool, FunctionToolSchema):
            functions.append(tool)
        elif isinstance(tool, MCPServerToolSchema):
            if tool.allowed_tools is not None or tool.blocked_tools is not None:
                raise ValueError(
                    "MCPServerTool allowed_tools/blocked_tools cannot be enforced over ACP "
                    f"(server {tool.server_label!r}); remove the filter or connect the server "
                    "as an MCP toolkit so AG2 executes its tools."
                )
            headers = [schema.HttpHeader(name=k, value=v) for k, v in (tool.headers or {}).items()]
            if tool.authorization_token:
                headers.append(schema.HttpHeader(name="Authorization", value=f"Bearer {tool.authorization_token}"))
            external.append(
                schema.HttpMcpServer(type="http", name=tool.server_label, url=tool.server_url, headers=headers)
            )
        else:
            raise UnsupportedToolError(tool.type, "acp")
    return functions, external


class _EmbeddedServer(uvicorn.Server):
    # uvicorn's serve() replaces the process-wide SIGINT/SIGTERM handlers for its
    # lifetime and replays captured signals on exit; an in-process gateway must
    # never touch the host application's signal handling.
    @contextmanager
    def capture_signals(self) -> Generator[None]:
        yield


@contextmanager
def _quiet_uvicorn_shutdown() -> Generator[None]:
    """Mute uvicorn's cancellation noise while the server is stopped on purpose.

    ``timeout_graceful_shutdown`` cancels in-flight requests by design, and uvicorn
    reports that at ERROR with a full ASGI traceback — on the path that is working
    correctly, which reads as a crash. ``uvicorn.error`` is a process-wide logger
    name, so this briefly mutes any concurrent gateway too; the window is bounded
    by ``close_timeout`` and covers only a deliberate teardown.
    """
    log = logging.getLogger("uvicorn.error")
    previous = log.level
    log.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        log.setLevel(previous)


class ToolGateway:
    """In-process streamable-HTTP MCP server serving the run's function tools.

    ``tools/call`` is executed through the run's event stream — send
    ``ToolCallEvent``, await the matching ``ToolResultEvent``/``ToolErrorEvent``
    — exactly like :func:`ag2.tools.executor._execute_call`, so tool middleware
    and observers apply. The live :class:`~ag2.context.ConversationContext` is
    read from ``state.context`` at call time (the bridge refreshes it each turn).

    ``tools`` is read live on every ``tools/list``/``tools/call``, so the owner
    may replace it between turns; the ACP-level ``mcp_servers`` entry pointing
    at this gateway is fixed for the session's lifetime.
    """

    def __init__(
        self,
        state: "BridgeState",
        tools: Sequence[FunctionToolSchema],
        *,
        name: str = GATEWAY_SERVER_NAME,
        address: "GatewayAddress | None" = None,
        startup_timeout: float = 30.0,
        close_timeout: float = 5.0,
    ) -> None:
        self.state = state
        self.tools = list(tools)
        self.name = name
        self.address = address or GatewayAddress()
        self.url: str | None = None
        self._startup_timeout = startup_timeout
        self._close_timeout = close_timeout
        self._uvicorn: uvicorn.Server | None = None
        self._task: asyncio.Task[None] | None = None
        # The random path segment is the gateway's credential. It reaches the CLI
        # agent inside ``mcp_servers.url`` and appears nowhere else — uvicorn's
        # access log is off, and the URL is never a command-line argument.
        self._path = f"{GATEWAY_PATH}/{secrets.token_urlsafe(32)}"

    def as_acp_server(self) -> schema.HttpMcpServer:
        assert self.url is not None, "start() must succeed before as_acp_server()"
        return schema.HttpMcpServer(type="http", name=self.name, url=self.url, headers=[])

    async def start(self) -> str:
        """Bind this gateway's address, start serving, and return the MCP URL.

        Defaults to ``127.0.0.1:<os-assigned port>``; a caller-supplied
        :class:`GatewayAddress` binds that instead so a remote agent can dial it.
        """
        server: Server = Server(name=self.name)
        gateway = self

        @server.list_tools()  # type: ignore[no-untyped-call, misc, untyped-decorator]
        async def _list_tools() -> list[MCPTool]:
            return [
                MCPTool(
                    name=t.function.name,
                    description=t.function.description or None,
                    inputSchema=t.function.parameters or {"type": "object", "properties": {}},
                )
                for t in gateway.tools
            ]

        # validate_input=False: argument coercion is FunctionTool's job (pydantic),
        # matching the executor path — the SDK's jsonschema check would reject
        # values like "5" for an int that the tool itself accepts.
        @server.call_tool(validate_input=False)  # type: ignore[no-untyped-call, misc, untyped-decorator]
        async def _call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult | list[TextContent | ImageContent]:
            return await gateway._execute(name, arguments or {})

        # Host/Origin validation stops a browser page reaching this port via DNS
        # rebinding (a page cannot forge Host). It does not stop a local process,
        # which can set any header it likes — that is what the secret path is for.
        # The allowlist tracks the bind address: a gateway bound off-loopback for
        # a remote agent would otherwise reject the very requests it exists for.
        # Loopback answers to several names — and the URL it hands out is one of
        # them — so all three stay allowed; anything else answers only to the
        # authority the agent was actually told to dial.
        hosts = ["127.0.0.1", "localhost", "[::1]"] if self.address.is_loopback else [self.address.authority]
        security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[f"{host}:*" for host in hosts],
            allowed_origins=[f"http://{host}:*" for host in hosts],
        )
        manager = StreamableHTTPSessionManager(
            app=server, stateless=True, json_response=True, security_settings=security
        )

        async def handle(scope: Scope, receive: Receive, send: Send) -> None:
            await manager.handle_request(scope, receive, send)

        @asynccontextmanager
        async def lifespan(_app: Starlette) -> AsyncGenerator[None]:
            async with manager.run():
                yield

        app = Starlette(routes=[Mount(self._path, app=handle)], lifespan=lifespan)

        family = socket.AF_INET6 if ":" in self.address.host else socket.AF_INET
        sock = socket.socket(family, socket.SOCK_STREAM)
        try:
            # Unlike the loopback default, a caller-supplied address can fail to
            # bind (port taken, address not on this host); close the fd rather
            # than leaking it on the way out.
            sock.bind((self.address.host, self.address.port))
            port = sock.getsockname()[1]
        except BaseException:
            sock.close()
            raise

        config = uvicorn.Config(
            app,
            log_level="warning",
            access_log=False,
            lifespan="on",
            # Bounds close(): in-flight tools/call requests (e.g. stuck on a hung
            # tool) are cancelled instead of blocking shutdown forever. Best-effort:
            # a tool that swallows CancelledError can still linger past close().
            timeout_graceful_shutdown=max(1, round(self._close_timeout)),
        )
        self._uvicorn = _EmbeddedServer(config)
        self._task = asyncio.ensure_future(self._uvicorn.serve(sockets=[sock]))
        self._task.add_done_callback(_log_serve_crash)
        try:
            deadline = asyncio.get_running_loop().time() + self._startup_timeout
            while not self._uvicorn.started:
                if self._task.done():
                    self._task.result()  # surfaces the startup exception if any
                    raise RuntimeError("MCP tool gateway exited during startup")
                if asyncio.get_running_loop().time() > deadline:
                    raise TimeoutError(f"MCP tool gateway did not start within {self._startup_timeout:.1f}s")
                await asyncio.sleep(0.01)
        except BaseException:
            task, self._task = self._task, None
            self._uvicorn = None
            if task is not None and not task.done():
                task.cancel()
                with suppress(BaseException):
                    await task
            # uvicorn closes passed-in sockets only after a successful startup;
            # close ours here so failed attempts don't leak the fd.
            sock.close()
            raise

        self.url = f"http://{self.address.authority}:{port}{self._path}"
        return self.url

    async def close(self) -> None:
        """Stop the HTTP server; idempotent and bounded by ``close_timeout``."""
        server, self._uvicorn = self._uvicorn, None
        task, self._task = self._task, None
        self.url = None
        if server is not None:
            server.should_exit = True
        if task is None:
            return
        try:
            # timeout_graceful_shutdown already caps uvicorn's wait on in-flight
            # requests; the margin only guards against the server wedging entirely.
            with _quiet_uvicorn_shutdown():
                await asyncio.wait_for(task, timeout=self._close_timeout + 1.0)
        except (TimeoutError, asyncio.TimeoutError):  # separate classes on Python 3.10
            logger.warning("MCP tool gateway did not shut down within %.1fs; cancelled", self._close_timeout)
        except asyncio.CancelledError:
            raise  # the caller was cancelled — propagate, wait_for already cancelled the task
        except Exception:
            logger.exception("MCP tool gateway shutdown raised")

    async def _execute(self, name: str, arguments: dict[str, Any]) -> CallToolResult | list[TextContent | ImageContent]:
        context = self.state.context
        if context is None:
            logger.warning("MCP tool gateway: tools/call %r received with no active AG2 run", name)
            raise RuntimeError("no active AG2 run to execute the tool in")

        call = ToolCallEvent(name, arguments=json.dumps(arguments))
        try:
            async with context.stream.get(
                (ToolErrorEvent.parent_id == call.id)
                | (ToolResultEvent.parent_id == call.id)
                | (ClientToolCallEvent.id == call.id)
            ) as pending:
                await context.send(call)
                event = await pending
        except Exception as e:
            logger.exception("MCP tool gateway: executing tool %r failed", name)
            return CallToolResult(
                content=[TextContent(type="text", text=str(e))],
                isError=True,
            )

        if isinstance(event, ToolErrorEvent):
            return CallToolResult(
                content=[TextContent(type="text", text=str(event.error))],
                isError=True,
            )

        if isinstance(event, ClientToolCallEvent):
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=(
                            f"tool {name!r} requires client-side execution and cannot be executed "
                            "through the ACP tool gateway."
                        ),
                    )
                ],
                isError=True,
            )

        assert isinstance(event, ToolResultEvent)  # the get() filter admits nothing else

        if event.result.final:
            logger.warning(
                "MCP tool gateway: tool %r returned final=True, but final-response semantics "
                "cannot be enforced over ACP — the CLI agent receives it as ordinary tool output",
                name,
            )

        blocks: list[TextContent | ImageContent] = []
        for part in event.result.parts:
            if isinstance(part, TextInput):
                blocks.append(TextContent(type="text", text=part.content))
            elif isinstance(part, DataInput):
                blocks.append(TextContent(type="text", text=json.dumps(part.data, default=str)))
            elif isinstance(part, BinaryInput) and str(part.media_type).startswith("image/"):
                blocks.append(
                    ImageContent(
                        type="image",
                        data=base64.b64encode(part.data).decode(),
                        mimeType=str(part.media_type),
                    )
                )
            elif isinstance(part, BinaryInput):
                blocks.append(TextContent(type="text", text=f"<binary {part.media_type}, {len(part.data)} bytes>"))
            elif isinstance(part, UrlInput):
                blocks.append(TextContent(type="text", text=part.url))
            elif isinstance(part, FileIdInput):
                blocks.append(TextContent(type="text", text=f"<file {part.file_id}>"))
            else:
                blocks.append(TextContent(type="text", text=str(part)))
        return blocks


def _log_serve_crash(task: "asyncio.Task[None]") -> None:
    """Surface an unexpected mid-serving crash as soon as it happens."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("MCP tool gateway server crashed: %r", exc)
