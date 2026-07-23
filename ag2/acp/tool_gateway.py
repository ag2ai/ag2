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

The ``mcp`` SDK ships with the ``acp`` extra, but ``uvicorn``/``starlette`` are
imported lazily inside :meth:`ToolGateway.start` so importing ``ag2.acp`` stays
cheap and a broken ``mcp`` install only surfaces when function tools actually
need exposing.
"""

import asyncio
import base64
import json
import logging
from collections.abc import Iterable, Sequence
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Any

from acp import schema

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
    import uvicorn

    from ag2.tools.schemas import ToolSchema

    from .bridge import BridgeState

logger = logging.getLogger(__name__)

GATEWAY_SERVER_NAME = "ag2"
GATEWAY_PATH = "/mcp"


class MCPCapabilityError(AG2Error):
    """The ACP agent cannot consume HTTP MCP servers, but tools must be exposed."""

    def __init__(self, agent: str) -> None:
        super().__init__(
            f"ACP agent {agent!r} does not support HTTP MCP servers "
            "(initialize returned mcp_capabilities.http=false), so AG2 cannot expose "
            "the agent's tools to it. Remove the tools or set expose_tools=False."
        )


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
        startup_timeout: float = 30.0,
        close_timeout: float = 5.0,
    ) -> None:
        self.state = state
        self.tools = list(tools)
        self.name = name
        self.url: str | None = None
        self._startup_timeout = startup_timeout
        self._close_timeout = close_timeout
        self._uvicorn: uvicorn.Server | None = None
        self._task: asyncio.Task[None] | None = None

    def as_acp_server(self) -> schema.HttpMcpServer:
        assert self.url is not None, "start() must succeed before as_acp_server()"
        return schema.HttpMcpServer(type="http", name=self.name, url=self.url, headers=[])

    async def start(self) -> str:
        """Bind 127.0.0.1:<os-assigned port>, start serving, return the MCP URL."""
        try:
            import socket

            import uvicorn
            from mcp.server.lowlevel import Server
            from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
            from mcp.server.transport_security import TransportSecuritySettings
            from mcp.types import Tool as MCPTool
            from starlette.applications import Starlette
            from starlette.routing import Mount
        except ImportError as e:  # pragma: no cover - exercised only with a broken install
            raise ImportError(
                "Exposing AG2 tools to an ACP agent requires the MCP SDK, which ships with "
                "the `ag2[acp]` extra — reinstall it (or set expose_tools=False on the ACP config)."
            ) from e

        class _EmbeddedServer(uvicorn.Server):
            # uvicorn's serve() replaces the process-wide SIGINT/SIGTERM handlers
            # for its lifetime and replays captured signals on exit; an in-process
            # gateway must never touch the host application's signal handling.
            @contextmanager
            def capture_signals(self) -> Any:
                yield

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
        async def _call_tool(name: str, arguments: dict[str, Any]) -> Any:
            return await gateway._execute(name, arguments or {})

        # Host/Origin validation: without it any local process — or a browser page
        # via DNS rebinding — could invoke the agent's tools on this port.
        security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=["127.0.0.1:*", "localhost:*"],
            allowed_origins=["http://127.0.0.1:*", "http://localhost:*"],
        )
        manager = StreamableHTTPSessionManager(
            app=server, stateless=True, json_response=True, security_settings=security
        )

        async def handle(scope: Any, receive: Any, send: Any) -> None:
            await manager.handle_request(scope, receive, send)

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def lifespan(_app: Any) -> Any:
            async with manager.run():
                yield

        app = Starlette(routes=[Mount(GATEWAY_PATH, app=handle)], lifespan=lifespan)

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

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

        self.url = f"http://127.0.0.1:{port}{GATEWAY_PATH}"
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
            await asyncio.wait_for(task, timeout=self._close_timeout + 1.0)
        except (TimeoutError, asyncio.TimeoutError):  # separate classes on Python 3.10
            logger.warning("MCP tool gateway did not shut down within %.1fs; cancelled", self._close_timeout)
        except asyncio.CancelledError:
            raise  # the caller was cancelled — propagate, wait_for already cancelled the task
        except Exception:
            logger.exception("MCP tool gateway shutdown raised")

    async def _execute(self, name: str, arguments: dict[str, Any]) -> Any:
        from mcp.types import CallToolResult, ImageContent, TextContent

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
