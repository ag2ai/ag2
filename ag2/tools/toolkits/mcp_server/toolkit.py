# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
from collections.abc import AsyncIterator, Iterable
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager
from dataclasses import replace
from typing import Any, get_args

import httpx
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import (
    AudioContent,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
    TextResourceContents,
)
from mcp.types import Tool as MCPTool

from ag2.annotations import Context, Variable
from ag2.events import (
    BinaryInput,
    BinaryType,
    Input,
    TextInput,
    ToolCallEvent,
    ToolErrorEvent,
    ToolResultEvent,
    UrlInput,
)
from ag2.middleware import (
    BaseMiddleware,
    ToolExecution,
    ToolMiddleware,
    ToolResultType,
)
from ag2.tools import ToolResult, Toolkit
from ag2.tools.final.function_tool import (
    FunctionDefinition,
    FunctionToolSchema,
)
from ag2.tools.schemas import ToolSchema
from ag2.tools.tool import Tool
from ag2.types import (
    AudioMediaType,
    DocumentMediaType,
    ImageMediaType,
    VideoMediaType,
)

from .types import MCPServerConfig, MCPStdioServerConfig

AnyMCPConfig = MCPServerConfig | MCPStdioServerConfig


@asynccontextmanager
async def _mcp_session(config: AnyMCPConfig) -> AsyncIterator[ClientSession]:
    """Open a short-lived MCP ``ClientSession`` for one operation.

    Dispatches on the config type — HTTP/streamable-http for
    :class:`MCPServerConfig`, stdio subprocess for :class:`MCPStdioServerConfig`.
    """
    if isinstance(config, MCPStdioServerConfig):
        params = StdioServerParameters(
            command=config.command,  # type: ignore[arg-type]
            args=list(config.args or []),  # type: ignore[arg-type]
            env=config.env,  # type: ignore[arg-type]
            cwd=config.cwd,  # type: ignore[arg-type]
            encoding=config.encoding,
        )
        async with (
            stdio_client(params) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()
            yield session
    else:
        async with (
            httpx.AsyncClient(
                headers=config.headers,  # type: ignore[arg-type]  # Variable already resolved by _resolve_config
                timeout=config.connection_timeout,
                proxy=config.proxy,
                verify=config.verify,
            ) as client,
            streamable_http_client(
                config.server_url,  # type: ignore[arg-type]  # Variable already resolved by _resolve_config
                http_client=client,
            ) as (read_stream, write_stream, _),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()
            yield session


class _MCPProxyTool(Tool):
    """A function-tool-shaped proxy that forwards calls to a remote MCP server."""

    __slots__ = ("name", "schema", "_config", "_middleware")

    def __init__(
        self,
        config: AnyMCPConfig,
        raw_tool: MCPTool,
        middleware: tuple[ToolMiddleware, ...] = (),
    ) -> None:
        self._config = config
        self._middleware = middleware
        self.name = raw_tool.name
        self.schema = FunctionToolSchema(
            function=FunctionDefinition(
                name=self.name,
                description=raw_tool.description or "",
                parameters=dict(raw_tool.inputSchema or {}),
            )
        )

    async def schemas(self, context: "Context") -> list[FunctionToolSchema]:
        return [self.schema]

    def register(
        self,
        stack: "ExitStack | AsyncExitStack",
        context: "Context",
        *,
        middleware: Iterable["BaseMiddleware"] = (),
    ) -> None:
        execution: ToolExecution = self
        for hook in reversed(self._middleware):
            execution = _wrap_middleware(hook, execution)
        for mw in middleware:
            execution = _wrap_middleware(mw.on_tool_execution, execution)

        async def execute(event: "ToolCallEvent", context: "Context") -> None:
            result = await execution(event, context)
            await context.send(result)

        # ``Event.field == value`` builds a Condition at runtime; mypy sees ``bool``.
        stack.enter_context(context.stream.where(ToolCallEvent.name == self.name).sub_scope(execute))  # type: ignore[arg-type]

    async def __call__(self, event: "ToolCallEvent", context: "Context") -> "ToolResultEvent | ToolErrorEvent":
        try:
            resolved = _resolve_config(self._config, context)
            async with _mcp_session(resolved) as session:
                result = await session.call_tool(self.name, event.serialized_arguments)

        except Exception as e:
            return ToolErrorEvent.from_call(event, error=e)

        if result.isError:
            return ToolErrorEvent.from_call(event, error=RuntimeError(str(result)))

        return ToolResultEvent.from_call(event, result=_extract_content(result))


class MCPToolkit(Toolkit):
    """Expose the tools of an MCP server as ordinary local tools.

    Accepts either:

    * a URL string or :class:`MCPServerConfig` for a remote (streamable-http)
      server, or
    * an :class:`MCPStdioServerConfig` for a locally-launched server
      communicating over stdin/stdout.

    Tool discovery is lazy: the first call to :meth:`schemas` performs the
    MCP handshake, lists the server's tools, and registers a proxy for each
    one. The agent never sees that these are MCP tools — they look and behave
    like ordinary :class:`FunctionTool` instances.
    """

    __slots__ = ("config", "_discovered", "_discover_lock")

    def __init__(
        self,
        server: str | MCPServerConfig | MCPStdioServerConfig,
        *,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        if isinstance(server, str):
            server = MCPServerConfig(server_url=server)
        self.config: AnyMCPConfig = server
        self._discovered = False
        self._discover_lock = asyncio.Lock()

        label = server.server_label if isinstance(server.server_label, str) else ""
        super().__init__(
            name=label or "mcp_toolkit",
            middleware=middleware,
        )

    async def schemas(self, context: "Context") -> Iterable[ToolSchema]:
        await self._discover_tools(context)
        return await super().schemas(context)

    async def _discover_tools(self, context: "Context") -> None:
        if self._discovered:
            return

        async with self._discover_lock:
            if self._discovered:
                return

            resolved = _resolve_config(self.config, context)

            async with _mcp_session(resolved) as session:
                raw_tools = (await session.list_tools()).tools

            # Both already resolved (Variable -> concrete) by _resolve_config above.
            allowed = resolved.allowed_tools
            blocked = set(resolved.blocked_tools or [])  # type: ignore[arg-type]

            for raw in raw_tools:
                if allowed is not None and raw.name not in allowed:  # type: ignore[operator]
                    continue
                if raw.name in blocked:
                    continue
                proxy = _MCPProxyTool(
                    config=self.config,
                    raw_tool=raw,
                    middleware=self._middleware,
                )
                self._tools[proxy.name] = proxy

            self._discovered = True


def _wrap_middleware(hook: "ToolMiddleware", inner: "ToolExecution") -> "ToolExecution":
    async def call(event: "ToolCallEvent", context: "Context") -> "ToolResultType":
        return await hook(inner, event, context)

    return call


def _extract_content(result: CallToolResult) -> ToolResult:
    """Convert MCP ``tools/call`` content blocks into a typed ``ToolResult``.

    Each MCP ``ContentBlock`` variant is mapped to the closest AG2 ``Input``
    type so non-text content (images, audio, blobs, resource links) reaches
    the agent / LLM without further unpacking.
    """
    parts = result.content
    if not parts:
        return ToolResult(result.model_dump_json(exclude_none=True))

    inputs: list[Input] = []
    for p in parts:
        if isinstance(p, TextContent):
            inputs.append(TextInput(content=p.text))
        elif isinstance(p, ImageContent):
            inputs.append(
                BinaryInput(
                    data=base64.b64decode(p.data),
                    media_type=p.mimeType,
                    kind=BinaryType.IMAGE,
                )
            )
        elif isinstance(p, AudioContent):
            inputs.append(
                BinaryInput(
                    data=base64.b64decode(p.data),
                    media_type=p.mimeType,
                    kind=BinaryType.AUDIO,
                )
            )
        elif isinstance(p, ResourceLink):
            inputs.append(UrlInput(url=str(p.uri), kind=_kind_from_mime(p.mimeType)))
        elif isinstance(p, EmbeddedResource):
            resource = p.resource
            if isinstance(resource, TextResourceContents):
                inputs.append(TextInput(content=resource.text))
            else:
                inputs.append(
                    BinaryInput(
                        data=base64.b64decode(resource.blob),
                        media_type=resource.mimeType or "application/octet-stream",
                        kind=_kind_from_mime(resource.mimeType),
                    )
                )
        else:
            # Future ContentBlock variant — preserve as JSON text rather than drop.
            inputs.append(TextInput(content=p.model_dump_json(exclude_none=True)))
    return ToolResult(parts=inputs)


_KIND_BY_MIME: dict[str, BinaryType] = {
    **dict.fromkeys(get_args(ImageMediaType), BinaryType.IMAGE),
    **dict.fromkeys(get_args(AudioMediaType), BinaryType.AUDIO),
    **dict.fromkeys(get_args(VideoMediaType), BinaryType.VIDEO),
    **dict.fromkeys(get_args(DocumentMediaType), BinaryType.DOCUMENT),
}


def _kind_from_mime(mime: str | None) -> BinaryType:
    if not mime:
        return BinaryType.BINARY
    return _KIND_BY_MIME.get(mime, BinaryType.BINARY)


def _resolve_value(value: Any, context: "Context") -> Any:
    if not isinstance(value, Variable):
        return value
    name = value.name
    if name in context.variables:
        return context.variables[name]
    if value.default is not Ellipsis:
        return value.default
    if value.default_factory is not Ellipsis:
        return value.default_factory()
    raise KeyError(f"Context variable {name!r} not found and no default provided")


def _resolve_config(config: AnyMCPConfig, context: "Context") -> AnyMCPConfig:
    if isinstance(config, MCPStdioServerConfig):
        return replace(
            config,
            command=_resolve_value(config.command, context),
            args=list(_resolve_value(config.args, context) or []),
            env=_resolve_value(config.env, context),
            cwd=_resolve_value(config.cwd, context),
            server_label=_resolve_value(config.server_label, context) or "",
            description=_resolve_value(config.description, context),
            allowed_tools=_resolve_value(config.allowed_tools, context),
            blocked_tools=_resolve_value(config.blocked_tools, context),
        )

    headers = dict(_resolve_value(config.headers, context) or {})
    auth = _resolve_value(config.authorization_token, context)
    if auth and "Authorization" not in headers:
        headers["Authorization"] = f"Bearer {auth}"

    return replace(
        config,
        server_url=_resolve_value(config.server_url, context),
        server_label=_resolve_value(config.server_label, context) or "",
        authorization_token=auth,
        description=_resolve_value(config.description, context),
        allowed_tools=_resolve_value(config.allowed_tools, context),
        blocked_tools=_resolve_value(config.blocked_tools, context),
        headers=headers or None,
    )
