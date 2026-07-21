# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Awaitable, Callable, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Annotated, Any, TypeAlias, overload

from fast_depends import dependency_provider
from fast_depends.pydantic.schema import get_schema
from mcp.server.session import ServerSession
from mcp.shared.context import RequestContext
from mcp.types import ContentBlock, TextContent, ToolAnnotations
from mcp.types import Tool as MCPTool

from ag2.annotations import ContextField
from ag2.utils import CONTEXT_OPTION_NAME, build_model

from ._async import call_user_fn

# The result a handler may produce: the content block(s) to send back, or a
# plain string (wrapped in a text block for convenience).
ToolResult: TypeAlias = "ContentBlock | Sequence[ContentBlock] | str"

# The MCP request context handed to a handler (``None`` outside a live request).
ToolContext: TypeAlias = "RequestContext[ServerSession, Any, Any] | None"

# A tool handler receives the call's ``arguments`` and the live MCP request
# context. Sync or async.
ToolHandler: TypeAlias = Callable[[dict[str, Any], ToolContext], "Awaitable[ToolResult] | ToolResult"]


# Annotate a ``@mcp_tool`` function parameter (any name) with this to receive
# the live MCP request context — session, client params, lifespan state:
#   async def my_tool(x: str, ctx: MCPRequestContext) -> ...
# Mirrors ``ag2.annotations.Context``; the parameter is excluded from the
# advertised ``inputSchema``.
MCPRequestContext = Annotated[RequestContext[ServerSession, Any, Any], ContextField(cast=False)]


@dataclass(frozen=True, slots=True)
class MCPFunctionTool:
    """A deterministic MCP tool served next to the agent's ``ask`` tool.

    Usually produced by :func:`mcp_tool`. Constructed directly, ``handler`` takes
    the raw ``tools/call`` ``arguments`` dict and the MCP request context, and
    returns the content block(s) — typically an :mod:`ag2.mcp_ui` resource, but
    any content block (or plain string) works. ``input_schema`` is the JSON
    Schema advertised in ``tools/list`` (defaults to an open object).

    ``title`` and ``annotations`` (``mcp.types.ToolAnnotations`` behavior hints
    such as ``readOnlyHint`` / ``destructiveHint``) are passed through to
    ``tools/list`` so hosts can decide e.g. whether to ask the user first.
    """

    name: str
    description: str
    handler: ToolHandler
    input_schema: dict[str, Any] = field(default_factory=lambda: {"type": "object"})
    title: str | None = None
    annotations: ToolAnnotations | None = None

    def _mcp_tool(self) -> MCPTool:
        return MCPTool(
            name=self.name,
            description=self.description,
            inputSchema=self.input_schema,
            title=self.title,
            annotations=self.annotations,
        )

    async def call(self, arguments: dict[str, Any], request_context: ToolContext = None) -> list[ContentBlock]:
        result = await call_user_fn(self.handler, arguments, request_context)
        if isinstance(result, str):
            return [TextContent(type="text", text=result)]
        if isinstance(result, ContentBlock):
            return [result]
        return list(result)


def _bind(call_model: Any) -> ToolHandler:
    """Wrap a ``fast_depends`` call model as a handler that unpacks ``arguments``.

    Mirrors ``ag2.a2ui.actions.A2UIAction.run``: the call's arguments become the
    function's keyword arguments (serializer-coerced), ``Depends``/``Inject``
    parameters resolve against the process dependency provider, and a
    :data:`MCPRequestContext`-annotated parameter receives the request context.
    """

    async def handler(arguments: dict[str, Any], request_context: ToolContext) -> Any:
        async with AsyncExitStack() as stack:
            return await call_model.asolve(
                **(arguments | {CONTEXT_OPTION_NAME: request_context}),
                stack=stack,
                cache_dependencies={},
                dependency_provider=dependency_provider,
            )

    return handler


@overload
def mcp_tool(
    function: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    title: str | None = None,
    annotations: ToolAnnotations | None = None,
    sync_to_thread: bool = True,
) -> MCPFunctionTool: ...


@overload
def mcp_tool(
    function: None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    title: str | None = None,
    annotations: ToolAnnotations | None = None,
    sync_to_thread: bool = True,
) -> Callable[[Callable[..., Any]], MCPFunctionTool]: ...


def mcp_tool(
    function: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    title: str | None = None,
    annotations: ToolAnnotations | None = None,
    sync_to_thread: bool = True,
) -> MCPFunctionTool | Callable[[Callable[..., Any]], MCPFunctionTool]:
    """Turn a function into a :class:`MCPFunctionTool` served alongside the agent's ``ask``.

    The tool ``name`` defaults to the function name, ``description`` to its
    docstring, and ``input_schema`` is derived from the typed signature. The
    function returns the MCP content block(s) for the result (e.g. an
    :mod:`ag2.mcp_ui` resource) or a plain string. A parameter annotated with
    :data:`MCPRequestContext` receives the live request context and is excluded
    from the advertised schema. Pass the result in ``MCPServer(tools=[...])``.

    Args:
        function: The function (when used as a bare ``@mcp_tool``).
        name: Tool name. Defaults to the function name.
        description: Tool description. Defaults to the function docstring.
        title: Human-readable display name for ``tools/list``.
        annotations: ``mcp.types.ToolAnnotations`` behavior hints
            (``readOnlyHint``, ``destructiveHint``, …) for the host.
        sync_to_thread: Run a sync function in a worker thread.
    """

    def make(f: Callable[..., Any]) -> MCPFunctionTool:
        call_model = build_model(f, sync_to_thread=sync_to_thread, serialize_result=False)
        schema = get_schema(call_model, exclude=(CONTEXT_OPTION_NAME,))
        if schema.get("type") != "object":
            schema = {"type": "object", "properties": {}}
        return MCPFunctionTool(
            name=name or f.__name__,
            description=description or f.__doc__ or "",
            handler=_bind(call_model),
            input_schema=schema,
            title=title,
            annotations=annotations,
        )

    if function is not None:
        return make(function)
    return make


class ToolProvider:
    """Serves a fixed set of custom :class:`MCPFunctionTool` over MCP.

    Unlike resources/prompts, MCP exposes a single ``tools/call`` handler, so this
    provider does not self-register decorators; :class:`~ag2.mcp.MCPServer` merges
    it into the one tool list / dispatcher it already owns.
    """

    __slots__ = ("_tools", "_by_name")

    def __init__(self, tools: Sequence[MCPFunctionTool]) -> None:
        self._tools = tuple(tools)
        self._by_name = {t.name: t for t in self._tools}

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._by_name)

    def list_mcp_tools(self) -> list[MCPTool]:
        return [t._mcp_tool() for t in self._tools]

    def has(self, name: str) -> bool:
        return name in self._by_name

    async def call(
        self, name: str, arguments: dict[str, Any], request_context: ToolContext = None
    ) -> list[ContentBlock]:
        return await self._by_name[name].call(arguments, request_context)
