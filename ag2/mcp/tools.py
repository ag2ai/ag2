# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Awaitable, Callable, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, TypeAlias, overload

from fast_depends import dependency_provider
from fast_depends.pydantic.schema import get_schema
from mcp.types import ContentBlock
from mcp.types import Tool as MCPTool

from ag2.utils import CONTEXT_OPTION_NAME, build_model

from ._async import call_user_fn

# A tool handler receives the call's ``arguments`` and returns the content
# block(s) to send back. Sync or async; a lone block is accepted for convenience.
ToolHandler: TypeAlias = Callable[
    [dict[str, Any]], "Awaitable[ContentBlock | Sequence[ContentBlock]] | ContentBlock | Sequence[ContentBlock]"
]


@dataclass(frozen=True, slots=True)
class MCPUITool:
    """A deterministic MCP tool served next to the agent's ``ask`` tool.

    Usually produced by :func:`mcp_tool`. Constructed directly, ``handler`` takes
    the raw ``tools/call`` ``arguments`` dict and returns the content block(s);
    ``input_schema`` is the JSON Schema advertised in ``tools/list`` (defaults to
    an open object).
    """

    name: str
    description: str
    handler: ToolHandler
    input_schema: dict[str, Any] = field(default_factory=lambda: {"type": "object"})

    def _mcp_tool(self) -> MCPTool:
        return MCPTool(name=self.name, description=self.description, inputSchema=self.input_schema)

    async def call(self, arguments: dict[str, Any]) -> list[ContentBlock]:
        result = await call_user_fn(self.handler, arguments)
        if isinstance(result, ContentBlock):
            return [result]
        return list(result)


def _bind(call_model: Any) -> ToolHandler:
    """Wrap a ``fast_depends`` call model as a handler that unpacks ``arguments``.

    Mirrors ``ag2.a2ui.actions.A2UIAction.run``: the call's arguments become the
    function's keyword arguments (serializer-coerced), and ``Depends``/``Inject``
    parameters resolve against the process dependency provider.
    """

    async def handler(arguments: dict[str, Any]) -> Any:
        async with AsyncExitStack() as stack:
            return await call_model.asolve(
                **arguments,
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
    sync_to_thread: bool = True,
) -> MCPUITool: ...


@overload
def mcp_tool(
    function: None = None, *, name: str | None = None, description: str | None = None, sync_to_thread: bool = True
) -> Callable[[Callable[..., Any]], MCPUITool]: ...


def mcp_tool(
    function: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    sync_to_thread: bool = True,
) -> MCPUITool | Callable[[Callable[..., Any]], MCPUITool]:
    """Turn a function into a :class:`MCPUITool` served alongside the agent's ``ask``.

    The tool ``name`` defaults to the function name, ``description`` to its
    docstring, and ``input_schema`` is derived from the typed signature. The
    function returns the MCP content block(s) for the result (e.g. an
    :mod:`ag2.mcp_ui` resource). Pass the result in ``MCPServer(tools=[...])``.

    Args:
        function: The function (when used as a bare ``@mcp_tool``).
        name: Tool name. Defaults to the function name.
        description: Tool description. Defaults to the function docstring.
        sync_to_thread: Run a sync function in a worker thread.
    """

    def make(f: Callable[..., Any]) -> MCPUITool:
        call_model = build_model(f, sync_to_thread=sync_to_thread, serialize_result=False)
        schema = get_schema(call_model, exclude=(CONTEXT_OPTION_NAME,))
        if schema.get("type") != "object":
            schema = {"type": "object", "properties": {}}
        return MCPUITool(
            name=name or f.__name__,
            description=description or f.__doc__ or "",
            handler=_bind(call_model),
            input_schema=schema,
        )

    if function is not None:
        return make(function)
    return make


class ToolProvider:
    """Serves a fixed set of custom :class:`MCPUITool` over MCP.

    Unlike resources/prompts, MCP exposes a single ``tools/call`` handler, so this
    provider does not self-register decorators; :class:`~ag2.mcp.MCPServer` merges
    it into the one tool list / dispatcher it already owns.
    """

    __slots__ = ("_tools", "_by_name")

    def __init__(self, tools: Sequence[MCPUITool]) -> None:
        self._tools = tuple(tools)
        self._by_name = {t.name: t for t in self._tools}

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._by_name)

    def list_mcp_tools(self) -> list[MCPTool]:
        return [t._mcp_tool() for t in self._tools]

    def has(self, name: str) -> bool:
        return name in self._by_name

    async def call(self, name: str, arguments: dict[str, Any]) -> list[ContentBlock]:
        return await self._by_name[name].call(arguments)
