# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field, is_dataclass
from types import GenericAlias
from typing import Annotated, Any, TypeAlias, get_type_hints, overload

from fast_depends import dependency_provider
from fast_depends.pydantic.schema import get_schema
from mcp.server.context import ServerRequestContext
from mcp.types import CallToolResult, ContentBlock, TextContent, ToolAnnotations
from mcp.types import Tool as MCPTool
from pydantic import BaseModel, JsonValue

from ag2.annotations import ContextField
from ag2.response import ResponseSchema
from ag2.utils import CONTEXT_OPTION_NAME, build_model

from ._async import call_user_fn
from .info import object_output_schema
from .mappers import to_structured_dict

# What a handler may return; :func:`_to_call_result` maps each arm onto the wire.
# Any dataclass is accepted too, alongside ``BaseModel``: it has no runtime type
# to name here, and naming it would cost this alias its resolvability.
ToolResult: TypeAlias = (
    "CallToolResult | str | ContentBlock | Sequence[ContentBlock] | Mapping[str, JsonValue] | BaseModel"
)

# The MCP request context handed to a handler (``None`` outside a live request).
ToolContext: TypeAlias = "ServerRequestContext[Any, Any] | None"

# A tool handler receives the call's ``arguments`` and the live MCP request
# context. Sync or async.
ToolHandler: TypeAlias = Callable[[dict[str, Any], ToolContext], "Awaitable[ToolResult] | ToolResult"]


# Annotate a ``@mcp_tool`` function parameter (any name) with this to receive
# the live MCP request context — session, client params, lifespan state:
#   async def my_tool(x: str, ctx: MCPRequestContext) -> ...
# Mirrors ``ag2.annotations.Context``; the parameter is excluded from the
# advertised ``inputSchema``.
MCPRequestContext = Annotated[ServerRequestContext[Any, Any], ContextField(cast=False)]


@dataclass(frozen=True, slots=True)
class MCPFunctionTool:
    """A deterministic MCP tool served next to the agent's ``ask`` tool.

    Usually produced by :func:`mcp_tool`. Constructed directly, ``handler`` takes
    the raw ``tools/call`` ``arguments`` dict and the MCP request context, and
    returns any :data:`ToolResult`.

    ``input_schema`` / ``output_schema`` are advertised in ``tools/list``, as are
    ``title`` and ``annotations`` (``mcp.types.ToolAnnotations`` behavior hints
    such as ``readOnlyHint``). ``meta`` reaches ``_meta``, the protocol's
    extension slot; an empty one is not sent at all.
    """

    name: str
    description: str
    handler: ToolHandler
    input_schema: dict[str, Any] = field(default_factory=lambda: {"type": "object"})
    title: str | None = None
    annotations: ToolAnnotations | None = None
    output_schema: dict[str, Any] | None = None
    meta: Mapping[str, Any] | None = None

    def _mcp_tool(self) -> MCPTool:
        return MCPTool(
            name=self.name,
            description=self.description,
            inputSchema=self.input_schema,
            outputSchema=self.output_schema,
            title=self.title,
            annotations=self.annotations,
            _meta=dict(self.meta) if self.meta else None,
        )

    async def call(self, arguments: dict[str, Any], request_context: ToolContext = None) -> CallToolResult:
        return _to_call_result(await call_user_fn(self.handler, arguments, request_context))


def _to_call_result(result: "ToolResult") -> CallToolResult:
    """Map a handler's return onto a ``tools/call`` result.

    Ordered by how much is derived: a ``CallToolResult`` is already the answer,
    and is also how a handler states its text and its data separately.
    """
    if isinstance(result, CallToolResult):
        return result
    if isinstance(result, str):
        return CallToolResult(content=[TextContent(type="text", text=result)])
    if isinstance(result, ContentBlock):
        return CallToolResult(content=[result])
    if isinstance(result, Mapping):
        data = dict(result)
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(data, default=str))],
            structuredContent=data,
        )
    if isinstance(result, Sequence):
        return CallToolResult(content=list(result))
    # A dataclass lands here, which is why this arm is wider than the alias.
    structured = to_structured_dict(result)
    if structured is None:
        raise TypeError(f"A tool handler cannot return {type(result).__name__}; see ag2.mcp.tools.ToolResult.")
    # ``str()`` and not the JSON dump: a type defining ``__str__`` is saying what
    # a reader should see, and a model is the reader here.
    return CallToolResult(content=[TextContent(type="text", text=str(result))], structuredContent=structured)


def derive_output_schema(f: Callable[..., Any]) -> dict[str, Any] | None:
    """The ``outputSchema`` implied by ``f``'s return annotation, or ``None``.

    Only a model or dataclass describes an object, which is what MCP requires
    ``structuredContent`` to be; every other annotation — ``CallToolResult``
    included — advertises none rather than failing.
    """
    try:
        annotation = get_type_hints(f).get("return")
    except Exception:  # pragma: no cover - a forward reference to a name that never resolves
        return None
    # Excluded before ``issubclass`` sees it: until 3.11 ``isinstance(list[int], type)``
    # was ``True`` while ``issubclass(list[int], X)`` raised.
    if not isinstance(annotation, type) or isinstance(annotation, GenericAlias):
        return None
    if issubclass(annotation, CallToolResult) or issubclass(annotation, ContentBlock):
        return None
    if not (issubclass(annotation, BaseModel) or is_dataclass(annotation)):
        return None
    return object_output_schema(ResponseSchema(annotation, embed=False))


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
    output_schema: dict[str, Any] | None = None,
    meta: Mapping[str, Any] | None = None,
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
    output_schema: dict[str, Any] | None = None,
    meta: Mapping[str, Any] | None = None,
    sync_to_thread: bool = True,
) -> Callable[[Callable[..., Any]], MCPFunctionTool]: ...


def mcp_tool(
    function: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    title: str | None = None,
    annotations: ToolAnnotations | None = None,
    output_schema: dict[str, Any] | None = None,
    meta: Mapping[str, Any] | None = None,
    sync_to_thread: bool = True,
) -> MCPFunctionTool | Callable[[Callable[..., Any]], MCPFunctionTool]:
    """Turn a function into a :class:`MCPFunctionTool` served alongside the agent's ``ask``.

    The tool ``name`` defaults to the function name, ``description`` to its
    docstring, ``input_schema`` is derived from the typed signature and
    ``output_schema`` from the return annotation. The function returns any
    :data:`ToolResult`. A parameter annotated with :data:`MCPRequestContext`
    receives the live request context and is excluded from the advertised
    schema. Pass the result in ``MCPServer(tools=[...])``.

    Args:
        function: The function (when used as a bare ``@mcp_tool``).
        name: Tool name. Defaults to the function name.
        description: Tool description. Defaults to the function docstring.
        title: Human-readable display name for ``tools/list``.
        annotations: ``mcp.types.ToolAnnotations`` behavior hints
            (``readOnlyHint``, ``destructiveHint``, …) for the host.
        output_schema: Overrides the schema derived from the return annotation.
        meta: ``_meta`` to advertise on the tool.
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
            output_schema=output_schema if output_schema is not None else derive_output_schema(f),
            meta=meta,
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

    def input_schema(self, name: str) -> dict[str, Any]:
        """The JSON Schema advertised for ``name``."""
        return self._by_name[name].input_schema

    async def call(self, name: str, arguments: dict[str, Any], request_context: ToolContext = None) -> CallToolResult:
        return await self._by_name[name].call(arguments, request_context)
