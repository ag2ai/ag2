# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Awaitable, Callable, Hashable, Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Annotated, Any, TypeAlias, overload

from fast_depends import dependency_provider
from fast_depends.pydantic.schema import get_schema
from mcp.server.context import ServerRequestContext
from mcp.server.mcpserver.context import Context as ResolverContext
from mcp.server.mcpserver.resolve import Resolve, build_resolver_plans, find_resolved_parameters, resolve_arguments
from mcp.types import ContentBlock, InputRequiredResult, InputResponseRequestParams, TextContent, ToolAnnotations
from mcp.types import Tool as MCPTool

from ag2.annotations import ContextField
from ag2.utils import CONTEXT_OPTION_NAME, build_model

from ._async import call_user_fn

# The result a handler may produce: the content block(s) to send back, or a
# plain string (wrapped in a text block for convenience).
ToolResult: TypeAlias = "ContentBlock | Sequence[ContentBlock] | str"

# The MCP request context handed to a handler (``None`` outside a live request).
ToolContext: TypeAlias = "ServerRequestContext[Any, Any] | None"

# What one round of a call carries back when a previous round asked the client
# something: the answers, and the state naming the questions they answer. Both
# arrive already verified by the ``RequestStateBoundary``.
InputRound: TypeAlias = "InputResponseRequestParams | None"

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

    Usually produced by :func:`mcp_tool`; build one directly to serve a handler
    that already speaks raw ``tools/call`` arguments.
    """

    name: str
    """The name the tool is advertised and called under."""

    description: str
    """What the tool does, shown to the model choosing whether to call it."""

    handler: ToolHandler
    """Runs the call: takes the raw ``arguments`` dict and the MCP request context."""

    input_schema: dict[str, Any] = field(default_factory=lambda: {"type": "object"})
    """The JSON Schema advertised in ``tools/list``; an open object by default."""

    title: str | None = None
    """Human-readable display name for hosts that show one."""

    annotations: ToolAnnotations | None = None
    """``mcp.types.ToolAnnotations`` behaviour hints — ``readOnlyHint``, ``destructiveHint`` — for the host."""

    # What this tool asks the client for, and the resolver DAG behind it. Filled
    # by :func:`mcp_tool`, which is their only producer; a hand-built tool asks
    # for nothing. Not constructor arguments: the plan type is private to the SDK
    # (``_ResolverPlan``), so it has no place in a signature a user reads.
    #
    # Being non-init, they are reinitialised empty by ``dataclasses.replace``,
    # which would hand back a tool that has quietly stopped asking. Rebuild
    # through the decorator instead of replacing a field on a decorated tool.
    _resolved_params: "Mapping[str, tuple[Resolve, bool]]" = field(default_factory=dict, init=False, repr=False)
    _resolver_plans: "Mapping[Hashable, Any]" = field(default_factory=dict, init=False, repr=False)

    def _mcp_tool(self) -> MCPTool:
        return MCPTool(
            name=self.name,
            description=self.description,
            inputSchema=self.input_schema,
            title=self.title,
            annotations=self.annotations,
        )

    async def call(
        self,
        arguments: dict[str, Any],
        request_context: ToolContext = None,
        *,
        input_round: InputRound = None,
    ) -> "list[ContentBlock] | InputRequiredResult":
        """Run the tool, or come back asking the client for what it is missing.

        With ``Resolve(...)`` parameters the SDK fills them first and the body
        then runs once; otherwise the outstanding questions come back instead.
        """
        if self._resolved_params:
            resolved = await resolve_arguments(
                self._resolved_params,
                self._resolver_plans,
                arguments,
                ResolverContext(request_context=request_context, input_params=input_round),
            )
            if isinstance(resolved, InputRequiredResult):
                return resolved
            arguments = {**arguments, **resolved}
        result = await call_user_fn(self.handler, arguments, request_context)
        if isinstance(result, str):
            return [TextContent(type="text", text=result)]
        if isinstance(result, ContentBlock):
            return [result]
        return list(result)


def _bind(call_model: Any) -> ToolHandler:
    """Wrap a ``fast_depends`` call model as a handler that unpacks ``arguments``.

    The call's arguments become the function's keyword arguments,
    ``Depends``/``Inject`` parameters resolve against the process dependency
    provider, and a :data:`MCPRequestContext` parameter receives the request
    context.
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

    A parameter annotated ``Annotated[T, Resolve(fn)]`` is filled by running
    ``fn`` before the body, and ``fn`` may return a request marker — ``Elicit``
    to ask the client's human, ``Sample`` to borrow its model, ``ListRoots`` to
    read its roots — which the framework puts to the client and injects the
    answer of::

        from typing import Annotated
        from ag2.mcp import Elicit, Resolve
        from pydantic import BaseModel


        class Colour(BaseModel):
            answer: str


        def pick_colour() -> Elicit[Colour]:
            return Elicit("What colour?", Colour)


        @mcp_tool
        def paint(room: str, colour: Annotated[Colour, Resolve(pick_colour)]) -> str:
            "Paint a room."
            return f"painted {room} {colour.answer}"

    A resolver body re-runs on every round of that exchange, so keep side
    effects in one idempotent. The tool body is the opposite: it does not run
    until every resolver is satisfied, and then runs exactly once. (The agent's
    own ``ask`` tool re-runs nothing — its turn is held open between rounds.)

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
        # Kept out of what ``tools/list`` advertises: a resolved parameter is
        # filled by its resolver, never by the caller, and advertising it would
        # ask a model for the very thing the tool goes and asks the client for.
        resolved_params = find_resolved_parameters(f)
        schema = get_schema(call_model, exclude=(CONTEXT_OPTION_NAME, *resolved_params))
        if schema.get("type") != "object":
            schema = {"type": "object", "properties": {}}
        built = MCPFunctionTool(
            name=name or f.__name__,
            description=description or f.__doc__ or "",
            handler=_bind(call_model),
            input_schema=schema,
            title=title,
            annotations=annotations,
        )
        # Frozen with slots, so the decorator sets its own state the only way
        # such a dataclass allows.
        object.__setattr__(built, "_resolved_params", dict(resolved_params))
        object.__setattr__(
            built, "_resolver_plans", build_resolver_plans(resolved_params, set(schema.get("properties") or ()))
        )
        return built

    if function is not None:
        return make(function)
    return make


class ToolProvider:
    """Serves a fixed set of custom :class:`MCPFunctionTool` over MCP."""

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

    async def call(
        self,
        name: str,
        arguments: dict[str, Any],
        request_context: ToolContext = None,
        *,
        input_round: InputRound = None,
    ) -> "list[ContentBlock] | InputRequiredResult":
        return await self._by_name[name].call(arguments, request_context, input_round=input_round)
