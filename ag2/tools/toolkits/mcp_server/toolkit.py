# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
from collections.abc import AsyncGenerator, Iterable
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager
from dataclasses import replace
from functools import partial
from typing import Any, get_args

import httpx2
from mcp import ClientSession
from mcp.client._input_required import run_input_required_driver
from mcp.client._probe import negotiate_auto
from mcp.client.session import ClientRequestContext
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.types import (
    AudioContent,
    CallToolResult,
    EmbeddedResource,
    ErrorData,
    ImageContent,
    InputRequest,
    InputRequiredResult,
    InputResponse,
    InputResponses,
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

from .answering import AnswerPolicy, InputRequestAnswerer
from .types import MCPServerConfig, MCPStdioServerConfig, ProtocolMode

AnyMCPConfig = MCPServerConfig | MCPStdioServerConfig


@asynccontextmanager
async def _mcp_session(
    config: AnyMCPConfig,
    **session_kwargs: Any,
) -> AsyncGenerator[ClientSession]:
    """Open a short-lived MCP ``ClientSession`` for one operation.

    Dispatches on the config type — HTTP/streamable-http for
    :class:`MCPServerConfig`, stdio subprocess for :class:`MCPStdioServerConfig`.

    ``session_kwargs`` are the answering callbacks for this operation (see
    :mod:`.answering`). Which ones are present is what the handshake advertises,
    so they are passed per-operation rather than fixed on the toolkit: the
    agent's human and the agent's model are reachable only from the live context
    the operation runs in.
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
            ClientSession(read_stream, write_stream, **session_kwargs) as session,
        ):
            await _settle_era(session, config.protocol_mode)
            yield session
    else:
        # ``httpx2``, not ``httpx``: that is the client type mcp 2.0's streamable-HTTP
        # transport takes. AG2 core stays on ``httpx`` — separate distributions with
        # separate module names, so the two coexist in one environment.
        async with (
            httpx2.AsyncClient(
                headers=config.headers,  # type: ignore[arg-type]  # Variable already resolved by _resolve_config
                timeout=config.connection_timeout,
                proxy=config.proxy,
                verify=config.verify,
                # A Starlette-mounted endpoint 307s the slashless form, which is the
                # form a caller naturally writes; without this the connection fails.
                follow_redirects=True,
            ) as client,
            streamable_http_client(
                config.server_url,  # type: ignore[arg-type]  # Variable already resolved by _resolve_config
                http_client=client,
            ) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream, **session_kwargs) as session,
        ):
            await _settle_era(session, config.protocol_mode)
            yield session


async def _settle_era(session: ClientSession, mode: ProtocolMode) -> None:
    """Establish which protocol revision this session speaks.

    ``"auto"`` asks the server (``server/discover``) and falls back to the
    handshake, so a modern-era server is met on the modern era — which is what
    lets a server return a question as the *result* of a call rather than over a
    standalone request. ``"legacy"`` performs the handshake only.
    """
    if mode == "auto":
        await negotiate_auto(session)
        return
    await session.initialize()


async def _dispatch_input_request(
    session: ClientSession, key: str, request: "InputRequest"
) -> "InputResponse | ErrorData":
    """Route one embedded input request through this session's answering callbacks.

    The same callback table the standalone server-to-client RPCs go through, so
    the handshake and modern eras cannot disagree about who answers what.
    """
    ctx = ClientRequestContext(
        session=session,
        request_id=key,
        meta=request.params.meta if request.params else None,
    )
    return await session.dispatch_input_request(ctx, request)


async def _retry_call(
    session: ClientSession,
    name: str,
    arguments: dict[str, Any],
    responses: "InputResponses | None",
    state: str | None,
) -> "CallToolResult | InputRequiredResult":
    """Re-issue the original call carrying the answers and the echoed state."""
    return await session.call_tool(
        name,
        arguments,
        input_responses=responses,
        request_state=state,
        allow_input_required=True,
    )


class _MCPProxyTool(Tool):
    """A function-tool-shaped proxy that forwards calls to a remote MCP server."""

    __slots__ = ("name", "schema", "_config", "_middleware", "_answering")

    def __init__(
        self,
        config: AnyMCPConfig,
        raw_tool: MCPTool,
        *,
        middleware: tuple[ToolMiddleware, ...] = (),
        answering: AnswerPolicy,
    ) -> None:
        self._config = config
        self._middleware = middleware
        self._answering = answering
        self.name = raw_tool.name
        self.schema = FunctionToolSchema(
            function=FunctionDefinition(
                name=self.name,
                description=raw_tool.description or "",
                parameters=dict(raw_tool.input_schema or {}),
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
            answerer = InputRequestAnswerer(self._answering, context)
            async with _mcp_session(resolved, **answerer.session_kwargs()) as session:
                result = await self._call(session, event, answerer)

        except Exception as e:
            return ToolErrorEvent.from_call(event, error=_unwrap(e))

        if result.is_error:
            return ToolErrorEvent.from_call(event, error=RuntimeError(str(result)))

        return ToolResultEvent.from_call(event, result=_extract_content(result))

    async def _call(
        self,
        session: ClientSession,
        event: "ToolCallEvent",
        answerer: InputRequestAnswerer,
    ) -> CallToolResult:
        """One ``tools/call``, answering and retrying for as long as the server asks.

        The whole loop lives inside the single operation that already opened the
        session, so nothing has to be held between calls: the pause is happening
        on the *remote* server and this end is simply waiting. ``request_state``
        is echoed back byte-exact and never inspected — it is the server's own
        sealed state, not ours to read.
        """
        arguments = event.serialized_arguments
        first = await session.call_tool(self.name, arguments, allow_input_required=True)
        if not isinstance(first, InputRequiredResult):
            return first
        # A bound on the rounds, so a server that re-asks the same thing forever
        # ends the call with an error naming that rather than looping the agent.
        return await run_input_required_driver(
            first,
            dispatch=partial(_dispatch_input_request, session),
            retry=partial(_retry_call, session, self.name, arguments),
            max_rounds=self._answering.max_rounds,
        )


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

    A server may answer a tool call by asking for something instead of returning
    a result — a question for the user, a model completion, the client's roots.
    ``answering`` says which of those this agent will supply, and everything in
    it is off by default: these hand the operator's own resources to a server
    they may not control. See :class:`.AnswerPolicy`. Whatever is enabled is
    answered inside the same operation that opened the session and the call is
    retried, so nothing is held between calls — the pause is on the remote
    server and this end is simply waiting.
    """

    __slots__ = ("config", "answering", "_discovered", "_discover_lock")

    def __init__(
        self,
        server: str | MCPServerConfig | MCPStdioServerConfig,
        *,
        middleware: Iterable[ToolMiddleware] = (),
        answering: AnswerPolicy | None = None,
    ) -> None:
        if isinstance(server, str):
            server = MCPServerConfig(server_url=server)
        self.config: AnyMCPConfig = server
        self.answering = answering if answering is not None else AnswerPolicy()
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
                    answering=self.answering,
                )
                self._tools[proxy.name] = proxy

            self._discovered = True


def _unwrap(error: Exception) -> Exception:
    """Peel task-group wrappers off a lone failure.

    A ``ClientSession`` runs its receive loop inside a task group, so anything
    raised while the session is open — the round bound being reached, a server
    refusing an input request — arrives here inside one ``ExceptionGroup`` per
    nesting level. The group's own message names nothing ("unhandled errors in a
    TaskGroup"), and that message is what the agent would otherwise be told the
    call failed for. A group carrying more than one failure is left alone:
    picking one of several would hide the rest.
    """
    while True:
        # Recognised by what it carries rather than by ``BaseExceptionGroup``,
        # which is a 3.11 builtin: on 3.10 the group is the ``exceptiongroup``
        # backport's class instead, and the name is not there to test against.
        members = getattr(error, "exceptions", None)
        if not isinstance(members, tuple) or len(members) != 1 or not isinstance(members[0], Exception):
            return error
        error = members[0]


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
                    media_type=p.mime_type,
                    kind=BinaryType.IMAGE,
                )
            )
        elif isinstance(p, AudioContent):
            inputs.append(
                BinaryInput(
                    data=base64.b64decode(p.data),
                    media_type=p.mime_type,
                    kind=BinaryType.AUDIO,
                )
            )
        elif isinstance(p, ResourceLink):
            inputs.append(UrlInput(url=str(p.uri), kind=_kind_from_mime(p.mime_type)))
        elif isinstance(p, EmbeddedResource):
            resource = p.resource
            if isinstance(resource, TextResourceContents):
                inputs.append(TextInput(content=resource.text))
            else:
                inputs.append(
                    BinaryInput(
                        data=base64.b64decode(resource.blob),
                        media_type=resource.mime_type or "application/octet-stream",
                        kind=_kind_from_mime(resource.mime_type),
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
