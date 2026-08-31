# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata
import logging
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from mcp.server.auth.middleware.auth_context import AuthContextMiddleware
from mcp.server.auth.middleware.bearer_auth import BearerAuthBackend, RequireAuthMiddleware
from mcp.server.auth.routes import build_resource_metadata_url, create_protected_resource_routes
from mcp.server.caching import CacheHint, CacheableMethod
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import CallToolRequestParams, CallToolResult, Icon, ListToolsResult, PaginatedRequestParams
from starlette.applications import Starlette
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.routing import BaseRoute, Mount, Route

from ag2.agent import Agent
from ag2.history import MemoryStorage

from .errors import MCPToolNameConflictError
from .executor import AgentExecutor, ContextProvider
from .mappers import input_validation_error, tool_error
from .prompts import Prompt, PromptProvider
from .resources import Resource, ResourceProvider, ResourceTemplate
from .security import Requirement
from .sessions import SessionConfig, SessionStore
from .tools import MCPFunctionTool, ToolProvider

if TYPE_CHECKING:
    from mcp.server.context import ServerRequestContext
    from starlette.types import Lifespan, Receive, Scope, Send

logger = logging.getLogger(__name__)

# An MCP ``Server`` lifespan: an async context manager yielding server-scoped
# state, reachable in every ``tools/call`` via ``request_context.lifespan_context``.
ServerLifespan = Callable[[Server], AbstractAsyncContextManager[Any]]

_DEFAULT_VERSION = "0.0.0"


def _package_version() -> str:
    try:
        return importlib.metadata.version("ag2")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - ag2 always installed in practice
        return _DEFAULT_VERSION


def _build_session_store(sessions: "bool | SessionConfig") -> SessionStore | None:
    if sessions is False:
        return None
    cfg = sessions if isinstance(sessions, SessionConfig) else SessionConfig()
    return SessionStore(
        max_sessions=cfg.max_sessions,
        ttl=cfg.ttl,
        storage=cfg.storage or MemoryStorage(),
    )


def _session_manager_lifespan(manager: StreamableHTTPSessionManager) -> "Lifespan[Any]":
    """An ASGI lifespan that runs the streamable-HTTP session manager.

    ``StreamableHTTPSessionManager`` must be entered via ``manager.run()`` before
    it can serve requests; this wires that into the app's lifespan so a standalone
    ``uvicorn`` run (which drives lifespan automatically) just works.
    """

    @asynccontextmanager
    async def lifespan(_: Starlette) -> AsyncGenerator[None]:
        async with manager.run():
            yield

    return lifespan


class MCPServer:
    """Wrap an AG2 :class:`Agent` as an MCP server.

    The agent is exposed as a single conversational tool (``ask`` by default)
    that runs :meth:`Agent.ask` and returns the reply — the inverse of the
    consume-side toolkit ``ag2.tools.MCPToolkit``, which connects *to*
    an MCP server.

    The instance is itself an ASGI3 application: it serves MCP over streamable
    HTTP and manages its own lifespan, so a standalone ``uvicorn`` run just works::

        app = MCPServer(agent, path="/mcp")
        uvicorn.run(app, host="127.0.0.1", port=8000)

    For local clients (Claude Desktop, Cursor, the MCP Inspector), :meth:`run_stdio`
    serves over stdin/stdout instead. The HTTP transport parameters (``path``,
    ``stateless``, ``json_response``, ``security``) are ignored over stdio.

    ``name`` / ``version`` / ``title`` / ``description`` / ``instructions`` /
    ``website_url`` / ``icons`` populate the ``initialize`` handshake's
    ``serverInfo`` + ``instructions``. ``instructions`` is client-facing "how to
    use this server" guidance — it is *not* derived from the agent's system
    prompt (which is internal); pass it explicitly when you want to advertise
    usage hints. ``title`` and ``description`` are likewise presentation-only
    and never derived from the agent.

    ``cache_hints`` fills ``ttlMs`` / ``cacheScope`` freshness hints on results
    of the cacheable methods (SEP-2549). The served tool set is fixed at
    construction, so ``{"tools/list": CacheHint(ttl_ms=...)}`` is always sound
    here; the same goes for ``resources`` / ``prompts``, which cannot change
    after init. Only protocol revision 2026-07-28 clients see the hints — older
    revisions drop the fields at serialization.

    ``sessions`` controls multi-turn history. By default (``True``) a
    conversation history accumulates across ``tools/call`` invocations; pass a
    :class:`~ag2.mcp.sessions.SessionConfig` to tune the bound / TTL / backend,
    or ``False`` to make every call stateless. Which conversation a call lands in
    is decided by the protocol era, since each era sanctions a different
    mechanism:

    | a conversation named? | handshake era (up to 2025-11-25)                     | modern era (2026-07-28) |
    |-----------------------|------------------------------------------------------|-------------------------|
    | yes                   | that conversation                                    | that conversation       |
    | no                    | the MCP session's own history (per-process on stdio)  | a fresh conversation    |

    The modern era has no MCP session and forbids deriving context from
    connection or process identity, so a caller there continues a conversation
    only by naming it. The name is an opaque handle the server mints and returns
    — in a text content block and in the result's ``_meta`` under
    ``ai.ag2/conversation`` — and never one the caller chooses. A handle the
    server does not recognise is a tool-level error, not a fresh conversation;
    under ``sessions=False`` — where the argument is not advertised and no handle
    is ever minted — presenting one is likewise refused rather than dropped.

    A conversation is bound to the principal that created it (the access token's
    subject, falling back to its client id) and that binding is revalidated on
    every call, so a leaked handle does not expose one caller's history to
    another. **With no** ``security`` **configured there is no principal to bind
    to, and the handle is then the only credential for the conversation it
    names** — it travels through readable content, so treat it as one.

    ``stateless`` governs the *handshake* era only: it stops the HTTP transport
    issuing an ``mcp-session-id``, so handshake-era calls have no session to key
    on and start fresh. Modern-era requests are single exchanges that never carry
    a session id in the first place, so the flag does not reach them. Pairing
    ``stateless=True`` with ``sessions=True`` is a valid configuration — no
    transport session, conversations named explicitly.

    ``resources`` / ``resource_templates`` / ``prompts`` expose MCP resources and
    prompts alongside the conversational tool; the corresponding capability is
    advertised only when a non-empty collection is supplied.
    """

    __slots__ = (
        "_agent",
        "_executor",
        "_server",
        "_name",
        "_version",
        "_title",
        "_description",
        "_instructions",
        "_website_url",
        "_icons",
        "_cache_hints",
        "_lifespan",
        "_session_store",
        "_resource_provider",
        "_prompt_provider",
        "_tool_provider",
        "_http",
    )

    def __init__(
        self,
        agent: Agent,
        *,
        name: str | None = None,
        version: str | None = None,
        title: str | None = None,
        description: str | None = None,
        instructions: str | None = None,
        website_url: str | None = None,
        icons: list[Icon] | None = None,
        cache_hints: Mapping[CacheableMethod, CacheHint] | None = None,
        tool_name: str = "ask",
        tool_description: str | None = None,
        stream_progress: bool = True,
        context_provider: "ContextProvider | None" = None,
        lifespan: "ServerLifespan | None" = None,
        sessions: "bool | SessionConfig" = True,
        resources: "Sequence[Resource]" = (),
        resource_templates: "Sequence[ResourceTemplate]" = (),
        prompts: "Sequence[Prompt]" = (),
        tools: "Sequence[MCPFunctionTool]" = (),
        path: str = "/mcp",
        stateless: bool = False,
        json_response: bool = False,
        security: Requirement | None = None,
    ) -> None:
        self._agent = agent
        self._name = name or agent.name
        self._version = version or _package_version()
        self._title = title
        self._description = description
        self._instructions = instructions
        self._website_url = website_url
        self._icons = icons
        self._cache_hints = cache_hints
        self._lifespan = lifespan
        self._session_store = _build_session_store(sessions)
        self._resource_provider = (
            ResourceProvider(resources, resource_templates) if (resources or resource_templates) else None
        )
        self._prompt_provider = PromptProvider(prompts) if prompts else None
        if tools:
            seen: set[str] = set()
            for tool in tools:
                if tool.name == tool_name:
                    raise MCPToolNameConflictError(tool.name)
                if tool.name in seen:
                    raise MCPToolNameConflictError(tool.name, reserved=False)
                seen.add(tool.name)
        self._tool_provider = ToolProvider(tools) if tools else None
        self._executor = AgentExecutor(
            agent,
            tool_name=tool_name,
            tool_description=tool_description,
            stream_progress=stream_progress,
            context_provider=context_provider,
            session_store=self._session_store,
        )
        self._server = self._build_server()
        routes, manager = self._streamable_routes(
            path=path, stateless=stateless, json_response=json_response, security=security
        )
        self._http: Starlette = Starlette(routes=routes, lifespan=_session_manager_lifespan(manager))

    @property
    def agent(self) -> Agent:
        return self._agent

    @property
    def server(self) -> Server:
        """The underlying low-level ``mcp`` server (for advanced wiring / tests)."""
        return self._server

    def _build_server(self) -> Server:
        """Build the low-level server, wiring every handler as a constructor callback.

        ``mcp`` 2.0 removed the decorator registration API; handlers now take a
        request context plus typed params and return a complete result model. A
        capability is advertised from the handlers actually registered, so the
        optional providers contribute their callbacks only when present.
        """
        kwargs: dict[str, Any] = {}
        if self._lifespan is not None:
            kwargs["lifespan"] = self._lifespan
        if self._resource_provider is not None:
            kwargs["on_list_resources"] = self._resource_provider.on_list_resources
            kwargs["on_read_resource"] = self._resource_provider.on_read_resource
            if self._resource_provider.has_templates:
                kwargs["on_list_resource_templates"] = self._resource_provider.on_list_resource_templates
        if self._prompt_provider is not None:
            kwargs["on_list_prompts"] = self._prompt_provider.on_list_prompts
            kwargs["on_get_prompt"] = self._prompt_provider.on_get_prompt
        return Server(
            name=self._name,
            version=self._version,
            title=self._title,
            description=self._description,
            instructions=self._instructions,
            website_url=self._website_url,
            icons=self._icons,
            cache_hints=self._cache_hints,
            on_list_tools=self._on_list_tools,
            on_call_tool=self._on_call_tool,
            **kwargs,
        )

    async def _on_list_tools(
        self, ctx: "ServerRequestContext[Any, Any]", params: PaginatedRequestParams | None
    ) -> ListToolsResult:
        tools = self._executor.list_tools()
        if self._tool_provider is not None:
            tools += self._tool_provider.list_mcp_tools()
        return ListToolsResult(tools=tools)

    async def _on_call_tool(
        self, ctx: "ServerRequestContext[Any, Any]", params: CallToolRequestParams
    ) -> CallToolResult:
        arguments = params.arguments or {}
        # A handler that raises would surface as a JSON-RPC error in mcp 2.0, where
        # 1.x's decorator turned it into a tool-level error result; keep the latter.
        # Validation sits inside that guard for the same reason: 1.x validated from
        # within the decorator, so even a malformed schema stayed a *tool* error.
        try:
            # 1.x's decorator validated arguments against the advertised schema
            # before dispatching; 2.0 validates nothing, so an unchecked argument
            # would reach the handler and surface as whatever it happened to raise.
            schema = self._advertised_input_schema(params.name)
            if schema is not None and (invalid := input_validation_error(arguments, schema)) is not None:
                return tool_error(invalid)
            # Custom tools run their handler directly; everything else is the
            # agent's conversational tool (name collisions are rejected at init).
            if self._tool_provider is not None and self._tool_provider.has(params.name):
                return CallToolResult(content=await self._tool_provider.call(params.name, arguments, ctx))
            return await self._executor.call(
                params.name,
                message=arguments.get("message", ""),
                context=arguments.get("context"),
                conversation=arguments.get("conversation"),
                request_context=ctx,
            )
        except Exception as e:
            # The wire carries the message only, so without this the stack is lost.
            logger.exception("MCP tools/call %r failed", params.name)
            # ``str`` is empty for a bare ``raise SomeError``; the class name is the
            # least a client can act on.
            return tool_error(str(e) or type(e).__name__)

    def _advertised_input_schema(self, name: str) -> dict[str, Any] | None:
        """The ``inputSchema`` ``tools/list`` advertises for ``name``, if any.

        ``None`` for a name nobody advertises, leaving the unknown-tool error to
        the dispatcher that already words it.
        """
        if self._tool_provider is not None and self._tool_provider.has(name):
            return self._tool_provider.input_schema(name)
        for tool in self._executor.list_tools():
            if tool.name == name:
                return tool.input_schema
        return None

    async def __call__(self, scope: "Scope", receive: "Receive", send: "Send") -> None:
        """ASGI3 entrypoint serving MCP over streamable HTTP.

        Handles the ``lifespan`` scope (running the streamable-HTTP session
        manager) and the ``http`` scope (MCP requests, bearer auth, and — when
        ``security`` is set — RFC 9728 Protected Resource Metadata at
        ``/.well-known/oauth-protected-resource``). Run it standalone::

            uvicorn.run(MCPServer(agent, path="/mcp"), host="127.0.0.1", port=8000)

        When ``security`` is given (build it with
        :func:`ag2.mcp.security.require`), missing/invalid tokens get
        ``401`` (with a ``WWW-Authenticate`` header pointing at the metadata) and
        insufficient scopes get ``403``. ``security.resource_url`` must point at
        this endpoint (its path component must equal ``path``).
        """
        await self._http(scope, receive, send)

    def _streamable_routes(
        self,
        *,
        path: str,
        stateless: bool,
        json_response: bool,
        security: Requirement | None,
    ) -> "tuple[list[BaseRoute], StreamableHTTPSessionManager]":
        """Build the streamable-HTTP routes + session manager for the ASGI app.

        Bearer auth is wrapped *around the MCP route* (not as app-level middleware)
        so it stays scoped if the route is mounted into a host app.
        """
        manager = StreamableHTTPSessionManager(
            app=self._server,
            stateless=stateless,
            json_response=json_response,
        )

        async def handle(scope: "Scope", receive: "Receive", send: "Send") -> None:
            await manager.handle_request(scope, receive, send)

        if security is None:
            return [Mount(path, app=handle)], manager

        metadata = security.to_metadata()
        resource_path = urlparse(str(metadata.resource)).path or "/"
        if resource_path.rstrip("/") != path.rstrip("/"):
            raise ValueError(
                f"security.resource_url path ({resource_path!r}) must match the MCP endpoint path ({path!r})."
            )
        guarded = AuthenticationMiddleware(
            AuthContextMiddleware(
                RequireAuthMiddleware(
                    handle,
                    list(security.required_scopes),
                    build_resource_metadata_url(metadata.resource),
                ),
            ),
            backend=BearerAuthBackend(security.verifier),
        )
        routes: list[BaseRoute] = [
            Route(path, endpoint=guarded),
            *create_protected_resource_routes(
                resource_url=metadata.resource,
                authorization_servers=metadata.authorization_servers,
                scopes_supported=metadata.scopes_supported,
                resource_name=metadata.resource_name,
                resource_documentation=metadata.resource_documentation,
            ),
        ]
        return routes, manager

    async def run_stdio(self) -> None:  # pragma: no cover - needs real stdio pipes
        """Serve the agent over stdio until the client disconnects."""
        async with stdio_server() as (read_stream, write_stream):
            await self._server.run(
                read_stream,
                write_stream,
                self._server.create_initialization_options(),
            )
