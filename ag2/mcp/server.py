# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from mcp.server.auth.middleware.auth_context import AuthContextMiddleware
from mcp.server.auth.middleware.bearer_auth import BearerAuthBackend, RequireAuthMiddleware
from mcp.server.auth.routes import build_resource_metadata_url, create_protected_resource_routes
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import CallToolResult, ContentBlock, Icon
from mcp.types import Tool as MCPTool
from starlette.applications import Starlette
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.routing import BaseRoute, Mount, Route

from ag2.agent import Agent
from ag2.history import MemoryStorage

from .executor import AgentExecutor, ContextProvider
from .prompts import Prompt, PromptProvider
from .resources import Resource, ResourceProvider, ResourceTemplate
from .security import Requirement
from .sessions import SessionConfig, SessionStore

if TYPE_CHECKING:
    from starlette.types import Lifespan, Receive, Scope, Send

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
    async def lifespan(_: Starlette) -> AsyncIterator[None]:
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

    ``name`` / ``version`` / ``instructions`` / ``website_url`` / ``icons``
    populate the ``initialize`` handshake's ``serverInfo`` + ``instructions``.
    ``instructions`` is client-facing "how to use this server" guidance — it is
    *not* derived from the agent's system prompt (which is internal); pass it
    explicitly when you want to advertise usage hints.

    ``sessions`` controls multi-turn history. By default (``True``) each MCP
    session (keyed by the transport's ``mcp-session-id``, or a per-process key
    over stdio) keeps its own conversation history that accumulates across calls;
    pass a :class:`~ag2.mcp.sessions.SessionConfig` to tune the bound /
    TTL / backend, or ``False`` to make every call stateless. A stateless HTTP
    transport (``stateless=True``) issues no session id, so it stays stateless
    regardless of this setting.

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
        "_instructions",
        "_website_url",
        "_icons",
        "_lifespan",
        "_session_store",
        "_resource_provider",
        "_prompt_provider",
        "_http",
    )

    def __init__(
        self,
        agent: Agent,
        *,
        name: str | None = None,
        version: str | None = None,
        instructions: str | None = None,
        website_url: str | None = None,
        icons: list[Icon] | None = None,
        tool_name: str = "ask",
        tool_description: str | None = None,
        stream_progress: bool = True,
        context_provider: "ContextProvider | None" = None,
        lifespan: "ServerLifespan | None" = None,
        sessions: "bool | SessionConfig" = True,
        resources: "Sequence[Resource]" = (),
        resource_templates: "Sequence[ResourceTemplate]" = (),
        prompts: "Sequence[Prompt]" = (),
        path: str = "/mcp",
        stateless: bool = False,
        json_response: bool = False,
        security: Requirement | None = None,
    ) -> None:
        self._agent = agent
        self._name = name or agent.name
        self._version = version or _package_version()
        self._instructions = instructions
        self._website_url = website_url
        self._icons = icons
        self._lifespan = lifespan
        self._session_store = _build_session_store(sessions)
        self._resource_provider = (
            ResourceProvider(resources, resource_templates) if (resources or resource_templates) else None
        )
        self._prompt_provider = PromptProvider(prompts) if prompts else None
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
        kwargs: dict[str, Any] = {}
        if self._lifespan is not None:
            kwargs["lifespan"] = self._lifespan
        server: Server = Server(
            name=self._name,
            version=self._version,
            instructions=self._instructions,
            website_url=self._website_url,
            icons=self._icons,
            **kwargs,
        )
        executor = self._executor

        # ``mcp``'s low-level decorators are untyped; ignore the resulting noise.
        @server.list_tools()  # type: ignore[no-untyped-call, misc]
        async def _list_tools() -> list[MCPTool]:
            return executor.list_tools()

        @server.call_tool()  # type: ignore[no-untyped-call, misc]
        async def _call_tool(
            name: str, arguments: dict[str, Any]
        ) -> list[ContentBlock] | tuple[list[ContentBlock], dict[str, Any]] | CallToolResult:
            arguments = arguments or {}
            return await executor.call(
                name,
                message=arguments.get("message", ""),
                context=arguments.get("context"),
                request_context=server.request_context,
            )

        if self._resource_provider is not None:
            self._resource_provider.register(server)
        if self._prompt_provider is not None:
            self._prompt_provider.register(server)

        return server

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
