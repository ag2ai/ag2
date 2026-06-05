# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from mcp.server.auth.middleware.auth_context import AuthContextMiddleware
from mcp.server.auth.middleware.bearer_auth import BearerAuthBackend, RequireAuthMiddleware
from mcp.server.auth.routes import build_resource_metadata_url, create_protected_resource_routes
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import CallToolResult, ContentBlock
from mcp.types import Tool as MCPTool
from starlette.applications import Starlette
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.routing import BaseRoute, Mount, Route

from autogen.beta.agent import Agent

from .authserver import authorization_server_routes
from .executor import AgentExecutor, ContextProvider
from .info import build_server_info
from .security import Requirement

if TYPE_CHECKING:
    from starlette.types import Receive, Scope, Send


class MCPServer:
    """Wrap an AG2 :class:`Agent` as an MCP server.

    The agent is exposed as a single conversational tool (``ask`` by default)
    that runs :meth:`Agent.ask` and returns the reply — the inverse of the
    consume-side toolkit ``autogen.beta.tools.toolkits.mcp_server.MCPServer``,
    which connects *to* an MCP server. The two classes share a name but live in
    different modules; alias one if you import both.

    Transport-agnostic state (the underlying low-level ``mcp`` server, the
    agent, the executor) lives on the instance; transport selection happens via
    the builders:

    * :meth:`build_streamable_http` returns a Starlette ASGI app for remote /
      production serving (run it with ``uvicorn``; attach CORS / auth to the
      returned app yourself).
    * :meth:`run_stdio` serves over stdin/stdout for local clients (Claude
      Desktop, Cursor, the MCP Inspector).
    """

    __slots__ = ("_agent", "_executor", "_server", "_name", "_version", "_instructions")

    def __init__(
        self,
        agent: Agent,
        *,
        name: str | None = None,
        version: str | None = None,
        instructions: str | None = None,
        tool_name: str = "ask",
        tool_description: str | None = None,
        stream_progress: bool = True,
        context_provider: "ContextProvider | None" = None,
    ) -> None:
        self._agent = agent
        self._name, self._version, self._instructions = build_server_info(
            agent, name=name, version=version, instructions=instructions
        )
        self._executor = AgentExecutor(
            agent,
            tool_name=tool_name,
            tool_description=tool_description,
            stream_progress=stream_progress,
            context_provider=context_provider,
        )
        self._server = self._build_server()

    @property
    def agent(self) -> Agent:
        return self._agent

    @property
    def server(self) -> Server:
        """The underlying low-level ``mcp`` server (for advanced wiring / tests)."""
        return self._server

    def _build_server(self) -> Server:
        server: Server = Server(name=self._name, version=self._version, instructions=self._instructions)
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

        return server

    def build_streamable_http(
        self,
        *,
        path: str = "/mcp",
        stateless: bool = False,
        json_response: bool = False,
        security: Requirement | None = None,
    ) -> Starlette:
        """Return a Starlette ASGI app serving MCP over streamable HTTP at ``path``.

        When ``security`` is given (build it with
        :func:`autogen.beta.mcp.security.require`), the app additionally serves
        RFC 9728 Protected Resource Metadata at ``/.well-known/oauth-protected-resource``
        and requires a valid bearer token on ``path`` — missing/invalid tokens get
        ``401`` (with a ``WWW-Authenticate`` header pointing at the metadata),
        insufficient scopes get ``403``. ``security.resource_url`` must point at this
        endpoint (its path component must equal ``path``).

        To embed MCP in an existing ASGI app instead of running this standalone,
        use :meth:`mount_into` (it wires the lifespan for you).
        """
        routes, manager = self._streamable_routes(
            path=path, stateless=stateless, json_response=json_response, security=security
        )

        @asynccontextmanager
        async def lifespan(_: Starlette) -> AsyncIterator[None]:
            async with manager.run():
                yield

        return Starlette(routes=routes, lifespan=lifespan)

    def mount_into(
        self,
        app: Starlette,
        *,
        path: str = "/mcp",
        stateless: bool = False,
        json_response: bool = False,
        security: Requirement | None = None,
    ) -> Starlette:
        """Mount this MCP server into an existing Starlette/FastAPI ``app``.

        Adds the MCP routes to ``app`` (the endpoint at ``path``, plus the RFC 9728
        ``/.well-known/oauth-protected-resource`` route when ``security`` is set) and
        composes the streamable-HTTP session-manager lifespan into ``app``'s lifespan,
        so you don't have to wire it yourself — without it, requests fail with
        "Task group is not initialized". Any existing lifespan is preserved.

        Routes are added at the app root, so ``path`` and the host-root
        ``.well-known`` route land exactly where MCP clients expect them. Bearer auth
        (when ``security`` is set) is scoped to the MCP route only, so the rest of
        ``app`` is unaffected. Call this during setup, before the app starts serving.
        """
        routes, manager = self._streamable_routes(
            path=path, stateless=stateless, json_response=json_response, security=security
        )
        app.router.routes.extend(routes)

        previous_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(scope_app: Any) -> AsyncIterator[Any]:
            async with manager.run(), previous_lifespan(scope_app) as state:
                yield state

        app.router.lifespan_context = lifespan
        return app

    def _streamable_routes(
        self,
        *,
        path: str,
        stateless: bool,
        json_response: bool,
        security: Requirement | None,
    ) -> "tuple[list[BaseRoute], StreamableHTTPSessionManager]":
        """Build the streamable-HTTP routes + session manager shared by
        :meth:`build_streamable_http` and :meth:`mount_into`.

        Bearer auth is wrapped *around the MCP route* (not as app-level middleware)
        so it stays scoped when the route is injected into a host app.
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
        # AS facade: also serve our own authorization-server metadata when the
        # requirement opts in (e.g. the IdP's discovery omits the DCR endpoint).
        if security.authorization_server is not None:
            routes.extend(authorization_server_routes(security.authorization_server))
        return routes, manager

    async def run_stdio(self) -> None:
        """Serve the agent over stdio until the client disconnects."""
        async with stdio_server() as (read_stream, write_stream):
            await self._server.run(
                read_stream,
                write_stream,
                self._server.create_initialization_options(),
            )
