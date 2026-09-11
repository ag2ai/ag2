# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata
import logging
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from functools import partial
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from mcp.server.auth.middleware.auth_context import AuthContextMiddleware, get_access_token
from mcp.server.auth.middleware.bearer_auth import BearerAuthBackend, RequireAuthMiddleware
from mcp.server.auth.routes import build_resource_metadata_url, create_protected_resource_routes
from mcp.server.caching import CacheHint, CacheableMethod
from mcp.server.lowlevel import Server
from mcp.server.request_state import RequestStateBoundary, RequestStateSecurity
from mcp.server.stdio import stdio_server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.shared.exceptions import MCPError
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    Icon,
    InputRequiredResult,
    InputResponseRequestParams,
    ListToolsResult,
    PaginatedRequestParams,
)
from starlette.applications import Starlette
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.routing import BaseRoute, Mount, Route

from ag2.agent import Agent
from ag2.history import MemoryStorage
from ag2.hitl import ElicitationPolicy

from .apps import EXTENSION_ID, MCPApp, binds_ui, client_supports_apps, collect_apps, visible_meta
from .errors import MCPToolNameConflictError
from .executor import AgentExecutor, ContextProvider
from .extensions import ExtensionMap, validated_extensions
from .mappers import input_validation_error, tool_error
from .pause import PausedRuns
from .prompts import Prompt, PromptProvider
from .resources import Resource, ResourceProvider, ResourceTemplate
from .security import Requirement
from .sessions import SessionConfig, SessionStore
from .tools import MCP_REQUEST_CONTEXT_DEP, MCPExecutionContext, MCPFunctionTool, MetaFilter, ToolProvider

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


def _ui_binding_filter(ctx: "ServerRequestContext[Any, Any]") -> MetaFilter:
    """A ``_meta`` filter withholding the MCP Apps binding from a client that cannot render it.

    Applied to every custom tool, whether it came from ``apps=`` or was
    registered by hand, so the two routes describe the same server. It is a
    no-op on a tool whose ``_meta`` has no ``ui`` key, which is most of them.
    """
    return partial(visible_meta, supports_apps=client_supports_apps(ctx))


def _session_manager_lifespan(manager: StreamableHTTPSessionManager, paused_runs: PausedRuns) -> "Lifespan[Any]":
    """An ASGI lifespan that runs the streamable-HTTP session manager.

    ``StreamableHTTPSessionManager`` must be entered via ``manager.run()`` before
    it can serve requests, so a standalone ``uvicorn`` run just works. Shutdown
    is also where the paused runs are reclaimed.
    """

    @asynccontextmanager
    async def lifespan(_: Starlette) -> AsyncGenerator[None]:
        try:
            async with manager.run():
                yield
        finally:
            paused_runs.reclaim_all()

    return lifespan


class MCPServer:
    """Wrap an AG2 :class:`Agent` as an MCP server.

    The agent is exposed as a single conversational tool (``ask`` by default) that
    runs :meth:`Agent.ask` and returns the reply — the inverse of the consume-side
    ``ag2.tools.MCPToolkit``. The instance is itself an ASGI3 application serving
    MCP over streamable HTTP; :meth:`run_stdio` serves over stdin/stdout instead,
    where the HTTP options are ignored.

    The full guide is ``website/docs/user-guide/tools/serving_mcp.mdx``.

    Args:
        agent: The agent to serve.
        name: Server name in the ``initialize`` handshake. Defaults to the agent's.
        version: Server version in the handshake. Defaults to the installed ag2's.
        title: Human-readable server name for hosts that show one.
        description: What this server is, for the handshake.
        instructions: Client-facing guidance on using this server. Not the agent's
            system prompt.
        website_url: A page about this server, for the handshake.
        icons: Icons a host may display for this server.
        cache_hints: ``ttlMs`` / ``cacheScope`` freshness hints (SEP-2549) per
            cacheable method. Only revision 2026-07-28 clients see them.
        tool_name: The conversational tool's name.
        tool_description: The conversational tool's description.
        stream_progress: Forward the agent's stream events to the client as
            progress notifications and log messages.
        context_provider: Build the agent's :class:`~ag2.context.ConversationContext`
            for each call yourself.
        lifespan: An ``mcp`` server lifespan whose yielded state each call reaches
            through ``request_context.lifespan_context``.
        sessions: Multi-turn history. ``True`` keeps one per conversation, a
            :class:`~ag2.mcp.sessions.SessionConfig` tunes the bound, TTL and
            backend, ``False`` makes every call stateless.
        elicitation_policy: Whether the served agent's own questions reach the
            human behind the calling client. ``"ask"`` sends them as an MCP
            elicitation; ``"decline"`` falls through to the agent's
            ``hitl_hook``. It does not gate an ``Elicit`` declared in a
            ``tools=`` entry's resolved parameter.
        client_model: Run the served agent's reasoning on the calling client's
            model, so a deployment holding no credentials can still serve an
            agent that needs one. Off by default: the caller then pays for every
            turn and supplies whichever model answers. MCP deprecated sampling
            in revision 2026-07-28, and the SDK warns on every borrowed
            request.
        resources: Resources exposed alongside the tool.
        resource_templates: Resource templates exposed alongside the tool.
        prompts: Prompts exposed alongside the tool.
        tools: Deterministic :func:`mcp_tool` tools served next to the agent's.
        path: The HTTP endpoint path.
        stateless: Stop the HTTP transport issuing an ``mcp-session-id``. A
            handshake-era switch; modern-era requests never carry one.
        json_response: Answer with JSON rather than SSE.
        security: OAuth 2.1 Resource Server requirements. With none configured a
            conversation has no principal, so its handle is the only credential
            for it.
        request_state_security: Advanced. Replaces the ephemeral policy that
            seals the state a paused run is resumed with, and whose TTL bounds
            how long a pause lives.

    Raises:
        MCPToolNameConflictError: A ``tools=`` entry collides with ``tool_name`` or
            with another entry.

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

    ``apps`` serves interactive documents — MCP Apps — alongside the agent: each
    :class:`~ag2.mcp.apps.MCPApp` contributes its tools to ``tools`` and its document
    to ``resources``, so ``apps=[app]`` is exactly shorthand for passing
    ``app.tools`` and ``app.resource`` by hand. Two apps claiming one document URI
    raise here, as a duplicate tool name does. A server holding at least one app
    also advertises ``io.modelcontextprotocol/ui`` in ``extensions``, and withholds
    each UI binding from a client that did not advertise it — see
    :mod:`ag2.mcp.apps`.

    ``extensions`` advertises SEP-2133 extension support: a mapping of
    reverse-DNS identifier to that extension's settings, written to
    ``ServerCapabilities.extensions``. Identifiers are validated here, so a
    malformed one fails at construction the way a tool-name conflict does. The
    two directions of SEP-2133 are **not** symmetric, and this one is the weaker:
    the field does not exist in the 2025-11-25 wire schema, so a handshake-era
    client never receives what is advertised here — only a 2026-07-28 client
    does. Reading what a *client* advertised works in both eras; see
    :func:`~ag2.mcp.extensions.client_extension`.
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
        "_paused_runs",
        "_resource_provider",
        "_prompt_provider",
        "_tool_provider",
        "_extensions",
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
        elicitation_policy: ElicitationPolicy = "ask",
        client_model: bool = False,
        resources: "Sequence[Resource]" = (),
        resource_templates: "Sequence[ResourceTemplate]" = (),
        prompts: "Sequence[Prompt]" = (),
        tools: "Sequence[MCPFunctionTool]" = (),
        apps: "Sequence[MCPApp]" = (),
        extensions: "ExtensionMap | None" = None,
        path: str = "/mcp",
        stateless: bool = False,
        json_response: bool = False,
        security: Requirement | None = None,
        request_state_security: RequestStateSecurity | None = None,
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
        # One lifetime, taken from the state token: once it has expired no client
        # can resume, so the run is unreachable. Two numbers could disagree.
        state_security = (
            request_state_security if request_state_security is not None else RequestStateSecurity.ephemeral()
        )
        self._paused_runs = PausedRuns(ttl=state_security.ttl)
        if apps:
            app_tools, app_resources = collect_apps(apps)
            tools = (*tools, *app_tools)
            resources = (*resources, *app_resources)
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
        self._extensions = validated_extensions(extensions) if extensions else {}
        if binds_ui(tools):
            # Read off the tools rather than off ``apps=``, so registering an app's
            # pieces by hand yields the same server. Deliberate decoration: the
            # specification defines only the client direction of SEP-2133 and says
            # nothing about servers advertising at all, and a handshake-era client
            # never receives it. Visible in discovery, inert otherwise. An explicit
            # setting wins.
            self._extensions.setdefault(EXTENSION_ID, {})
        self._executor = AgentExecutor(
            agent,
            tool_name=tool_name,
            tool_description=tool_description,
            stream_progress=stream_progress,
            context_provider=context_provider,
            session_store=self._session_store,
            elicitation_policy=elicitation_policy,
            client_model=client_model,
            paused_runs=self._paused_runs,
        )
        if self._session_store is not None:
            # A run abandoned without either bound elapsing goes with its
            # conversation.
            self._session_store.on_evict = self._paused_runs.discard_conversation
        self._server = self._build_server()
        # The lowlevel tier installs no default policy, so it is installed here
        # and state never leaves this process unsealed.
        self._server.middleware.append(RequestStateBoundary(state_security, default_audience=self._name))
        self._server.extensions.update(self._extensions)
        routes, manager = self._streamable_routes(
            path=path, stateless=stateless, json_response=json_response, security=security
        )
        self._http: Starlette = Starlette(routes=routes, lifespan=_session_manager_lifespan(manager, self._paused_runs))

    @property
    def agent(self) -> Agent:
        """The agent this server serves."""
        return self._agent

    @property
    def server(self) -> Server:
        """The underlying low-level ``mcp`` server (for advanced wiring / tests)."""
        return self._server

    def _build_server(self) -> Server:
        """Build the low-level server, wiring every handler as a constructor callback.

        A capability is advertised from the handlers actually registered, so the
        optional providers contribute their callbacks only when present.
        """
        kwargs: dict[str, Any] = {}
        if self._lifespan is not None:
            kwargs["lifespan"] = self._lifespan
        if self._resource_provider is not None:
            kwargs["on_list_resources"] = self._on_list_resources
            kwargs["on_read_resource"] = self._on_read_resource
            if self._resource_provider.has_templates:
                kwargs["on_list_resource_templates"] = self._on_list_resource_templates
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

    async def _on_list_resources(
        self, ctx: "ServerRequestContext[Any, Any]", params: PaginatedRequestParams | None
    ) -> Any:
        assert self._resource_provider is not None
        return await self._resource_provider.on_list_resources(await self._request_context(ctx), params)

    async def _on_list_resource_templates(
        self, ctx: "ServerRequestContext[Any, Any]", params: PaginatedRequestParams | None
    ) -> Any:
        assert self._resource_provider is not None
        return await self._resource_provider.on_list_resource_templates(await self._request_context(ctx), params)

    async def _on_read_resource(self, ctx: "ServerRequestContext[Any, Any]", params: Any) -> Any:
        assert self._resource_provider is not None
        return await self._resource_provider.on_read_resource(await self._request_context(ctx), params)

    async def _on_list_tools(
        self, ctx: "ServerRequestContext[Any, Any]", params: PaginatedRequestParams | None
    ) -> ListToolsResult:
        tools = self._executor.list_tools()
        if self._tool_provider is not None:
            context = await self._request_context(ctx)
            tools += self._tool_provider.list_mcp_tools(context, _ui_binding_filter(ctx))
        return ListToolsResult(tools=tools)

    async def _on_call_tool(
        self, ctx: "ServerRequestContext[Any, Any]", params: CallToolRequestParams
    ) -> "CallToolResult | InputRequiredResult":
        arguments = params.arguments or {}
        # 2.0 surfaces a raising handler as a JSON-RPC error and validates no
        # arguments; 1.x's decorator did both as *tool* errors. Keep that, which
        # is why validation sits inside the guard rather than ahead of it.
        try:
            schema = self._advertised_input_schema(params.name)
            if schema is not None and (invalid := input_validation_error(arguments, schema)) is not None:
                return tool_error(invalid)
            # Custom tools run their handler directly; everything else is the
            # agent's conversational tool (name collisions are rejected at init).
            if self._tool_provider is not None and self._tool_provider.has(params.name):
                # Such a tool drives its own round trip through the SDK's
                # resolvers, so this round's answers and state go to it rather
                # than to the paused-run registry; the two never share a call.
                outcome = await self._tool_provider.call(
                    params.name,
                    arguments,
                    await self._request_context(ctx),
                    input_round=InputResponseRequestParams(
                        inputResponses=params.input_responses,
                        requestState=params.request_state,
                    ),
                )
                if isinstance(outcome, InputRequiredResult):
                    return outcome
                return outcome
            return await self._executor.call(
                params.name,
                message=arguments.get("message", ""),
                context=arguments.get("context"),
                conversation=arguments.get("conversation"),
                input_responses=params.input_responses,
                request_state=params.request_state,
                request_context=ctx,
            )
        except MCPError:
            # A protocol-level refusal the model cannot reword its way out of, so
            # it must not be flattened into a tool result the way a failure is.
            raise
        except Exception as e:
            # The wire carries the message only, so without this the stack is lost.
            logger.exception("MCP tools/call %r failed", params.name)
            # ``str`` is empty for a bare ``raise SomeError``; the class name is the
            # least a client can act on.
            return tool_error(str(e) or type(e).__name__)

    async def _request_context(self, ctx: "ServerRequestContext[Any, Any]") -> MCPExecutionContext:
        """The request-scoped context a resource read, tool listing or call resolves against.

        Only the two fields of :class:`AskContext` that a non-conversational
        request can use are carried over: ``prompt`` and ``tools`` shape an agent
        turn, and this request is not one. The provider is called per request, so
        the parallel document read and tool call stay independent.
        """
        context = MCPExecutionContext(dependencies={MCP_REQUEST_CONTEXT_DEP: ctx})
        if self._executor.context_provider is None:
            return context
        provided = await self._executor.context_provider(get_access_token())
        if provided.variables is not None:
            context.variables.update(provided.variables)
        if provided.dependencies is not None:
            context.dependencies.update(provided.dependencies)
            context.dependencies[MCP_REQUEST_CONTEXT_DEP] = ctx
        return context

    def _advertised_input_schema(self, name: str) -> dict[str, Any] | None:
        """The ``inputSchema`` ``tools/list`` advertises for ``name``, or ``None``."""
        if self._tool_provider is not None and self._tool_provider.has(name):
            return self._tool_provider.input_schema(name)
        for tool in self._executor.list_tools():
            if tool.name == name:
                return tool.input_schema
        return None

    async def __call__(self, scope: "Scope", receive: "Receive", send: "Send") -> None:
        """ASGI3 entrypoint serving MCP over streamable HTTP.

        Run it standalone::

            uvicorn.run(MCPServer(agent, path="/mcp"), host="127.0.0.1", port=8000)

        With ``security`` set (build it with :func:`ag2.mcp.security.require`),
        missing or invalid tokens get ``401`` and insufficient scopes ``403``,
        and RFC 9728 metadata is served at
        ``/.well-known/oauth-protected-resource``. ``security.resource_url``
        must point at this endpoint (its path must equal ``path``).
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
        """Build the streamable-HTTP routes and session manager for the ASGI app.

        Bearer auth wraps the MCP route rather than the app, so it stays scoped
        if the route is mounted into a host app.
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
        try:
            async with stdio_server() as (read_stream, write_stream):
                await self._server.run(
                    read_stream,
                    write_stream,
                    self._server.create_initialization_options(),
                )
        finally:
            # The HTTP app reclaims these from its lifespan; this transport has
            # none, so it says the same thing here.
            self._paused_runs.reclaim_all()
