# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Serve an interactive document — an MCP App — alongside an AG2 agent.

An **app** is a document plus the tools that render it::

    shop = MCPApp("ui://shop/card", CARD_HTML)

    @shop.tool
    async def show_item(item_id: str) -> Item:
        \"\"\"Show a product card.\"\"\"
        return await catalog.get(item_id)

    server = MCPServer(agent, apps=[shop])

The constraint that shapes this whole API: a host reads the document **in
parallel with** the call that references it, so the document cannot be built
from the call's arguments. It is a static body registered as a resource; the
per-call data reaches it as ``structuredContent`` on the result, which the host
redelivers into the frame as a ``ui/notifications/tool-result`` notification.
That is the opposite of :mod:`ag2.mcp_ui`, which builds HTML inside the handler
from the arguments it was given. The two are alternatives; this module does not
replace that one.

Nothing here is a dead end. :attr:`MCPApp.tools` are ordinary
:class:`~ag2.mcp.MCPFunctionTool`\\ s, :attr:`MCPApp.resource` is an ordinary
:class:`~ag2.mcp.Resource`, and ``apps=[app]`` is defined as shorthand for
passing those two to ``tools=`` and ``resources=`` by hand.
"""

import json
import os
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, overload

from mcp.server.apps import APP_MIME_TYPE, EXTENSION_ID, ResourceCsp, ResourcePermissions, Visibility
from mcp.server.apps import client_supports_apps as client_supports_apps
from mcp.types import CallToolResult, ToolAnnotations

from ag2.annotations import Variable
from ag2.tools.builtin._resolve import resolve_variable

from ._async import call_user_fn
from .errors import MCPAppFrozenError, MCPAppURIError, MCPDuplicateAppURIError
from .resources import Resource as AG2Resource
from .tools import (
    MCPExecutionContext,
    MCPFunctionTool,
    ToolContext,
    call_with_context,
    mcp_tool,
    resolve_context_value,
    to_call_result,
)

# The key ``_meta`` reserves for MCP Apps, on a tool and on a resource alike.
UI_META_KEY = "ui"

# Where the answering tool's name is stamped on a call result, following the
# existing ``ai.ag2/conversation`` handle key. MCP Apps gives the document no
# correlation identifier of its own — the notification carrying a result names
# neither the call nor the tool — so a document serving two tools would not
# otherwise know which result arrived.
TOOL_META_KEY = "ai.ag2/tool"

# The revision of the View-to-Host dialect the runtime announces in
# ``ui/initialize``, which the specification requires it to state. Restated here
# rather than imported: the Python SDK ships only the server side of MCP Apps
# (``mcp.server.apps`` is tools, resources and the client check), and has no
# constant for the dialect spoken inside the frame.
APP_PROTOCOL_VERSION = "2026-01-26"

# The document content: inline HTML, a path-like file read per request, a
# request-time variable, or a sync/async callable producing HTML per read.
AppContent = str | os.PathLike[str] | Variable | Callable[..., "Awaitable[str] | str"]

# A document field a host displays, which may be resolved per request.
AppText = str | Variable


@dataclass(frozen=True, slots=True)
class AppSandbox:
    """Sandbox policy for an MCP app document."""

    csp: ResourceCsp | Variable | None = None
    permissions: ResourcePermissions | Variable | None = None
    domain: str | Variable | None = None


class MCPApp:
    """A document and the tools that render it.

    ``uri`` must use the ``ui://`` scheme — a host discards anything else — and
    is validated here rather than on the wire.

    ``content`` is the document body, in one of four forms, told apart by type
    alone rather than by inspecting the filesystem:

    * a ``str`` — always literal HTML, never a filename;
    * an ``os.PathLike[str]`` — always a file, read on **every** resource read,
      so a rebuilt bundle is picked up without restarting the server;
    * a ``Variable`` — resolved per request from the ``AskContext`` the server's
      ``context_provider`` returned;
    * a sync or async callable invoked per read, matching
      :class:`~ag2.mcp.Resource`'s reader contract. It may declare ``Variable``,
      ``Depends``/``Inject`` and :data:`~ag2.mcp.MCPRequestContext` parameters,
      which resolve like a tool's.

    An overload per form makes each one visible in an IDE.

    ``name`` identifies the resource and defaults to the URI; ``title`` and
    ``description`` are what a client shows for it in ``resources/list``, so they
    are worth setting only alongside ``listed=True``.

    ``listed`` defaults to false, keeping the document out of ``resources/list``
    — the specification permits omitting UI-only resources, and a person browsing
    a server's resources should not meet a file that is not one. Set it true for
    a document meant to be discoverable.

    ``csp``, ``permissions``, ``domain`` and ``prefers_border`` are the document's
    sandbox policy, written into the resource's ``_meta.ui``; they are the SDK's
    own models, so the wire spelling is never guessed at. ``meta`` is a raw
    passthrough merged alongside, for whatever the specification adds next — and
    here a ``ui`` key in it *merges over* the typed parameters rather than being
    refused, because on a document that slot is shared. On a tool it is not:
    ``@app.tool``'s ``meta`` rejects a ``ui`` key outright, since there the only
    thing in that slot is the binding the decorator itself writes.

    ``inject_runtime`` controls injection of the document runtime (see
    :meth:`runtime_script`). Turn it off for a document that brings its own
    bundle: the body is then served byte-for-byte as given.
    """

    __slots__ = (
        "_uri",
        "_content",
        "_name",
        "_title",
        "_description",
        "_listed",
        "_inject_runtime",
        "_sandbox",
        "_meta",
        "_tools",
        "_frozen",
    )

    @overload
    def __init__(
        self,
        uri: str,
        content: str,
        *,
        name: str | None = None,
        title: AppText | None = None,
        description: AppText | None = None,
        listed: bool = False,
        inject_runtime: bool | Variable = True,
        sandbox: AppSandbox | None = None,
        csp: ResourceCsp | Variable | None = None,
        permissions: ResourcePermissions | Variable | None = None,
        domain: str | Variable | None = None,
        prefers_border: bool | Variable | None = None,
        meta: Mapping[str, Any] | None = None,
        tools: Sequence[Callable[..., Any] | MCPFunctionTool] = (),
    ) -> None: ...

    @overload
    def __init__(
        self,
        uri: str,
        content: os.PathLike[str],
        *,
        name: str | None = None,
        title: AppText | None = None,
        description: AppText | None = None,
        listed: bool = False,
        inject_runtime: bool | Variable = True,
        sandbox: AppSandbox | None = None,
        csp: ResourceCsp | Variable | None = None,
        permissions: ResourcePermissions | Variable | None = None,
        domain: str | Variable | None = None,
        prefers_border: bool | Variable | None = None,
        meta: Mapping[str, Any] | None = None,
        tools: Sequence[Callable[..., Any] | MCPFunctionTool] = (),
    ) -> None: ...

    @overload
    def __init__(
        self,
        uri: str,
        content: Callable[..., Awaitable[str] | str],
        *,
        name: str | None = None,
        title: AppText | None = None,
        description: AppText | None = None,
        listed: bool = False,
        inject_runtime: bool | Variable = True,
        sandbox: AppSandbox | None = None,
        csp: ResourceCsp | Variable | None = None,
        permissions: ResourcePermissions | Variable | None = None,
        domain: str | Variable | None = None,
        prefers_border: bool | Variable | None = None,
        meta: Mapping[str, Any] | None = None,
        tools: Sequence[Callable[..., Any] | MCPFunctionTool] = (),
    ) -> None: ...

    @overload
    def __init__(
        self,
        uri: str,
        content: Variable,
        *,
        name: str | None = None,
        title: AppText | None = None,
        description: AppText | None = None,
        listed: bool = False,
        inject_runtime: bool | Variable = True,
        sandbox: AppSandbox | None = None,
        csp: ResourceCsp | Variable | None = None,
        permissions: ResourcePermissions | Variable | None = None,
        domain: str | Variable | None = None,
        prefers_border: bool | Variable | None = None,
        meta: Mapping[str, Any] | None = None,
        tools: Sequence[Callable[..., Any] | MCPFunctionTool] = (),
    ) -> None: ...

    def __init__(
        self,
        uri: str,
        content: AppContent,
        *,
        name: str | None = None,
        title: AppText | None = None,
        description: AppText | None = None,
        listed: bool = False,
        inject_runtime: bool | Variable = True,
        sandbox: AppSandbox | None = None,
        csp: ResourceCsp | Variable | None = None,
        permissions: ResourcePermissions | Variable | None = None,
        domain: str | Variable | None = None,
        prefers_border: bool | Variable | None = None,
        meta: Mapping[str, Any] | None = None,
        tools: Sequence[Callable[..., Any] | MCPFunctionTool] = (),
    ) -> None:
        if not uri.startswith("ui://"):
            raise MCPAppURIError(uri)
        self._uri = uri
        self._content = content
        self._name = name or uri
        self._title = title
        self._description = description
        self._listed = listed
        self._inject_runtime = inject_runtime
        self._sandbox = sandbox or AppSandbox(csp=csp, permissions=permissions, domain=domain)
        self._meta = _document_meta(
            csp=self._sandbox.csp,
            permissions=self._sandbox.permissions,
            domain=self._sandbox.domain,
            prefers_border=prefers_border,
            extra=meta,
        )
        self._tools: list[MCPFunctionTool] = []
        self._frozen = False
        for tool in tools:
            self.tool(tool)

    @property
    def uri(self) -> str:
        """The ``ui://`` URI the document is served at and the tools point to."""
        return self._uri

    @property
    def tools(self) -> tuple[MCPFunctionTool, ...]:
        """The tools declared on this app, as ordinary tools.

        Passing these to ``MCPServer(tools=...)`` is exactly what ``apps=``
        does with them.
        """
        return tuple(self._tools)

    @property
    def resource(self) -> AG2Resource:
        """The document, as an ordinary resource.

        Passing this to ``MCPServer(resources=...)`` is exactly what ``apps=``
        does with it.
        """
        return AG2Resource(
            uri=self._uri,
            name=self._name,
            read=_AppResourceReader(self),
            title=self._title,
            description=self._description,
            mime_type=APP_MIME_TYPE,
            meta=self._meta or None,
            listed=self._listed,
        )

    async def _read(self, context: MCPExecutionContext) -> str:
        body = await _resolve_content(self._content, context)
        inject_runtime = resolve_context_value(self._inject_runtime, context)
        return _inject(body, self.runtime_script()) if inject_runtime else body

    def runtime_script(self) -> str:
        """The ``<script>`` element performing the MCP Apps handshake, as text.

        A host discards every message from a view that has not completed the
        ``ui/initialize`` exchange, and says nothing about it — so a hand-written
        button on a document without this is silently dead. The script performs
        that exchange and defines a global ``ag2ui`` exposing the View-side
        surface: ``callTool``, ``sendMessage``, ``openLink``, ``readResource``,
        ``updateContext``, ``log``, ``requestDisplayMode``, ``downloadFile`` and
        ``sampling``; the ``onToolInput`` / ``onToolInputPartial`` /
        ``onToolResult`` / ``onHostContextChanged`` / ``onCancelled``
        subscriptions; the negotiated
        ``ag2ui.host.capabilities`` and ``ag2ui.host.context``; an ``ag2ui.ready``
        promise; and automatic size reporting.

        This is the same text injection uses, exposed for a document that would
        rather place it itself (with ``inject_runtime=False``).
        """
        return _runtime_script(self._name)

    @overload
    def tool(
        self,
        function: Callable[..., Any] | MCPFunctionTool,
        *,
        name: str | None = None,
        description: str | None = None,
        title: str | None = None,
        annotations: ToolAnnotations | None = None,
        visibility: Sequence[Visibility] | None = None,
        output_schema: dict[str, Any] | None = None,
        meta: Mapping[str, Any] | None = None,
        sync_to_thread: bool = True,
    ) -> MCPFunctionTool: ...

    @overload
    def tool(
        self,
        function: None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        title: str | None = None,
        annotations: ToolAnnotations | None = None,
        visibility: Sequence[Visibility] | None = None,
        output_schema: dict[str, Any] | None = None,
        meta: Mapping[str, Any] | None = None,
        sync_to_thread: bool = True,
    ) -> Callable[[Callable[..., Any]], MCPFunctionTool]: ...

    def tool(
        self,
        function: Callable[..., Any] | MCPFunctionTool | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        title: str | None = None,
        annotations: ToolAnnotations | None = None,
        visibility: Sequence[Visibility] | None = None,
        output_schema: dict[str, Any] | None = None,
        meta: Mapping[str, Any] | None = None,
        sync_to_thread: bool = True,
    ) -> "MCPFunctionTool | Callable[[Callable[..., Any]], MCPFunctionTool]":
        """Declare a tool that renders this app's document.

        Takes everything :func:`~ag2.mcp.mcp_tool` takes, plus ``visibility``
        (where the host surfaces the tool: ``["model", "app"]``). The tool is an
        ordinary :class:`~ag2.mcp.MCPFunctionTool` carrying the document's URI in
        ``_meta.ui.resourceUri``, and is collected into :attr:`tools`.

        Several tools may be declared on one app — a card that both renders and
        updates itself is one app, not two — and each result is stamped with the
        answering tool's name under :data:`TOOL_META_KEY` so the document can
        route it. The stamp reaches a ``CallToolResult`` the handler assembled
        itself, which is the one respect in which such a result is modified; a
        handler that sets that key keeps its own value.

        Args:
            function: The function (when used as a bare ``@app.tool``).
            name: Tool name. Defaults to the function name.
            description: Tool description. Defaults to the function docstring.
            title: Human-readable display name for ``tools/list``.
            annotations: ``mcp.types.ToolAnnotations`` behavior hints.
            visibility: Where the host surfaces the tool (``_meta.ui.visibility``).
            output_schema: Overrides the schema derived from the return annotation.
            meta: Additional ``_meta`` keys, merged alongside the ``ui`` entry.
            sync_to_thread: Run a sync function in a worker thread.

        Raises:
            ValueError: If ``meta`` carries a ``ui`` key. The binding between a
                tool and its document is this decorator's to write; a caller
                setting it by hand would be overwriting the app's own URI.
            MCPAppFrozenError: If the app is already registered with a server,
                which has read its tools; the tool would never be served.
        """
        if meta and UI_META_KEY in meta:
            raise ValueError(
                f"@MCPApp.tool owns _meta[{UI_META_KEY!r}]; pass visibility= rather than a 'ui' meta key, "
                "or declare the tool with mcp_tool() if it should not be bound to this document."
            )
        if self._frozen:
            raise MCPAppFrozenError(self._uri, name or getattr(function, "__name__", None))
        ui: dict[str, Any] = {"resourceUri": self._uri}
        if visibility is not None:
            ui["visibility"] = list(visibility)
        merged = {**(meta or {}), UI_META_KEY: ui}

        def make(f: Callable[..., Any] | MCPFunctionTool) -> MCPFunctionTool:
            if isinstance(f, MCPFunctionTool):
                if (
                    name is not None
                    or description is not None
                    or output_schema is not None
                    or sync_to_thread is not True
                ):
                    raise ValueError(
                        "Existing MCPFunctionTool objects already define name, description, schema, and execution."
                    )
                if f.meta and UI_META_KEY in f.meta and f.meta[UI_META_KEY].get("resourceUri") != self._uri:
                    raise ValueError("Existing MCPFunctionTool is already bound to a different MCP app document.")
                built = _replace_tool(
                    f,
                    title=title if title is not None else f.title,
                    annotations=annotations if annotations is not None else f.annotations,
                    meta={**dict(f.meta or {}), **merged},
                )
            else:
                built = mcp_tool(
                    f,
                    name=name,
                    description=description,
                    title=title,
                    annotations=annotations,
                    output_schema=output_schema,
                    meta=merged,
                    sync_to_thread=sync_to_thread,
                )
            stamped = _replace_tool(built, handler=_stamping(built.handler, built.name))
            self._tools.append(stamped)
            return stamped

        return make(function) if function is not None else make

    def _freeze(self) -> None:
        """Close the app's tool composition, once a server has read it."""
        self._frozen = True


def _replace_tool(tool: MCPFunctionTool, **changes: Any) -> MCPFunctionTool:
    """Replace public tool fields while retaining private resolver state."""
    built = replace(tool, **changes)
    object.__setattr__(built, "_resolved_params", tool._resolved_params)
    object.__setattr__(built, "_resolver_plans", tool._resolver_plans)
    return built


class _AppResourceReader:
    __slots__ = ("_app",)

    def __init__(self, app: MCPApp) -> None:
        self._app = app

    async def __call__(self, context: MCPExecutionContext) -> str:
        return await self._app._read(context)


def _stamping(handler: Callable[..., Any], tool_name: str) -> Callable[[dict[str, Any], ToolContext], Any]:
    """Wrap ``handler`` so every result names the tool that produced it.

    The stamp is applied at every rung of the result ladder, including a
    ``CallToolResult`` the author assembled themselves — **the one respect in
    which such a result is modified**. It is skipped when the author's own
    ``_meta`` already occupies :data:`TOOL_META_KEY`, so the escape hatch stays
    open, and every other key they set travels alongside it untouched.

    Mapping the ladder here rather than leaving it to
    :meth:`~ag2.mcp.MCPFunctionTool.call` is what makes one wrapper enough: the
    result this returns is already a ``CallToolResult``, which that method
    passes through untouched.
    """

    async def stamp(arguments: dict[str, Any], request_context: ToolContext) -> CallToolResult:
        result = to_call_result(await call_user_fn(handler, arguments, request_context))
        meta = dict(result.meta or {})
        if TOOL_META_KEY in meta:
            return result
        meta[TOOL_META_KEY] = tool_name
        return result.model_copy(update={"meta": meta})

    return stamp


def visible_meta(meta: "Mapping[str, Any] | None", *, supports_apps: bool) -> "Mapping[str, Any] | None":
    """``meta`` as a client that may or may not render apps should see it.

    The UI binding is a promise that the client can read the document and render
    it. To a client that did not advertise the extension it is a promise about a
    document it will never fetch, so it is withheld — correctness rather than
    policy, and hence not switchable. Everything else the author put in ``_meta``
    is unaffected, and the tool still lists and still calls: a UI-bound tool
    returns text alongside its data either way.
    """
    if supports_apps or not meta or UI_META_KEY not in meta:
        return meta
    return {key: value for key, value in meta.items() if key != UI_META_KEY}


def binds_ui(tools: "Iterable[MCPFunctionTool]") -> bool:
    """Whether any of ``tools`` advertises a UI binding.

    Read off the tools themselves rather than off the apps they came from, so a
    server given ``app.tools`` by hand advertises the extension exactly as one
    given ``apps=[app]`` does.
    """
    return any(tool.meta and UI_META_KEY in tool.meta for tool in tools)


def collect_apps(apps: "Iterable[MCPApp]") -> "tuple[tuple[MCPFunctionTool, ...], tuple[AG2Resource, ...]]":
    """The tools and resources ``apps`` contributes, refusing a shared document URI.

    Two apps claiming one URI would silently shadow each other — one document
    would win the registration and the other app's tools would point at a body
    nobody wrote — so it raises where the server is built, as a duplicate tool
    name already does.
    """
    tools: list[MCPFunctionTool] = []
    resources: list[AG2Resource] = []
    seen: set[str] = set()
    for app in apps:
        if app.uri in seen:
            raise MCPDuplicateAppURIError(app.uri)
        seen.add(app.uri)
        app._freeze()
        tools.extend(app.tools)
        resources.append(app.resource)
    return tuple(tools), tuple(resources)


async def _resolve_content(content: AppContent, context: MCPExecutionContext) -> str:
    resolved = resolve_variable(content, context, param_name="content")
    if isinstance(resolved, str):
        return resolved
    if isinstance(resolved, os.PathLike):
        try:
            return Path(resolved).read_text()
        except OSError as e:
            raise ValueError(f"Could not read MCP app document from {resolved!s}: {e.strerror or e}") from e
    if callable(resolved):
        value = await call_with_context(resolved, context)
        if isinstance(value, str):
            return value
        raise TypeError(f"MCP app content provider returned {type(value).__name__}; expected str.")
    raise TypeError(f"MCP app content resolved to {type(resolved).__name__}; expected str, path-like, or callable.")


def _document_meta(
    *,
    csp: "ResourceCsp | Variable | None",
    permissions: "ResourcePermissions | Variable | None",
    domain: str | Variable | None,
    prefers_border: bool | Variable | None,
    extra: "Mapping[str, Any] | None",
) -> dict[str, Any]:
    """The document's ``_meta``: its sandbox policy under ``ui``, plus a passthrough."""
    ui: dict[str, Any] = {}
    if csp is not None:
        ui["csp"] = csp
    if permissions is not None:
        ui["permissions"] = permissions
    if domain is not None:
        ui["domain"] = domain
    if prefers_border is not None:
        ui["prefersBorder"] = prefers_border
    meta: dict[str, Any] = dict(extra or {})
    if ui:
        # The caller's own ``ui`` keys win: the typed parameters are the spelling
        # aid, not the authority on a slot the passthrough exists to reach.
        meta[UI_META_KEY] = {**ui, **meta.get(UI_META_KEY, {})}
    return meta


# An opening tag is its name followed by whitespace or ``>`` — matching on
# ``"<head "`` alone would miss a tag whose attributes start on the next line.
_HEAD_TAG = re.compile(r"<head(?=[\s>])", re.IGNORECASE)
_HTML_TAG = re.compile(r"<html(?=[\s>])", re.IGNORECASE)


def _inject(html: str, script: str) -> str:
    """Place ``script`` at the head of ``html``.

    After ``<head>`` when there is one, after ``<html …>`` when there is not, and
    at the very front for a fragment — which is what a small document usually is.
    The runtime must run before the author's own script, which may call it.
    """
    for pattern in (_HEAD_TAG, _HTML_TAG):
        match = pattern.search(html)
        if match:
            end = html.find(">", match.start()) + 1
            return html[:end] + script + html[end:]
    return script + html


def _js(value: object) -> str:
    """``value`` as a JavaScript literal that cannot close the surrounding script."""
    return json.dumps(value).replace("</", "<\\/")


def _runtime_script(app_name: str) -> str:
    """The document runtime, specialised with the app's name for the handshake.

    Written to the constraints the specification's own CSP imposes: inline
    script is permitted and is how the ecosystem ships, but ``eval`` and dynamic
    function construction are not available, and no network call may be made
    unless the app declared ``connect`` domains — so this makes none. Messages go
    to ``window.parent`` with a ``"*"`` target origin, because a sandboxed view
    has an opaque origin and pinning is not workable; inbound messages are
    filtered on ``event.source`` instead, and anything that is not JSON-RPC is
    ignored.
    """
    return (
        _RUNTIME_TEMPLATE
        .replace("__AG2_APP_NAME__", _js(app_name))
        .replace("__AG2_TOOL_KEY__", _js(TOOL_META_KEY))
        .replace("__AG2_PROTOCOL_VERSION__", _js(APP_PROTOCOL_VERSION))
    )


# ``ui/message`` sends its content as an array. The specification's prose says a
# single object, but its shipped types, its generated schema and the reference
# host's validator all require the array, and the wire is what a host validates.
_RUNTIME_TEMPLATE = """<script>
(function () {
  "use strict";
  if (window.ag2ui) { return; }

  var TOOL_KEY = __AG2_TOOL_KEY__;
  var nextId = 1;
  var pending = {};
  var subscribers = {
    toolInput: [], toolInputPartial: [], toolResult: [], hostContextChanged: [], cancelled: []
  };
  var byTool = {};
  var settle = {};
  var ready = new Promise(function (resolve, reject) { settle.resolve = resolve; settle.reject = reject; });
  // A host that refuses the handshake rejects this promise whether or not the
  // document awaited it; without a handler here that becomes an unhandled
  // rejection. Awaiting ag2ui.ready still throws — this only marks it handled.
  ready.catch(function () {});
  var host = { protocolVersion: null, info: null, capabilities: {}, context: {} };
  var negotiated = false;

  function post(message) { window.parent.postMessage(message, "*"); }

  function request(method, params) {
    var id = nextId++;
    return new Promise(function (resolve, reject) {
      pending[id] = { resolve: resolve, reject: reject };
      post({ jsonrpc: "2.0", id: id, method: method, params: params || {} });
    });
  }

  function notify(method, params) { post({ jsonrpc: "2.0", method: method, params: params || {} }); }

  function needs(capability, method) {
    if (!negotiated) {
      throw new Error("ag2ui." + method + ": await ag2ui.ready before calling the host.");
    }
    if (!host.capabilities || !host.capabilities[capability]) {
      throw new Error(
        "ag2ui." + method + ": the host did not advertise '" + capability + "'. " +
        "A host discards a call it did not advertise without answering, so this is refused here instead."
      );
    }
  }

  function reportError(e) {
    if (window.console && window.console.error) { window.console.error("ag2ui subscriber failed", e); }
  }

  function emit(list, payload) {
    for (var i = 0; i < list.length; i++) {
      try { list[i](payload); } catch (e) { reportError(e); }
    }
  }

  function subscribe(list, fn) {
    list.push(fn);
    return function () {
      var at = list.indexOf(fn);
      if (at !== -1) { list.splice(at, 1); }
    };
  }

  function onNotification(method, params) {
    if (method === "ui/notifications/tool-input") { emit(subscribers.toolInput, params); return; }
    if (method === "ui/notifications/tool-input-partial") {
      emit(subscribers.toolInputPartial, params);
      return;
    }
    if (method === "ui/notifications/tool-result") {
      emit(subscribers.toolResult, params);
      var meta = params ? params._meta : null;
      var name = meta ? meta[TOOL_KEY] : null;
      if (name && byTool[name]) { emit(byTool[name], params); }
      return;
    }
    if (method === "ui/notifications/host-context-changed") {
      var merged = {};
      var key;
      for (key in host.context) { merged[key] = host.context[key]; }
      for (key in (params || {})) { merged[key] = params[key]; }
      host.context = merged;
      emit(subscribers.hostContextChanged, host.context);
      return;
    }
    if (method === "ui/notifications/tool-cancelled") { emit(subscribers.cancelled, params); return; }
  }

  function onRequest(message) {
    // Teardown is answered so the host may proceed; ping because a health check
    // this view refused would read as a view that had stopped responding.
    if (message.method === "ui/resource-teardown" || message.method === "ping") {
      post({ jsonrpc: "2.0", id: message.id, result: {} });
      return;
    }
    post({
      jsonrpc: "2.0",
      id: message.id,
      error: { code: -32601, message: "This view does not serve " + message.method + "." }
    });
  }

  window.addEventListener("message", function (event) {
    if (event.source !== window.parent) { return; }
    var message = event.data;
    if (!message || message.jsonrpc !== "2.0") { return; }
    if (typeof message.method === "string") {
      if (message.id === undefined || message.id === null) { onNotification(message.method, message.params); }
      else { onRequest(message); }
      return;
    }
    var entry = pending[message.id];
    if (!entry) { return; }
    delete pending[message.id];
    if (message.error) { entry.reject(new Error(message.error.message || "The host refused the call.")); }
    else { entry.resolve(message.result); }
  });

  var lastWidth = -1;
  var lastHeight = -1;

  function reportSize() {
    var root = document.documentElement;
    if (!root) { return; }
    var width = Math.ceil(root.scrollWidth);
    var height = Math.ceil(root.scrollHeight);
    if (width === lastWidth && height === lastHeight) { return; }
    lastWidth = width;
    lastHeight = height;
    notify("ui/notifications/size-changed", { width: width, height: height });
  }

  function watchSize() {
    reportSize();
    if (typeof ResizeObserver === "function" && document.documentElement) {
      new ResizeObserver(reportSize).observe(document.documentElement);
    }
  }

  window.ag2ui = {
    ready: ready,
    host: host,
    callTool: function (name, args) {
      needs("serverTools", "callTool");
      return request("tools/call", { name: name, arguments: args || {} });
    },
    sendMessage: function (text) {
      needs("message", "sendMessage");
      return request("ui/message", { role: "user", content: [{ type: "text", text: text }] });
    },
    openLink: function (url) {
      needs("openLinks", "openLink");
      return request("ui/open-link", { url: url });
    },
    readResource: function (uri) {
      needs("serverResources", "readResource");
      return request("resources/read", { uri: uri });
    },
    updateContext: function (payload) {
      needs("updateModelContext", "updateContext");
      return request("ui/update-model-context", payload || {});
    },
    log: function (level, data) {
      needs("logging", "log");
      notify("notifications/message", { level: level, data: data });
    },
    requestDisplayMode: function (mode) {
      var available = host.context ? host.context.availableDisplayModes : null;
      if (!available || available.indexOf(mode) === -1) {
        throw new Error("ag2ui.requestDisplayMode: the host did not offer the '" + mode + "' display mode.");
      }
      return request("ui/request-display-mode", { mode: mode }).then(function (result) {
        // The host answers with the mode it actually applied, which may not be
        // the one asked for. Recording it keeps ag2ui.host.context from reading
        // stale until the host happens to send host-context-changed.
        if (result && result.displayMode) { host.context.displayMode = result.displayMode; }
        return result;
      });
    },
    downloadFile: function (contents) {
      needs("downloadFile", "downloadFile");
      return request("ui/download-file", { contents: contents });
    },
    sampling: function (params) {
      needs("sampling", "sampling");
      return request("sampling/createMessage", params || {});
    },
    onToolInput: function (fn) { return subscribe(subscribers.toolInput, fn); },
    onToolInputPartial: function (fn) { return subscribe(subscribers.toolInputPartial, fn); },
    onToolResult: function (nameOrFn, maybeFn) {
      if (typeof nameOrFn === "function") { return subscribe(subscribers.toolResult, nameOrFn); }
      if (!byTool[nameOrFn]) { byTool[nameOrFn] = []; }
      return subscribe(byTool[nameOrFn], maybeFn);
    },
    onHostContextChanged: function (fn) { return subscribe(subscribers.hostContextChanged, fn); },
    onCancelled: function (fn) { return subscribe(subscribers.cancelled, fn); },
    reportSize: reportSize
  };

  request("ui/initialize", {
    appInfo: { name: __AG2_APP_NAME__, version: "1" },
    // Every display mode, because a document that reports its own size lays
    // out in any of them. Declaring none is not the safe default it looks
    // like: a host may decline a mode the view never claimed.
    appCapabilities: { availableDisplayModes: ["inline", "fullscreen", "pip"] },
    protocolVersion: __AG2_PROTOCOL_VERSION__
  }).then(function (result) {
    host.protocolVersion = result.protocolVersion || null;
    host.info = result.hostInfo || null;
    host.capabilities = result.hostCapabilities || {};
    host.context = result.hostContext || {};
    negotiated = true;
    notify("ui/notifications/initialized", {});
    watchSize();
    settle.resolve(host);
  }, settle.reject);
})();
</script>"""


__all__ = (
    "APP_MIME_TYPE",
    "APP_PROTOCOL_VERSION",
    "EXTENSION_ID",
    "TOOL_META_KEY",
    "UI_META_KEY",
    "AppContent",
    "AppSandbox",
    "MCPApp",
    "ResourceCsp",
    "ResourcePermissions",
    "Visibility",
    "binds_ui",
    "client_supports_apps",
    "collect_apps",
    "visible_meta",
)
