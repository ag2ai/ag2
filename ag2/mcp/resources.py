# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.types import (
    BlobResourceContents,
    ListResourceTemplatesResult,
    ListResourcesResult,
    PaginatedRequestParams,
    ReadResourceRequestParams,
    ReadResourceResult,
    TextResourceContents,
)
from mcp.types import Resource as MCPResource
from mcp.types import ResourceTemplate as MCPResourceTemplate

from ag2.annotations import Variable

from ._async import call_user_fn
from .errors import MCPResourceNotFoundError
from .tools import MCPExecutionContext, call_with_context, resolve_context_value

# Resource bodies are either text (``str``) or binary (``bytes``); the reader may
# be sync or async. A template reader additionally receives the variables matched
# out of the request URI.
ResourceContent = str | bytes
ReadFn = Callable[..., Awaitable[ResourceContent] | ResourceContent]
TemplateReadFn = Callable[[dict[str, str]], Awaitable[ResourceContent] | ResourceContent]


@dataclass(frozen=True, slots=True)
class Resource:
    """A static MCP resource exposed at a fixed ``uri``.

    ``read`` returns the body (``str`` → text, ``bytes`` → binary) and may be sync
    or async. ``mime_type`` defaults to ``text/plain`` for text and
    ``application/octet-stream`` for bytes when left ``None``.

    ``meta`` reaches ``_meta`` on both the listing entry and the read result. It
    is a generic passthrough into the protocol's extension slot — ag2 puts
    nothing of its own there — so an extension's keys (``ui`` for MCP Apps, say)
    go in verbatim. An empty mapping puts no ``_meta`` on the wire at all.

    ``title`` is the display name a client shows in ``resources/list``; ``name``
    stays the identifier. It reaches the listing entry only — the wire's read
    result has no such field — so it is visible only on a listed resource.

    ``listed`` keeps the resource out of ``resources/list`` when false. It stays
    readable by URI: the listing is a browsing surface, and a body that only
    makes sense to a machine that was told its URI does not belong there.

    ``title``, ``description``, ``mime_type`` and the values inside ``meta`` may
    each be a ``Variable``, resolved per request against the ``AskContext`` the
    server's ``context_provider`` returned.
    """

    uri: str
    name: str
    read: ReadFn
    title: str | Variable | None = None
    description: str | Variable | None = None
    mime_type: str | Variable | None = None
    meta: Mapping[str, Any] | None = None
    listed: bool = True


@dataclass(frozen=True, slots=True)
class ResourceTemplate:
    """A dynamic MCP resource addressed by an RFC 6570 URI template.

    Only the simple-string (``{var}``) and reserved (``{+var}``) expansion forms
    are supported: ``{var}`` matches a single path segment, ``{+var}`` matches
    across ``/``. ``read`` receives the matched variables as a ``{name: value}``
    dict and returns the body (sync or async).

    Example::

        ResourceTemplate(
            uri_template="file:///{+path}",
            name="file",
            read=lambda vars: Path(vars["path"]).read_text(),
        )
    """

    uri_template: str
    name: str
    read: TemplateReadFn
    description: str | None = None
    mime_type: str | None = None


_VAR = re.compile(r"\{(\+?)(\w+)\}")


def _compile_template(uri_template: str) -> "re.Pattern[str]":
    """Compile an RFC 6570 ``{var}`` / ``{+var}`` template into a match regex."""
    parts: list[str] = []
    last = 0
    for m in _VAR.finditer(uri_template):
        parts.append(re.escape(uri_template[last : m.start()]))
        reserved, name = m.group(1), m.group(2)
        # {+var} (reserved expansion) may span '/'; plain {var} is one segment.
        parts.append(f"(?P<{name}>.+)" if reserved else f"(?P<{name}>[^/]+)")
        last = m.end()
    parts.append(re.escape(uri_template[last:]))
    return re.compile("^" + "".join(parts) + "$")


class ResourceProvider:
    """Serves a fixed set of :class:`Resource` / :class:`ResourceTemplate` over MCP."""

    __slots__ = ("_resources", "_templates", "_by_uri", "_compiled")

    def __init__(self, resources: Sequence[Resource], templates: Sequence[ResourceTemplate]) -> None:
        self._resources = tuple(resources)
        self._templates = tuple(templates)
        self._by_uri = {r.uri: r for r in self._resources}
        self._compiled = [(_compile_template(t.uri_template), t) for t in self._templates]

    @property
    def has_templates(self) -> bool:
        """Whether to register a ``resources/templates/list`` handler at all.

        The capability is advertised from the registered handlers, so a server
        with only static resources must not carry this one.
        """
        return bool(self._templates)

    async def on_list_resources(
        self, ctx: MCPExecutionContext, params: PaginatedRequestParams | None
    ) -> ListResourcesResult:
        return ListResourcesResult(resources=[_to_mcp_resource(r, ctx) for r in self._resources if r.listed])

    async def on_list_resource_templates(
        self, ctx: MCPExecutionContext, params: PaginatedRequestParams | None
    ) -> ListResourceTemplatesResult:
        return ListResourceTemplatesResult(resourceTemplates=[_to_mcp_template(t) for t in self._templates])

    async def on_read_resource(self, ctx: MCPExecutionContext, params: ReadResourceRequestParams) -> ReadResourceResult:
        contents = await self.read(params.uri, ctx)
        return ReadResourceResult(contents=[_to_wire_contents(params.uri, c) for c in contents])

    async def read(self, uri: str, context: MCPExecutionContext | None = None) -> list[ReadResourceContents]:
        resource = self._by_uri.get(uri)
        if resource is not None:
            data = await _call_resource_read(resource.read, context)
            return [
                ReadResourceContents(
                    content=data,
                    mime_type=resolve_context_value(resource.mime_type, context),
                    meta=_meta(resource.meta, context),
                )
            ]
        for pattern, template in self._compiled:
            match = pattern.match(uri)
            if match is not None:
                data = await call_user_fn(template.read, match.groupdict())
                return [ReadResourceContents(content=data, mime_type=template.mime_type)]
        raise MCPResourceNotFoundError(uri)


async def _call_resource_read(read: ReadFn, context: MCPExecutionContext | None) -> ResourceContent:
    if context is None:
        return await call_user_fn(read)
    return await call_with_context(read, context)


def _to_wire_contents(uri: str, contents: ReadResourceContents) -> TextResourceContents | BlobResourceContents:
    """Map a read body onto the wire contents variant matching its type.

    ``mcp`` 1.x did this inside the ``read_resource`` decorator; the 2.0 handler
    returns a complete result, so the mapping — and its MIME-type defaults, which
    :class:`Resource` documents — moves here.
    """
    meta = _meta(contents.meta)
    if isinstance(contents.content, bytes):
        return BlobResourceContents(
            uri=uri,
            mimeType=contents.mime_type or "application/octet-stream",
            blob=base64.b64encode(contents.content).decode("ascii"),
            _meta=meta,
        )
    return TextResourceContents(
        uri=uri,
        mimeType=contents.mime_type or "text/plain",
        text=contents.content,
        _meta=meta,
    )


def _meta(meta: Mapping[str, Any] | None, context: MCPExecutionContext | None = None) -> dict[str, Any] | None:
    """``meta`` as a wire ``_meta``, or ``None`` so an empty one is never sent."""
    if not meta:
        return None
    resolved = resolve_context_value(meta, context)
    return dict(resolved) if resolved else None


def _to_mcp_resource(resource: Resource, context: MCPExecutionContext | None = None) -> MCPResource:
    return MCPResource(
        uri=resource.uri,
        name=resource.name,
        title=resolve_context_value(resource.title, context),
        description=resolve_context_value(resource.description, context),
        mimeType=resolve_context_value(resource.mime_type, context),
        _meta=_meta(resource.meta, context),
    )


def _to_mcp_template(template: ResourceTemplate) -> MCPResourceTemplate:
    return MCPResourceTemplate(
        uriTemplate=template.uri_template,
        name=template.name,
        description=template.description,
        mimeType=template.mime_type,
    )


__all__ = (
    "Resource",
    "ResourceProvider",
    "ResourceTemplate",
)
