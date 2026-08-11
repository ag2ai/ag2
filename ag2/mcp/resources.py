# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import re
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.types import Resource as MCPResource
from mcp.types import ResourceTemplate as MCPResourceTemplate
from pydantic import AnyUrl

from ._async import call_user_fn
from .errors import MCPResourceNotFoundError

if TYPE_CHECKING:
    from mcp.server.lowlevel import Server

# Resource bodies are either text (``str``) or binary (``bytes``); the reader may
# be sync or async. A template reader additionally receives the variables matched
# out of the request URI.
ResourceContent = str | bytes
ReadFn = Callable[[], Awaitable[ResourceContent] | ResourceContent]
TemplateReadFn = Callable[[dict[str, str]], Awaitable[ResourceContent] | ResourceContent]


@dataclass(frozen=True, slots=True)
class Resource:
    """A static MCP resource exposed at a fixed ``uri``.

    ``read`` returns the body (``str`` → text, ``bytes`` → binary) and may be sync
    or async. ``mime_type`` defaults per the MCP SDK (``text/plain`` for text,
    ``application/octet-stream`` for bytes) when left ``None``.
    """

    uri: str
    name: str
    read: ReadFn
    description: str | None = None
    mime_type: str | None = None


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

    def register(self, server: "Server") -> None:
        provider = self

        # ``mcp``'s low-level decorators are untyped; ignore the resulting noise.
        @server.list_resources()  # type: ignore[no-untyped-call, misc]
        async def _list_resources() -> list[MCPResource]:
            return [_to_mcp_resource(r) for r in provider._resources]

        @server.read_resource()  # type: ignore[no-untyped-call, misc]
        async def _read_resource(uri: AnyUrl) -> Iterable[ReadResourceContents]:
            return await provider.read(str(uri))

        if self._templates:

            @server.list_resource_templates()  # type: ignore[no-untyped-call, misc]
            async def _list_resource_templates() -> list[MCPResourceTemplate]:
                return [_to_mcp_template(t) for t in provider._templates]

    async def read(self, uri: str) -> list[ReadResourceContents]:
        resource = self._by_uri.get(uri)
        if resource is not None:
            data = await call_user_fn(resource.read)
            return [ReadResourceContents(content=data, mime_type=resource.mime_type)]
        for pattern, template in self._compiled:
            match = pattern.match(uri)
            if match is not None:
                data = await call_user_fn(template.read, match.groupdict())
                return [ReadResourceContents(content=data, mime_type=template.mime_type)]
        raise MCPResourceNotFoundError(uri)


def _to_mcp_resource(resource: Resource) -> MCPResource:
    return MCPResource(
        uri=AnyUrl(resource.uri),
        name=resource.name,
        description=resource.description,
        mimeType=resource.mime_type,
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
