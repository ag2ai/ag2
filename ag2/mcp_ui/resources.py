# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64
from typing import Any, Literal, TypeAlias

from mcp.types import BlobResourceContents, EmbeddedResource, TextResourceContents
from mcp_ui_server import UIResource
from mcp_ui_server.exceptions import InvalidContentError, InvalidURIError
from mcp_ui_server.types import UI_METADATA_PREFIX

__all__ = (
    "external_url",
    "raw_html",
    "remote_dom",
)

# Metadata passed through to the client: ``ui_metadata`` is prefixed with
# ``mcpui.dev/ui-`` so the client recognizes it; ``metadata`` is written verbatim
# into the resource ``_meta``.
UIMetadata = dict[str, Any]
Encoding: TypeAlias = Literal["text", "blob"]

_UI_SCHEME = "ui://"

# The client selects its renderer from the MIME type, so it identifies the content
# kind on the wire; Remote-DOM additionally carries the framework as a parameter.
_HTML_MIME = "text/html"
_URL_MIME = "text/uri-list"
_REMOTE_DOM_MIME = {
    "react": "application/vnd.mcp-ui.remote-dom+javascript; framework=react",
    "webcomponents": "application/vnd.mcp-ui.remote-dom+javascript; framework=webcomponents",
}


def _require_ui_uri(uri: str) -> None:
    """Reject a ``uri`` outside the ``ui://`` scheme.

    Called from :func:`_build`, and ahead of any builder-specific validation so
    that the guard every builder shares is the one a caller trips first.
    """
    if not uri.startswith(_UI_SCHEME):
        raise InvalidURIError(f"URI must start with '{_UI_SCHEME}' but got: {uri}")


def _build(
    uri: str,
    content: str,
    *,
    content_name: str,
    mime_type: str,
    encoding: Encoding,
    ui_metadata: UIMetadata | None,
    metadata: UIMetadata | None,
) -> EmbeddedResource:
    """Construct the UI resource contents that ``mcp-ui-server``'s builder would.

    Reproduces ``mcp_ui_server.create_ui_resource`` rather than calling it: that
    builder hardcodes ``AnyUrl(uri)`` for the resource uri, which ``mcp`` 2.0
    retyped to ``str``, so every call raises a validation error. Reported upstream
    with a proposed two-line fix as MCP-UI-Org/mcp-ui#216 — once that ships, this
    can go back to delegating. The dependency stays for the action helpers, the
    ``UIResource`` type and the wire constants above.
    """
    _require_ui_uri(uri)
    if not content:
        raise InvalidContentError(f"{content_name} must be provided as a non-empty string")

    meta = {f"{UI_METADATA_PREFIX}{key}": value for key, value in (ui_metadata or {}).items()}
    # Caller-supplied metadata is verbatim, and wins over the prefixed UI keys.
    meta |= metadata or {}

    resource: TextResourceContents | BlobResourceContents
    if encoding == "blob":
        resource = BlobResourceContents(
            uri=uri,
            mimeType=mime_type,
            blob=base64.b64encode(content.encode("utf-8")).decode("ascii"),
            _meta=meta or None,
        )
    elif encoding == "text":
        resource = TextResourceContents(uri=uri, mimeType=mime_type, text=content, _meta=meta or None)
    else:
        # Unreachable for a caller who honours ``Encoding``; treating an unknown
        # value as text would hand back a resource nobody asked for.
        raise InvalidContentError(f"Invalid encoding type: {encoding}")
    return UIResource(resource=resource)


def raw_html(
    uri: str,
    html: str,
    *,
    encoding: Encoding = "text",
    ui_metadata: UIMetadata | None = None,
    metadata: UIMetadata | None = None,
) -> EmbeddedResource:
    """A UI resource rendering an inline HTML string (``uri`` must start with ``ui://``)."""
    return _build(
        uri,
        html,
        content_name="html",
        mime_type=_HTML_MIME,
        encoding=encoding,
        ui_metadata=ui_metadata,
        metadata=metadata,
    )


def external_url(
    uri: str,
    url: str,
    *,
    encoding: Encoding = "text",
    ui_metadata: UIMetadata | None = None,
    metadata: UIMetadata | None = None,
) -> EmbeddedResource:
    """A UI resource the client renders by embedding ``url`` in an ``iframe``."""
    return _build(
        uri,
        url,
        content_name="url",
        mime_type=_URL_MIME,
        encoding=encoding,
        ui_metadata=ui_metadata,
        metadata=metadata,
    )


def remote_dom(
    uri: str,
    script: str,
    *,
    framework: Literal["react", "webcomponents"] = "react",
    encoding: Encoding = "text",
    ui_metadata: UIMetadata | None = None,
    metadata: UIMetadata | None = None,
) -> EmbeddedResource:
    """A UI resource carrying a Remote-DOM ``script`` the client mounts (``react`` or ``webcomponents``)."""
    _require_ui_uri(uri)
    mime_type = _REMOTE_DOM_MIME.get(framework)
    if mime_type is None:
        raise InvalidContentError(f"framework must be 'react' or 'webcomponents', got: {framework}")
    return _build(
        uri,
        script,
        content_name="script",
        mime_type=mime_type,
        encoding=encoding,
        ui_metadata=ui_metadata,
        metadata=metadata,
    )
