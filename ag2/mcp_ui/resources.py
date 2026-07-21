# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal, TypeAlias

from mcp.types import EmbeddedResource
from mcp_ui_server import create_ui_resource

__all__ = (
    "external_url",
    "raw_html",
    "remote_dom",
)

# Metadata passed through to the client: ``ui_metadata`` is prefixed by
# ``mcp-ui-server`` with ``mcpui.dev/ui-`` so the client recognizes it;
# ``metadata`` is written verbatim into the resource ``_meta``.
UIMetadata = dict[str, Any]
Encoding: TypeAlias = Literal["text", "blob"]


def _build(options: dict[str, Any]) -> EmbeddedResource:
    return create_ui_resource(options)


def _options(
    uri: str,
    content: dict[str, Any],
    encoding: Encoding,
    ui_metadata: UIMetadata | None,
    metadata: UIMetadata | None,
) -> dict[str, Any]:
    options: dict[str, Any] = {"uri": uri, "content": content, "encoding": encoding}
    if ui_metadata:
        options["uiMetadata"] = ui_metadata
    if metadata:
        options["metadata"] = metadata
    return options


def raw_html(
    uri: str,
    html: str,
    *,
    encoding: Encoding = "text",
    ui_metadata: UIMetadata | None = None,
    metadata: UIMetadata | None = None,
) -> EmbeddedResource:
    """A UI resource rendering an inline HTML string (``uri`` must start with ``ui://``)."""
    return _build(_options(uri, {"type": "rawHtml", "htmlString": html}, encoding, ui_metadata, metadata))


def external_url(
    uri: str,
    url: str,
    *,
    encoding: Encoding = "text",
    ui_metadata: UIMetadata | None = None,
    metadata: UIMetadata | None = None,
) -> EmbeddedResource:
    """A UI resource the client renders by embedding ``url`` in an ``iframe``."""
    return _build(_options(uri, {"type": "externalUrl", "iframeUrl": url}, encoding, ui_metadata, metadata))


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
    content = {"type": "remoteDom", "script": script, "framework": framework}
    return _build(_options(uri, content, encoding, ui_metadata, metadata))
