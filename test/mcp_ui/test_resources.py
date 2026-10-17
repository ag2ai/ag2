# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64
from collections.abc import Callable

import pytest
from mcp.types import BlobResourceContents, TextResourceContents
from mcp_ui_server import UIResource
from mcp_ui_server.exceptions import InvalidContentError, InvalidURIError

from ag2 import mcp_ui

# Each builder invoked with a caller-supplied ``uri`` and otherwise-valid content,
# so a validation test can drive all three through their shared guards.
BUILDERS_TAKING_A_URI: list[Callable[[str], UIResource]] = [
    lambda uri: mcp_ui.raw_html(uri, "<h1>Hi</h1>"),
    lambda uri: mcp_ui.external_url(uri, "https://docs.ag2.ai/"),
    lambda uri: mcp_ui.remote_dom(uri, "root.appendChild(el)"),
]


class TestRawHtml:
    def test_builds_inline_html_resource(self) -> None:
        block = mcp_ui.raw_html("ui://ag2/greeting", "<h1>Hi</h1>")

        assert block == UIResource(
            resource=TextResourceContents(uri="ui://ag2/greeting", mimeType="text/html", text="<h1>Hi</h1>"),
        )

    def test_blob_encoding_base64_encodes_payload(self) -> None:
        block = mcp_ui.raw_html("ui://ag2/greeting", "<h1>Hi</h1>", encoding="blob")

        assert block == UIResource(
            resource=BlobResourceContents(
                uri="ui://ag2/greeting",
                mimeType="text/html",
                blob=base64.b64encode(b"<h1>Hi</h1>").decode(),
            ),
        )


def test_external_url_builds_iframe_resource() -> None:
    block = mcp_ui.external_url("ui://ag2/docs", "https://docs.ag2.ai/")

    assert block == UIResource(
        resource=TextResourceContents(uri="ui://ag2/docs", mimeType="text/uri-list", text="https://docs.ag2.ai/"),
    )


class TestRemoteDom:
    def test_carries_script_and_react_framework(self) -> None:
        block = mcp_ui.remote_dom("ui://ag2/dom", "root.appendChild(el)", framework="react")

        assert block == UIResource(
            resource=TextResourceContents(
                uri="ui://ag2/dom",
                mimeType="application/vnd.mcp-ui.remote-dom+javascript; framework=react",
                text="root.appendChild(el)",
            ),
        )

    def test_webcomponents_framework(self) -> None:
        block = mcp_ui.remote_dom("ui://ag2/dom", "root.appendChild(el)", framework="webcomponents")

        assert block == UIResource(
            resource=TextResourceContents(
                uri="ui://ag2/dom",
                mimeType="application/vnd.mcp-ui.remote-dom+javascript; framework=webcomponents",
                text="root.appendChild(el)",
            ),
        )


def test_ui_metadata_is_prefixed_and_metadata_is_verbatim() -> None:
    block = mcp_ui.raw_html(
        "ui://ag2/greeting",
        "<h1>Hi</h1>",
        ui_metadata={"preferred-frame-size": ["800px", "600px"]},
        metadata={"title": "Greeting"},
    )

    assert block == UIResource(
        resource=TextResourceContents(
            uri="ui://ag2/greeting",
            mimeType="text/html",
            text="<h1>Hi</h1>",
            _meta={
                "mcpui.dev/ui-preferred-frame-size": ["800px", "600px"],
                "title": "Greeting",
            },
        ),
    )


@pytest.mark.parametrize("build", BUILDERS_TAKING_A_URI, ids=["raw_html", "external_url", "remote_dom"])
def test_uri_without_ui_scheme_is_rejected(build: Callable[[str], UIResource]) -> None:
    with pytest.raises(InvalidURIError, match="must start with 'ui://'"):
        build("https://ag2.ai/greeting")


# The same three builders, called with a valid ``uri`` and empty content.
BUILDERS_TAKING_CONTENT: list[Callable[[str], UIResource]] = [
    lambda content: mcp_ui.raw_html("ui://ag2/greeting", content),
    lambda content: mcp_ui.external_url("ui://ag2/docs", content),
    lambda content: mcp_ui.remote_dom("ui://ag2/dom", content),
]


@pytest.mark.parametrize("build", BUILDERS_TAKING_CONTENT, ids=["raw_html", "external_url", "remote_dom"])
def test_empty_content_is_rejected(build: Callable[[str], UIResource]) -> None:
    """Empty content is a caller mistake, not an empty resource."""
    with pytest.raises(InvalidContentError):
        build("")


def test_remote_dom_rejects_unknown_framework() -> None:
    with pytest.raises(InvalidContentError, match="react"):
        mcp_ui.remote_dom("ui://ag2/dom", "root.appendChild(el)", framework="svelte")  # type: ignore[arg-type]


def test_an_unknown_encoding_is_rejected() -> None:
    """Outside the ``Encoding`` literal nothing is a valid wire form, and silently
    treating it as text would ship the caller a resource they did not ask for.
    """
    with pytest.raises(InvalidContentError, match="encoding"):
        mcp_ui.raw_html("ui://ag2/greeting", "<h1>Hi</h1>", encoding="base64")  # type: ignore[arg-type]


def test_the_uri_is_checked_before_the_framework() -> None:
    """Both are wrong here. The ``uri`` guard is the one every builder shares, so
    it is the error a caller gets from all three rather than one special case.
    """
    with pytest.raises(InvalidURIError):
        mcp_ui.remote_dom("https://ag2.ai/dom", "root.appendChild(el)", framework="svelte")  # type: ignore[arg-type]
