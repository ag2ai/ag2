# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import base64

from mcp.types import BlobResourceContents, TextResourceContents
from mcp_ui_server import UIResource

from ag2 import mcp_ui


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
