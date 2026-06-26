# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
from google.protobuf.json_format import MessageToDict

from ag2.a2ui._types import A2UIVersion
from ag2.a2ui.a2a import get_a2ui_agent_extension
from ag2.a2ui.a2a.extension import try_activate_a2ui_extension
from ag2.a2ui.constants import (
    A2UI_DEFAULT_CATALOG_ID_BY_VERSION,
    A2UI_EXTENSION_URI_BY_VERSION,
)

# The default protocol version is v0.9; these handles keep the assertions terse.
A2UI_DEFAULT_CATALOG_ID = A2UI_DEFAULT_CATALOG_ID_BY_VERSION["v0.9"]
A2UI_EXTENSION_URI = A2UI_EXTENSION_URI_BY_VERSION["v0.9"]


def _params(ext) -> dict:
    return MessageToDict(ext.params, preserving_proto_field_name=True)


class _StubContext:
    """Minimal stand-in for the bits of RequestContext the helper reads."""

    def __init__(self, requested_extensions: list[str] | None = None) -> None:
        self.requested_extensions = requested_extensions or []
        self.metadata: dict[str, Any] | None = None


class TestAgentExtension:
    def test_default_includes_basic_catalog(self) -> None:
        ext = get_a2ui_agent_extension()
        assert ext.uri == A2UI_EXTENSION_URI
        assert "A2UI" in ext.description
        assert _params(ext) == {"supportedCatalogIds": [A2UI_DEFAULT_CATALOG_ID]}

    def test_custom_supported_catalog_ids(self) -> None:
        ext = get_a2ui_agent_extension(supported_catalog_ids=["https://mycompany.com/cat.json"])
        assert _params(ext) == {"supportedCatalogIds": ["https://mycompany.com/cat.json"]}

    def test_multiple_supported_catalogs(self) -> None:
        ext = get_a2ui_agent_extension(
            supported_catalog_ids=[
                A2UI_DEFAULT_CATALOG_ID,
                "https://mycompany.com/cat.json",
            ]
        )
        assert _params(ext)["supportedCatalogIds"] == [
            A2UI_DEFAULT_CATALOG_ID,
            "https://mycompany.com/cat.json",
        ]

    def test_accepts_inline_catalogs_flag_uses_spec_field_name(self) -> None:
        ext = get_a2ui_agent_extension(accepts_inline_catalogs=True)
        params = _params(ext)
        assert params["acceptsInlineCatalogs"] is True
        # Spec field name is exactly "acceptsInlineCatalogs" — not the legacy name.
        assert "acceptsInlineCustomCatalog" not in params

    def test_inline_catalogs_omitted_when_false(self) -> None:
        ext = get_a2ui_agent_extension(accepts_inline_catalogs=False)
        assert "acceptsInlineCatalogs" not in _params(ext)

    @pytest.mark.parametrize("version", ["v0.9", "v0.9.1", "v1.0"])
    def test_per_version_uri_and_default_catalog(self, version: A2UIVersion) -> None:
        ext = get_a2ui_agent_extension(version=version)
        assert ext.uri == A2UI_EXTENSION_URI_BY_VERSION[version]
        assert version in ext.description
        assert _params(ext) == {"supportedCatalogIds": [A2UI_DEFAULT_CATALOG_ID_BY_VERSION[version]]}

    def test_v1_0_uses_v1_namespace(self) -> None:
        ext = get_a2ui_agent_extension(version="v1.0")
        assert ext.uri == "https://a2ui.org/a2a-extension/a2ui/v1.0"
        assert "v1_0" in _params(ext)["supportedCatalogIds"][0]


class TestTryActivateExtension:
    def test_activates_when_client_requests_uri(self) -> None:
        ctx = _StubContext(requested_extensions=[A2UI_EXTENSION_URI])
        assert try_activate_a2ui_extension(ctx) is True  # type: ignore[arg-type]
        assert ctx.metadata is not None
        assert ctx.metadata["activated_extensions"] == [A2UI_EXTENSION_URI]

    def test_not_activated_when_uri_absent(self) -> None:
        ctx = _StubContext(requested_extensions=["https://example.com/other"])
        assert try_activate_a2ui_extension(ctx) is False  # type: ignore[arg-type]
        assert ctx.metadata is None

    def test_not_activated_when_no_extensions(self) -> None:
        ctx = _StubContext()
        assert try_activate_a2ui_extension(ctx) is False  # type: ignore[arg-type]

    def test_idempotent_no_duplicate_activation(self) -> None:
        ctx = _StubContext(requested_extensions=[A2UI_EXTENSION_URI])
        try_activate_a2ui_extension(ctx)  # type: ignore[arg-type]
        try_activate_a2ui_extension(ctx)  # type: ignore[arg-type]
        assert ctx.metadata is not None
        assert ctx.metadata["activated_extensions"] == [A2UI_EXTENSION_URI]

    def test_preserves_existing_activated_extensions(self) -> None:
        ctx = _StubContext(requested_extensions=[A2UI_EXTENSION_URI])
        ctx.metadata = {"activated_extensions": ["https://example.com/other"]}
        try_activate_a2ui_extension(ctx)  # type: ignore[arg-type]
        assert ctx.metadata["activated_extensions"] == [
            "https://example.com/other",
            A2UI_EXTENSION_URI,
        ]

    @pytest.mark.parametrize("version", ["v0.9", "v0.9.1", "v1.0"])
    def test_activates_matching_version_uri(self, version: A2UIVersion) -> None:
        uri = A2UI_EXTENSION_URI_BY_VERSION[version]
        ctx = _StubContext(requested_extensions=[uri])
        assert try_activate_a2ui_extension(ctx, version=version) is True  # type: ignore[arg-type]
        assert ctx.metadata is not None
        assert ctx.metadata["activated_extensions"] == [uri]

    def test_does_not_activate_on_version_mismatch(self) -> None:
        # Client requested v0.9 but the agent serves v1.0 — no activation.
        ctx = _StubContext(requested_extensions=[A2UI_EXTENSION_URI_BY_VERSION["v0.9"]])
        assert try_activate_a2ui_extension(ctx, version="v1.0") is False  # type: ignore[arg-type]
        assert ctx.metadata is None
