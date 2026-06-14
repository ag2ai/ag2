# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from google.protobuf.json_format import MessageToDict

from autogen.beta.a2ui.a2a import get_a2ui_agent_extension
from autogen.beta.a2ui.a2a.extension import try_activate_a2ui_extension
from autogen.beta.a2ui.constants import A2UI_DEFAULT_CATALOG_ID, A2UI_EXTENSION_URI


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
