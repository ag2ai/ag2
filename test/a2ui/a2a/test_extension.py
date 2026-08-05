# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from a2a.server.context import ServerCallContext
from google.protobuf.json_format import MessageToDict

from ag2.a2ui._types import A2UIVersion
from ag2.a2ui.a2a import get_a2ui_agent_extension, get_activated_extensions
from ag2.a2ui.a2a.extension import ACTIVATED_EXTENSIONS_KEY, try_activate_a2ui_extension
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
    """Minimal stand-in for the bits of RequestContext the helper reads.

    ``call_context`` is a real ``ServerCallContext`` rather than a
    hand-rolled double: activation records into its ``state``, and an
    invented stand-in could accept a write the real type rejects — which
    is exactly how the previous ``metadata`` carrier went unnoticed
    (``RequestContext.metadata`` is read-only, so nothing survived). See
    ``test_extension_e2e.py`` for the round-trip that pins it.
    """

    def __init__(self, requested_extensions: list[str] | None = None) -> None:
        self.requested_extensions = requested_extensions or []
        self.call_context = ServerCallContext()

    @property
    def activated(self) -> list[str] | None:
        """Recorded activations, or ``None`` when nothing was recorded."""
        return self.call_context.state.get(ACTIVATED_EXTENSIONS_KEY)


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
        assert ctx.activated == [A2UI_EXTENSION_URI]

    def test_not_activated_when_uri_absent(self) -> None:
        ctx = _StubContext(requested_extensions=["https://example.com/other"])
        assert try_activate_a2ui_extension(ctx) is False  # type: ignore[arg-type]
        assert ctx.activated is None

    def test_not_activated_when_no_extensions(self) -> None:
        ctx = _StubContext()
        assert try_activate_a2ui_extension(ctx) is False  # type: ignore[arg-type]

    def test_idempotent_no_duplicate_activation(self) -> None:
        ctx = _StubContext(requested_extensions=[A2UI_EXTENSION_URI])
        try_activate_a2ui_extension(ctx)  # type: ignore[arg-type]
        try_activate_a2ui_extension(ctx)  # type: ignore[arg-type]
        assert ctx.activated == [A2UI_EXTENSION_URI]

    def test_preserves_existing_activated_extensions(self) -> None:
        ctx = _StubContext(requested_extensions=[A2UI_EXTENSION_URI])
        ctx.call_context.state[ACTIVATED_EXTENSIONS_KEY] = ["https://example.com/other"]
        try_activate_a2ui_extension(ctx)  # type: ignore[arg-type]
        assert ctx.activated == [
            "https://example.com/other",
            A2UI_EXTENSION_URI,
        ]

    @pytest.mark.parametrize("version", ["v0.9", "v0.9.1", "v1.0"])
    def test_activates_matching_version_uri(self, version: A2UIVersion) -> None:
        uri = A2UI_EXTENSION_URI_BY_VERSION[version]
        ctx = _StubContext(requested_extensions=[uri])
        assert try_activate_a2ui_extension(ctx, version=version) is True  # type: ignore[arg-type]
        assert ctx.activated == [uri]

    def test_does_not_activate_on_version_mismatch(self) -> None:
        # Client requested v0.9 but the agent serves v1.0 — no activation.
        ctx = _StubContext(requested_extensions=[A2UI_EXTENSION_URI_BY_VERSION["v0.9"]])
        assert try_activate_a2ui_extension(ctx, version="v1.0") is False  # type: ignore[arg-type]
        assert ctx.activated is None


class TestGetActivatedExtensions:
    def test_empty_before_any_activation(self) -> None:
        # Returns a list rather than raising — indexing the state key
        # directly would KeyError here, which is the trap this avoids.
        assert get_activated_extensions(_StubContext()) == []  # type: ignore[arg-type]

    def test_reports_what_was_activated(self) -> None:
        ctx = _StubContext(requested_extensions=[A2UI_EXTENSION_URI])
        try_activate_a2ui_extension(ctx)  # type: ignore[arg-type]

        assert get_activated_extensions(ctx) == [A2UI_EXTENSION_URI]  # type: ignore[arg-type]

    def test_returns_a_copy_the_caller_cannot_corrupt(self) -> None:
        ctx = _StubContext(requested_extensions=[A2UI_EXTENSION_URI])
        try_activate_a2ui_extension(ctx)  # type: ignore[arg-type]

        get_activated_extensions(ctx).append("https://example.com/injected")  # type: ignore[arg-type]

        assert get_activated_extensions(ctx) == [A2UI_EXTENSION_URI]  # type: ignore[arg-type]
