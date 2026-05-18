# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from google.protobuf.json_format import MessageToDict

from autogen.beta.a2ui import A2UI_DEFAULT_CATALOG_ID, A2UI_EXTENSION_URI
from autogen.beta.a2ui.a2a import get_a2ui_agent_extension


def _params(ext) -> dict:
    return MessageToDict(ext.params, preserving_proto_field_name=True)


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
