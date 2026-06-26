# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.a2ui.a2a import (
    A2UI_CLIENT_CAPABILITIES_METADATA_KEY,
    A2UIClientCapabilities,
    parse_client_capabilities,
)


class TestParseClientCapabilities:
    def test_full_payload(self) -> None:
        metadata = {
            A2UI_CLIENT_CAPABILITIES_METADATA_KEY: {
                "v0.9": {
                    "supportedCatalogIds": [
                        "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json",
                    ],
                    "inlineCatalogs": [{"catalogId": "https://mycompany.com/inline.json"}],
                }
            }
        }
        assert parse_client_capabilities(metadata) == A2UIClientCapabilities(
            supported_catalog_ids=["https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json"],
            inline_catalogs=[{"catalogId": "https://mycompany.com/inline.json"}],
        )

    def test_minimal_supported_catalog_ids_only(self) -> None:
        metadata = {A2UI_CLIENT_CAPABILITIES_METADATA_KEY: {"v0.9": {"supportedCatalogIds": ["x"]}}}
        caps = parse_client_capabilities(metadata)
        assert caps == A2UIClientCapabilities(supported_catalog_ids=["x"], inline_catalogs=[])

    def test_missing_metadata_returns_none(self) -> None:
        assert parse_client_capabilities(None) is None
        assert parse_client_capabilities({}) is None

    def test_wrong_version_returns_none(self) -> None:
        metadata = {A2UI_CLIENT_CAPABILITIES_METADATA_KEY: {"v0.8": {"supportedCatalogIds": []}}}
        assert parse_client_capabilities(metadata) is None

    def test_malformed_payload_returns_none(self) -> None:
        assert parse_client_capabilities({A2UI_CLIENT_CAPABILITIES_METADATA_KEY: "not-a-dict"}) is None
