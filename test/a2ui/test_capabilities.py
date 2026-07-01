# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import logging

from ag2.a2ui import A2UIClientCapabilities
from ag2.a2ui.capabilities import (
    A2UI_CLIENT_CAPABILITIES_METADATA_KEY,
    capabilities_to_prompt,
    parse_client_capabilities,
)

_AGENT_CATALOG = "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json"


class TestCapabilitiesToPrompt:
    def test_none_returns_empty(self) -> None:
        assert capabilities_to_prompt(None, catalog_id=_AGENT_CATALOG) == ""

    def test_empty_payload_returns_empty(self) -> None:
        assert capabilities_to_prompt(A2UIClientCapabilities(), catalog_id=_AGENT_CATALOG) == ""

    def test_match_lists_catalogs_without_mismatch_note(self) -> None:
        caps = A2UIClientCapabilities(supported_catalog_ids=[_AGENT_CATALOG])
        prompt = capabilities_to_prompt(caps, catalog_id=_AGENT_CATALOG)
        assert "## A2UI Client Capabilities" in prompt
        assert _AGENT_CATALOG in prompt
        assert "did NOT list" not in prompt

    def test_mismatch_adds_note_and_warns(self, caplog) -> None:
        caps = A2UIClientCapabilities(supported_catalog_ids=["https://other.example/catalog.json"])
        with caplog.at_level(logging.WARNING):
            prompt = capabilities_to_prompt(caps, catalog_id=_AGENT_CATALOG)
        assert "did NOT list" in prompt
        assert _AGENT_CATALOG in prompt
        assert any("did not advertise" in r.message for r in caplog.records)

    def test_inline_catalogs_are_summarized(self) -> None:
        caps = A2UIClientCapabilities(
            supported_catalog_ids=[_AGENT_CATALOG],
            inline_catalogs=[
                {
                    "catalogId": "https://acme.example/cat.json",
                    "components": {"Gauge": {}, "Chart": {}},
                    "functions": {"refresh": {}},
                }
            ],
        )
        prompt = capabilities_to_prompt(caps, catalog_id=_AGENT_CATALOG)
        assert "inline catalog" in prompt
        assert "https://acme.example/cat.json" in prompt
        assert "components: Chart, Gauge" in prompt  # sorted
        assert "functions: refresh" in prompt

    def test_inline_catalog_without_defs_is_dropped(self) -> None:
        # An inline catalog with neither components nor functions yields no line,
        # so with no supportedCatalogIds the whole fragment is empty.
        caps = A2UIClientCapabilities(inline_catalogs=[{"catalogId": "x"}])
        assert capabilities_to_prompt(caps, catalog_id=_AGENT_CATALOG) == ""


class TestParseClientCapabilitiesVersionKey:
    def test_reads_requested_version(self) -> None:
        metadata = {A2UI_CLIENT_CAPABILITIES_METADATA_KEY: {"v1.0": {"supportedCatalogIds": ["x"]}}}
        caps = parse_client_capabilities(metadata, version_key="v1.0")
        assert caps == A2UIClientCapabilities(supported_catalog_ids=["x"], inline_catalogs=[])

    def test_default_version_misses_v1_payload(self) -> None:
        metadata = {A2UI_CLIENT_CAPABILITIES_METADATA_KEY: {"v1.0": {"supportedCatalogIds": ["x"]}}}
        assert parse_client_capabilities(metadata) is None

    def test_coerces_ids_to_str(self) -> None:
        metadata = {A2UI_CLIENT_CAPABILITIES_METADATA_KEY: {"v0.9": {"supportedCatalogIds": [1, 2]}}}
        caps = parse_client_capabilities(metadata)
        assert caps == A2UIClientCapabilities(supported_catalog_ids=["1", "2"], inline_catalogs=[])

    def test_drops_non_dict_inline_catalogs(self) -> None:
        metadata = {
            A2UI_CLIENT_CAPABILITIES_METADATA_KEY: {
                "v0.9": {"supportedCatalogIds": ["x"], "inlineCatalogs": [{"catalogId": "ok"}, "nope", 5]}
            }
        }
        caps = parse_client_capabilities(metadata)
        assert caps == A2UIClientCapabilities(supported_catalog_ids=["x"], inline_catalogs=[{"catalogId": "ok"}])
