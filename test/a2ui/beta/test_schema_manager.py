# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Any

import pytest
from a2ui.core.basic_catalog import BasicCatalog

from ag2.a2ui.beta._types import A2UIVersion
from ag2.a2ui.beta.actions import A2UIEventAction
from ag2.a2ui.beta.parser import A2UIResponseParser
from ag2.a2ui.beta.schema_manager import A2UISchemaManager

BADGE_CATALOG_ID = "https://example.com/badge_catalog.json"


def _badge_catalog(**extra_defs: Any) -> dict[str, Any]:
    """A self-contained catalog with one component, using the ref forms core resolves."""
    return {
        "$id": BADGE_CATALOG_ID,
        "components": {
            "Badge": {
                "type": "object",
                "allOf": [
                    {"$ref": "common_types.json#/$defs/ComponentCommon"},
                    {"$ref": "catalog.json#/$defs/CatalogComponentCommon"},
                ],
                "properties": {"component": {"const": "Badge"}, "label": {"type": "string"}},
                "required": ["component", "label"],
                "unevaluatedProperties": False,
            }
        },
        "$defs": {
            "anyComponent": {"oneOf": [{"$ref": "#/components/Badge"}], "discriminator": {"propertyName": "component"}},
            "CatalogComponentCommon": {"type": "object", "properties": {"weight": {"type": "number"}}},
            **extra_defs,
        },
    }


class TestA2UISchemaManager:
    def test_default_init(self) -> None:
        manager = A2UISchemaManager()
        assert manager.protocol_version == "v0.9"
        assert "v0_9" in manager.catalog_id
        assert "Text" in manager.catalog.components

    def test_unsupported_version_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported A2UI protocol version"):
            A2UISchemaManager(protocol_version="v0.7")  # type: ignore[arg-type]

    def test_v0_8_is_not_supported(self) -> None:
        with pytest.raises(ValueError, match="Unsupported A2UI protocol version"):
            A2UISchemaManager(protocol_version="v0.8")  # type: ignore[arg-type]

    @pytest.mark.parametrize("version", ["v0.9", "v0.9.1", "v1.0"])
    def test_catalog_comes_from_core(self, version: A2UIVersion) -> None:
        manager = A2UISchemaManager(protocol_version=version)
        assert manager.version_string == version
        assert manager.catalog_id == BasicCatalog(version).id
        assert {"Text", "Image", "Button", "Column", "Row"} <= set(manager.catalog.components)

    def test_v0_9_1_reuses_v0_9_catalog_id(self) -> None:
        manager = A2UISchemaManager(protocol_version="v0.9.1")
        assert manager.catalog_id == "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json"

    def test_default_catalog_id_is_canonical_resolving_url(self) -> None:
        manager = A2UISchemaManager()
        assert manager.catalog_id == "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json"

    def test_v1_0_catalog_id(self) -> None:
        manager = A2UISchemaManager(protocol_version="v1.0")
        assert manager.catalog_id == "https://a2ui.org/specification/v1_0/catalogs/basic/catalog.json"

    def test_v1_0_server_to_client_schema_follows_core(self) -> None:
        defs = A2UISchemaManager(protocol_version="v1.0").server_to_client_schema["$defs"]
        assert isinstance(defs, dict)
        assert {"CallRendererFunctionMessage", "AgentFunctionResponseMessage"} <= set(defs)
        assert "CallFunctionMessage" not in defs
        assert "ActionResponseMessage" not in defs

    def test_v1_0_prompt_mentions_core_message_types(self) -> None:
        manager = A2UISchemaManager(protocol_version="v1.0")
        prompt = manager.generate_prompt_section(include_schema=False)
        assert "v1.0" in prompt
        assert "callRendererFunction" in prompt
        assert "agentFunctionResponse" in prompt
        assert "actionResponse" not in prompt
        assert "wantResponse" not in prompt

    def test_server_to_client_schema_structure(self) -> None:
        schema = A2UISchemaManager().server_to_client_schema
        assert schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema"
        assert "oneOf" in schema
        defs = schema["$defs"]
        assert "CreateSurfaceMessage" in defs
        assert "UpdateComponentsMessage" in defs
        assert "UpdateDataModelMessage" in defs
        assert "DeleteSurfaceMessage" in defs

    def test_catalog_rules_loaded(self) -> None:
        manager = A2UISchemaManager()
        prompt = manager.generate_prompt_section(include_schema=False, include_rules=True)
        assert "REQUIRED PROPERTIES" in prompt
        assert "Text" in prompt

    def test_construction_does_not_touch_the_file_system(self, tmp_path: Path) -> None:
        manager = A2UISchemaManager(custom_catalog=tmp_path / "missing.json")
        with pytest.raises(FileNotFoundError, match="Custom catalog file not found"):
            _ = manager.catalog


class TestCustomCatalog:
    """A custom catalog is a standalone unit, as in a2ui-core: it is not merged with the basic one."""

    def test_catalog_id_from_catalog(self) -> None:
        assert A2UISchemaManager(custom_catalog=_badge_catalog()).catalog_id == BADGE_CATALOG_ID

    def test_catalog_id_key_is_accepted(self) -> None:
        catalog = _badge_catalog()
        catalog["catalogId"] = catalog.pop("$id")
        assert A2UISchemaManager(custom_catalog=catalog).catalog_id == BADGE_CATALOG_ID

    def test_without_id_raises(self) -> None:
        # The custom catalog is loaded lazily (construction is side-effect-free),
        # so the error surfaces on first accessor use.
        manager = A2UISchemaManager(custom_catalog={"components": {}})
        with pytest.raises(ValueError, match="Custom catalog must include"):
            _ = manager.catalog_id

    def test_non_string_id_raises(self) -> None:
        manager = A2UISchemaManager(custom_catalog={"$id": 1, "components": {}})
        with pytest.raises(ValueError, match="must be a string"):
            _ = manager.catalog_id

    def test_loaded_from_a_file(self, tmp_path: Path) -> None:
        path = tmp_path / "catalog.json"
        path.write_text(json.dumps(_badge_catalog()), encoding="utf-8")
        assert A2UISchemaManager(custom_catalog=path).catalog_id == BADGE_CATALOG_ID

    def test_invalid_json_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "catalog.json"
        path.write_text("{", encoding="utf-8")
        with pytest.raises(ValueError, match="is not valid JSON"):
            _ = A2UISchemaManager(custom_catalog=path).catalog

    def test_catalog_has_only_its_own_components(self) -> None:
        manager = A2UISchemaManager(custom_catalog=_badge_catalog())
        assert set(manager.catalog.components) == {"Badge"}

    def test_surface_with_the_catalog_own_component_validates(self) -> None:
        manager = A2UISchemaManager(custom_catalog=_badge_catalog())
        parser = A2UIResponseParser(version_string="v0.9", catalog=manager.catalog)
        ops = _badge_surface([{"id": "root", "component": "Badge", "label": "New", "weight": 1}])
        result = parser.validate(ops)  # type: ignore[arg-type]
        assert result.is_valid is True, result.errors

    def test_basic_component_absent_from_the_catalog_is_rejected(self) -> None:
        manager = A2UISchemaManager(custom_catalog=_badge_catalog())
        parser = A2UIResponseParser(version_string="v0.9", catalog=manager.catalog)
        result = parser.validate(_badge_surface([{"id": "root", "component": "Text", "text": "Hi"}]))  # type: ignore[arg-type]
        assert result.is_valid is False
        assert "Unrecognized component type 'Text'" in result.errors[0]

    def test_basic_catalog_id_is_not_available_on_a_custom_catalog(self) -> None:
        manager = A2UISchemaManager(custom_catalog=_badge_catalog())
        parser = A2UIResponseParser(version_string="v0.9", catalog=manager.catalog)
        ops = [{"version": "v0.9", "createSurface": {"surfaceId": "s1", "catalogId": BasicCatalog("v0.9").id}}]
        assert parser.validate(ops).is_valid is False  # type: ignore[arg-type]

    def test_unresolvable_ref_surfaces_as_a_bounded_validation_error(self) -> None:
        catalog = _badge_catalog()
        catalog["components"]["Badge"]["allOf"][1] = {"$ref": "catalog.json#/$defs/Missing"}
        manager = A2UISchemaManager(custom_catalog=catalog)
        parser = A2UIResponseParser(version_string="v0.9", catalog=manager.catalog)
        result = parser.validate(_badge_surface([{"id": "root", "component": "Badge", "label": "x"}]))  # type: ignore[arg-type]
        assert result.is_valid is False
        assert "Missing" in result.errors[0]
        assert len(result.errors[0]) < 2100

    def test_prompt_lists_only_catalog_components_and_skips_basic_rules(self) -> None:
        manager = A2UISchemaManager(custom_catalog=_badge_catalog(), custom_catalog_rules="Badge labels are short.")
        prompt = manager.generate_prompt_section(include_schema=False)
        assert "**Custom catalog components:** Badge" in prompt
        assert "**Badge**: Properties: label" in prompt
        assert "Basic catalog" not in prompt
        assert "REQUIRED PROPERTIES" not in prompt
        assert "Badge labels are short." in prompt
        assert BADGE_CATALOG_ID in prompt

    def test_example_message_uses_the_catalog_id(self) -> None:
        prompt = A2UISchemaManager(custom_catalog=_badge_catalog()).generate_prompt_section(include_schema=False)
        assert f'"catalogId": "{BADGE_CATALOG_ID}"' in prompt


def _badge_surface(components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"version": "v0.9", "createSurface": {"surfaceId": "s1", "catalogId": BADGE_CATALOG_ID}},
        {"version": "v0.9", "updateComponents": {"surfaceId": "s1", "components": components}},
    ]


class TestPromptSection:
    def test_includes_format(self) -> None:
        manager = A2UISchemaManager()
        prompt = manager.generate_prompt_section()
        assert "A2UI Response Format" in prompt
        assert "v0.9" in prompt
        assert "<a2ui-json>" in prompt
        assert "</a2ui-json>" in prompt
        assert "createSurface" in prompt
        assert "updateComponents" in prompt

    def test_includes_components(self) -> None:
        manager = A2UISchemaManager()
        prompt = manager.generate_prompt_section()
        assert "Available Components" in prompt
        assert "Text" in prompt

    def test_includes_rules_by_default(self) -> None:
        manager = A2UISchemaManager()
        prompt = manager.generate_prompt_section(include_rules=True)
        assert "Component Rules" in prompt
        assert "REQUIRED PROPERTIES" in prompt

    def test_excludes_rules(self) -> None:
        manager = A2UISchemaManager()
        prompt = manager.generate_prompt_section(include_rules=False)
        assert "Component Rules" not in prompt

    def test_includes_schema_by_default(self) -> None:
        manager = A2UISchemaManager()
        prompt = manager.generate_prompt_section(include_schema=True)
        assert "A2UI Message Schema" in prompt
        assert '"$schema"' in prompt

    def test_excludes_schema(self) -> None:
        manager = A2UISchemaManager()
        prompt = manager.generate_prompt_section(include_schema=False)
        assert "A2UI Message Schema" not in prompt

    def test_uses_official_tags(self) -> None:
        manager = A2UISchemaManager()
        prompt = manager.generate_prompt_section()
        assert "<a2ui-json>" in prompt
        assert "</a2ui-json>" in prompt
        assert "---a2ui_JSON---" not in prompt

    def test_event_action_in_prompt(self) -> None:
        manager = A2UISchemaManager()
        prompt = manager.generate_prompt_section(
            actions=[
                A2UIEventAction(
                    name="book_table",
                    description="Book a table",
                    example_context={"restaurant_id": "abc123"},
                ),
            ],
        )
        assert "Server Events" in prompt
        assert "book_table" in prompt
        assert '"event"' in prompt
        assert "Client Functions" not in prompt
