# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2.a2ui.actions import A2UIEventAction
from ag2.a2ui.parser import A2UIResponseParser
from ag2.a2ui.schema_manager import A2UISchemaManager


class TestA2UISchemaManager:
    def test_default_init(self) -> None:
        manager = A2UISchemaManager()
        assert manager.protocol_version == "v0.9"
        assert "v0_9" in manager.catalog_id
        assert manager.server_to_client_schema is not None
        assert manager.basic_catalog_schema is not None
        assert manager.common_types_schema is not None

    def test_unsupported_version_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported A2UI protocol version"):
            A2UISchemaManager(protocol_version="v0.7")

    def test_v0_9_1_init_loads_specs(self) -> None:
        manager = A2UISchemaManager(protocol_version="v0.9.1")
        assert manager.protocol_version == "v0.9.1"
        assert manager.version_string == "v0.9.1"
        # v0.9.1 reuses v0.9's canonical identifiers (shared catalog/$ids).
        assert manager.catalog_id == "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json"
        assert manager.server_to_client_schema is not None
        assert manager.basic_catalog_schema is not None
        # Registry must build without unresolved cross-file $refs.
        assert manager.build_schema_registry() is not None

    def test_default_catalog_id_is_canonical_resolving_url(self) -> None:
        # Must match the official catalog id renderers advertise (not a 404).
        manager = A2UISchemaManager()
        assert manager.catalog_id == "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json"

    def test_v1_0_init_loads_specs(self) -> None:
        manager = A2UISchemaManager(protocol_version="v1.0")
        assert manager.protocol_version == "v1.0"
        assert manager.version_string == "v1.0"
        assert "v1_0" in manager.catalog_id
        assert manager.server_to_client_schema is not None
        assert manager.basic_catalog_schema is not None
        assert manager.common_types_schema is not None
        # v1.0 adds two server->client message types beyond v0.9's four.
        defs = manager.server_to_client_schema.get("$defs", {})
        assert isinstance(defs, dict)
        assert "CallFunctionMessage" in defs
        assert "ActionResponseMessage" in defs

    def test_v1_0_prompt_mentions_new_message_types(self) -> None:
        manager = A2UISchemaManager(protocol_version="v1.0")
        prompt = manager.generate_prompt_section(include_schema=False)
        assert "v1.0" in prompt
        assert "callFunction" in prompt
        assert "actionResponse" in prompt

    def test_custom_catalog_id_from_catalog(self) -> None:
        manager = A2UISchemaManager(custom_catalog={"$id": "https://mycompany.com/custom.json", "components": {}})
        assert manager.catalog_id == "https://mycompany.com/custom.json"

    def test_custom_catalog_without_id_raises(self) -> None:
        # The custom catalog is loaded/validated lazily (construction is
        # side-effect-free), so the error surfaces on first accessor use.
        manager = A2UISchemaManager(custom_catalog={"components": {}})
        with pytest.raises(ValueError, match="Custom catalog must include"):
            _ = manager.catalog_id

    def test_server_to_client_schema_structure(self) -> None:
        manager = A2UISchemaManager()
        schema = manager.server_to_client_schema
        assert schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema"
        assert "oneOf" in schema
        assert "$defs" in schema
        defs = schema["$defs"]
        assert "CreateSurfaceMessage" in defs
        assert "UpdateComponentsMessage" in defs
        assert "UpdateDataModelMessage" in defs
        assert "DeleteSurfaceMessage" in defs

    def test_basic_catalog_has_components(self) -> None:
        manager = A2UISchemaManager()
        catalog = manager.basic_catalog_schema
        components = catalog.get("components", {})
        assert "Text" in components
        assert "Image" in components
        assert "Button" in components
        assert "Column" in components
        assert "Row" in components

    def test_catalog_rules_loaded(self) -> None:
        manager = A2UISchemaManager()
        prompt = manager.generate_prompt_section(include_schema=False, include_rules=True)
        assert "REQUIRED PROPERTIES" in prompt
        assert "Text" in prompt

    def test_get_component_schemas_includes_basic(self) -> None:
        manager = A2UISchemaManager()
        schemas = manager.get_component_schemas()
        assert "Text" in schemas
        assert "Button" in schemas

    def test_get_component_schemas_includes_custom(self) -> None:
        manager = A2UISchemaManager(
            custom_catalog={
                "$id": "https://mycompany.com/custom.json",
                "components": {"MyWidget": {"type": "object"}},
            }
        )
        schemas = manager.get_component_schemas()
        assert "MyWidget" in schemas
        assert "Text" in schemas  # basic still present

    def test_build_schema_registry_returns_registry(self) -> None:
        manager = A2UISchemaManager()
        registry = manager.build_schema_registry()
        assert registry is not None

    def test_client_to_server_schema_loaded(self) -> None:
        manager = A2UISchemaManager()
        schema = manager.client_to_server_schema
        assert schema is not None
        # client_to_server.json declares action / error oneOf at top level.
        props = schema.get("properties", {})
        assert "action" in props
        assert "error" in props


class TestMergedCatalog:
    """Regression: custom catalog must EXTEND, not REPLACE, the basic catalog."""

    def test_basic_text_validates_when_custom_catalog_present(self) -> None:
        manager = A2UISchemaManager(
            custom_catalog={
                "$id": "https://mycompany.com/cat.json",
                "components": {"MyWidget": {"type": "object", "required": ["foo"]}},
            }
        )
        parser = A2UIResponseParser(
            version_string="v0.9",
            server_to_client_schema=manager.server_to_client_schema,
            schema_registry=manager.build_schema_registry(),
            component_schemas=manager.get_component_schemas(),
            catalog_id=manager.catalog_id,
        )
        ops = [
            {
                "version": "v0.9",
                "updateComponents": {
                    "surfaceId": "s1",
                    "components": [{"id": "root", "component": "Text", "text": "Hi"}],
                },
            }
        ]
        result = parser.validate(ops)
        assert result.is_valid is True

    def test_merged_catalog_exposes_both_basic_and_custom_components(self) -> None:
        manager = A2UISchemaManager(
            custom_catalog={
                "$id": "https://mycompany.com/cat.json",
                "components": {"MyWidget": {"type": "object"}},
            }
        )
        schemas = manager.get_component_schemas()
        assert "Text" in schemas
        assert "Button" in schemas
        assert "MyWidget" in schemas


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
