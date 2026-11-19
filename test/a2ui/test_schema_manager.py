# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import re
from typing import Any

import pytest
from dirty_equals import IsPartialDict
from jsonschema import Draft202012Validator

from ag2.a2ui._types import A2UIVersion
from ag2.a2ui.actions import A2UIEventAction
from ag2.a2ui.parser import A2UIResponseParser, A2UIValidationResult
from ag2.a2ui.schema_manager import A2UISchemaManager

from ._helpers import update_components, validating_parser


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
        assert manager.version_string == "1.0"
        assert "v1_0" in manager.catalog_id
        assert manager.server_to_client_schema is not None
        assert manager.basic_catalog_schema is not None
        assert manager.common_types_schema is not None
        # v1.0 adds two agent->renderer message types beyond v0.9's four.
        defs = manager.server_to_client_schema.get("$defs", {})
        assert isinstance(defs, dict)
        assert "CallRendererFunctionMessage" in defs
        assert "AgentFunctionResponseMessage" in defs
        assert manager.client_to_server_schema == IsPartialDict({
            "$id": "https://a2ui.org/specification/v1_0/renderer_to_agent.json",
        })

    def test_v1_0_prompt_mentions_new_message_types(self) -> None:
        manager = A2UISchemaManager(protocol_version="v1.0")
        prompt = manager.generate_prompt_section(include_schema=False)
        assert "v1.0" in prompt
        assert "callRendererFunction" in prompt
        assert "agentFunctionResponse" in prompt

    def test_v1_0_prompt_teaches_reserved_binding_keys(self) -> None:
        prompt = A2UISchemaManager(protocol_version="v1.0").generate_prompt_section(include_schema=False)
        assert '{"@path": "/customTime"}' in prompt
        assert '"@call"' in prompt

    def test_v1_0_prompt_includes_the_catalog_instructions(self) -> None:
        manager = A2UISchemaManager(protocol_version="v1.0")
        instructions = manager.basic_catalog_schema["instructions"]
        assert isinstance(instructions, str)
        assert instructions.strip() in manager.generate_prompt_section(include_schema=False)

    def test_v0_9_prompt_has_no_catalog_instructions(self) -> None:
        assert "instructions" not in A2UISchemaManager(protocol_version="v0.9").basic_catalog_schema

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

    def test_function_definition_references_escape_json_pointer_tokens(self) -> None:
        manager = A2UISchemaManager(
            protocol_version="v1.0",
            custom_catalog={
                "$id": "https://example.com/catalog.json",
                "functions": {
                    "lookup/~definition": {
                        "type": "object",
                        "returnType": "string",
                        "requiresUserActivation": False,
                        "properties": {"@call": {"const": "lookup"}},
                        "required": ["@call"],
                    },
                },
            },
        )
        registry = manager.build_schema_registry()
        validator = Draft202012Validator({"$ref": f"{manager.catalog_id}#/$defs/anyFunction"}, registry=registry)

        assert validator.is_valid({"@call": "lookup"})
        assert not validator.is_valid({"@call": "lookup/~definition"})

    @pytest.mark.parametrize("version", ["v0.9", "v0.9.1", "v1.0"])
    def test_component_definition_references_escape_json_pointer_tokens(self, version: A2UIVersion) -> None:
        manager = A2UISchemaManager(
            protocol_version=version,
            custom_catalog={
                "$id": "https://example.com/catalog.json",
                "components": {
                    "widget/~definition": {
                        "type": "object",
                        "properties": {"component": {"const": "Widget"}},
                        "required": ["component"],
                    },
                },
            },
        )
        validator = Draft202012Validator(
            {"$ref": f"{manager.catalog_id}#/$defs/anyComponent"}, registry=manager.build_schema_registry()
        )

        assert validator.is_valid({"component": "Widget"})
        assert not validator.is_valid({"component": "widget/~definition"})


def _catalog_with_ref(ref: str) -> dict[str, Any]:
    return {
        "$id": "https://example.com/catalog.json",
        "$defs": {"Local": {"type": "string"}},
        "components": {
            "Label": {
                "type": "object",
                "properties": {"component": {"const": "Label"}, "text": {"$ref": ref}},
                "required": ["component", "text"],
            },
        },
    }


class TestV1WireVersion:
    """A2UI v1.0 is named ``"v1.0"`` but its messages carry the bare ``"1.0"`` (a2ui-project/a2ui#3726)."""

    @pytest.mark.parametrize(
        "message",
        [
            {
                "createSurface": {
                    "surfaceId": "s1",
                    "catalogId": "https://a2ui.org/specification/v1_0/catalogs/basic/catalog.json",
                }
            },
            {
                "updateComponents": {
                    "surfaceId": "s1",
                    "components": [{"id": "root", "component": "Text", "text": "Hi"}],
                }
            },
            {"updateDataModel": {"surfaceId": "s1", "path": "/name", "value": "Ada"}},
            {"deleteSurface": {"surfaceId": "s1"}},
        ],
    )
    def test_messages_carry_the_bare_wire_version(self, message: dict[str, Any]) -> None:
        parser = validating_parser("v1.0")

        assert parser.validate([{"version": "1.0", **message}]) == A2UIValidationResult(is_valid=True)
        assert parser.validate([{"version": "v1.0", **message}]) == A2UIValidationResult(
            is_valid=False, errors=["Operation 0: '1.0' was expected (at /version)"]
        )

    def test_prompt_teaches_the_wire_version(self) -> None:
        prompt = A2UISchemaManager(protocol_version="v1.0").generate_prompt_section(include_schema=False)

        assert '"version": "1.0"' in prompt
        assert '"version": "v1.0"' not in prompt


class TestV1Catalog:
    """The v1.0 basic catalog and the ``$ref`` rules for custom catalogs (a2ui_protocol.md, "Restricted $ref Targets")."""

    def test_the_catalog_instruction_examples_validate(self) -> None:
        instructions = A2UISchemaManager(protocol_version="v1.0").basic_catalog_schema["instructions"]
        assert isinstance(instructions, str)
        examples = re.findall(r"```json\n(.*?)\n```", instructions, flags=re.DOTALL)
        assert examples

        parser = validating_parser("v1.0")
        for example in examples:
            assert parser.validate(json.loads(example)) == A2UIValidationResult(is_valid=True)

    @pytest.mark.parametrize(
        "ref",
        [
            "basic_catalog.json#/components/Text",
            "other_catalog.json#/$defs/Shared",
            "https://a2ui.org/specification/v1_0/catalogs/basic/catalog.json#/components/Text",
            "my_common_types.json#/$defs/DynamicString",
            "catalog.json#/components/Text",
            "common_types.json#/$defs/Surface",
            "common_types.json#DynamicString",
            "https://a2ui.org/specification/v1_0/common_types.json#/$defs/DynamicString",
            "https://evil.example/x/common_types.json#/$defs/DynamicString",
            "DynamicString",
            "#/$defs/anyComponent",
            "#/$defs/anyFunction",
            "#/$defs/Missing",
            "#/components/Missing",
            "#/functions/required",
            "#/properties/text",
        ],
    )
    def test_custom_catalog_rejects_references_the_spec_does_not_allow(self, ref: str) -> None:
        manager = A2UISchemaManager(protocol_version="v1.0", custom_catalog=_catalog_with_ref(ref))

        with pytest.raises(ValueError, match=re.escape(f"'{ref}' at /components/Label/properties/text/$ref")):
            manager.build_schema_registry()

    @pytest.mark.parametrize(
        "ref",
        [
            "#/$defs/Local",
            "common_types.json#/$defs/DynamicString",
        ],
    )
    def test_custom_catalog_may_reference_itself_and_common_types(self, ref: str) -> None:
        parser = validating_parser("v1.0", _catalog_with_ref(ref))
        label = {"id": "root", "component": "Label", "text": "Hi"}

        assert parser.validate([update_components("v1.0", label)]) == A2UIValidationResult(is_valid=True)

    def test_custom_catalog_may_reference_its_components_and_functions(self) -> None:
        catalog = _catalog_with_ref("common_types.json#/$defs/DynamicString")
        catalog["functions"] = {
            "isSet": {
                "type": "object",
                "returnType": "boolean",
                "properties": {"@call": {"const": "isSet"}, "args": {"type": "object"}},
                "required": ["@call", "args"],
            },
        }
        catalog["$defs"] = {
            "anyComponent": {"oneOf": [{"$ref": "#/components/Label"}]},
            "anyFunction": {"oneOf": [{"$ref": "#/functions/isSet"}]},
        }
        parser = validating_parser("v1.0", catalog)
        label = {"id": "root", "component": "Label", "text": "Hi"}

        assert parser.validate([update_components("v1.0", label)]) == A2UIValidationResult(is_valid=True)

    @pytest.mark.parametrize(
        ("keyword", "annotation"),
        [
            ("examples", [{"$ref": "other_catalog.json#/$defs/Shared"}]),
            ("default", {"$ref": "other_catalog.json#/$defs/Shared"}),
            ("const", {"$ref": "other_catalog.json#/$defs/Shared"}),
            ("enum", [{"$ref": "other_catalog.json#/$defs/Shared"}]),
            ("metadata", {"$ref": "other_catalog.json#/$defs/Shared"}),
        ],
    )
    def test_custom_catalog_ignores_ref_in_annotations(self, keyword: str, annotation: Any) -> None:
        catalog = _catalog_with_ref("common_types.json#/$defs/DynamicString")
        catalog["components"]["Label"]["properties"]["hint"] = {keyword: annotation}
        parser = validating_parser("v1.0", catalog)
        label = {"id": "root", "component": "Label", "text": "Hi"}

        assert parser.validate([update_components("v1.0", label)]) == A2UIValidationResult(is_valid=True)

    def test_text_field_requires_a_value(self) -> None:
        parser = validating_parser("v1.0")
        field = {"id": "root", "component": "TextField", "label": "Name"}

        assert parser.validate([update_components("v1.0", field)]) == A2UIValidationResult(
            is_valid=False, errors=["Operation 0: Component 'root' (TextField): 'value' is a required property"]
        )
        assert parser.validate([update_components("v1.0", {**field, "value": "Ada"})]) == A2UIValidationResult(
            is_valid=True
        )

    def test_earlier_versions_keep_cross_catalog_references(self) -> None:
        parser = validating_parser("v0.9", _catalog_with_ref("basic_catalog.json#/components/Text"))
        label = {"id": "root", "component": "Label", "text": {"id": "t", "component": "Text", "text": "Hi"}}

        assert parser.validate([update_components("v0.9", label)]) == A2UIValidationResult(is_valid=True)
        rejected = parser.validate([update_components("v0.9", {**label, "text": "Hi"})])
        assert rejected.is_valid is False
        assert set(rejected.errors) == {"Operation 0: Component 'root' (Label): 'Hi' is not of type 'object'"}


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
