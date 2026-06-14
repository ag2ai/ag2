# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta.a2ui.parser import (
    A2UIParseResult,
    A2UIResponseParser,
    A2UIValidationResult,
    strip_markdown_fences,
)
from autogen.beta.a2ui.schema_manager import A2UISchemaManager


class TestA2UIResponseParser:
    def test_parse_no_a2ui(self) -> None:
        parser = A2UIResponseParser(version_string="v0.9")
        result = parser.parse("Just a plain text response.")
        assert result.text == "Just a plain text response."
        assert result.operations == []
        assert result.has_a2ui is False
        assert result.parse_error is None

    def test_parse_with_a2ui(self) -> None:
        parser = A2UIResponseParser(version_string="v0.9")
        response = (
            "Here is your UI.\n---a2ui_JSON---\n"
            '[{"version": "v0.9", "createSurface": {"surfaceId": "s1", '
            '"catalogId": "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json"}}]'
        )
        result = parser.parse(response)
        assert result.text == "Here is your UI."
        assert result.has_a2ui is True
        assert len(result.operations) == 1
        assert result.operations[0]["createSurface"]["surfaceId"] == "s1"
        assert result.parse_error is None

    def test_parse_single_object(self) -> None:
        parser = A2UIResponseParser(version_string="v0.9")
        response = (
            'UI below.\n---a2ui_JSON---\n{"version": "v0.9", "createSurface": {"surfaceId": "s1", "catalogId": "test"}}'
        )
        result = parser.parse(response)
        assert result.has_a2ui is True
        assert len(result.operations) == 1

    def test_parse_invalid_json(self) -> None:
        parser = A2UIResponseParser(version_string="v0.9")
        response = "Text\n---a2ui_JSON---\n{invalid json}"
        result = parser.parse(response)
        assert result.has_a2ui is True
        assert result.operations == []
        assert result.parse_error is not None
        assert "Invalid JSON" in result.parse_error

    def test_parse_non_array_non_object(self) -> None:
        parser = A2UIResponseParser(version_string="v0.9")
        response = 'Text\n---a2ui_JSON---\n"just a string"'
        result = parser.parse(response)
        assert result.has_a2ui is True
        assert result.operations == []
        assert result.parse_error is not None
        assert "Expected JSON array or object" in result.parse_error

    def test_parse_empty_after_delimiter(self) -> None:
        parser = A2UIResponseParser(version_string="v0.9")
        result = parser.parse("Text\n---a2ui_JSON---\n")
        assert result.has_a2ui is False
        assert result.text == "Text"

    def test_parse_custom_delimiter(self) -> None:
        parser = A2UIResponseParser(version_string="v0.9", delimiter="<<<A2UI>>>")
        response = 'Text\n<<<A2UI>>>\n[{"version": "v0.9", "deleteSurface": {"surfaceId": "s1"}}]'
        result = parser.parse(response)
        assert result.has_a2ui is True
        assert len(result.operations) == 1

    def test_parse_multiple_operations(self) -> None:
        parser = A2UIResponseParser(version_string="v0.9")
        response = (
            "Here.\n---a2ui_JSON---\n"
            '[{"version": "v0.9", "createSurface": {"surfaceId": "s1", "catalogId": "test"}}, '
            '{"version": "v0.9", "updateComponents": {"surfaceId": "s1", '
            '"components": [{"id": "root", "component": "Text", "text": "Hello"}]}}]'
        )
        result = parser.parse(response)
        assert len(result.operations) == 2
        assert "createSurface" in result.operations[0]
        assert "updateComponents" in result.operations[1]


class TestA2UIResponseParserValidation:
    @pytest.fixture()
    def parser_with_schema(self) -> A2UIResponseParser:
        manager = A2UISchemaManager()
        return A2UIResponseParser(
            version_string="v0.9",
            server_to_client_schema=manager.server_to_client_schema,
        )

    def test_validate_valid_create_surface(self, parser_with_schema: A2UIResponseParser) -> None:
        ops = [
            {
                "version": "v0.9",
                "createSurface": {
                    "surfaceId": "s1",
                    "catalogId": "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json",
                },
            }
        ]
        result = parser_with_schema.validate(ops)
        assert result.is_valid is True
        assert result.errors == []

    def test_validate_valid_delete_surface(self, parser_with_schema: A2UIResponseParser) -> None:
        ops = [{"version": "v0.9", "deleteSurface": {"surfaceId": "s1"}}]
        result = parser_with_schema.validate(ops)
        assert result.is_valid is True

    def test_validate_missing_version(self, parser_with_schema: A2UIResponseParser) -> None:
        ops = [{"createSurface": {"surfaceId": "s1", "catalogId": "test"}}]
        result = parser_with_schema.validate(ops)
        assert result.is_valid is False
        assert len(result.errors) == 1

    def test_validate_missing_required_field(self, parser_with_schema: A2UIResponseParser) -> None:
        ops = [{"version": "v0.9", "createSurface": {"surfaceId": "s1"}}]
        result = parser_with_schema.validate(ops)
        assert result.is_valid is False

    def test_validate_no_schema_always_valid(self) -> None:
        parser = A2UIResponseParser(version_string="v0.9", server_to_client_schema=None)
        result = parser.validate([{"anything": "goes"}])
        assert result.is_valid is True

    def test_validate_multiple_ops_one_invalid(self, parser_with_schema: A2UIResponseParser) -> None:
        ops = [
            {
                "version": "v0.9",
                "createSurface": {"surfaceId": "s1", "catalogId": "test"},
            },
            {"version": "v0.9", "createSurface": {"surfaceId": "s2"}},
        ]
        result = parser_with_schema.validate(ops)
        assert result.is_valid is False

    def test_validate_non_dict_operation_does_not_raise(self, parser_with_schema: A2UIResponseParser) -> None:
        # A non-iterable, non-dict operation (e.g. a bare number) must not crash
        # validation with a TypeError — it is reported as invalid instead.
        ops = [
            {"version": "v0.9", "createSurface": {"surfaceId": "s1", "catalogId": "test"}},
            123,
        ]
        result = parser_with_schema.validate(ops)  # type: ignore[arg-type]
        assert result.is_valid is False
        assert result.errors
        assert len(result.errors) == 1
        assert "Operation 1" in result.errors[0]


class TestA2UIResponseParserV091Validation:
    @pytest.fixture()
    def parser_v091(self) -> A2UIResponseParser:
        manager = A2UISchemaManager(protocol_version="v0.9.1")
        return A2UIResponseParser(
            version_string=manager.version_string,
            server_to_client_schema=manager.server_to_client_schema,
            schema_registry=manager.build_schema_registry(),
            component_schemas=manager.get_component_schemas(),
            catalog_id=manager.catalog_id,
        )

    def test_accepts_v0_9_1_version_string(self, parser_v091: A2UIResponseParser) -> None:
        ops = [{"version": "v0.9.1", "deleteSurface": {"surfaceId": "s1"}}]
        assert parser_v091.validate(ops).is_valid is True

    def test_also_accepts_v0_9_version_string(self, parser_v091: A2UIResponseParser) -> None:
        # v0.9.1's version field is the enum ["v0.9", "v0.9.1"] — both are valid.
        ops = [{"version": "v0.9", "deleteSurface": {"surfaceId": "s1"}}]
        assert parser_v091.validate(ops).is_valid is True


class TestA2UIResponseParserV1Validation:
    @pytest.fixture()
    def parser_v1(self) -> A2UIResponseParser:
        manager = A2UISchemaManager(protocol_version="v1.0")
        return A2UIResponseParser(
            version_string=manager.version_string,
            server_to_client_schema=manager.server_to_client_schema,
            schema_registry=manager.build_schema_registry(),
            component_schemas=manager.get_component_schemas(),
            catalog_id=manager.catalog_id,
        )

    def test_validate_call_function(self, parser_v1: A2UIResponseParser) -> None:
        # 'openUrl' is a function defined in the v1.0 basic catalog.
        ops = [
            {
                "version": "v1.0",
                "functionCallId": "fc-1",
                "callFunction": {"call": "openUrl", "args": {"url": "https://example.com"}},
            }
        ]
        result = parser_v1.validate(ops)
        assert result.is_valid is True
        assert result.errors == []

    def test_validate_action_response_value(self, parser_v1: A2UIResponseParser) -> None:
        ops = [{"version": "v1.0", "actionId": "act-1", "actionResponse": {"value": 42}}]
        result = parser_v1.validate(ops)
        assert result.is_valid is True

    def test_validate_action_response_error(self, parser_v1: A2UIResponseParser) -> None:
        ops = [
            {
                "version": "v1.0",
                "actionId": "act-1",
                "actionResponse": {"error": {"code": "NOT_FOUND", "message": "missing"}},
            }
        ]
        result = parser_v1.validate(ops)
        assert result.is_valid is True

    def test_action_response_rejects_value_and_error_together(self, parser_v1: A2UIResponseParser) -> None:
        # Spec: actionResponse carries exactly one of value | error (oneOf).
        ops = [
            {
                "version": "v1.0",
                "actionId": "act-1",
                "actionResponse": {"value": 1, "error": {"code": "c", "message": "m"}},
            }
        ]
        result = parser_v1.validate(ops)
        assert result.is_valid is False

    def test_call_function_requires_function_call_id(self, parser_v1: A2UIResponseParser) -> None:
        ops = [{"version": "v1.0", "callFunction": {"call": "openUrl"}}]
        result = parser_v1.validate(ops)
        assert result.is_valid is False

    def test_create_surface_valid_in_v1(self, parser_v1: A2UIResponseParser) -> None:
        # createSurface is shared across versions; the v1.0 parser accepts it
        # with the v1.0 version string and catalog id.
        ops = [
            {
                "version": "v1.0",
                "createSurface": {
                    "surfaceId": "s1",
                    "catalogId": "https://a2ui.org/specification/v1_0/catalogs/basic/catalog.json",
                },
            }
        ]
        result = parser_v1.validate(ops)
        assert result.is_valid is True


class TestPerComponentValidation:
    @pytest.fixture()
    def parser_with_components(self) -> A2UIResponseParser:
        manager = A2UISchemaManager()
        return A2UIResponseParser(
            version_string="v0.9",
            server_to_client_schema=manager.server_to_client_schema,
            schema_registry=manager.build_schema_registry(),
            component_schemas=manager.get_component_schemas(),
            catalog_id=manager.catalog_id,
        )

    def test_button_missing_child_gives_actionable_error(self, parser_with_components: A2UIResponseParser) -> None:
        ops = [
            {
                "version": "v0.9",
                "updateComponents": {
                    "surfaceId": "s1",
                    "components": [
                        {
                            "id": "btn1",
                            "component": "Button",
                            "text": "Click me",
                            "action": {"event": {"name": "click"}},
                        }
                    ],
                },
            }
        ]
        result = parser_with_components.validate(ops)
        assert result.is_valid is False
        assert any("btn1" in e and "Button" in e for e in result.errors)

    def test_valid_button_with_child_passes(self, parser_with_components: A2UIResponseParser) -> None:
        ops = [
            {
                "version": "v0.9",
                "updateComponents": {
                    "surfaceId": "s1",
                    "components": [
                        {"id": "root", "component": "Column", "children": ["btn1"]},
                        {
                            "id": "btn1",
                            "component": "Button",
                            "child": "btn1_text",
                            "action": {"event": {"name": "click"}},
                        },
                        {"id": "btn1_text", "component": "Text", "text": "Click me"},
                    ],
                },
            }
        ]
        result = parser_with_components.validate(ops)
        assert result.is_valid is True

    def test_update_components_without_root_fails(self, parser_with_components: A2UIResponseParser) -> None:
        ops = [
            {
                "version": "v0.9",
                "updateComponents": {
                    "surfaceId": "s1",
                    "components": [
                        {"id": "txt1", "component": "Text", "text": "Hi"},
                    ],
                },
            }
        ]
        result = parser_with_components.validate(ops)
        assert result.is_valid is False
        assert any("'root'" in e for e in result.errors)

    def test_multiple_component_errors(self, parser_with_components: A2UIResponseParser) -> None:
        ops = [
            {
                "version": "v0.9",
                "updateComponents": {
                    "surfaceId": "s1",
                    "components": [
                        {
                            "id": "btn1",
                            "component": "Button",
                            "text": "Bad",
                            "action": {"event": {"name": "x"}},
                        },
                        {"id": "txt1", "component": "Text"},
                    ],
                },
            }
        ]
        result = parser_with_components.validate(ops)
        assert result.is_valid is False
        assert len(result.errors) >= 2
        assert any("btn1" in e for e in result.errors)
        assert any("txt1" in e for e in result.errors)


class TestPerComponentValidationWithoutCatalogId:
    def test_per_component_validation_works_without_catalog_id(self) -> None:
        manager = A2UISchemaManager()
        parser = A2UIResponseParser(
            version_string="v0.9",
            server_to_client_schema=manager.server_to_client_schema,
            schema_registry=manager.build_schema_registry(),
            component_schemas=manager.get_component_schemas(),
            # catalog_id intentionally omitted — should still drill into components.
        )
        ops = [
            {
                "version": "v0.9",
                "updateComponents": {
                    "surfaceId": "s1",
                    "components": [
                        {"id": "root", "component": "Column", "children": ["btn1"]},
                        {
                            "id": "btn1",
                            "component": "Button",
                            "text": "Bad — Button has no child",
                            "action": {"event": {"name": "click"}},
                        },
                    ],
                },
            }
        ]
        result = parser.validate(ops)
        assert result.is_valid is False
        assert any("btn1" in e and "Button" in e for e in result.errors)


class TestFormatValidationError:
    def test_with_validation_errors(self) -> None:
        parser = A2UIResponseParser(version_string="v0.9")
        parse_result = A2UIParseResult(text="Hello", operations=[], has_a2ui=True)
        validation_result = A2UIValidationResult(
            is_valid=False,
            errors=["Operation 0: 'catalogId' is a required property"],
        )
        feedback = parser.format_validation_error(parse_result, validation_result)
        assert "validation errors" in feedback
        assert "'catalogId' is a required property" in feedback
        assert "fix these errors" in feedback

    def test_with_parse_error(self) -> None:
        parser = A2UIResponseParser(version_string="v0.9")
        parse_result = A2UIParseResult(
            text="Hello", operations=[], has_a2ui=True, parse_error="Invalid JSON: unexpected token"
        )
        validation_result = A2UIValidationResult(is_valid=False, errors=[])
        feedback = parser.format_validation_error(parse_result, validation_result)
        assert "Invalid JSON" in feedback

    def test_includes_version_hint(self) -> None:
        parser = A2UIResponseParser(version_string="v0.9")
        parse_result = A2UIParseResult(text="", operations=[], has_a2ui=True)
        validation_result = A2UIValidationResult(is_valid=False, errors=["missing version"])
        feedback = parser.format_validation_error(parse_result, validation_result)
        assert "v0.9" in feedback


class TestStripMarkdownFences:
    def test_no_fences(self) -> None:
        assert strip_markdown_fences('[{"key": "value"}]') == '[{"key": "value"}]'

    def test_json_fence(self) -> None:
        assert strip_markdown_fences('```json\n[{"key": "value"}]\n```') == '[{"key": "value"}]'

    def test_plain_fence(self) -> None:
        assert strip_markdown_fences('```\n{"key": "value"}\n```') == '{"key": "value"}'

    def test_fence_no_newline(self) -> None:
        assert strip_markdown_fences('```{"key": "value"}```') == '{"key": "value"}'

    def test_whitespace_around_fences(self) -> None:
        assert strip_markdown_fences("  ```json\n[1, 2]\n```  ") == "[1, 2]"

    def test_empty_string(self) -> None:
        assert strip_markdown_fences("") == ""

    def test_only_fences(self) -> None:
        assert strip_markdown_fences("```json\n```") == ""


class TestParseWithMarkdownFences:
    def test_parse_json_wrapped_in_fences(self) -> None:
        parser = A2UIResponseParser(version_string="v0.9")
        response = (
            "Here is your UI.\n---a2ui_JSON---\n```json\n"
            '[{"version": "v0.9", "deleteSurface": {"surfaceId": "s1"}}]\n```'
        )
        result = parser.parse(response)
        assert result.has_a2ui is True
        assert len(result.operations) == 1
        assert result.parse_error is None

    def test_parse_plain_fence(self) -> None:
        parser = A2UIResponseParser(version_string="v0.9")
        response = 'Text\n---a2ui_JSON---\n```\n{"version": "v0.9", "deleteSurface": {"surfaceId": "s1"}}\n```'
        result = parser.parse(response)
        assert result.has_a2ui is True
        assert len(result.operations) == 1
        assert result.parse_error is None
