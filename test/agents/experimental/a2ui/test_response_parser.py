# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.agents.experimental.a2ui.response_parser import A2UIResponseParser, strip_markdown_fences


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
            '"catalogId": "https://a2ui.org/specification/v0_9/basic_catalog.json"}}]'
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
        from autogen.agents.experimental.a2ui.schema_manager import A2UISchemaManager

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
                    "catalogId": "https://a2ui.org/specification/v0_9/basic_catalog.json",
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
                "createSurface": {
                    "surfaceId": "s1",
                    "catalogId": "test",
                },
            },
            {"version": "v0.9", "createSurface": {"surfaceId": "s2"}},  # missing catalogId
        ]
        result = parser_with_schema.validate(ops)
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "Operation 1" in result.errors[0]


class TestPerComponentValidation:
    """Tests for per-component error drill-down in updateComponents operations."""

    @pytest.fixture()
    def parser_with_components(self) -> A2UIResponseParser:
        from autogen.agents.experimental.a2ui.schema_manager import A2UISchemaManager

        manager = A2UISchemaManager()
        return A2UIResponseParser(
            version_string="v0.9",
            server_to_client_schema=manager.server_to_client_schema,
            schema_registry=manager.build_schema_registry(),
            component_schemas=manager.get_component_schemas(),
            catalog_id=manager.catalog_id,
        )

    def test_button_missing_child_gives_actionable_error(self, parser_with_components: A2UIResponseParser) -> None:
        """A Button with 'text' instead of 'child' should produce a clear error."""
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
        """A properly formed Button with child reference should pass component-level validation."""
        ops = [
            {
                "version": "v0.9",
                "updateComponents": {
                    "surfaceId": "s1",
                    "components": [
                        {
                            "id": "btn1",
                            "component": "Button",
                            "child": "btn1_text",
                            "action": {"event": {"name": "click"}},
                        },
                        {
                            "id": "btn1_text",
                            "component": "Text",
                            "text": "Click me",
                        },
                    ],
                },
            }
        ]
        result = parser_with_components.validate(ops)
        assert result.is_valid is True

    def test_multiple_component_errors(self, parser_with_components: A2UIResponseParser) -> None:
        """Multiple invalid components should each produce their own error."""
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
                        {
                            "id": "txt1",
                            "component": "Text",
                            # missing 'text' property
                        },
                    ],
                },
            }
        ]
        result = parser_with_components.validate(ops)
        assert result.is_valid is False
        assert len(result.errors) >= 2
        assert any("btn1" in e for e in result.errors)
        assert any("txt1" in e for e in result.errors)


class TestFormatValidationError:
    def test_format_with_validation_errors(self) -> None:
        from autogen.agents.experimental.a2ui.response_parser import A2UIParseResult, A2UIValidationResult

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

    def test_format_with_parse_error(self) -> None:
        from autogen.agents.experimental.a2ui.response_parser import A2UIParseResult, A2UIValidationResult

        parser = A2UIResponseParser(version_string="v0.9")
        parse_result = A2UIParseResult(
            text="Hello", operations=[], has_a2ui=True, parse_error="Invalid JSON: unexpected token"
        )
        validation_result = A2UIValidationResult(is_valid=False, errors=[])
        feedback = parser.format_validation_error(parse_result, validation_result)
        assert "Invalid JSON" in feedback

    def test_format_includes_version_hint(self) -> None:
        from autogen.agents.experimental.a2ui.response_parser import A2UIParseResult, A2UIValidationResult

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
        """Edge case: opening fence with no newline."""
        assert strip_markdown_fences('```{"key": "value"}```') == '{"key": "value"}'

    def test_whitespace_around_fences(self) -> None:
        assert strip_markdown_fences("  ```json\n[1, 2]\n```  ") == "[1, 2]"

    def test_empty_string(self) -> None:
        assert strip_markdown_fences("") == ""

    def test_only_fences(self) -> None:
        assert strip_markdown_fences("```json\n```") == ""


class TestParseWithMarkdownFences:
    """Test that parse() strips markdown fences from JSON after the delimiter."""

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
