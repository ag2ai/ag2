# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Any

import pytest
from a2ui.core.basic_catalog import BasicCatalog

from ag2.a2ui.beta._types import A2UIVersion
from ag2.a2ui.beta.parser import (
    A2UIParseResult,
    A2UIResponseParser,
    A2UIValidationResult,
    strip_markdown_fences,
)
from ag2.a2ui.beta.schema_manager import A2UISchemaManager


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
            "Here is your UI.\n<a2ui-json>\n"
            '[{"version": "v0.9", "createSurface": {"surfaceId": "s1", '
            '"catalogId": "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json"}}]\n'
            "</a2ui-json>"
        )
        result = parser.parse(response)
        assert result.text == "Here is your UI."
        assert result.has_a2ui is True
        assert result.operations == [
            {
                "version": "v0.9",
                "createSurface": {
                    "surfaceId": "s1",
                    "catalogId": "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json",
                },
            }
        ]
        assert result.parse_error is None

    def test_parse_single_object(self) -> None:
        parser = A2UIResponseParser(version_string="v0.9")
        response = (
            "UI below.\n<a2ui-json>\n"
            '{"version": "v0.9", "createSurface": {"surfaceId": "s1", "catalogId": "test"}}\n</a2ui-json>'
        )
        result = parser.parse(response)
        assert result.has_a2ui is True
        assert len(result.operations) == 1

    def test_parse_jsonl_inside_tag(self) -> None:
        # Canonical A2UI wire format: one JSON message per line inside the tag.
        parser = A2UIResponseParser(version_string="v0.9")
        response = (
            "Here.\n<a2ui-json>\n"
            '{"version": "v0.9", "createSurface": {"surfaceId": "s1", "catalogId": "test"}}\n'
            '{"version": "v0.9", "deleteSurface": {"surfaceId": "s1"}}\n'
            "</a2ui-json>"
        )
        result = parser.parse(response)
        assert result.has_a2ui is True
        assert result.parse_error is None
        assert len(result.operations) == 2
        assert "createSurface" in result.operations[0]
        assert "deleteSurface" in result.operations[1]

    def test_parse_invalid_json(self) -> None:
        parser = A2UIResponseParser(version_string="v0.9")
        response = "Text\n<a2ui-json>\n{invalid json}\n</a2ui-json>"
        result = parser.parse(response)
        assert result.has_a2ui is True
        assert result.operations == []
        assert result.parse_error is not None
        assert "Invalid JSON" in result.parse_error

    def test_parse_non_array_non_object(self) -> None:
        parser = A2UIResponseParser(version_string="v0.9")
        response = 'Text\n<a2ui-json>\n"just a string"\n</a2ui-json>'
        result = parser.parse(response)
        assert result.has_a2ui is True
        assert result.operations == []
        assert result.parse_error is not None
        assert "Expected JSON" in result.parse_error

    @pytest.mark.parametrize(
        "block",
        [
            pytest.param('[{"deleteSurface": {"surfaceId": "s"}}, 1]', id="array-with-a-scalar"),
            pytest.param('[{"deleteSurface": {"surfaceId": "s"}}, {"bogus": {}}]', id="array-with-an-unknown-envelope"),
            pytest.param('{"version": "v0.9"}', id="object-without-a-payload"),
            pytest.param('{"createSurface": {}, "deleteSurface": {}}', id="object-with-two-payloads"),
            pytest.param('{"deleteSurface": {"surfaceId": "s"}}\n{"bogus": {}}', id="jsonl-with-an-unknown-envelope"),
        ],
    )
    def test_parse_rejects_a_block_that_is_not_made_of_messages(self, block: str) -> None:
        parser = A2UIResponseParser(version_string="v0.9")
        result = parser.parse(f"Text\n<a2ui-json>\n{block}\n</a2ui-json>")
        assert result.has_a2ui is True
        assert result.operations == []
        assert result.parse_error is not None

    def test_parse_names_the_offending_operation(self) -> None:
        parser = A2UIResponseParser(version_string="v0.9")
        result = parser.parse('<a2ui-json>[{"deleteSurface": {"surfaceId": "s"}}, {"bogus": {}}]</a2ui-json>')
        assert result.parse_error is not None
        assert result.parse_error.startswith("Operation 1 is not an A2UI message")

    def test_parse_empty_inside_tag(self) -> None:
        parser = A2UIResponseParser(version_string="v0.9")
        result = parser.parse("Text\n<a2ui-json>\n</a2ui-json>")
        assert result.has_a2ui is False
        assert result.text == "Text"

    def test_parse_missing_close_tag_is_tolerated(self) -> None:
        # The model forgot </a2ui-json>: take the block to the end of the response.
        parser = A2UIResponseParser(version_string="v0.9")
        response = 'Prose <a2ui-json>[{"version": "v0.9", "deleteSurface": {"surfaceId": "s1"}}]'
        result = parser.parse(response)
        assert result.text == "Prose"
        assert result.has_a2ui is True
        assert len(result.operations) == 1
        assert result.parse_error is None

    def test_parse_prose_before_and_after_tag(self) -> None:
        parser = A2UIResponseParser(version_string="v0.9")
        response = 'Before. <a2ui-json>[{"version": "v0.9", "deleteSurface": {"surfaceId": "s1"}}]</a2ui-json> After.'
        result = parser.parse(response)
        assert result.text == "Before. After."
        assert result.has_a2ui is True
        assert len(result.operations) == 1

    def test_parse_multiple_operations(self) -> None:
        parser = A2UIResponseParser(version_string="v0.9")
        response = (
            "Here.\n<a2ui-json>\n"
            '[{"version": "v0.9", "createSurface": {"surfaceId": "s1", "catalogId": "test"}}, '
            '{"version": "v0.9", "updateComponents": {"surfaceId": "s1", '
            '"components": [{"id": "root", "component": "Text", "text": "Hello"}]}}]\n'
            "</a2ui-json>"
        )
        result = parser.parse(response)
        assert len(result.operations) == 2
        assert "createSurface" in result.operations[0]
        assert "updateComponents" in result.operations[1]


VERSIONS: list[A2UIVersion] = ["v0.9", "v0.9.1", "v1.0"]


def _parser(version: A2UIVersion = "v0.9") -> A2UIResponseParser:
    manager = A2UISchemaManager(protocol_version=version)
    return A2UIResponseParser(version_string=manager.version_string, catalog=manager.catalog)


def _surface(version: A2UIVersion, components: list[dict[str, Any]], surface_id: str = "s1") -> list[dict[str, Any]]:
    return [
        {"version": version, "createSurface": {"surfaceId": surface_id, "catalogId": BasicCatalog(version).id}},
        {"version": version, "updateComponents": {"surfaceId": surface_id, "components": components}},
    ]


class TestA2UIResponseParserValidation:
    @pytest.mark.parametrize("version", VERSIONS)
    def test_valid_surface(self, version: A2UIVersion) -> None:
        ops = _surface(
            version,
            [
                {"id": "root", "component": "Column", "children": ["t"]},
                {"id": "t", "component": "Text", "text": "Hi"},
            ],
        )
        result = _parser(version).validate(ops)  # type: ignore[arg-type]
        assert result.is_valid is True
        assert result.errors == []

    @pytest.mark.parametrize("version", VERSIONS)
    def test_valid_delete_surface(self, version: A2UIVersion) -> None:
        result = _parser(version).validate([{"version": version, "deleteSurface": {"surfaceId": "s1"}}])
        assert result.is_valid is True

    def test_missing_version(self) -> None:
        result = _parser().validate([{"createSurface": {"surfaceId": "s1", "catalogId": "test"}}])  # type: ignore[list-item]
        assert result.is_valid is False
        assert len(result.errors) == 1

    def test_unsupported_version(self) -> None:
        result = _parser().validate([{"version": "v9", "deleteSurface": {"surfaceId": "s1"}}])  # type: ignore[list-item]
        assert result.is_valid is False
        assert "Unsupported protocol version" in result.errors[0]

    def test_missing_required_field(self) -> None:
        result = _parser().validate([{"version": "v0.9", "createSurface": {"surfaceId": "s1"}}])
        assert result.is_valid is False
        assert "catalogId" in result.errors[0]

    def test_unknown_catalog_id(self) -> None:
        result = _parser().validate([{"version": "v0.9", "createSurface": {"surfaceId": "s1", "catalogId": "test"}}])
        assert result.is_valid is False
        assert "test" in result.errors[0]

    def test_no_catalog_always_valid(self) -> None:
        parser = A2UIResponseParser(version_string="v0.9", catalog=None)
        result = parser.validate([{"anything": "goes"}])  # type: ignore[list-item]
        assert result.is_valid is True

    def test_non_dict_operation_does_not_raise(self) -> None:
        ops = [{"version": "v0.9", "deleteSurface": {"surfaceId": "s1"}}, 123]
        result = _parser().validate(ops)  # type: ignore[arg-type]
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "Operation 1" in result.errors[0]

    def test_non_dict_update_components_does_not_raise(self) -> None:
        result = _parser().validate([{"version": "v0.9", "updateComponents": "not-an-object"}])  # type: ignore[list-item]
        assert result.is_valid is False
        assert result.errors

    def test_invalid_operations_are_reported_by_index(self) -> None:
        ops = [
            {"version": "v0.9", "deleteSurface": {"surfaceId": "s1"}},
            {"version": "v0.9", "createSurface": {"surfaceId": "s2"}},
        ]
        result = _parser().validate(ops)  # type: ignore[arg-type]
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0].startswith("Operation 1:")

    def test_v0_9_1_accepts_v0_9_version_string(self) -> None:
        # v0.9.1's version field accepts both strings.
        ops = [{"version": "v0.9", "deleteSurface": {"surfaceId": "s1"}}]
        assert _parser("v0.9.1").validate(ops).is_valid is True  # type: ignore[arg-type]


@pytest.mark.parametrize("version", VERSIONS)
class TestIntegrityValidation:
    """What ``a2ui-core`` rejects and the hand-rolled validator accepted."""

    def test_orphan_component_is_rejected(self, version: A2UIVersion) -> None:
        ops = _surface(
            version,
            [{"id": "root", "component": "Text", "text": "a"}, {"id": "lost", "component": "Text", "text": "b"}],
        )
        result = _parser(version).validate(ops)  # type: ignore[arg-type]
        assert result.is_valid is False
        assert "not reachable from 'root'" in result.errors[0]

    def test_dangling_child_id_is_rejected(self, version: A2UIVersion) -> None:
        ops = _surface(version, [{"id": "root", "component": "Column", "children": ["nope"]}])
        result = _parser(version).validate(ops)  # type: ignore[arg-type]
        assert result.is_valid is False
        assert "non-existent component 'nope'" in result.errors[0]

    def test_cycle_is_rejected(self, version: A2UIVersion) -> None:
        ops = _surface(
            version,
            [
                {"id": "root", "component": "Column", "children": ["a"]},
                {"id": "a", "component": "Column", "children": ["root"]},
            ],
        )
        result = _parser(version).validate(ops)  # type: ignore[arg-type]
        assert result.is_valid is False
        assert "Circular reference" in result.errors[0]

    def test_missing_root_is_rejected(self, version: A2UIVersion) -> None:
        ops = _surface(version, [{"id": "txt1", "component": "Text", "text": "Hi"}])
        result = _parser(version).validate(ops)  # type: ignore[arg-type]
        assert result.is_valid is False
        assert any("'root'" in e for e in result.errors)

    def test_unknown_component_is_rejected(self, version: A2UIVersion) -> None:
        ops = _surface(version, [{"id": "root", "component": "Nope"}])
        result = _parser(version).validate(ops)  # type: ignore[arg-type]
        assert result.is_valid is False
        assert "Unrecognized component type 'Nope'" in result.errors[0]

    def test_components_split_across_messages_form_one_tree(self, version: A2UIVersion) -> None:
        ops = _surface(version, [{"id": "root", "component": "Column", "children": ["a"]}])
        ops.append({
            "version": version,
            "updateComponents": {"surfaceId": "s1", "components": [{"id": "a", "component": "Text", "text": "x"}]},
        })
        result = _parser(version).validate(ops)  # type: ignore[arg-type]
        assert result.is_valid is True, result.errors

    def test_update_for_a_surface_created_earlier_is_validated(self, version: A2UIVersion) -> None:
        ops = [
            {
                "version": version,
                "updateComponents": {
                    "surfaceId": "old",
                    "components": [{"id": "root", "component": "Text", "text": "x"}],
                },
            }
        ]
        assert _parser(version).validate(ops).is_valid is True  # type: ignore[arg-type]

    def test_errors_after_a_failed_surface_are_not_repeated(self, version: A2UIVersion) -> None:
        ops = _surface(version, [{"id": "root", "component": "Nope"}])
        ops.append({"version": version, "updateDataModel": {"surfaceId": "s1", "value": {"a": 1}}})
        assert len(_parser(version).validate(ops).errors) == 1  # type: ignore[arg-type]

    def test_input_operations_are_not_modified(self, version: A2UIVersion) -> None:
        ops = _surface(version, [{"id": "root", "component": "Column", "children": ["a"]}])
        ops.append({
            "version": version,
            "updateComponents": {"surfaceId": "s1", "components": [{"id": "a", "component": "Text", "text": "x"}]},
        })
        snapshot = copy.deepcopy(ops)
        _parser(version).validate(ops)  # type: ignore[arg-type]
        assert ops == snapshot


class TestA2UIResponseParserV1Validation:
    @pytest.fixture()
    def parser_v1(self) -> A2UIResponseParser:
        return _parser("v1.0")

    def _call(self, **overrides: Any) -> dict[str, Any]:
        call: dict[str, Any] = {
            "functionCallId": "fc-1",
            "callFunction": {"@call": "openUrl", "catalogId": BasicCatalog("v1.0").id, "args": {"url": "https://x.io"}},
        }
        call.update(overrides)
        return {"version": "v1.0", "callRendererFunction": call}

    def test_call_renderer_function(self, parser_v1: A2UIResponseParser) -> None:
        result = parser_v1.validate([self._call()])  # type: ignore[list-item]
        assert result.is_valid is True, result.errors

    def test_call_renderer_function_requires_catalog_id(self, parser_v1: A2UIResponseParser) -> None:
        result = parser_v1.validate([self._call(callFunction={"@call": "openUrl"})])  # type: ignore[list-item]
        assert result.is_valid is False

    def test_call_renderer_function_requires_function_call_id(self, parser_v1: A2UIResponseParser) -> None:
        message = self._call()
        del message["callRendererFunction"]["functionCallId"]
        assert parser_v1.validate([message]).is_valid is False  # type: ignore[list-item]

    def test_agent_function_response_value(self, parser_v1: A2UIResponseParser) -> None:
        ops = [{"version": "v1.0", "agentFunctionResponse": {"functionCallId": "fc-1", "value": 42}}]
        assert parser_v1.validate(ops).is_valid is True  # type: ignore[arg-type]

    def test_agent_function_response_error(self, parser_v1: A2UIResponseParser) -> None:
        ops = [
            {
                "version": "v1.0",
                "agentFunctionResponse": {"functionCallId": "fc-1", "error": {"code": "NOT_FOUND", "message": "x"}},
            }
        ]
        assert parser_v1.validate(ops).is_valid is True  # type: ignore[arg-type]

    def test_agent_function_response_requires_function_call_id(self, parser_v1: A2UIResponseParser) -> None:
        ops = [{"version": "v1.0", "agentFunctionResponse": {"value": 1}}]
        assert parser_v1.validate(ops).is_valid is False  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "message",
        [
            {"version": "v1.0", "actionId": "a", "actionResponse": {"value": 1}},
            {"version": "v1.0", "functionCallId": "f", "callFunction": {"call": "openUrl"}},
        ],
    )
    def test_removed_v1_rc_messages_are_rejected(self, parser_v1: A2UIResponseParser, message: dict[str, Any]) -> None:
        result = parser_v1.validate([message])  # type: ignore[list-item]
        assert result.is_valid is False
        assert "exactly one update action" in result.errors[0]

    def test_create_surface_valid_in_v1(self, parser_v1: A2UIResponseParser) -> None:
        ops = [{"version": "v1.0", "createSurface": {"surfaceId": "s1", "catalogId": BasicCatalog("v1.0").id}}]
        assert parser_v1.validate(ops).is_valid is True  # type: ignore[arg-type]


class TestPerComponentValidation:
    @pytest.fixture()
    def parser_with_components(self) -> A2UIResponseParser:
        return _parser("v0.9")

    def test_button_missing_child_gives_actionable_error(self, parser_with_components: A2UIResponseParser) -> None:
        ops = _surface(
            "v0.9",
            [
                {"id": "root", "component": "Column", "children": ["btn1"]},
                {"id": "btn1", "component": "Button", "text": "Click me", "action": {"event": {"name": "click"}}},
            ],
        )
        result = parser_with_components.validate(ops)  # type: ignore[arg-type]
        assert result.is_valid is False
        assert "'child' is a required property" in result.errors[0]
        assert "btn1" in result.errors[0] and "Button" in result.errors[0]

    def test_valid_button_with_child_passes(self, parser_with_components: A2UIResponseParser) -> None:
        ops = _surface(
            "v0.9",
            [
                {"id": "root", "component": "Column", "children": ["btn1"]},
                {
                    "id": "btn1",
                    "component": "Button",
                    "child": "btn1_text",
                    "action": {"event": {"name": "click"}},
                },
                {"id": "btn1_text", "component": "Text", "text": "Click me"},
            ],
        )
        assert parser_with_components.validate(ops).is_valid is True  # type: ignore[arg-type]

    def test_every_component_error_of_a_message_is_reported(self, parser_with_components: A2UIResponseParser) -> None:
        ops = _surface(
            "v0.9",
            [
                {"id": "root", "component": "Column", "children": ["btn1", "txt1"]},
                {"id": "btn1", "component": "Button", "text": "Bad", "action": {"event": {"name": "x"}}},
                {"id": "txt1", "component": "Text"},
            ],
        )
        result = parser_with_components.validate(ops)  # type: ignore[arg-type]
        assert result.is_valid is False
        assert "btn1" in result.errors[0]
        assert "txt1" in result.errors[0]


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

    def test_single_line_fence_with_language_tag(self) -> None:
        assert strip_markdown_fences('```json {"key": "value"}```') == '{"key": "value"}'

    def test_single_line_fence_with_language_tag_before_array(self) -> None:
        assert strip_markdown_fences("```json[1, 2]```") == "[1, 2]"

    def test_single_line_fence_with_uppercase_language_tag(self) -> None:
        assert strip_markdown_fences('```JSON {"a": 1}```') == '{"a": 1}'

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
            "Here is your UI.\n<a2ui-json>\n```json\n"
            '[{"version": "v0.9", "deleteSurface": {"surfaceId": "s1"}}]\n```\n</a2ui-json>'
        )
        result = parser.parse(response)
        assert result.has_a2ui is True
        assert len(result.operations) == 1
        assert result.parse_error is None

    def test_parse_plain_fence(self) -> None:
        parser = A2UIResponseParser(version_string="v0.9")
        response = (
            'Text\n<a2ui-json>\n```\n{"version": "v0.9", "deleteSurface": {"surfaceId": "s1"}}\n```\n</a2ui-json>'
        )
        result = parser.parse(response)
        assert result.has_a2ui is True
        assert len(result.operations) == 1
        assert result.parse_error is None

    def test_parse_single_line_json_fence(self) -> None:
        parser = A2UIResponseParser(version_string="v0.9")
        response = (
            "Here is your UI.\n<a2ui-json>\n```json "
            '[{"version": "v0.9", "deleteSurface": {"surfaceId": "s1"}}]```\n</a2ui-json>'
        )
        result = parser.parse(response)
        assert result.has_a2ui is True
        assert len(result.operations) == 1
        assert result.parse_error is None
