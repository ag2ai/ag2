# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2.a2ui._types import JsonObject
from ag2.a2ui.parser import (
    A2UIParseResult,
    A2UIResponseParser,
    A2UIValidationResult,
    strip_markdown_fences,
)
from ag2.a2ui.schema_manager import A2UISchemaManager


def _create_surface_with_extension(key: str) -> JsonObject:
    return {
        "version": "v1.0",
        "createSurface": {
            "surfaceId": "s",
            "catalogId": "https://a2ui.org/specification/v1_0/catalogs/basic/catalog.json",
            "metadata": {"extensions": {key: {}}},
        },
    }


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

    def test_validate_non_dict_update_components_does_not_raise(self, parser_with_schema: A2UIResponseParser) -> None:
        # An ``updateComponents`` whose value is not an object (e.g. a bare
        # string) passes the membership check but must not crash the 'root'-node
        # scan with an AttributeError — it is reported as invalid instead.
        ops = [{"version": "v0.9", "updateComponents": "not-an-object"}]
        result = parser_with_schema.validate(ops)
        assert result.is_valid is False
        assert result.errors


class TestA2UIResponseParserV091Validation:
    @pytest.fixture()
    def parser_v091(self) -> A2UIResponseParser:
        manager = A2UISchemaManager(protocol_version="v0.9.1")
        return A2UIResponseParser(
            version_string=manager.version_string,
            server_to_client_schema=manager.server_to_client_schema,
            schema_registry=manager.build_schema_registry(),
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
        )

    @pytest.fixture()
    def parser_v1_agent_function(self) -> A2UIResponseParser:
        manager = A2UISchemaManager(
            protocol_version="v1.0",
            custom_catalog={
                "$id": "https://mycompany.com/cat.json",
                "functions": {
                    "lookup": {
                        "type": "object",
                        "returnType": "string",
                        "allowedCallers": "agentOnly",
                        "properties": {"@call": {"const": "lookup"}},
                        "required": ["@call"],
                    },
                },
            },
        )
        return A2UIResponseParser(
            version_string=manager.version_string,
            server_to_client_schema=manager.server_to_client_schema,
            schema_registry=manager.build_schema_registry(),
        )

    def test_call_renderer_function_accepts_a_function_the_agent_may_call(
        self, parser_v1_agent_function: A2UIResponseParser
    ) -> None:
        ops = [
            {
                "version": "v1.0",
                "callRendererFunction": {
                    "functionCallId": "fc-1",
                    "callFunction": {"@call": "lookup", "catalogId": "https://mycompany.com/cat.json"},
                },
            }
        ]
        assert parser_v1_agent_function.validate(ops).is_valid is True

    @pytest.mark.parametrize("message_key", ["createSurface", "updateComponents"])
    def test_unknown_component_property_is_reported_on_its_component(
        self, parser_v1: A2UIResponseParser, message_key: str
    ) -> None:
        # The envelope's ``oneOf`` alone only says "no branch matched"; the LLM needs the property.
        slider = {"id": "root", "component": "Slider", "min": 1, "max": 3, "step": 1, "value": {"@path": "/n"}}
        ops = [{"version": "v1.0", message_key: {"surfaceId": "s", "components": [slider]}}]

        result = parser_v1.validate(ops)

        assert result.errors == [
            "Operation 0: Component 'root' (Slider): Unevaluated properties are not allowed ('step' was unexpected)"
        ]

    def test_call_renderer_function_rejects_a_renderer_only_function(self, parser_v1: A2UIResponseParser) -> None:
        # 'openUrl' is in the v1.0 basic catalog, which leaves allowedCallers at its
        # default (rendererOnly): the renderer would reject the agent's call.
        ops = [
            {
                "version": "v1.0",
                "callRendererFunction": {
                    "functionCallId": "fc-1",
                    "callFunction": {
                        "@call": "openUrl",
                        "catalogId": "https://a2ui.org/specification/v1_0/catalogs/basic/catalog.json",
                        "args": {"url": "https://example.com"},
                    },
                },
            }
        ]
        result = parser_v1.validate(ops)
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "openUrl" in result.errors[0]

    def test_agent_function_response_is_not_generated_by_the_llm(self, parser_v1: A2UIResponseParser) -> None:
        ops = [{"version": "v1.0", "agentFunctionResponse": {"functionCallId": "fc-1", "value": 42}}]
        result = parser_v1.validate(ops)
        assert result.is_valid is False
        assert "do not emit it" in result.errors[0]

    @pytest.mark.parametrize("key", ["a\u0301", "_private", "x2", "дом"])
    def test_extension_keys_follow_uax31_identifiers(self, parser_v1: A2UIResponseParser, key: str) -> None:
        # A letter plus a combining mark is a valid identifier (XID_Continue covers marks).
        assert parser_v1.validate([_create_surface_with_extension(key)]).is_valid is True

    @pytest.mark.parametrize("key", ["\u00b2", "1x", "a-b", ""])
    def test_extension_keys_reject_non_identifiers(self, parser_v1: A2UIResponseParser, key: str) -> None:
        # "²" is \w-class but cannot start an identifier; digits and dashes never can.
        assert parser_v1.validate([_create_surface_with_extension(key)]).is_valid is False

    def test_function_response_rejects_value_and_error_together(self, parser_v1: A2UIResponseParser) -> None:
        # Spec: a FunctionResponse carries exactly one of value | error (oneOf).
        ops = [
            {
                "version": "v1.0",
                "agentFunctionResponse": {
                    "functionCallId": "fc-1",
                    "value": 1,
                    "error": {"code": "c", "message": "m"},
                },
            }
        ]
        result = parser_v1.validate(ops)
        assert result.is_valid is False

    def test_call_renderer_function_requires_function_call_id(self, parser_v1: A2UIResponseParser) -> None:
        ops = [{"version": "v1.0", "callRendererFunction": {"callFunction": {"@call": "openUrl"}}}]
        result = parser_v1.validate(ops)
        assert result.is_valid is False

    @pytest.mark.parametrize(
        "legacy",
        [
            {"functionCallId": "fc-1", "callFunction": {"call": "openUrl"}},
            {"actionId": "act-1", "actionResponse": {"value": 42}},
        ],
    )
    def test_pre_rc_function_messages_are_rejected(self, parser_v1: A2UIResponseParser, legacy: dict) -> None:
        # ``callFunction`` / ``actionResponse`` were replaced by the RC's role-based messages.
        assert parser_v1.validate([{"version": "v1.0", **legacy}]).is_valid is False

    def test_create_surface_carries_initial_components_and_data_model(self, parser_v1: A2UIResponseParser) -> None:
        ops = [
            {
                "version": "v1.0",
                "createSurface": {
                    "surfaceId": "s1",
                    "components": [{"id": "root", "component": "Text", "text": {"@path": "/greeting"}}],
                    "dataModel": {"greeting": "hi"},
                    "metadata": {"extensions": {"a2ui_trace": "t-1"}},
                },
            }
        ]
        assert parser_v1.validate(ops).is_valid is True

    def test_create_surface_inline_components_need_a_root(self, parser_v1: A2UIResponseParser) -> None:
        ops = [
            {
                "version": "v1.0",
                "createSurface": {"surfaceId": "s1", "components": [{"id": "a", "component": "Text", "text": "hi"}]},
            }
        ]
        result = parser_v1.validate(ops)
        assert result.is_valid is False
        assert "root" in result.errors[0]

    def test_create_surface_inline_component_errors_are_drilled_into(self, parser_v1: A2UIResponseParser) -> None:
        ops = [
            {
                "version": "v1.0",
                "createSurface": {"surfaceId": "s1", "components": [{"id": "root", "component": "Text", "text": 5}]},
            }
        ]
        result = parser_v1.validate(ops)
        assert result.is_valid is False
        assert result.errors[0].startswith("Operation 0: Component 'root' (Text)")

    def test_create_surface_rejects_the_removed_theme(self, parser_v1: A2UIResponseParser) -> None:
        ops = [{"version": "v1.0", "createSurface": {"surfaceId": "s1", "theme": {"primaryColor": "#fff"}}}]
        assert parser_v1.validate(ops).is_valid is False

    def test_create_surface_rejects_malformed_extension_keys(self, parser_v1: A2UIResponseParser) -> None:
        ops = [{"version": "v1.0", "createSurface": {"surfaceId": "s1", "metadata": {"extensions": {"@x": 1}}}}]
        assert parser_v1.validate(ops).is_valid is False

    def test_update_data_model_requires_value(self, parser_v1: A2UIResponseParser) -> None:
        missing = [{"version": "v1.0", "updateDataModel": {"surfaceId": "s1", "path": "/a"}}]
        deleting = [{"version": "v1.0", "updateDataModel": {"surfaceId": "s1", "path": "/a", "value": None}}]
        assert parser_v1.validate(missing).is_valid is False
        assert parser_v1.validate(deleting).is_valid is True

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


def test_per_component_validation_works_without_catalog_id() -> None:
    manager = A2UISchemaManager()
    parser = A2UIResponseParser(
        version_string="v0.9",
        server_to_client_schema=manager.server_to_client_schema,
        schema_registry=manager.build_schema_registry(),
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

    def test_tells_the_llm_not_to_address_the_correction_to_the_user(self) -> None:
        parser = A2UIResponseParser(version_string="v0.9")
        parse_result = A2UIParseResult(text="", operations=[], has_a2ui=True)
        validation_result = A2UIValidationResult(is_valid=False, errors=["missing version"])
        feedback = parser.format_validation_error(parse_result, validation_result)
        assert "do not mention the errors" in feedback


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
