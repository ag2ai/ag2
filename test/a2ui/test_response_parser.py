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


class TestSurfaceCatalogValidation:
    @pytest.fixture()
    def parser(self) -> A2UIResponseParser:
        manager = A2UISchemaManager(
            protocol_version="v1.0",
            custom_catalog={
                "$id": "https://example.com/catalog.json",
                "functions": {
                    "openUrl": {
                        "type": "object",
                        "returnType": "void",
                        "allowedCallers": "agentOnly",
                        "properties": {
                            "@call": {"const": "openUrl"},
                            "args": {
                                "type": "object",
                                "properties": {"url": {"type": "string"}},
                                "required": ["url"],
                            },
                        },
                        "required": ["@call", "args"],
                    },
                },
            },
        )
        return A2UIResponseParser(
            version_string=manager.version_string,
            server_to_client_schema=manager.server_to_client_schema,
            schema_registry=manager.build_schema_registry(),
        )

    @pytest.mark.parametrize("message_key", ["createSurface", "updateComponents"])
    @pytest.mark.parametrize("surface_catalog", ["basic", "custom"])
    @pytest.mark.parametrize("call_catalog", [None, "basic", "custom"])
    def test_function_inherits_the_surface_catalog_unless_explicitly_overridden(
        self,
        parser: A2UIResponseParser,
        message_key: str,
        surface_catalog: str,
        call_catalog: str | None,
    ) -> None:
        catalogs = {
            "basic": "https://a2ui.org/specification/v1_0/catalogs/basic/catalog.json",
            "custom": "https://example.com/catalog.json",
        }
        call: JsonObject = {"@call": "openUrl", "args": {"url": "https://example.com"}}
        if call_catalog is not None:
            call["catalogId"] = catalogs[call_catalog]
        components: list[JsonObject] = [
            {"id": "root", "component": "Button", "child": "label", "action": {"functionCall": call}},
            {"id": "label", "component": "Text", "text": "Open"},
        ]
        create: JsonObject = {"surfaceId": "s", "catalogId": catalogs[surface_catalog]}
        if message_key == "createSurface":
            create["components"] = components
            operations = [{"version": "v1.0", "createSurface": create}]
        else:
            operations = [
                {"version": "v1.0", "createSurface": create},
                {"version": "v1.0", "updateComponents": {"surfaceId": "s", "components": components}},
            ]

        result = parser.validate(operations)

        # The basic function is rendererOnly; the custom override is agentOnly.
        assert result.is_valid is ((call_catalog or surface_catalog) == "basic"), result.errors

    @pytest.mark.parametrize("call_catalog", [None, "basic", "custom"])
    def test_an_earlier_responses_surface_uses_only_the_available_catalog_context(
        self, parser: A2UIResponseParser, call_catalog: str | None
    ) -> None:
        basic_id = "https://a2ui.org/specification/v1_0/catalogs/basic/catalog.json"
        custom_id = "https://example.com/catalog.json"
        # A parser is shared by clients: this earlier response must not establish
        # a persistent binding for another validation call's identical surface id.
        assert parser.validate([
            {"version": "v1.0", "createSurface": {"surfaceId": "s", "catalogId": custom_id}}
        ]).is_valid
        call: JsonObject = {"@call": "openUrl", "args": {"url": "https://example.com"}}
        if call_catalog is not None:
            call["catalogId"] = basic_id if call_catalog == "basic" else custom_id
        operations = [
            {
                "version": "v1.0",
                "updateComponents": {
                    "surfaceId": "s",
                    "components": [
                        {"id": "root", "component": "Button", "child": "label", "action": {"functionCall": call}},
                        {"id": "label", "component": "Text", "text": "Open"},
                    ],
                },
            }
        ]

        result = parser.validate(operations)

        # With no binding available, the implicit call has a valid interpretation
        # in the basic catalog. An explicit custom call remains forbidden.
        assert result.is_valid is (call_catalog != "custom"), result.errors

    def test_surface_catalogs_are_kept_separate_within_a_response(self, parser: A2UIResponseParser) -> None:
        operations = [
            {
                "version": "v1.0",
                "createSurface": {
                    "surfaceId": "basic",
                    "catalogId": "https://a2ui.org/specification/v1_0/catalogs/basic/catalog.json",
                },
            },
            {
                "version": "v1.0",
                "createSurface": {"surfaceId": "custom", "catalogId": "https://example.com/catalog.json"},
            },
            {
                "version": "v1.0",
                "updateComponents": {
                    "surfaceId": "basic",
                    "components": [
                        {
                            "id": "root",
                            "component": "Button",
                            "child": "label",
                            "action": {"functionCall": {"@call": "openUrl", "args": {"url": "https://example.com"}}},
                        },
                        {"id": "label", "component": "Text", "text": "Open"},
                    ],
                },
            },
        ]

        result = parser.validate(operations)

        assert result.is_valid, result.errors

    @pytest.mark.parametrize("surface_catalog", ["basic", "custom"])
    def test_component_catalog_does_not_override_its_functions_surface_default(
        self, parser: A2UIResponseParser, surface_catalog: str
    ) -> None:
        catalogs = {
            "basic": "https://a2ui.org/specification/v1_0/catalogs/basic/catalog.json",
            "custom": "https://example.com/catalog.json",
        }
        result = parser.validate([
            {
                "version": "v1.0",
                "createSurface": {
                    "surfaceId": "s",
                    "catalogId": catalogs[surface_catalog],
                    "components": [
                        {
                            "id": "root",
                            "component": "Button",
                            "catalogId": catalogs["custom" if surface_catalog == "basic" else "basic"],
                            "child": "label",
                            "action": {"functionCall": {"@call": "openUrl", "args": {"url": "https://example.com"}}},
                        },
                        {"id": "label", "component": "Text", "text": "Open"},
                    ],
                },
            },
        ])

        assert result.is_valid is (surface_catalog == "basic"), result.errors


def test_component_can_select_basic_catalog_over_custom_surface() -> None:
    manager = A2UISchemaManager(
        protocol_version="v1.0",
        custom_catalog={
            "$id": "https://example.com/catalog.json",
            "components": {
                "Text": {
                    "type": "object",
                    "properties": {"component": {"const": "Text"}, "customText": {"type": "string"}},
                    "required": ["component", "customText"],
                },
            },
        },
    )
    parser = A2UIResponseParser(
        version_string=manager.version_string,
        server_to_client_schema=manager.server_to_client_schema,
        schema_registry=manager.build_schema_registry(),
    )

    result = parser.validate([
        {
            "version": "v1.0",
            "createSurface": {
                "surfaceId": "s",
                "catalogId": manager.catalog_id,
                "components": [
                    {
                        "id": "root",
                        "component": "Text",
                        "catalogId": "https://a2ui.org/specification/v1_0/catalogs/basic/catalog.json",
                        "text": "Hello",
                    },
                ],
            },
        },
    ])

    assert result.is_valid, result.errors


class TestComponentCatalogValidation:
    @pytest.fixture()
    def parser(self) -> A2UIResponseParser:
        manager = A2UISchemaManager(
            protocol_version="v1.0",
            custom_catalog={
                "$id": "https://example.com/catalog.json",
                "components": {
                    "Text": {
                        "type": "object",
                        "properties": {"component": {"const": "Text"}, "customText": {"type": "string"}},
                        "required": ["component", "customText"],
                    },
                },
            },
        )
        return A2UIResponseParser(
            version_string=manager.version_string,
            server_to_client_schema=manager.server_to_client_schema,
            schema_registry=manager.build_schema_registry(),
        )

    @pytest.mark.parametrize("message_key", ["createSurface", "updateComponents"])
    @pytest.mark.parametrize("surface_catalog", ["basic", "custom"])
    @pytest.mark.parametrize("component_catalog", [None, "basic", "custom", "unknown"])
    @pytest.mark.parametrize("text_field", ["text", "customText"])
    def test_components_use_the_selected_catalog_definition(
        self,
        parser: A2UIResponseParser,
        message_key: str,
        surface_catalog: str,
        component_catalog: str | None,
        text_field: str,
    ) -> None:
        catalogs = {
            "basic": "https://a2ui.org/specification/v1_0/catalogs/basic/catalog.json",
            "custom": "https://example.com/catalog.json",
            "unknown": "https://unknown.example/catalog.json",
        }
        component: JsonObject = {"id": "root", "component": "Text", text_field: "Hello"}
        if component_catalog is not None:
            component["catalogId"] = catalogs[component_catalog]
        create: JsonObject = {"surfaceId": "s", "catalogId": catalogs[surface_catalog]}
        if message_key == "createSurface":
            create["components"] = [component]
            operations = [{"version": "v1.0", "createSurface": create}]
        else:
            operations = [
                {"version": "v1.0", "createSurface": create},
                {"version": "v1.0", "updateComponents": {"surfaceId": "s", "components": [component]}},
            ]

        result = parser.validate(operations)

        selected = component_catalog or surface_catalog
        expected = (selected == "basic" and text_field == "text") or (
            selected == "custom" and text_field == "customText"
        )
        assert result.is_valid is expected, result.errors

    def test_component_catalog_defaults_are_isolated_between_responses(self, parser: A2UIResponseParser) -> None:
        assert parser.validate([
            {
                "version": "v1.0",
                "createSurface": {"surfaceId": "s", "catalogId": "https://example.com/catalog.json"},
            },
        ]).is_valid

        result = parser.validate([
            {
                "version": "v1.0",
                "updateComponents": {
                    "surfaceId": "s",
                    "components": [{"id": "root", "component": "Text", "text": "Hello"}],
                },
            },
        ])

        assert result.is_valid, result.errors


class TestCatalogFunctionDefinitions:
    @pytest.fixture()
    def function_schema(self) -> JsonObject:
        return {
            "type": "object",
            "returnType": "string",
            "requiresUserActivation": False,
            "properties": {
                "@call": {"const": "lookup"},
                "args": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                },
            },
            "required": ["@call", "args"],
        }

    @pytest.mark.parametrize("call_kind", ["rpc", "binding"])
    @pytest.mark.parametrize("allowed_callers", [None, "rendererOnly", "agentOnly", "rendererOrAgent"])
    def test_callers_use_function_definitions_independently_of_schema_keys(
        self, function_schema: JsonObject, call_kind: str, allowed_callers: str | None
    ) -> None:
        if allowed_callers is not None:
            function_schema["allowedCallers"] = allowed_callers
        manager = A2UISchemaManager(
            protocol_version="v1.0",
            custom_catalog={
                "$id": "https://example.com/catalog.json",
                "functions": {"lookup/~definition": function_schema},
            },
        )
        parser = A2UIResponseParser(
            version_string=manager.version_string,
            server_to_client_schema=manager.server_to_client_schema,
            schema_registry=manager.build_schema_registry(),
        )
        call: JsonObject = {"@call": "lookup", "catalogId": manager.catalog_id, "args": {"id": "x"}}
        if call_kind == "rpc":
            operation = {"version": "v1.0", "callRendererFunction": {"functionCallId": "fc", "callFunction": call}}
        else:
            operation = {
                "version": "v1.0",
                "createSurface": {
                    "surfaceId": "s",
                    "components": [
                        {"id": "root", "component": "Button", "child": "label", "action": {"functionCall": call}},
                        {"id": "label", "component": "Text", "text": "Lookup"},
                    ],
                },
            }

        result = parser.validate([operation])

        expected = (
            allowed_callers in ("agentOnly", "rendererOrAgent")
            if call_kind == "rpc"
            else allowed_callers != "agentOnly"
        )
        assert result.is_valid is expected, result.errors

    @pytest.mark.parametrize(
        ("call", "expected_error"),
        [
            ({"@call": "missing", "args": {"id": "x"}}, "unknown @call 'missing'"),
            ({"@call": "lookup", "args": {}}, "'id' is a required property"),
        ],
    )
    def test_definition_keys_do_not_obscure_function_validation_errors(
        self, function_schema: JsonObject, call: JsonObject, expected_error: str
    ) -> None:
        function_schema["allowedCallers"] = "agentOnly"
        manager = A2UISchemaManager(
            protocol_version="v1.0",
            custom_catalog={
                "$id": "https://example.com/catalog.json",
                "functions": {"lookupDefinition": function_schema},
            },
        )
        parser = A2UIResponseParser(
            version_string=manager.version_string,
            server_to_client_schema=manager.server_to_client_schema,
            schema_registry=manager.build_schema_registry(),
        )
        result = parser.validate([
            {
                "version": "v1.0",
                "callRendererFunction": {
                    "functionCallId": "fc",
                    "callFunction": {**call, "catalogId": manager.catalog_id},
                },
            }
        ])

        assert not result.is_valid
        assert len(result.errors) == 1
        assert expected_error in result.errors[0]


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
