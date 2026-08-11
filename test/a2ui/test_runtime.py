# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from ag2.a2ui import a2ui_action
from ag2.a2ui._runtime import _A2UIRuntime
from ag2.a2ui.actions import A2UIEventAction, collect_action_declarations
from ag2.a2ui.middleware import A2UIExtractionMiddleware, A2UIValidationMiddleware


class TestRuntimeConstruction:
    def test_default_init(self) -> None:
        rt = _A2UIRuntime()
        assert rt.protocol_version == "v0.9"
        assert "v0_9" in rt.catalog_id

    def test_system_message_contains_a2ui(self) -> None:
        rt = _A2UIRuntime()
        prompt = rt.system_prompt_section
        assert "A2UI" in prompt
        assert "v0.9" in prompt
        assert "<a2ui-json>" in prompt
        assert "</a2ui-json>" in prompt
        assert "---a2ui_JSON---" not in prompt
        assert "createSurface" in prompt

    def test_custom_system_message_prepended(self) -> None:
        rt = _A2UIRuntime(system_message="You are a restaurant agent.")
        prompt = rt.system_prompt_section
        assert prompt.startswith("You are a restaurant agent.")
        assert "A2UI Response Format" in prompt

    def test_unsupported_version_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported A2UI protocol version"):
            _A2UIRuntime(protocol_version="v0.7")

    def test_custom_catalog_id_from_catalog(self) -> None:
        rt = _A2UIRuntime(
            custom_catalog={"$id": "https://mycompany.com/custom.json", "components": {}},
        )
        assert rt.catalog_id == "https://mycompany.com/custom.json"
        assert "mycompany.com/custom.json" in rt.system_prompt_section

    def test_custom_catalog_without_id_raises(self) -> None:
        with pytest.raises(ValueError, match="Custom catalog must include"):
            _A2UIRuntime(custom_catalog={"components": {}})

    def test_prompt_uses_official_tags(self) -> None:
        prompt = _A2UIRuntime().system_prompt_section
        # The official A2UI "Standard Prompt Tags" wrap the UI JSON block.
        assert "<a2ui-json>" in prompt
        assert "</a2ui-json>" in prompt

    def test_exclude_schema_from_prompt(self) -> None:
        prompt = _A2UIRuntime(include_schema_in_prompt=False).system_prompt_section
        assert "A2UI Message Schema" not in prompt

    def test_exclude_rules_from_prompt(self) -> None:
        prompt = _A2UIRuntime(include_rules_in_prompt=False).system_prompt_section
        assert "Component Rules" not in prompt

    def test_schema_manager_accessible(self) -> None:
        rt = _A2UIRuntime()
        assert rt.schema_manager is not None
        assert rt.schema_manager.protocol_version == "v0.9"

    def test_response_parser_accessible(self) -> None:
        rt = _A2UIRuntime()
        assert rt.parser is not None
        result = rt.parser.parse("No A2UI here.")
        assert result.has_a2ui is False

    def test_parser_extracts_a2ui(self) -> None:
        rt = _A2UIRuntime()
        response = (
            "Here is your UI.\n<a2ui-json>\n"
            '[{"version": "v0.9", "createSurface": {"surfaceId": "s1", "catalogId": "test"}}]\n'
            "</a2ui-json>"
        )
        result = rt.parser.parse(response)
        assert result.has_a2ui is True
        assert len(result.operations) == 1

    def test_system_prompt_section_property(self) -> None:
        section = _A2UIRuntime().system_prompt_section
        assert "A2UI Response Format" in section
        assert "v0.9" in section

    def test_validation_middleware_attached(self) -> None:
        rt = _A2UIRuntime(validate_responses=True, validation_retries=3)
        factories = rt.middleware_factories()
        assert len(factories) == 1
        assert isinstance(factories[0], A2UIValidationMiddleware)

    def test_extraction_middleware_when_validation_disabled(self) -> None:
        # Validation off still attaches extraction: A2UI must be published and
        # stripped from prose regardless of validation (spec: separate channels).
        rt = _A2UIRuntime(validate_responses=False)
        factories = rt.middleware_factories()
        assert len(factories) == 1
        assert isinstance(factories[0], A2UIExtractionMiddleware)


class TestRuntimeActions:
    def test_action_type_defaults_to_event(self) -> None:
        action = A2UIEventAction(name="test_action", description="Test")
        assert action.action_type == "event"

    def test_event_action_in_prompt(self) -> None:
        @a2ui_action(
            name="book_table",
            description="Book a table",
            example_context={"restaurant_id": "abc123"},
        )
        def book_table(restaurant_id: str) -> str:
            return restaurant_id

        rt = _A2UIRuntime(actions=collect_action_declarations([book_table]))
        prompt = rt.system_prompt_section
        assert "Server Events" in prompt
        assert "book_table" in prompt
        assert '"event"' in prompt
        assert "Client Functions" not in prompt

    def test_actions_declared_on_server(self) -> None:
        @a2ui_action(description="Schedule posts", example_context={"time": "2:00 PM"})
        def schedule(time: str) -> str:
            return time

        rt = _A2UIRuntime(actions=collect_action_declarations([schedule]))
        prompt = rt.system_prompt_section
        assert "Server Events" in prompt
        assert "schedule" in prompt
        assert {a.name for a in rt.actions} == {"schedule"}

    def test_get_action_resolves_registered_action(self) -> None:
        @a2ui_action(description="Save data")
        def save(value: str) -> str:
            return value

        rt = _A2UIRuntime(actions=collect_action_declarations([save]))
        save_action = rt.get_action("save")
        assert save_action is not None
        assert save_action.action_type == "event"
        assert save_action.name == "save"
        assert rt.get_action("nonexistent") is None
