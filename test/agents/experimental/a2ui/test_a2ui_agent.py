# Copyright (c) 2023 - 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from autogen.agents.experimental.a2ui import A2UIAction, A2UIAgent


class TestA2UIAgent:
    def test_default_init(self) -> None:
        agent = A2UIAgent(name="test_agent", llm_config=False)
        assert agent.name == "test_agent"
        assert agent.protocol_version == "v0.9"
        assert "v0_9" in agent.catalog_id

    def test_system_message_contains_a2ui(self) -> None:
        agent = A2UIAgent(name="test_agent", llm_config=False)
        msg = agent.system_message
        assert "A2UI" in msg
        assert "v0.9" in msg
        assert "---a2ui_JSON---" in msg
        assert "createSurface" in msg

    def test_custom_system_message_prepended(self) -> None:
        agent = A2UIAgent(
            name="test_agent",
            system_message="You are a restaurant agent.",
            llm_config=False,
        )
        msg = agent.system_message
        assert msg.startswith("You are a restaurant agent.")
        assert "A2UI Response Format" in msg

    def test_unsupported_version_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported A2UI protocol version"):
            A2UIAgent(name="test_agent", protocol_version="v0.7", llm_config=False)

    def test_custom_catalog_id_from_catalog(self) -> None:
        agent = A2UIAgent(
            name="test_agent",
            custom_catalog={"$id": "https://mycompany.com/custom.json", "components": {}},
            llm_config=False,
        )
        assert agent.catalog_id == "https://mycompany.com/custom.json"
        assert "mycompany.com/custom.json" in agent.system_message

    def test_custom_catalog_without_id_raises(self) -> None:
        with pytest.raises(ValueError, match="Custom catalog must include"):
            A2UIAgent(
                name="test_agent",
                custom_catalog={"components": {}},
                llm_config=False,
            )

    def test_custom_delimiter(self) -> None:
        agent = A2UIAgent(
            name="test_agent",
            response_delimiter="<<<A2UI>>>",
            llm_config=False,
        )
        assert "<<<A2UI>>>" in agent.system_message
        assert "---a2ui_JSON---" not in agent.system_message

    def test_exclude_schema_from_prompt(self) -> None:
        agent = A2UIAgent(
            name="test_agent",
            include_schema_in_prompt=False,
            llm_config=False,
        )
        assert "A2UI Message Schema" not in agent.system_message

    def test_exclude_rules_from_prompt(self) -> None:
        agent = A2UIAgent(
            name="test_agent",
            include_rules_in_prompt=False,
            llm_config=False,
        )
        assert "Component Rules" not in agent.system_message

    def test_schema_manager_accessible(self) -> None:
        agent = A2UIAgent(name="test_agent", llm_config=False)
        assert agent.schema_manager is not None
        assert agent.schema_manager.protocol_version == "v0.9"

    def test_response_parser_accessible(self) -> None:
        agent = A2UIAgent(name="test_agent", llm_config=False)
        assert agent.response_parser is not None
        result = agent.response_parser.parse("No A2UI here.")
        assert result.has_a2ui is False

    def test_response_parser_extracts_a2ui(self) -> None:
        agent = A2UIAgent(name="test_agent", llm_config=False)
        response = (
            "Here is your UI.\n---a2ui_JSON---\n"
            '[{"version": "v0.9", "createSurface": {"surfaceId": "s1", '
            '"catalogId": "test"}}]'
        )
        result = agent.response_parser.parse(response)
        assert result.has_a2ui is True
        assert len(result.operations) == 1

    def test_update_system_message_preserves_a2ui(self) -> None:
        agent = A2UIAgent(name="test_agent", llm_config=False)
        agent.update_system_message("You are now a travel agent.")
        msg = agent.system_message
        assert msg.startswith("You are now a travel agent.")
        assert "A2UI Response Format" in msg
        assert "createSurface" in msg

    def test_update_system_message_no_duplicate(self) -> None:
        agent = A2UIAgent(name="test_agent", llm_config=False)
        # If user manually includes the A2UI section, don't double it
        a2ui_section = agent.a2ui_prompt_section
        agent.update_system_message(f"Custom message.\n\n{a2ui_section}")
        msg = agent.system_message
        assert msg.count("A2UI Response Format") == 1

    def test_a2ui_prompt_section_property(self) -> None:
        agent = A2UIAgent(name="test_agent", llm_config=False)
        section = agent.a2ui_prompt_section
        assert "A2UI Response Format" in section
        assert "v0.9" in section

    def test_validation_retries_property(self) -> None:
        agent = A2UIAgent(name="test_agent", llm_config=False, validate_responses=True, validation_retries=3)
        assert agent.validation_retries == 3

    def test_import_from_experimental(self) -> None:
        from autogen.agents.experimental import A2UIAgent as ImportedAgent

        assert ImportedAgent is A2UIAgent

    def test_action_type_defaults_to_event(self) -> None:
        action = A2UIAction(name="test_action", description="Test")
        assert action.action_type == "event"

    def test_event_action_in_prompt(self) -> None:
        agent = A2UIAgent(
            name="test_agent",
            llm_config=False,
            actions=[
                A2UIAction(
                    name="book_table",
                    tool_name="book_restaurant",
                    description="Book a table",
                    example_context={"restaurant_id": "abc123"},
                ),
            ],
        )
        msg = agent.system_message
        assert "Server Events" in msg
        assert "book_table" in msg
        assert '"event"' in msg
        assert "Client Functions" not in msg

    def test_function_call_action_in_prompt(self) -> None:
        agent = A2UIAgent(
            name="test_agent",
            llm_config=False,
            actions=[
                A2UIAction(
                    name="openUrl",
                    action_type="functionCall",
                    description="Open a URL",
                    example_args={"url": "https://example.com"},
                ),
            ],
        )
        msg = agent.system_message
        assert "Client Functions" in msg
        assert "openUrl" in msg
        assert '"functionCall"' in msg
        assert "Server Events" not in msg

    def test_function_call_prompt_uses_call_not_name(self) -> None:
        """The functionCall example in the prompt must use 'call' (not 'name') per A2UI v0.9 spec."""
        agent = A2UIAgent(
            name="test_agent",
            llm_config=False,
            actions=[
                A2UIAction(
                    name="openUrl",
                    action_type="functionCall",
                    description="Open a URL",
                    example_args={"url": "https://example.com"},
                ),
            ],
        )
        msg = agent.system_message
        # The prompt example should use "call", not "name", inside functionCall
        assert '"call":' in msg or '"call": ' in msg
        assert '"returnType"' in msg

    def test_function_call_validates_against_schema(self) -> None:
        """A functionCall button with 'call' and 'returnType' should pass A2UI schema validation."""
        agent = A2UIAgent(
            name="test_agent",
            llm_config=False,
            validate_responses=True,
            actions=[
                A2UIAction(
                    name="openUrl",
                    action_type="functionCall",
                    description="Open a URL",
                    example_args={"url": "https://example.com"},
                ),
            ],
        )
        # A response with a functionCall button using the correct v0.9 format
        response = (
            "Here is a link.\n---a2ui_JSON---\n"
            "["
            '{"version": "v0.9", "createSurface": {"surfaceId": "s1", '
            '"catalogId": "https://a2ui.org/specification/v0_9/basic_catalog.json"}},'
            '{"version": "v0.9", "updateComponents": {"surfaceId": "s1", "components": ['
            '{"id": "btn", "component": "Button", "child": "btn_text", '
            '"action": {"functionCall": {"call": "openUrl", "args": {"url": "https://example.com"}, "returnType": "void"}}},'
            '{"id": "btn_text", "component": "Text", "text": "Open Link"}'
            "]}}"
            "]"
        )
        parse_result, errors = agent._validate_a2ui_response(response)
        assert parse_result.has_a2ui is True
        assert errors is None, f"Validation errors: {errors}"

    def test_mixed_actions_in_prompt(self) -> None:
        agent = A2UIAgent(
            name="test_agent",
            llm_config=False,
            actions=[
                A2UIAction(
                    name="schedule",
                    description="Schedule posts",
                    example_context={"time": "2:00 PM"},
                ),
                A2UIAction(
                    name="openUrl",
                    action_type="functionCall",
                    description="Open URL",
                    example_args={"url": "https://example.com"},
                ),
            ],
        )
        msg = agent.system_message
        assert "Server Events" in msg
        assert "Client Functions" in msg
        assert "schedule" in msg
        assert "openUrl" in msg

    def test_get_action_both_types(self) -> None:
        actions = [
            A2UIAction(name="save", description="Save data"),
            A2UIAction(name="openUrl", action_type="functionCall", description="Open URL"),
        ]
        agent = A2UIAgent(name="test_agent", llm_config=False, actions=actions)
        save_action = agent.get_action("save")
        assert save_action is not None
        assert save_action.action_type == "event"
        open_action = agent.get_action("openUrl")
        assert open_action is not None
        assert open_action.action_type == "functionCall"
        assert agent.get_action("nonexistent") is None


class TestA2UIAgentValidationRetry:
    """Tests for the validation retry loop."""

    def _make_agent(self, retries: int = 1) -> A2UIAgent:
        return A2UIAgent(
            name="test_agent",
            llm_config={"model": "gpt-4o-mini", "api_key": "fake"},
            validate_responses=True,
            validation_retries=retries,
        )

    def test_valid_response_passes_through(self) -> None:
        """A valid A2UI response should be returned immediately."""
        agent = self._make_agent()
        valid_response = (
            "Here is your UI.\n---a2ui_JSON---\n"
            '[{"version": "v0.9", "createSurface": {"surfaceId": "s1", '
            '"catalogId": "https://a2ui.org/specification/v0_9/basic_catalog.json"}}]'
        )

        # Mock generate_oai_reply to return a valid response
        call_count = 0

        def mock_generate(messages: Any = None, sender: Any = None, **kwargs: Any) -> tuple[bool, str]:
            nonlocal call_count
            call_count += 1
            return True, valid_response

        agent.generate_oai_reply = mock_generate  # type: ignore[assignment]
        final, reply = agent._a2ui_validating_reply(messages=[{"role": "user", "content": "show UI"}])
        assert final is True
        assert reply == valid_response
        assert call_count == 1  # No retry needed

    def test_plain_text_bypasses_validation(self) -> None:
        """Plain text responses (no A2UI) should pass through without validation."""
        agent = self._make_agent()

        def mock_generate(messages: Any = None, sender: Any = None, **kwargs: Any) -> tuple[bool, str]:
            return True, "Just a plain text response."

        agent.generate_oai_reply = mock_generate  # type: ignore[assignment]
        final, reply = agent._a2ui_validating_reply(messages=[{"role": "user", "content": "hello"}])
        assert final is True
        assert reply == "Just a plain text response."

    def test_retry_on_invalid_then_success(self) -> None:
        """Invalid response should trigger retry, second attempt succeeds."""
        agent = self._make_agent(retries=1)

        invalid_response = (
            "Here.\n---a2ui_JSON---\n"
            '[{"version": "v0.9", "createSurface": {"surfaceId": "s1"}}]'  # missing catalogId
        )
        valid_response = (
            "Here.\n---a2ui_JSON---\n"
            '[{"version": "v0.9", "createSurface": {"surfaceId": "s1", '
            '"catalogId": "https://a2ui.org/specification/v0_9/basic_catalog.json"}}]'
        )

        call_count = 0

        def mock_generate(messages: Any = None, sender: Any = None, **kwargs: Any) -> tuple[bool, str]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return True, invalid_response
            return True, valid_response

        agent.generate_oai_reply = mock_generate  # type: ignore[assignment]
        final, reply = agent._a2ui_validating_reply(messages=[{"role": "user", "content": "show UI"}])
        assert final is True
        assert reply == valid_response
        assert call_count == 2

    def test_retry_exhausted_returns_text_only(self) -> None:
        """When all retries fail, return the text-only portion."""
        agent = self._make_agent(retries=2)

        invalid_response = (
            "Some text here.\n---a2ui_JSON---\n"
            '[{"version": "v0.9", "createSurface": {"surfaceId": "s1"}}]'  # always invalid
        )

        def mock_generate(messages: Any = None, sender: Any = None, **kwargs: Any) -> tuple[bool, str]:
            return True, invalid_response

        agent.generate_oai_reply = mock_generate  # type: ignore[assignment]
        final, reply = agent._a2ui_validating_reply(messages=[{"role": "user", "content": "show UI"}])
        assert final is True
        assert reply == "Some text here."  # Text-only, no A2UI

    def test_retry_feeds_back_errors(self) -> None:
        """Verify that retry messages include validation error feedback."""
        agent = self._make_agent(retries=1)

        invalid_response = 'Text.\n---a2ui_JSON---\n[{"version": "v0.9", "createSurface": {"surfaceId": "s1"}}]'
        valid_response = (
            'Text.\n---a2ui_JSON---\n[{"version": "v0.9", "createSurface": {"surfaceId": "s1", "catalogId": "test"}}]'
        )

        captured_messages: list[list[dict[str, Any]]] = []
        call_count = 0

        def mock_generate(messages: Any = None, sender: Any = None, **kwargs: Any) -> tuple[bool, str]:
            nonlocal call_count
            call_count += 1
            if messages:
                captured_messages.append(list(messages))
            if call_count == 1:
                return True, invalid_response
            return True, valid_response

        agent.generate_oai_reply = mock_generate  # type: ignore[assignment]
        agent._a2ui_validating_reply(messages=[{"role": "user", "content": "show UI"}])

        # Second call should have the error feedback appended
        assert len(captured_messages) == 2
        retry_messages = captured_messages[1]
        # Should have: original user msg + assistant invalid response + user feedback
        assert len(retry_messages) == 3
        assert retry_messages[1]["role"] == "assistant"
        assert retry_messages[2]["role"] == "user"
        assert "validation errors" in retry_messages[2]["content"]

    def test_json_parse_error_triggers_retry(self) -> None:
        """A response with invalid JSON should also trigger retry."""
        agent = self._make_agent(retries=1)

        broken_json = "Text.\n---a2ui_JSON---\n{not valid json}"
        valid_response = 'Text.\n---a2ui_JSON---\n[{"version": "v0.9", "deleteSurface": {"surfaceId": "s1"}}]'

        call_count = 0

        def mock_generate(messages: Any = None, sender: Any = None, **kwargs: Any) -> tuple[bool, str]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return True, broken_json
            return True, valid_response

        agent.generate_oai_reply = mock_generate  # type: ignore[assignment]
        final, reply = agent._a2ui_validating_reply(messages=[{"role": "user", "content": "show UI"}])
        assert final is True
        assert reply == valid_response
        assert call_count == 2


class TestValidateA2UIResponse:
    """Tests for the extracted _validate_a2ui_response helper."""

    def _make_agent(self) -> A2UIAgent:
        return A2UIAgent(
            name="test_agent",
            llm_config=False,
            validate_responses=True,
        )

    def test_plain_text_returns_no_errors(self) -> None:
        agent = self._make_agent()
        parse_result, errors = agent._validate_a2ui_response("Just plain text.")
        assert errors is None
        assert parse_result.has_a2ui is False

    def test_valid_a2ui_returns_no_errors(self) -> None:
        agent = self._make_agent()
        response = (
            "UI.\n---a2ui_JSON---\n"
            '[{"version": "v0.9", "createSurface": {"surfaceId": "s1", '
            '"catalogId": "https://a2ui.org/specification/v0_9/basic_catalog.json"}}]'
        )
        parse_result, errors = agent._validate_a2ui_response(response)
        assert errors is None
        assert parse_result.has_a2ui is True

    def test_invalid_a2ui_returns_errors(self) -> None:
        agent = self._make_agent()
        response = (
            "UI.\n---a2ui_JSON---\n"
            '[{"version": "v0.9", "createSurface": {"surfaceId": "s1"}}]'  # missing catalogId
        )
        parse_result, errors = agent._validate_a2ui_response(response)
        assert errors is not None
        assert len(errors) > 0

    def test_invalid_json_returns_parse_error(self) -> None:
        agent = self._make_agent()
        response = "Text.\n---a2ui_JSON---\n{bad json}"
        parse_result, errors = agent._validate_a2ui_response(response)
        assert errors is not None
        assert any("Invalid JSON" in e for e in errors)


class TestBuildRetryMessages:
    """Tests for the extracted _build_retry_messages helper."""

    def _make_agent(self) -> A2UIAgent:
        return A2UIAgent(
            name="test_agent",
            llm_config=False,
            validate_responses=True,
        )

    def test_appends_response_and_feedback(self) -> None:
        agent = self._make_agent()
        from autogen.agents.experimental.a2ui.response_parser import A2UIParseResult

        original = [{"role": "user", "content": "show UI"}]
        parse_result = A2UIParseResult(text="Text", operations=[], has_a2ui=True)
        errors = ["missing catalogId"]

        result = agent._build_retry_messages(original, "response text", parse_result, errors)

        assert len(result) == 3
        assert result[0] == original[0]
        assert result[1] == {"role": "assistant", "content": "response text"}
        assert result[2]["role"] == "user"
        assert "validation errors" in result[2]["content"]
        assert "missing catalogId" in result[2]["content"]

    def test_does_not_mutate_original(self) -> None:
        agent = self._make_agent()
        from autogen.agents.experimental.a2ui.response_parser import A2UIParseResult

        original = [{"role": "user", "content": "show UI"}]
        parse_result = A2UIParseResult(text="Text", operations=[], has_a2ui=True)

        agent._build_retry_messages(original, "resp", parse_result, ["err"])
        assert len(original) == 1  # Original not mutated
