# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from autogen.beta import Agent
from autogen.beta.a2ui import A2UIAction, A2UIAgent
from autogen.beta.a2ui.middleware import A2UIValidationMiddleware
from autogen.beta.testing import TestConfig


class TestA2UIAgentConstruction:
    def test_default_init(self) -> None:
        agent = A2UIAgent(name="test_agent")
        assert agent.name == "test_agent"
        assert agent.protocol_version == "v0.9"
        assert "v0_9" in agent.catalog_id

    def test_is_beta_agent(self) -> None:
        agent = A2UIAgent(name="test_agent")
        assert isinstance(agent, Agent)

    def test_system_message_contains_a2ui(self) -> None:
        agent = A2UIAgent(name="test_agent")
        prompt = "\n".join(agent._system_prompt)
        assert "A2UI" in prompt
        assert "v0.9" in prompt
        assert "<a2ui-json>" in prompt
        assert "</a2ui-json>" in prompt
        assert "---a2ui_JSON---" not in prompt
        assert "createSurface" in prompt

    def test_custom_system_message_prepended(self) -> None:
        agent = A2UIAgent(
            name="test_agent",
            system_message="You are a restaurant agent.",
        )
        prompt = "\n".join(agent._system_prompt)
        assert prompt.startswith("You are a restaurant agent.")
        assert "A2UI Response Format" in prompt

    def test_unsupported_version_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported A2UI protocol version"):
            A2UIAgent(name="test_agent", protocol_version="v0.7")

    def test_custom_catalog_id_from_catalog(self) -> None:
        agent = A2UIAgent(
            name="test_agent",
            custom_catalog={"$id": "https://mycompany.com/custom.json", "components": {}},
        )
        assert agent.catalog_id == "https://mycompany.com/custom.json"
        assert "mycompany.com/custom.json" in "\n".join(agent._system_prompt)

    def test_custom_catalog_without_id_raises(self) -> None:
        with pytest.raises(ValueError, match="Custom catalog must include"):
            A2UIAgent(name="test_agent", custom_catalog={"components": {}})

    def test_prompt_uses_official_tags(self) -> None:
        agent = A2UIAgent(name="test_agent")
        prompt = "\n".join(agent._system_prompt)
        # The official A2UI "Standard Prompt Tags" wrap the UI JSON block.
        assert "<a2ui-json>" in prompt
        assert "</a2ui-json>" in prompt

    def test_exclude_schema_from_prompt(self) -> None:
        agent = A2UIAgent(name="test_agent", include_schema_in_prompt=False)
        prompt = "\n".join(agent._system_prompt)
        assert "A2UI Message Schema" not in prompt

    def test_exclude_rules_from_prompt(self) -> None:
        agent = A2UIAgent(name="test_agent", include_rules_in_prompt=False)
        prompt = "\n".join(agent._system_prompt)
        assert "Component Rules" not in prompt

    def test_schema_manager_accessible(self) -> None:
        agent = A2UIAgent(name="test_agent")
        assert agent.schema_manager is not None
        assert agent.schema_manager.protocol_version == "v0.9"

    def test_response_parser_accessible(self) -> None:
        agent = A2UIAgent(name="test_agent")
        assert agent.parser is not None
        result = agent.parser.parse("No A2UI here.")
        assert result.has_a2ui is False

    def test_parser_extracts_a2ui(self) -> None:
        agent = A2UIAgent(name="test_agent")
        response = (
            "Here is your UI.\n<a2ui-json>\n"
            '[{"version": "v0.9", "createSurface": {"surfaceId": "s1", "catalogId": "test"}}]\n'
            "</a2ui-json>"
        )
        result = agent.parser.parse(response)
        assert result.has_a2ui is True
        assert len(result.operations) == 1

    def test_a2ui_prompt_section_property(self) -> None:
        agent = A2UIAgent(name="test_agent")
        section = agent.a2ui_prompt_section
        assert "A2UI Response Format" in section
        assert "v0.9" in section

    def test_validation_retries_property(self) -> None:
        agent = A2UIAgent(name="test_agent", validate_responses=True, validation_retries=3)
        assert agent.validation_retries == 3

    def test_validation_middleware_attached(self) -> None:
        agent = A2UIAgent(name="test_agent", validate_responses=True)
        assert any(isinstance(m, A2UIValidationMiddleware) for m in agent._middleware)

    def test_no_middleware_when_validation_disabled(self) -> None:
        agent = A2UIAgent(name="test_agent", validate_responses=False)
        assert not any(isinstance(m, A2UIValidationMiddleware) for m in agent._middleware)


class TestA2UIAgentActions:
    def test_action_type_defaults_to_event(self) -> None:
        action = A2UIAction(name="test_action", description="Test")
        assert action.action_type == "event"

    def test_event_action_in_prompt(self) -> None:
        agent = A2UIAgent(
            name="test_agent",
            actions=[
                A2UIAction(
                    name="book_table",
                    tool_name="book_restaurant",
                    description="Book a table",
                    example_context={"restaurant_id": "abc123"},
                ),
            ],
        )
        prompt = "\n".join(agent._system_prompt)
        assert "Server Events" in prompt
        assert "book_table" in prompt
        assert '"event"' in prompt
        assert "Client Functions" not in prompt

    def test_function_call_action_in_prompt(self) -> None:
        agent = A2UIAgent(
            name="test_agent",
            actions=[
                A2UIAction(
                    name="openUrl",
                    action_type="functionCall",
                    description="Open a URL",
                    example_args={"url": "https://example.com"},
                ),
            ],
        )
        prompt = "\n".join(agent._system_prompt)
        assert "Client Functions" in prompt
        assert "openUrl" in prompt
        assert '"functionCall"' in prompt
        assert "Server Events" not in prompt

    def test_mixed_actions_in_prompt(self) -> None:
        agent = A2UIAgent(
            name="test_agent",
            actions=[
                A2UIAction(name="schedule", description="Schedule posts", example_context={"time": "2:00 PM"}),
                A2UIAction(
                    name="openUrl",
                    action_type="functionCall",
                    description="Open URL",
                    example_args={"url": "https://example.com"},
                ),
            ],
        )
        prompt = "\n".join(agent._system_prompt)
        assert "Server Events" in prompt
        assert "Client Functions" in prompt
        assert "schedule" in prompt
        assert "openUrl" in prompt

    def test_get_action_both_types(self) -> None:
        actions = [
            A2UIAction(name="save", description="Save data"),
            A2UIAction(name="openUrl", action_type="functionCall", description="Open URL"),
        ]
        agent = A2UIAgent(name="test_agent", actions=actions)
        save_action = agent.get_action("save")
        assert save_action is not None
        assert save_action.action_type == "event"
        open_action = agent.get_action("openUrl")
        assert open_action is not None
        assert open_action.action_type == "functionCall"
        assert agent.get_action("nonexistent") is None

    def test_function_call_prompt_uses_call_not_name(self) -> None:
        agent = A2UIAgent(
            name="test_agent",
            actions=[
                A2UIAction(
                    name="openUrl",
                    action_type="functionCall",
                    description="Open a URL",
                    example_args={"url": "https://example.com"},
                ),
            ],
        )
        prompt = "\n".join(agent._system_prompt)
        assert '"call":' in prompt or '"call": ' in prompt
        assert '"returnType"' in prompt


@pytest.mark.asyncio()
class TestA2UIAgentAsk:
    async def test_plain_text_ask(self) -> None:
        agent = A2UIAgent(
            name="test_agent",
            config=TestConfig("Hello, no UI needed."),
            validate_responses=False,
        )
        reply = await agent.ask("Hi")
        assert reply.body == "Hello, no UI needed."

    async def test_valid_a2ui_strips_prose_into_response(self) -> None:
        valid_response = (
            "Here is your UI.\n<a2ui-json>\n"
            '[{"version": "v0.9", "createSurface": {"surfaceId": "s1", '
            '"catalogId": "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json"}}]\n'
            "</a2ui-json>"
        )
        agent = A2UIAgent(
            name="test_agent",
            config=TestConfig(valid_response),
            validate_responses=True,
        )
        reply = await agent.ask("Show me a UI")
        # The durable response keeps prose only — the A2UI message rides the
        # stream as an A2UIMessageEvent (see test_middleware).
        assert reply.body == "Here is your UI."
