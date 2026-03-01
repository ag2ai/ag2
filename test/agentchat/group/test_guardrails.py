# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from autogen.agentchat.group.guardrails import (
    Guardrail,
    GuardrailResult,
    LLMGuardrail,
    RegexGuardrail,
    ToolCallLLMGuardrail,
)
from autogen.agentchat.group.targets.transition_target import TransitionTarget


class TestGuardrailResult:
    def test_init_default(self) -> None:
        """Test GuardrailResult initialization with default values."""
        result = GuardrailResult(activated=True, guardrail=MagicMock(spec=Guardrail))
        assert result.activated is True
        assert result.justification == "No justification provided"

    def test_init_with_justification(self) -> None:
        """Test GuardrailResult initialization with custom justification."""
        justification = "Custom justification message"
        result = GuardrailResult(activated=False, justification=justification, guardrail=MagicMock(spec=Guardrail))
        assert result.activated is False
        assert result.justification == justification

    def test_str_representation(self) -> None:
        """Test string representation of GuardrailResult."""
        result = GuardrailResult(
            activated=True, justification="Test justification", guardrail=MagicMock(spec=Guardrail)
        )
        expected = "Guardrail Result: True\nJustification: Test justification"
        assert str(result) == expected

    def test_parse_valid_json(self) -> None:
        """Test parsing valid JSON string to GuardrailResult."""
        json_str = '{"activated": true, "justification": "Test justification"}'
        result = GuardrailResult.parse(json_str, guardrail=MagicMock(spec=Guardrail))
        assert result.activated is True
        assert result.justification == "Test justification"

    def test_parse_valid_json_minimal(self) -> None:
        """Test parsing minimal valid JSON string."""
        json_str = '{"activated": false}'
        result = GuardrailResult.parse(json_str, guardrail=MagicMock(spec=Guardrail))
        assert result.activated is False
        assert result.justification == "No justification provided"

    def test_parse_invalid_json(self) -> None:
        """Test parsing invalid JSON string raises ValueError."""
        invalid_json = '{"activated": true, "justification": "Test"'  # Missing closing brace

        with pytest.raises(ValueError) as excinfo:
            GuardrailResult.parse(invalid_json, guardrail=MagicMock())

        assert "Failed to parse GuardrailResult from text" in str(excinfo.value)

    def test_parse_invalid_structure(self) -> None:
        """Test parsing JSON with invalid structure raises ValueError."""
        invalid_structure = '{"invalid_field": true}'

        with pytest.raises(ValueError) as excinfo:
            GuardrailResult.parse(invalid_structure, guardrail=MagicMock())

        assert "Failed to parse GuardrailResult from text" in str(excinfo.value)


class TestGuardrail:
    @pytest.fixture
    def mock_target(self) -> TransitionTarget:
        """Create a mock TransitionTarget for testing."""
        return MagicMock(spec=TransitionTarget)

    @pytest.fixture
    def concrete_guardrail(self, mock_target: TransitionTarget) -> Guardrail:
        """Create a concrete implementation of Guardrail for testing."""

        class ConcreteGuardrail(Guardrail):
            def check(self, context: str | list[dict[str, Any]]) -> GuardrailResult:
                return GuardrailResult(activated=True, justification="Test check", guardrail=self)

        return ConcreteGuardrail(name="test_guardrail", condition="test condition", target=mock_target)

    def test_init_default_activation_message(self, mock_target: TransitionTarget) -> None:
        """Test Guardrail initialization with default activation message."""

        class ConcreteGuardrail(Guardrail):
            def check(self, context: str | list[dict[str, Any]]) -> GuardrailResult:
                return GuardrailResult(activated=True, guardrail=self)

        guardrail = ConcreteGuardrail(name="test_guardrail", condition="test condition", target=mock_target)

        assert guardrail.name == "test_guardrail"
        assert guardrail.condition == "test condition"
        assert guardrail.target == mock_target
        assert guardrail.activation_message == "Guardrail 'test_guardrail' has been activated."

    def test_init_custom_activation_message(self, mock_target: TransitionTarget) -> None:
        """Test Guardrail initialization with custom activation message."""

        class ConcreteGuardrail(Guardrail):
            def check(self, context: str | list[dict[str, Any]]) -> GuardrailResult:
                return GuardrailResult(activated=True, guardrail=self)

        custom_message = "Custom activation message"
        guardrail = ConcreteGuardrail(
            name="test_guardrail", condition="test condition", target=mock_target, activation_message=custom_message
        )

        assert guardrail.activation_message == custom_message


class TestLLMGuardrail:
    @pytest.fixture
    def mock_target(self) -> TransitionTarget:
        """Create a mock TransitionTarget for testing."""
        return MagicMock(spec=TransitionTarget)

    @pytest.fixture
    def mock_llm_config(self) -> MagicMock:
        """Create a mock LLMConfig for testing."""
        mock_config = MagicMock()
        mock_config.deepcopy.return_value = mock_config
        mock_config.model_dump.return_value = {"model": "test-model"}
        return mock_config

    @pytest.fixture
    def mock_openai_wrapper(self) -> MagicMock:
        """Create a mock OpenAIWrapper for testing."""
        mock_wrapper = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"activated": true, "justification": "Test response"}'
        mock_wrapper.create.return_value = mock_response
        return mock_wrapper

    def test_init_valid_config(self, mock_target: TransitionTarget, mock_llm_config: MagicMock) -> None:
        """Test LLMGuardrail initialization with valid config."""
        with patch("autogen.agentchat.group.guardrails.OpenAIWrapper") as mock_openai:
            guardrail = LLMGuardrail(
                name="test_llm_guardrail", condition="test condition", target=mock_target, llm_config=mock_llm_config
            )

            assert guardrail.name == "test_llm_guardrail"
            assert guardrail.condition == "test condition"
            assert guardrail.target == mock_target
            assert guardrail.llm_config == mock_llm_config
            assert "test condition" in guardrail.check_prompt
            mock_openai.assert_called_once()

    def test_check_with_string_context(self, mock_target: TransitionTarget, mock_llm_config: MagicMock) -> None:
        """Test LLMGuardrail check method with string context."""
        with patch("autogen.agentchat.group.guardrails.OpenAIWrapper") as mock_openai:
            mock_wrapper = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '{"activated": true, "justification": "Condition met"}'
            mock_wrapper.create.return_value = mock_response
            mock_openai.return_value = mock_wrapper

            guardrail = LLMGuardrail(
                name="test_llm_guardrail", condition="test condition", target=mock_target, llm_config=mock_llm_config
            )

            result = guardrail.check("Test context string")

            assert result.activated is True
            assert result.justification == "Condition met"

            # Verify the correct messages were sent to the LLM
            mock_wrapper.create.assert_called_once()
            call_args = mock_wrapper.create.call_args[1]
            messages = call_args["messages"]

            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert "test condition" in messages[0]["content"]
            assert messages[1]["role"] == "user"
            assert messages[1]["content"] == "Test context string"

    def test_check_with_list_context(self, mock_target: TransitionTarget, mock_llm_config: MagicMock) -> None:
        """Test LLMGuardrail check method with list context."""
        with patch("autogen.agentchat.group.guardrails.OpenAIWrapper") as mock_openai:
            mock_wrapper = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '{"activated": false, "justification": "No violation"}'
            mock_wrapper.create.return_value = mock_response
            mock_openai.return_value = mock_wrapper

            guardrail = LLMGuardrail(
                name="test_llm_guardrail", condition="test condition", target=mock_target, llm_config=mock_llm_config
            )

            context_messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there"}]

            result = guardrail.check(context_messages)

            assert result.activated is False
            assert result.justification == "No violation"

            # Verify the correct messages were sent to the LLM
            mock_wrapper.create.assert_called_once()
            call_args = mock_wrapper.create.call_args[1]
            messages = call_args["messages"]

            assert len(messages) == 3  # system + 2 context messages
            assert messages[0]["role"] == "system"
            assert messages[1:] == context_messages

    def test_check_response_format_configuration(
        self, mock_target: TransitionTarget, mock_llm_config: MagicMock
    ) -> None:
        """Test that LLMGuardrail properly configures response format."""
        with patch("autogen.agentchat.group.guardrails.OpenAIWrapper"):
            LLMGuardrail(
                name="test_llm_guardrail", condition="test condition", target=mock_target, llm_config=mock_llm_config
            )

            # Verify that response_format was set (GuardrailCheckResponse) and config was copied
            mock_llm_config.deepcopy.assert_called_once()


class TestToolCallLLMGuardrail:
    @pytest.fixture
    def mock_target(self) -> TransitionTarget:
        return MagicMock(spec=TransitionTarget)

    @pytest.fixture
    def mock_llm_config(self) -> MagicMock:
        config = MagicMock()
        config.deepcopy.return_value = config
        config.model_dump.return_value = {"model": "test", "api_key": "test"}
        return config

    def test_init_prepends_tool_call_condition(self, mock_target: TransitionTarget, mock_llm_config: MagicMock) -> None:
        """Test that ToolCallLLMGuardrail prepends tool-call context to the condition."""
        with patch("autogen.agentchat.group.guardrails.OpenAIWrapper"):
            guardrail = ToolCallLLMGuardrail(
                name="tool_guard",
                condition="arguments contain PII.",
                target=mock_target,
                llm_config=mock_llm_config,
            )
        assert "Here are arguments to a Tool call function." in guardrail.condition
        assert "arguments contain PII." in guardrail.condition

    def test_check_returns_not_activated_when_no_tool_calls(
        self, mock_target: TransitionTarget, mock_llm_config: MagicMock
    ) -> None:
        """Test that check returns activated=False when context has no tool calls."""
        with patch("autogen.agentchat.group.guardrails.OpenAIWrapper"):
            guardrail = ToolCallLLMGuardrail(
                name="tool_guard",
                condition="test",
                target=mock_target,
                llm_config=mock_llm_config,
            )
        result = guardrail.check(context=[])
        assert result.activated is False
        assert result.guardrail is guardrail
        result = guardrail.check(context=[{"role": "user", "content": "hello"}])
        assert result.activated is False

    def test_check_delegates_to_base_when_tool_calls_present(
        self, mock_target: TransitionTarget, mock_llm_config: MagicMock
    ) -> None:
        """Test that check delegates to LLMGuardrail when tool_calls are in the last message."""
        with patch("autogen.agentchat.group.guardrails.OpenAIWrapper") as mock_wrapper:
            mock_wrapper.return_value.create.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(content='{"activated": false, "justification": "No issue."}'))]
            )
            guardrail = ToolCallLLMGuardrail(
                name="tool_guard",
                condition="test",
                target=mock_target,
                llm_config=mock_llm_config,
            )
            context: list[dict[str, Any]] = [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "tc_1",
                            "function": {"name": "run_code", "arguments": '{"code": "1+1"}'},
                            "type": "function",
                        }
                    ],
                },
            ]
            result = guardrail.check(context=context)
        assert result.activated is False
        assert result.justification == "No issue."
        assert result.guardrail is guardrail
        mock_wrapper.return_value.create.assert_called_once()


class TestRegexGuardrail:
    @pytest.fixture
    def mock_target(self) -> TransitionTarget:
        """Create a mock TransitionTarget for testing."""
        return MagicMock(spec=TransitionTarget)

    def test_init_valid_regex(self, mock_target: TransitionTarget) -> None:
        """Test RegexGuardrail initialization with valid regex."""
        regex_pattern = r"\b(password|secret)\b"
        guardrail = RegexGuardrail(name="test_regex_guardrail", condition=regex_pattern, target=mock_target)

        assert guardrail.name == "test_regex_guardrail"
        assert guardrail.condition == regex_pattern
        assert guardrail.target == mock_target
        assert isinstance(guardrail.regex, re.Pattern)

    def test_init_invalid_regex(self, mock_target: TransitionTarget) -> None:
        """Test RegexGuardrail initialization with invalid regex raises ValueError."""
        invalid_regex = r"[unclosed"

        with pytest.raises(ValueError) as excinfo:
            RegexGuardrail(name="test_regex_guardrail", condition=invalid_regex, target=mock_target)

        assert "Invalid regex pattern" in str(excinfo.value)
        assert invalid_regex in str(excinfo.value)

    def test_check_string_context_match(self, mock_target: TransitionTarget) -> None:
        """Test RegexGuardrail check with string context that matches."""
        regex_pattern = r"\b(password|secret)\b"
        guardrail = RegexGuardrail(name="test_regex_guardrail", condition=regex_pattern, target=mock_target)

        context = "Please don't share your password with anyone"
        result = guardrail.check(context)

        assert result.activated is True
        assert "Match found -> password" in result.justification

    def test_check_string_context_no_match(self, mock_target: TransitionTarget) -> None:
        """Test RegexGuardrail check with string context that doesn't match."""
        regex_pattern = r"\b(password|secret)\b"
        guardrail = RegexGuardrail(name="test_regex_guardrail", condition=regex_pattern, target=mock_target)

        context = "This is a safe message"
        result = guardrail.check(context)

        assert result.activated is False
        assert result.justification == "No match found in the context."

    def test_check_list_context_match(self, mock_target: TransitionTarget) -> None:
        """Test RegexGuardrail check with list context that matches."""
        regex_pattern = r"\b(password|secret)\b"
        guardrail = RegexGuardrail(name="test_regex_guardrail", condition=regex_pattern, target=mock_target)

        context = [
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "Your secret is safe with me"},
        ]
        result = guardrail.check(context)

        assert result.activated is True
        assert "Match found -> secret" in result.justification

    def test_check_list_context_no_match(self, mock_target: TransitionTarget) -> None:
        """Test RegexGuardrail check with list context that doesn't match."""
        regex_pattern = r"\b(password|secret)\b"
        guardrail = RegexGuardrail(name="test_regex_guardrail", condition=regex_pattern, target=mock_target)

        context = [
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "How can I help you today?"},
        ]
        result = guardrail.check(context)

        assert result.activated is False
        assert result.justification == "No match found in the context."

    def test_check_list_context_missing_content(self, mock_target: TransitionTarget) -> None:
        """Test RegexGuardrail check with list context missing content field."""
        regex_pattern = r"\b(password|secret)\b"
        guardrail = RegexGuardrail(name="test_regex_guardrail", condition=regex_pattern, target=mock_target)

        context = [
            {"role": "user"},  # Missing content field
            {"role": "assistant", "content": "Hello"},
        ]
        result = guardrail.check(context)

        assert result.activated is False
        assert result.justification == "No match found in the context."

    def test_check_case_sensitive_match(self, mock_target: TransitionTarget) -> None:
        """Test RegexGuardrail check with case-sensitive pattern."""
        regex_pattern = r"\bPASSWORD\b"  # Uppercase only
        guardrail = RegexGuardrail(name="test_regex_guardrail", condition=regex_pattern, target=mock_target)

        # Should not match lowercase
        result = guardrail.check("Enter your password")
        assert result.activated is False

        # Should match uppercase
        result = guardrail.check("Enter your PASSWORD")
        assert result.activated is True
        assert "Match found -> PASSWORD" in result.justification

    def test_check_case_insensitive_pattern(self, mock_target: TransitionTarget) -> None:
        """Test RegexGuardrail check with case-insensitive pattern."""
        regex_pattern = r"(?i)\bpassword\b"  # Case insensitive
        guardrail = RegexGuardrail(name="test_regex_guardrail", condition=regex_pattern, target=mock_target)

        # Should match both cases
        result = guardrail.check("Enter your password")
        assert result.activated is True

        result = guardrail.check("Enter your PASSWORD")
        assert result.activated is True

    def test_check_complex_regex_pattern(self, mock_target: TransitionTarget) -> None:
        """Test RegexGuardrail check with complex regex pattern."""
        # Pattern to match email addresses
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        guardrail = RegexGuardrail(name="email_guardrail", condition=email_pattern, target=mock_target)

        # Should match valid email
        result = guardrail.check("Contact me at john.doe@example.com")
        assert result.activated is True
        assert "Match found -> john.doe@example.com" in result.justification

        # Should not match invalid email format
        result = guardrail.check("Contact me at john.doe@")
        assert result.activated is False

    def test_check_multiple_matches_returns_first(self, mock_target: TransitionTarget) -> None:
        """Test that RegexGuardrail returns on first match found."""
        regex_pattern = r"\b(password|secret)\b"
        guardrail = RegexGuardrail(name="test_regex_guardrail", condition=regex_pattern, target=mock_target)

        context = [
            {"role": "user", "content": "Don't share your password"},
            {"role": "assistant", "content": "Your secret is safe"},
        ]
        result = guardrail.check(context)

        assert result.activated is True
        # Should return the first match found (password)
        assert "Match found -> password" in result.justification
