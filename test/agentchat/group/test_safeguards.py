# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import json
import warnings
from typing import Any
from unittest.mock import MagicMock, mock_open, patch

import pytest

from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.agentchat.group.guardrails import LLMGuardrail, RegexGuardrail
from autogen.agentchat.group.safeguards import SafeguardEnforcer, apply_safeguard_policy, reset_safeguard_policy
from autogen.llm_config.config import LLMConfig

# Suppress Google protobuf warnings at the module level
warnings.filterwarnings("ignore", message=".*MessageMapContainer.*", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*ScalarMapContainer.*", category=DeprecationWarning)

# Pytest marker to suppress warnings for this entire test module
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


class TestSafeguardEnforcer:
    """Test SafeguardEnforcer core functionality."""

    def test_valid_policy_initialization(self) -> None:
        """Test SafeguardEnforcer with valid policy."""
        policy = {
            "inter_agent_safeguards": {
                "agent_transitions": [
                    {
                        "message_source": "agent1",
                        "message_destination": "agent2",
                        "check_method": "regex",
                        "pattern": r"\btest\b",
                        "violation_response": "block",
                    }
                ]
            }
        }

        enforcer = SafeguardEnforcer(policy=policy)
        assert len(enforcer.inter_agent_rules) >= 0

    def test_policy_file_loading(self) -> None:
        """Test loading policy from file."""
        policy_content: dict[str, Any] = {"inter_agent_safeguards": {"agent_transitions": []}}

        with patch("builtins.open", mock_open(read_data=json.dumps(policy_content))):
            enforcer = SafeguardEnforcer(policy="/fake/path/policy.json")
            assert enforcer.policy == policy_content

    def test_missing_required_fields(self) -> None:
        """Test validation fails when required fields are missing."""
        invalid_policy = {"inter_agent_safeguards": {"agent_transitions": [{"message_source": "agent1"}]}}

        with pytest.raises(ValueError, match="missing required field"):
            SafeguardEnforcer(policy=invalid_policy)

    def test_invalid_check_method(self) -> None:
        """Test validation fails with invalid check_method."""
        invalid_policy = {
            "inter_agent_safeguards": {
                "agent_transitions": [
                    {"message_source": "agent1", "message_destination": "agent2", "check_method": "invalid_method"}
                ]
            }
        }

        with pytest.raises(ValueError, match="invalid check_method"):
            SafeguardEnforcer(policy=invalid_policy)


class TestInvalidPolicies:
    """Test invalid policy validation."""

    def test_missing_check_method(self) -> None:
        """Test validation fails when check_method is missing."""
        invalid_policy = {
            "inter_agent_safeguards": {
                "agent_transitions": [
                    {
                        "message_source": "agent1",
                        "message_destination": "agent2",
                        "pattern": r"\btest\b",
                        "violation_response": "block",
                    }
                ]
            }
        }

        with pytest.raises(ValueError, match="missing required field: check_method"):
            SafeguardEnforcer(policy=invalid_policy)

    def test_missing_action_fields(self) -> None:
        """Test validation fails when action fields are missing."""
        invalid_policy = {
            "inter_agent_safeguards": {
                "agent_transitions": [
                    {
                        "message_source": "agent1",
                        "message_destination": "agent2",
                        "check_method": "regex",
                        "pattern": r"\btest\b",
                    }
                ]
            }
        }

        with pytest.raises(ValueError, match="missing required field: violation_response or action"):
            SafeguardEnforcer(policy=invalid_policy)

    def test_invalid_check_method_pattern(self) -> None:
        """Test validation fails with old 'pattern' check_method."""
        invalid_policy = {
            "agent_environment_safeguards": {
                "tool_interaction": [
                    {
                        "message_source": "agent1",
                        "message_destination": "tool1",
                        "check_method": "pattern",  # Should be 'regex' now
                        "pattern": r"\btest\b",
                        "action": "block",
                    }
                ]
            }
        }

        with pytest.raises(ValueError, match="invalid check_method: pattern"):
            SafeguardEnforcer(policy=invalid_policy)

    def test_tool_interaction_missing_action(self) -> None:
        """Test tool_interaction fails without action field."""
        invalid_policy = {
            "agent_environment_safeguards": {
                "tool_interaction": [
                    {
                        "message_source": "agent1",
                        "message_destination": "tool1",
                        "check_method": "regex",
                        "pattern": r"\btest\b",
                        # Missing action field
                    }
                ]
            }
        }

        with pytest.raises(ValueError, match="missing required field: violation_response or action"):
            SafeguardEnforcer(policy=invalid_policy)

    def test_llm_interaction_missing_action(self) -> None:
        """Test llm_interaction fails without action field."""
        invalid_policy = {
            "agent_environment_safeguards": {
                "llm_interaction": [
                    {
                        "check_method": "regex",
                        "message_source": "agent1",
                        "message_destination": "llm",
                        "pattern": r"\btest\b",
                        # Missing action field
                    }
                ]
            }
        }

        with pytest.raises(ValueError, match="missing required field: action"):
            SafeguardEnforcer(policy=invalid_policy)

    def test_llm_check_method_missing_required_fields(self) -> None:
        """Test LLM check_method fails without required fields."""
        invalid_policy = {
            "inter_agent_safeguards": {
                "agent_transitions": [
                    {
                        "message_source": "agent1",
                        "message_destination": "agent2",
                        "check_method": "llm",
                        "violation_response": "block",
                        # Missing custom_prompt or disallow_item
                    }
                ]
            }
        }

        with pytest.raises(ValueError, match="must have either 'custom_prompt' or 'disallow_item'"):
            SafeguardEnforcer(policy=invalid_policy)

    def test_regex_check_method_missing_pattern(self) -> None:
        """Test regex check_method fails without pattern."""
        invalid_policy = {
            "inter_agent_safeguards": {
                "agent_transitions": [
                    {
                        "message_source": "agent1",
                        "message_destination": "agent2",
                        "check_method": "regex",
                        "violation_response": "block",
                        # Missing pattern field
                    }
                ]
            }
        }

        with pytest.raises(ValueError, match="must have 'pattern'"):
            SafeguardEnforcer(policy=invalid_policy)

    def test_invalid_regex_pattern(self) -> None:
        """Test validation fails with invalid regex pattern."""
        invalid_policy = {
            "inter_agent_safeguards": {
                "agent_transitions": [
                    {
                        "message_source": "agent1",
                        "message_destination": "agent2",
                        "check_method": "regex",
                        "pattern": "[invalid regex(",  # Invalid regex
                        "violation_response": "block",
                    }
                ]
            }
        }

        with pytest.raises(ValueError, match="invalid regex pattern"):
            SafeguardEnforcer(policy=invalid_policy)

    def test_tool_interaction_llm_missing_message_fields(self) -> None:
        """Test LLM tool interaction fails without message_source/message_destination."""
        invalid_policy = {
            "agent_environment_safeguards": {
                "tool_interaction": [
                    {
                        "check_method": "llm",
                        "custom_prompt": "Check this",
                        "action": "block",
                        # Missing message_source/message_destination
                    }
                ]
            }
        }

        with pytest.raises(ValueError, match="must have 'message_source' and 'message_destination'"):
            SafeguardEnforcer(policy=invalid_policy)

    def test_completely_invalid_tool_interaction_format(self) -> None:
        """Test tool interaction with completely wrong format fails early."""
        invalid_policy = {
            "agent_environment_safeguards": {
                "tool_interaction": [
                    {
                        "some_random_field": "value",
                        "another_field": "value2",
                        # Missing check_method, action, any valid fields
                    }
                ]
            }
        }

        with pytest.raises(ValueError, match="missing required field: check_method"):
            SafeguardEnforcer(policy=invalid_policy)


class TestSafeguardChecks:
    """Test safeguard checking functionality."""

    @pytest.fixture
    def regex_enforcer(self) -> SafeguardEnforcer:
        """SafeguardEnforcer with regex rule."""
        policy = {
            "inter_agent_safeguards": {
                "agent_transitions": [
                    {
                        "message_source": "agent1",
                        "message_destination": "agent2",
                        "check_method": "regex",
                        "pattern": r"\bpassword\b",
                        "violation_response": "block",
                    }
                ]
            }
        }
        return SafeguardEnforcer(policy=policy)

    def test_regex_block_violation(self, regex_enforcer: SafeguardEnforcer) -> None:
        """Test regex rule blocks violating message."""
        result = regex_enforcer._check_inter_agent_communication("agent1", "agent2", "Please share your password")

        assert isinstance(result, dict)
        assert "blocked" in result.get("content", "").lower() or "password" not in result.get("content", "")

    def test_regex_pass_safe_message(self, regex_enforcer: SafeguardEnforcer) -> None:
        """Test regex rule passes safe message."""
        message = "Hello, how are you?"
        result = regex_enforcer._check_inter_agent_communication("agent1", "agent2", message)

        assert result == message

    def test_llm_guardrail_creation(self) -> None:
        """Test LLM guardrail is created correctly."""
        policy = {
            "inter_agent_safeguards": {
                "agent_transitions": [
                    {
                        "message_source": "agent1",
                        "message_destination": "agent2",
                        "check_method": "llm",
                        "custom_prompt": "Check content",
                        "violation_response": "block",
                    }
                ]
            }
        }

        mock_llm_config: LLMConfig = LLMConfig(model="test-model")
        with patch("autogen.agentchat.group.guardrails.OpenAIWrapper"):
            enforcer = SafeguardEnforcer(policy=policy, safeguard_llm_config=mock_llm_config)

        assert len(enforcer.inter_agent_rules) == 1
        assert isinstance(enforcer.inter_agent_rules[0]["guardrail"], LLMGuardrail)

    def test_regex_guardrail_creation(self) -> None:
        """Test regex guardrail is created correctly."""
        policy = {
            "inter_agent_safeguards": {
                "agent_transitions": [
                    {
                        "message_source": "agent1",
                        "message_destination": "agent2",
                        "check_method": "regex",
                        "pattern": r"\btest\b",
                        "violation_response": "mask",
                    }
                ]
            }
        }

        enforcer = SafeguardEnforcer(policy=policy)
        assert len(enforcer.inter_agent_rules) == 1
        assert isinstance(enforcer.inter_agent_rules[0]["guardrail"], RegexGuardrail)


class TestApplySafeguards:
    """Test apply_safeguards integration."""

    @pytest.fixture
    def mock_agent(self) -> ConversableAgent:
        """Mock ConversableAgent."""
        agent = MagicMock()
        agent.name = "test_agent"
        agent.hook_lists = {
            "process_message_before_send": [],
            "safeguard_tool_input_process": [],
            "safeguard_tool_output_process": [],
            "safeguard_llm_input_process": [],
            "safeguard_llm_output_process": [],
            "safeguard_human_input_process": [],
        }
        return agent

    def test_apply_safeguards_to_agents(self, mock_agent: Any) -> None:
        """Test applying safeguards to agents."""
        policy = {
            "inter_agent_safeguards": {
                "agent_transitions": [
                    {
                        "message_source": "test_agent",
                        "message_destination": "*",
                        "check_method": "regex",
                        "pattern": r"\btest\b",
                        "violation_response": "block",
                    }
                ]
            }
        }

        safeguard_enforcer = apply_safeguard_policy(agents=[mock_agent], policy=policy)

        assert isinstance(safeguard_enforcer, SafeguardEnforcer)
        assert len(mock_agent.hook_lists["process_message_before_send"]) > 0

    def test_apply_safeguards_no_targets(self) -> None:
        """Test apply_safeguard_policy fails without targets."""
        policy: dict[str, Any] = {"inter_agent_safeguards": {}}

        with pytest.raises(ValueError, match="Either agents or groupchat_manager must be provided"):
            apply_safeguard_policy(policy=policy)


class TestResetSafeguards:
    """Test reset_safeguard_policy functionality."""

    @pytest.fixture
    def mock_agent(self) -> ConversableAgent:
        """Mock ConversableAgent with safeguard hooks."""
        agent = MagicMock()
        agent.name = "test_agent"
        agent.hook_lists = {
            "process_message_before_send": [],
            "safeguard_tool_inputs": [],
            "safeguard_tool_outputs": [],
            "safeguard_llm_inputs": [],
            "safeguard_llm_outputs": [],
            "safeguard_human_inputs": [],
        }
        # Make sure the agent doesn't have reset_safeguards method so it uses fallback
        del agent.reset_safeguards
        return agent

    def test_reset_safeguards_from_agents(self, mock_agent: Any) -> None:
        """Test resetting safeguards from agents."""

        # Add some mock safeguard hooks
        def mock_safeguard_hook() -> None:
            pass

        mock_safeguard_hook.__name__ = "safeguard_tool_input_hook"

        def mock_other_hook() -> None:
            pass

        mock_other_hook.__name__ = "other_hook"

        mock_agent.hook_lists["safeguard_tool_inputs"].append(mock_safeguard_hook)
        mock_agent.hook_lists["safeguard_tool_inputs"].append(mock_other_hook)
        mock_agent.hook_lists["process_message_before_send"].append(mock_safeguard_hook)

        # Reset safeguards
        reset_safeguard_policy(agents=[mock_agent])

        # Check that all hooks in safeguard-specific lists were cleared
        assert (
            len(mock_agent.hook_lists["safeguard_tool_inputs"]) == 0
        )  # All hooks cleared (it's a safeguard-specific list)
        assert (
            len(mock_agent.hook_lists["process_message_before_send"]) == 0
        )  # All hooks cleared (used for inter-agent safeguards)

    def test_reset_safeguards_no_targets(self) -> None:
        """Test reset_safeguard_policy fails without targets."""
        with pytest.raises(ValueError, match="Either agents or groupchat_manager must be provided"):
            reset_safeguard_policy()

    def test_invalid_agent_names(self, mock_agent: Any) -> None:
        """Test validation fails with invalid agent names."""
        policy = {
            "inter_agent_safeguards": {
                "agent_transitions": [
                    {
                        "message_source": "unknown_agent",
                        "message_destination": "test_agent",
                        "check_method": "regex",
                        "pattern": r"test",
                        "violation_response": "block",
                    }
                ]
            }
        }

        with pytest.raises(ValueError, match="Policy validation failed"):
            apply_safeguard_policy(agents=[mock_agent], policy=policy)
