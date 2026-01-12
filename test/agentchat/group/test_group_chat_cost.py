# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for group chat cost tracking.

These tests verify that cost/usage is properly tracked for ALL agents
in a group chat, not just the initiating agent and manager.
"""

from typing import Any, cast
from unittest.mock import MagicMock, patch

from autogen import ConversableAgent, gather_usage_summary
from autogen.agentchat.agent import Agent
from autogen.agentchat.chat import ChatResult, CostDict
from autogen.agentchat.group.multi_agent_chat import initiate_group_chat
from autogen.agentchat.group.patterns.auto import AutoPattern


def create_mock_client(model_name: str, total_cost: float, prompt_tokens: int, completion_tokens: int) -> MagicMock:
    """Create a mock LLM client with usage summary."""
    client = MagicMock()
    client.total_usage_summary = {
        "total_cost": total_cost,
        model_name: {
            "cost": total_cost,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    client.actual_usage_summary = {
        "total_cost": total_cost,
        model_name: {
            "cost": total_cost,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return client


class TestGroupChatCostTracking:
    """Test that group chat properly gathers cost from ALL agents."""

    def test_gather_usage_summary_includes_all_agents(self) -> None:
        """Test that gather_usage_summary correctly aggregates cost from multiple agents."""
        # Create mock agents with different usage
        agent1 = MagicMock()
        agent1.client = create_mock_client("gpt-4o-mini", 0.1, 100, 50)

        agent2 = MagicMock()
        agent2.client = create_mock_client("gpt-4o-mini", 0.2, 200, 100)

        agent3 = MagicMock()
        agent3.client = create_mock_client("gpt-4o", 0.5, 150, 75)

        # Agent without client (like UserProxyAgent)
        agent_no_client = MagicMock()
        agent_no_client.client = None

        # Gather usage from all agents
        usage = gather_usage_summary([agent1, agent2, agent3, agent_no_client])

        # Verify aggregation
        assert round(usage["usage_including_cached_inference"]["total_cost"], 8) == 0.8
        assert round(usage["usage_including_cached_inference"]["gpt-4o-mini"]["cost"], 8) == 0.3
        assert usage["usage_including_cached_inference"]["gpt-4o-mini"]["prompt_tokens"] == 300
        assert usage["usage_including_cached_inference"]["gpt-4o-mini"]["completion_tokens"] == 150
        assert round(usage["usage_including_cached_inference"]["gpt-4o"]["cost"], 8) == 0.5

    def test_initiate_group_chat_cost_includes_all_agents(self) -> None:
        """Test that initiate_group_chat recalculates cost to include all group agents."""
        # Create real agents
        llm_config = {"config_list": [{"model": "gpt-4o-mini", "api_key": "fake-key"}]}

        agent1 = ConversableAgent(
            name="agent1",
            system_message="You are agent 1",
            llm_config=llm_config,
            human_input_mode="NEVER",
        )
        agent2 = ConversableAgent(
            name="agent2",
            system_message="You are agent 2",
            llm_config=llm_config,
            human_input_mode="NEVER",
        )
        agent3 = ConversableAgent(
            name="agent3",
            system_message="You are agent 3",
            llm_config=llm_config,
            human_input_mode="NEVER",
        )

        # Set up mock clients with usage for each agent
        agent1.client = create_mock_client("gpt-4o-mini", 0.1, 100, 50)
        agent2.client = create_mock_client("gpt-4o-mini", 0.2, 200, 100)
        agent3.client = create_mock_client("gpt-4o-mini", 0.3, 300, 150)

        # Create pattern
        pattern = AutoPattern(
            initial_agent=agent1,
            agents=[agent1, agent2, agent3],
            group_manager_args={"llm_config": llm_config},
        )

        # Mock initiate_chat to avoid actual LLM calls
        with patch.object(ConversableAgent, "initiate_chat") as mock_initiate_chat:
            # Create a mock chat result that would normally only include 2 agents
            mock_chat_result = ChatResult()
            mock_chat_result.chat_history = [{"role": "user", "content": "test"}]
            mock_chat_result.summary = "test summary"
            # Simulate the bug: only 2 agents' cost would be included
            mock_chat_result.cost = cast(CostDict, gather_usage_summary([agent1]))  # Only one agent's cost
            mock_initiate_chat.return_value = mock_chat_result

            # Call initiate_group_chat
            chat_result, context_vars, last_speaker = initiate_group_chat(
                pattern=pattern,
                messages="Test message",
                max_rounds=1,
            )

            # The fix should recalculate cost to include ALL agents
            # Total should be 0.1 + 0.2 + 0.3 = 0.6
            # (plus any manager cost, but manager doesn't have a mock client here)
            total_cost = chat_result.cost["usage_including_cached_inference"]["total_cost"]
            assert round(total_cost, 8) == 0.6, f"Expected 0.6 total cost, got {total_cost}"

            # Verify token aggregation
            assert chat_result.cost["usage_including_cached_inference"]["gpt-4o-mini"]["prompt_tokens"] == 600
            assert chat_result.cost["usage_including_cached_inference"]["gpt-4o-mini"]["completion_tokens"] == 300

    def test_initiate_group_chat_cost_with_manager_client(self) -> None:
        """Test that manager's cost is also included when manager has an LLM client."""
        llm_config = {"config_list": [{"model": "gpt-4o-mini", "api_key": "fake-key"}]}

        agent1 = ConversableAgent(
            name="agent1",
            system_message="You are agent 1",
            llm_config=llm_config,
            human_input_mode="NEVER",
        )

        # Set up mock client for agent
        agent1.client = create_mock_client("gpt-4o-mini", 0.1, 100, 50)

        # Create pattern with manager that has llm_config
        pattern = AutoPattern(
            initial_agent=agent1,
            agents=[agent1],
            group_manager_args={"llm_config": llm_config},
        )

        with patch.object(ConversableAgent, "initiate_chat") as mock_initiate_chat:
            mock_chat_result = ChatResult()
            mock_chat_result.chat_history = [{"role": "user", "content": "test"}]
            mock_chat_result.summary = "test summary"
            mock_chat_result.cost = {
                "usage_including_cached_inference": {"total_cost": 0},
                "usage_excluding_cached_inference": {"total_cost": 0},
            }
            mock_initiate_chat.return_value = mock_chat_result

            # We need to patch prepare_group_chat to return a manager with a mock client
            original_prepare = pattern.prepare_group_chat

            def patched_prepare(*args: Any, **kwargs: Any) -> Any:
                result = original_prepare(*args, **kwargs)
                # result[8] is the manager
                manager = result[8]
                manager.client = create_mock_client("gpt-4o-mini", 0.2, 200, 100)
                return result

            with patch.object(pattern, "prepare_group_chat", side_effect=patched_prepare):
                chat_result, context_vars, last_speaker = initiate_group_chat(
                    pattern=pattern,
                    messages="Test message",
                    max_rounds=1,
                )

                # Total should be 0.1 (agent1) + 0.2 (manager) = 0.3
                total_cost = chat_result.cost["usage_including_cached_inference"]["total_cost"]
                assert round(total_cost, 8) == 0.3, f"Expected 0.3 total cost, got {total_cost}"


class TestTwoAgentVsGroupChatCostParity:
    """Test that cost tracking behaves consistently between two-agent and group chat scenarios."""

    def test_two_agent_cost_tracking(self) -> None:
        """Verify two-agent chat correctly tracks cost from both agents."""
        llm_config = {"config_list": [{"model": "gpt-4o-mini", "api_key": "fake-key"}]}

        agent1 = ConversableAgent(
            name="agent1",
            system_message="You are agent 1",
            llm_config=llm_config,
            human_input_mode="NEVER",
        )
        agent2 = ConversableAgent(
            name="agent2",
            system_message="You are agent 2",
            llm_config=llm_config,
            human_input_mode="NEVER",
        )

        agent1.client = create_mock_client("gpt-4o-mini", 0.1, 100, 50)
        agent2.client = create_mock_client("gpt-4o-mini", 0.2, 200, 100)

        # In two-agent chat, gather_usage_summary is called with [self, recipient]
        # which should include both agents
        usage = gather_usage_summary([agent1, agent2])

        assert round(usage["usage_including_cached_inference"]["total_cost"], 8) == 0.3
        assert usage["usage_including_cached_inference"]["gpt-4o-mini"]["prompt_tokens"] == 300
        assert usage["usage_including_cached_inference"]["gpt-4o-mini"]["completion_tokens"] == 150

    def test_group_chat_cost_tracking_parity(self) -> None:
        """Verify group chat with same agents has equivalent cost tracking."""
        llm_config = {"config_list": [{"model": "gpt-4o-mini", "api_key": "fake-key"}]}

        agent1 = ConversableAgent(
            name="agent1",
            system_message="You are agent 1",
            llm_config=llm_config,
            human_input_mode="NEVER",
        )
        agent2 = ConversableAgent(
            name="agent2",
            system_message="You are agent 2",
            llm_config=llm_config,
            human_input_mode="NEVER",
        )

        agent1.client = create_mock_client("gpt-4o-mini", 0.1, 100, 50)
        agent2.client = create_mock_client("gpt-4o-mini", 0.2, 200, 100)

        # Simulate what the fix does: gather from all groupchat agents + manager
        # (manager without client won't contribute cost)
        manager = MagicMock()
        manager.client = None  # Manager without LLM client

        groupchat_agents: list[Agent] = [agent1, agent2]
        all_agents: list[Agent] = list(groupchat_agents) + [manager]
        usage = gather_usage_summary(all_agents)

        # Should be same as two-agent case since manager has no client
        assert round(usage["usage_including_cached_inference"]["total_cost"], 8) == 0.3
        assert usage["usage_including_cached_inference"]["gpt-4o-mini"]["prompt_tokens"] == 300
        assert usage["usage_including_cached_inference"]["gpt-4o-mini"]["completion_tokens"] == 150
