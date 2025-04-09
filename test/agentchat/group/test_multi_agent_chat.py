# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional
from unittest.mock import MagicMock

import pytest

from autogen.agentchat.chat import ChatResult
from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.agentchat.group.context_variables import ContextVariables
from autogen.agentchat.group.group_tool_executor import GroupToolExecutor
from autogen.agentchat.group.handoffs import Handoffs
from autogen.agentchat.group.multi_agent_chat import (
    a_initiate_group_chat,
)
from autogen.agentchat.group.patterns.pattern import Pattern
from autogen.agentchat.groupchat import GroupChat, GroupChatManager
from autogen.agentchat.user_proxy_agent import UserProxyAgent


# Helper function to create a mock agent
def create_mock_agent(name: str, handoffs: Optional[Handoffs] = None) -> MagicMock:
    agent = MagicMock(spec=ConversableAgent)
    agent.name = name
    agent.context_variables = ContextVariables()
    agent.register_hook = MagicMock()
    agent.register_reply = MagicMock()
    agent.chat_messages = {}
    agent._group_manager = None
    agent.tools = []
    agent.llm_config = {"config_list": [{"model": "test-model"}]}  # Needed for _add_single_function
    agent._function_map = {}
    agent._add_single_function = MagicMock()
    agent.remove_tool_for_llm = MagicMock()

    # Mock handoffs structure
    if handoffs:
        agent.handoffs = handoffs
    else:
        agent.handoffs = MagicMock(spec=Handoffs)
        agent.handoffs.llm_conditions = []
        agent.handoffs.context_conditions = []
        agent.handoffs.after_work = None
        agent.handoffs.get_llm_conditions_requiring_wrapping = MagicMock(return_value=[])
        agent.handoffs.get_context_conditions_requiring_wrapping = MagicMock(return_value=[])
        agent.handoffs.set_llm_function_names = MagicMock()

    # Mock initiate_chat to return a dummy ChatResult
    chat_result = ChatResult(chat_history=[], cost={"cost": {}}, summary="", human_input=[])
    agent.initiate_chat = MagicMock(return_value=chat_result)

    return agent


@pytest.fixture
def agent1() -> MagicMock:
    return create_mock_agent("agent1")


@pytest.fixture
def agent2() -> MagicMock:
    return create_mock_agent("agent2")


@pytest.fixture
def agent3() -> MagicMock:
    return create_mock_agent("agent3")


@pytest.fixture
def user_proxy() -> MagicMock:
    agent = MagicMock(spec=UserProxyAgent)
    agent.name = "user_proxy"
    agent.context_variables = ContextVariables()
    agent._group_manager = None
    # Mock initiate_chat
    chat_result = ChatResult(chat_history=[], cost={"cost": {}}, summary="", human_input=[])
    agent.initiate_chat = MagicMock(return_value=chat_result)
    return agent


@pytest.fixture
def context_vars() -> ContextVariables:
    return ContextVariables(data={"initial_key": "initial_value"})


@pytest.fixture
def mock_group_chat() -> MagicMock:
    gc = MagicMock(spec=GroupChat)
    gc.messages = []
    gc.agents = []
    gc.select_speaker_prompt = MagicMock(return_value="Select speaker prompt")
    gc.agent_by_name = MagicMock(side_effect=lambda name: next((a for a in gc.agents if a.name == name), None))
    return gc


@pytest.fixture
def mock_group_chat_manager() -> MagicMock:
    manager = MagicMock(spec=GroupChatManager)
    manager.name = "manager"
    manager.context_variables = ContextVariables()
    manager.groupchat = MagicMock(spec=GroupChat)
    manager.groupchat.agents = []
    manager.llm_config = {"config_list": [{"model": "test-model"}]}
    # Mock initiate_chat for manager as well
    chat_result = ChatResult(chat_history=[], cost={"cost": {}}, summary="", human_input=[])
    manager.initiate_chat = MagicMock(return_value=chat_result)
    manager.resume = MagicMock(return_value=(None, None))  # Default resume mock
    manager.last_speaker = None
    return manager


@pytest.fixture
def mock_tool_executor() -> MagicMock:
    executor = MagicMock(spec=GroupToolExecutor)
    executor.name = "_Group_Tool_Executor"
    executor.context_variables = ContextVariables()
    executor.has_next_target = MagicMock(return_value=False)
    executor.get_next_target = MagicMock(return_value=None)
    executor.clear_next_target = MagicMock()
    executor.set_next_target = MagicMock()
    executor.register_agents_functions = MagicMock()
    return executor


@pytest.fixture
def pattern() -> MagicMock:
    pattern = MagicMock(spec=Pattern)
    return pattern


# TODO TESTS FOR INITIATE_GROUP_CHAT


# Test async function placeholder
@pytest.mark.asyncio
async def test_a_initiate_group_chat(pattern: MagicMock) -> None:
    """Test that the async function raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        await a_initiate_group_chat(pattern, "Hello", 20)
