# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Any, Optional, cast
from unittest.mock import MagicMock, patch

import pytest

from autogen.agentchat.agent import Agent
from autogen.agentchat.chat import ChatResult
from autogen.agentchat.conversable_agent import ConversableAgent
from autogen.agentchat.group.available_condition import AvailableCondition
from autogen.agentchat.group.context_condition import StringContextCondition
from autogen.agentchat.group.context_variables import ContextVariables
from autogen.agentchat.group.group_tool_executor import GroupToolExecutor
from autogen.agentchat.group.group_utils import (
    _create_on_condition_handoff_function,
    _run_oncontextconditions,
    cleanup_temp_user_messages,
    create_group_transition,
    create_on_condition_handoff_functions,
    determine_next_agent,
    ensure_handoff_agents_in_group,
    establish_group_agent,
    get_last_agent_speaker,
    link_agents_to_group_manager,
    make_remove_function,
    prepare_exclude_transit_messages,
    process_initial_messages,
    setup_context_variables,
    update_conditional_functions,
    wrap_agent_handoff_targets,
)
from autogen.agentchat.group.handoffs import Handoffs
from autogen.agentchat.group.llm_condition import StringLLMCondition
from autogen.agentchat.group.on_condition import OnCondition
from autogen.agentchat.group.on_context_condition import OnContextCondition
from autogen.agentchat.group.targets.group_manager_target import GroupManagerTarget
from autogen.agentchat.group.targets.transition_target import (
    AgentNameTarget,
    AgentTarget,
    NestedChatTarget,
    StayTarget,
    TerminateTarget,
    TransitionTarget,
)
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

    agent.handoffs = MagicMock()
    agent.handoffs.after_work = None

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


class TestHelperFunctions:
    def test_update_conditional_functions(self, agent1: MagicMock) -> None:
        """Test updating functions based on available condition."""
        # Setup OnCondition with an available condition
        available_cond = MagicMock(spec=AvailableCondition)
        available_cond.is_available.return_value = True
        on_cond = OnCondition(
            target=AgentTarget(agent=agent1),
            condition=StringLLMCondition(prompt="prompt"),
            available=available_cond,
            llm_function_name="test_func_name",
        )
        agent1.handoffs.llm_conditions = [on_cond]

        # Mock a tool to be removed
        mock_tool = MagicMock()
        mock_tool.name = "test_func_name"
        agent1.tools = [mock_tool]

        update_conditional_functions(agent=agent1, messages=[])

        # Assertions
        available_cond.is_available.assert_called_once_with(agent1, [])
        agent1.remove_tool_for_llm.assert_called_once_with(mock_tool)
        agent1._add_single_function.assert_called_once()
        # Check the first argument of _add_single_function (the function itself)
        assert callable(agent1._add_single_function.call_args[0][0])
        # Check the other arguments
        assert agent1._add_single_function.call_args[0][1] == "test_func_name"
        assert agent1._add_single_function.call_args[0][2] == "prompt"

        # Test when not available
        agent1.remove_tool_for_llm.reset_mock()
        agent1._add_single_function.reset_mock()
        available_cond.is_available.return_value = False

        update_conditional_functions(agent=agent1, messages=[])

        available_cond.is_available.assert_called_with(agent1, [])
        agent1.remove_tool_for_llm.assert_called_with(mock_tool)
        agent1._add_single_function.assert_not_called()

    def test_establish_group_agent(self, agent1: MagicMock) -> None:
        """Test setting up group agent attributes and hooks."""
        establish_group_agent(agent1)

        assert agent1.register_hook.call_count == 1
        assert agent1.register_hook.call_args[0][0] == "update_agent_state"
        assert agent1.register_hook.call_args[0][1] == update_conditional_functions

        assert agent1.register_reply.call_count == 1
        assert agent1.register_reply.call_args[1]["trigger"] == ([Agent, None])
        assert agent1.register_reply.call_args[1]["reply_func"] == _run_oncontextconditions
        assert agent1.register_reply.call_args[1]["position"] == 0

        assert hasattr(agent1, "_get_display_name")
        assert agent1._get_display_name() == f"Group agent --> {agent1.name}"
        assert agent1._group_is_established is True

    def test_link_agents_to_group_manager(
        self, agent1: MagicMock, agent2: MagicMock, mock_group_chat_manager: MagicMock
    ) -> None:
        """Test linking agents to the manager."""
        agents = [agent1, agent2]
        typed_agents = cast(list[Agent], agents)
        link_agents_to_group_manager(typed_agents, mock_group_chat_manager)
        assert agent1._group_manager == mock_group_chat_manager
        assert agent2._group_manager == mock_group_chat_manager

    def test_run_oncontextconditions_triggered(self, agent1: MagicMock, mock_tool_executor: MagicMock) -> None:
        """Test OnContextCondition triggering a handoff."""
        # Setup OnContextCondition
        context_cond = MagicMock(spec=StringContextCondition)
        context_cond.evaluate.return_value = True
        target = AgentTarget(agent=agent1)  # Target doesn't matter much here, just needs to be valid
        available_cond = MagicMock(spec=AvailableCondition)
        available_cond.is_available.return_value = True
        on_context_cond = OnContextCondition(target=target, condition=context_cond, available=available_cond)
        agent1.handoffs.context_conditions = [on_context_cond]
        agent1.chat_messages = {"sender": [{"role": "user", "content": "test"}]}

        # Mock the group manager and its agents list including the tool executor
        mock_manager = MagicMock()
        mock_manager.groupchat.agents = [agent1, mock_tool_executor]
        agent1._group_manager = mock_manager

        success, message = _run_oncontextconditions(agent1)

        assert success is True
        assert isinstance(message, str), "Expected message to be a string"
        assert "[Handing off to agent1]" in message
        context_cond.evaluate.assert_called_once_with(agent1.context_variables)
        mock_tool_executor.set_next_target.assert_called_once_with(target)

    def test_run_oncontextconditions_not_triggered(self, agent1: MagicMock) -> None:
        """Test OnContextCondition not triggering."""
        context_cond = MagicMock(spec=StringContextCondition)
        context_cond.evaluate.return_value = False
        available_cond = MagicMock(spec=AvailableCondition)
        available_cond.is_available.return_value = True
        on_context_cond = OnContextCondition(
            target=AgentTarget(agent=agent1), condition=context_cond, available=available_cond
        )
        agent1.handoffs.context_conditions = [on_context_cond]
        agent1.chat_messages = {"sender": [{"role": "user", "content": "test"}]}

        success, message = _run_oncontextconditions(agent1)

        assert success is False
        assert message is None
        context_cond.evaluate.assert_called_once_with(agent1.context_variables)

    def test_create_on_condition_handoff_function(self, agent1: MagicMock) -> None:
        """Test the creation of the handoff function."""
        target = AgentTarget(agent=agent1)
        func = _create_on_condition_handoff_function(target)
        assert callable(func)
        assert func() == target

    def test_create_on_condition_handoff_functions(self, agent1: MagicMock, agent2: MagicMock) -> None:
        """Test creating functions for multiple OnConditions."""
        target1 = AgentTarget(agent=agent2)
        cond1 = OnCondition(target=target1, condition=StringLLMCondition(prompt="prompt1"))
        target2 = AgentNameTarget(agent_name="agent3")
        cond2 = OnCondition(target=target2, condition=StringLLMCondition(prompt="prompt2"))
        agent1.handoffs.llm_conditions = [cond1, cond2]

        create_on_condition_handoff_functions(agent1)

        agent1.handoffs.set_llm_function_names.assert_called_once()
        assert agent1._add_single_function.call_count == 2
        # Check calls (note: function names are set by set_llm_function_names which is mocked)
        assert agent1._add_single_function.call_args_list[0][0][1] == cond1.llm_function_name
        assert agent1._add_single_function.call_args_list[0][0][2] == "prompt1"
        assert agent1._add_single_function.call_args_list[1][0][1] == cond2.llm_function_name
        assert agent1._add_single_function.call_args_list[1][0][2] == "prompt2"

    def test_ensure_handoff_agents_in_group(self, agent1: MagicMock, agent2: MagicMock) -> None:
        """Test validation of handoff targets."""
        # Valid case
        target = AgentTarget(agent=agent2)
        cond = OnCondition(target=target, condition=StringLLMCondition(prompt="prompt"))
        agent1.handoffs.llm_conditions = [cond]
        agent1.handoffs.after_work = AgentNameTarget(agent_name=agent1.name)
        ensure_handoff_agents_in_group([agent1, agent2])  # Should not raise

        # Invalid case (LLM condition)
        target_invalid = AgentNameTarget(agent_name="non_existent_agent")
        cond_invalid = OnCondition(target=target_invalid, condition=StringLLMCondition(prompt="prompt"))
        agent1.handoffs.llm_conditions = [cond_invalid]
        with pytest.raises(ValueError, match="Agent in OnCondition Hand-offs must be in the agents list"):
            ensure_handoff_agents_in_group([agent1, agent2])

        # Invalid case (Context condition)
        agent1.handoffs.llm_conditions = []
        context_cond_invalid = OnContextCondition(
            target=target_invalid, condition=StringContextCondition(variable_name="var")
        )
        agent1.handoffs.context_conditions = [context_cond_invalid]
        with pytest.raises(ValueError, match="Agent in OnContextCondition Hand-offs must be in the agents list"):
            ensure_handoff_agents_in_group([agent1, agent2])

    @patch("autogen.agentchat.group.group_utils.make_remove_function")
    def test_prepare_exclude_transit_messages(
        self, mock_make_remove: MagicMock, agent1: MagicMock, agent2: MagicMock
    ) -> None:
        """Test setting up the hook to remove transit messages."""
        remove_func = MagicMock()
        mock_make_remove.return_value = remove_func

        target = AgentTarget(agent=agent2)
        cond = OnCondition(
            target=target, condition=StringLLMCondition(prompt="prompt"), llm_function_name="handoff_func"
        )
        agent1.handoffs.llm_conditions = [cond]

        prepare_exclude_transit_messages([agent1, agent2])

        mock_make_remove.assert_called_once_with(["handoff_func"])  # Called once and applied to all agents
        agent1.register_hook.assert_called_with("process_all_messages_before_reply", remove_func)
        agent2.register_hook.assert_called_with("process_all_messages_before_reply", remove_func)

    def test_wrap_agent_handoff_targets(self) -> None:
        agent1 = ConversableAgent(name="agent1")
        agent2 = ConversableAgent(name="agent2")
        """Test wrapping NestedChatTargets."""
        # Setup NestedChatTarget in OnCondition
        nested_target = NestedChatTarget(
            nested_chat_config={
                "chat_queue": [
                    {
                        "recipient": agent1,
                        "summary_method": "reflection_with_llm",
                        "summary_prompt": "Summarise the conversation into bullet points.",
                    },
                    {
                        "recipient": agent2,
                        "message": "Write a poem about the context.",
                        "max_turns": 1,
                        "summary_method": "last_msg",
                    },
                ],
                "use_async": False,
            }
        )
        nested_target_2 = NestedChatTarget(
            nested_chat_config={
                "chat_queue": [
                    {
                        "recipient": agent1,
                        "summary_method": "reflection_with_llm",
                        "summary_prompt": "Summarise the conversation into bullet points.",
                    },
                    {
                        "recipient": agent2,
                        "message": "Write a poem about the context.",
                        "max_turns": 1,
                        "summary_method": "last_msg",
                    },
                ],
                "use_async": False,
            }
        )
        on_cond_nested = OnCondition(target=nested_target, condition=StringLLMCondition(prompt="prompt"))
        agent1.handoffs.add_llm_condition(on_cond_nested)

        on_context_cond_nested = OnContextCondition(
            target=nested_target_2, condition=StringContextCondition(variable_name="var")
        )
        agent1.handoffs.add_context_condition(on_context_cond_nested)

        wrapped_list: list[ConversableAgent] = []
        wrap_agent_handoff_targets(agent1, wrapped_list)

        # Check that the original targets were replaced
        assert isinstance(on_cond_nested.target, AgentTarget)
        assert isinstance(on_context_cond_nested.target, AgentTarget)

    def test_process_initial_messages(self, agent1: MagicMock, agent2: MagicMock, user_proxy: MagicMock) -> None:
        """Test processing of initial messages."""
        agents = [agent1, agent2]
        wrapped_agents: list[ConversableAgent] = []

        # Case 1: String message, no user_proxy -> temp user
        typed_agents = cast(list[ConversableAgent], agents)
        msgs, last_agent, names, temps = process_initial_messages("Hello", None, typed_agents, wrapped_agents)
        assert msgs == [{"role": "user", "content": "Hello"}]
        assert last_agent is not None
        assert last_agent.name == "_User"
        assert names == ["agent1", "agent2"]
        assert len(temps) == 1 and temps[0].name == "_User"

        # Case 2: String message, with user_proxy -> user_proxy is last
        msgs, last_agent, names, temps = process_initial_messages("Hello", user_proxy, typed_agents, wrapped_agents)
        assert msgs == [{"role": "user", "content": "Hello"}]
        assert last_agent == user_proxy
        assert names == ["agent1", "agent2"]
        assert temps == []

        # Case 3: List of messages, last from group agent
        initial_msgs = [{"role": "assistant", "name": "agent1", "content": "Hi back"}]
        msgs, last_agent, names, temps = process_initial_messages(
            initial_msgs, user_proxy, typed_agents, wrapped_agents
        )
        assert msgs == initial_msgs
        assert last_agent == agent1
        assert names == ["agent1", "agent2"]
        assert temps == []

        # Case 4: List of messages, last from user_proxy
        initial_msgs = [{"role": "user", "name": "user_proxy", "content": "My turn"}]
        msgs, last_agent, names, temps = process_initial_messages(
            initial_msgs, user_proxy, typed_agents, wrapped_agents
        )
        assert msgs == initial_msgs
        assert last_agent == user_proxy
        assert names == ["agent1", "agent2"]
        assert temps == []

        # Case 5: List of messages, last from unknown agent
        initial_msgs = [{"role": "assistant", "name": "unknown", "content": "Who am I?"}]
        with pytest.raises(ValueError, match="Invalid group agent name in last message: unknown"):
            process_initial_messages(initial_msgs, user_proxy, typed_agents, wrapped_agents)

        # Case 6: List of messages, last has no name (implies user, use user_proxy if available)
        initial_msgs = [{"role": "user", "content": "Question"}]
        msgs, last_agent, names, temps = process_initial_messages(
            initial_msgs, user_proxy, typed_agents, wrapped_agents
        )
        assert msgs == initial_msgs
        assert last_agent == user_proxy
        assert names == ["agent1", "agent2"]
        assert temps == []

        # Case 7: List of messages, last has no name (implies user, no user_proxy -> temp user)
        msgs, last_agent, names, temps = process_initial_messages(initial_msgs, None, typed_agents, wrapped_agents)
        assert msgs == initial_msgs
        assert last_agent is not None
        assert last_agent.name == "_User"  # Should create a temp user
        assert names == ["agent1", "agent2"]
        assert len(temps) == 1 and temps[0].name == "_User"

    def test_setup_context_variables(
        self,
        agent1: MagicMock,
        agent2: MagicMock,
        mock_tool_executor: MagicMock,
        mock_group_chat_manager: MagicMock,
        context_vars: ContextVariables,
    ) -> None:
        """Test assigning the common context variables object."""
        agents = [agent1, agent2]
        typed_agents = cast(list[ConversableAgent], agents)
        setup_context_variables(mock_tool_executor, typed_agents, mock_group_chat_manager, context_vars)

        assert agent1.context_variables is context_vars
        assert agent2.context_variables is context_vars
        assert mock_tool_executor.context_variables is context_vars
        assert mock_group_chat_manager.context_variables is context_vars

    def test_cleanup_temp_user_messages(self) -> None:
        """Test removing the temporary user name from messages."""
        chat_result = ChatResult(chat_history=[], cost={"cost": {}}, summary="", human_input=[])
        chat_result.chat_history = [
            {"role": "user", "name": "_User", "content": "Hello"},
            {"role": "assistant", "name": "agent1", "content": "Hi"},
            {"role": "user", "name": "_User", "content": "Question?"},
        ]
        cleanup_temp_user_messages(chat_result)
        assert chat_result.chat_history == [
            {"role": "user", "content": "Hello"},  # name removed
            {"role": "assistant", "name": "agent1", "content": "Hi"},
            {"role": "user", "content": "Question?"},  # name removed
        ]

        # Test with no temp user messages
        chat_result.chat_history = [
            {"role": "user", "name": "real_user", "content": "Hello"},
            {"role": "assistant", "name": "agent1", "content": "Hi"},
        ]
        original_history = copy.deepcopy(chat_result.chat_history)
        cleanup_temp_user_messages(chat_result)
        assert chat_result.chat_history == original_history  # No changes

    def test_get_last_agent_speaker(
        self, mock_group_chat: MagicMock, agent1: MagicMock, agent2: MagicMock, mock_tool_executor: MagicMock
    ) -> None:
        """Test finding the last agent speaker in messages."""
        mock_group_chat.agents = [agent1, agent2, mock_tool_executor]
        group_agent_names = ["agent1", "agent2"]

        # Case 1: Last message from agent2
        mock_group_chat.messages = [
            {"role": "user", "content": "..."},
            {"role": "assistant", "name": "agent1", "content": "..."},
            {"role": "tool", "name": "_Group_Tool_Executor", "content": "..."},
            {"role": "assistant", "name": "agent2", "content": "Final word"},
        ]
        last_speaker = get_last_agent_speaker(mock_group_chat, group_agent_names, mock_tool_executor)
        assert last_speaker == agent2

        # Case 2: Last message from tool executor, previous from agent1
        mock_group_chat.messages = [
            {"role": "user", "content": "..."},
            {"role": "assistant", "name": "agent1", "content": "..."},
            {"role": "tool", "name": "_Group_Tool_Executor", "content": "..."},
        ]
        last_speaker = get_last_agent_speaker(mock_group_chat, group_agent_names, mock_tool_executor)
        assert last_speaker == agent1

        # Case 3: No group agent found
        mock_group_chat.messages = [
            {"role": "user", "content": "..."},
            {"role": "tool", "name": "_Group_Tool_Executor", "content": "..."},
        ]
        with pytest.raises(ValueError, match="No group agent found in the message history"):
            get_last_agent_speaker(mock_group_chat, group_agent_names, mock_tool_executor)

    @patch("autogen.agentchat.group.group_utils.get_last_agent_speaker")
    def test_determine_next_agent(
        self,
        mock_get_last_agent_speaker: MagicMock,
        mock_group_chat: MagicMock,
        agent1: MagicMock,
        agent2: MagicMock,
        user_proxy: MagicMock,
        mock_tool_executor: MagicMock,
    ) -> None:
        """Test the logic for determining the next agent."""
        group_agent_names = ["agent1", "agent2"]
        mock_group_chat.agents = [agent1, agent2, user_proxy, mock_tool_executor]
        mock_get_last_agent_speaker.return_value = agent1

        # Case 1: First response -> initial agent
        mock_last_speaker = MagicMock(spec="ConversableAgent")
        mock_group_after_work = MagicMock(spec=TransitionTarget)
        next_agent = determine_next_agent(
            mock_last_speaker,
            mock_group_chat,
            agent1,
            True,
            mock_tool_executor,
            group_agent_names,
            user_proxy,
            mock_group_after_work,
        )
        assert next_agent == agent1

        # Case 2: Last message is tool call -> tool executor
        mock_group_chat.messages = [{"role": "assistant", "name": "agent1", "tool_calls": []}]
        next_agent = determine_next_agent(
            agent1,
            mock_group_chat,
            agent1,
            False,
            mock_tool_executor,
            group_agent_names,
            user_proxy,
            mock_group_after_work,
        )
        assert next_agent == mock_tool_executor

        # Case 3: Tool executor has next target (AgentTarget)
        mock_group_chat.messages = [
            {"role": "tool", "name": "_Group_Tool_Executor", "content": "..."}
        ]  # Last msg not tool call
        next_target = AgentTarget(agent=agent2)
        mock_tool_executor.has_next_target.return_value = True
        mock_tool_executor.get_next_target.return_value = next_target
        next_agent = determine_next_agent(
            mock_tool_executor,
            mock_group_chat,
            agent1,
            False,
            mock_tool_executor,
            group_agent_names,
            user_proxy,
            mock_group_after_work,
        )
        assert next_agent == agent2
        mock_tool_executor.clear_next_target.assert_called_once()

        # Case 4: Tool executor has next target (Terminate)
        mock_tool_executor.clear_next_target.reset_mock()
        awot_next_target = TerminateTarget()
        mock_tool_executor.has_next_target.return_value = True
        mock_tool_executor.get_next_target.return_value = awot_next_target
        next_agent = determine_next_agent(
            mock_tool_executor,
            mock_group_chat,
            agent1,
            False,
            mock_tool_executor,
            group_agent_names,
            user_proxy,
            mock_group_after_work,
        )
        assert next_agent is None  # Terminate -> None
        mock_tool_executor.clear_next_target.assert_called_once()

        # Reset tool executor mocks for subsequent tests
        mock_tool_executor.has_next_target.return_value = False
        mock_tool_executor.get_next_target.return_value = None
        mock_tool_executor.clear_next_target.reset_mock()

        # Case 5: User last spoke -> return to previous agent (agent1 in this mock setup)
        mock_group_chat.messages = [{"role": "user", "name": "user_proxy", "content": "..."}]
        next_agent = determine_next_agent(
            user_proxy,
            mock_group_chat,
            agent1,
            False,
            mock_tool_executor,
            group_agent_names,
            user_proxy,
            mock_group_after_work,
        )
        assert next_agent == agent1  # agent1 is returned by mock_get_last_speaker

        # Case 6: After Work (Agent level - stay)
        mock_group_chat.messages = [{"role": "assistant", "name": "agent1", "content": "..."}]
        agent1.handoffs.after_work = StayTarget()
        next_agent = determine_next_agent(
            agent1,
            mock_group_chat,
            agent1,
            False,
            mock_tool_executor,
            group_agent_names,
            user_proxy,
            mock_group_after_work,
        )
        assert next_agent == agent1

        # Case 7: After Work (Group level - terminate)
        agent1.handoffs.after_work = None  # Remove agent level
        group_after_work = TerminateTarget()
        next_agent = determine_next_agent(
            agent1, mock_group_chat, agent1, False, mock_tool_executor, group_agent_names, user_proxy, group_after_work
        )
        assert next_agent is None

        # Case 8: After Work (Group level - group_manager)
        mock_get_last_agent_speaker.reset_mock()
        group_after_work_group_manager = GroupManagerTarget()
        next_agent = determine_next_agent(
            agent1,
            mock_group_chat,
            agent1,
            False,
            mock_tool_executor,
            group_agent_names,
            user_proxy,
            group_after_work_group_manager,
        )
        assert next_agent == "auto"
        mock_get_last_agent_speaker.assert_called_with(
            mock_group_chat, group_agent_names, mock_tool_executor
        )  # No specific message

    def test_create_group_transition(
        self,
        agent1: MagicMock,
        agent2: MagicMock,
        user_proxy: MagicMock,
        mock_tool_executor: MagicMock,
        mock_group_chat: MagicMock,
    ) -> None:
        """Test the creation and behavior of the group transition function."""
        group_agent_names = ["agent1", "agent2"]
        group_after_work = TerminateTarget()

        transition_func = create_group_transition(
            agent1, mock_tool_executor, group_agent_names, user_proxy, group_after_work
        )

        # First call: should use initial_agent logic
        with patch("autogen.agentchat.group.group_utils.determine_next_agent") as mock_determine:
            mock_determine.return_value = agent1
            result1 = transition_func(
                user_proxy, mock_group_chat
            )  # last_speaker doesn't matter much here due to mocking
            mock_determine.assert_called_once_with(
                last_speaker=user_proxy,
                groupchat=mock_group_chat,
                initial_agent=agent1,
                use_initial_agent=True,  # First call
                tool_executor=mock_tool_executor,
                group_agent_names=group_agent_names,
                user_agent=user_proxy,
                group_after_work=group_after_work,
            )
            assert result1 == agent1

        # Second call: should not use initial_agent logic
        with patch("autogen.agentchat.group.group_utils.determine_next_agent") as mock_determine:
            mock_determine.return_value = agent2
            result2 = transition_func(agent1, mock_group_chat)
            mock_determine.assert_called_once_with(
                last_speaker=agent1,
                groupchat=mock_group_chat,
                initial_agent=agent1,
                use_initial_agent=False,  # Second call
                tool_executor=mock_tool_executor,
                group_agent_names=group_agent_names,
                user_agent=user_proxy,
                group_after_work=group_after_work,
            )
            assert result2 == agent2

    def test_make_remove_function(self) -> None:
        """Test the function created by make_remove_function."""
        tool_names_to_remove = ["transit_function_1", "secret_handoff"]
        remove_func = make_remove_function(tool_names_to_remove)

        messages: list[dict[str, Any]] = [
            {
                "role": "assistant",
                "content": "Thinking...",
                "tool_calls": [
                    {"id": "call_abc", "function": {"name": "normal_tool", "arguments": "{}"}},
                    {"id": "call_def", "function": {"name": "transit_function_1", "arguments": "{}"}},
                ],
            },
            {
                "role": "tool",
                "tool_responses": [
                    {"tool_call_id": "call_abc", "role": "tool", "content": "Normal result"},
                    {"tool_call_id": "call_def", "role": "tool", "content": "Transit result"},
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [  # Message should be removed entirely
                    {"id": "call_ghi", "function": {"name": "secret_handoff", "arguments": "{}"}},
                ],
            },
            {
                "role": "tool",
                "tool_responses": [  # Message should be removed entirely
                    {"tool_call_id": "call_ghi", "role": "tool", "content": "Secret result"},
                ],
            },
            {"role": "assistant", "content": "Final answer"},
        ]

        processed_messages = remove_func(messages)

        expected_messages = [
            {
                "role": "assistant",
                "content": "Thinking...",
                "tool_calls": [  # transit_function_1 removed
                    {"id": "call_abc", "function": {"name": "normal_tool", "arguments": "{}"}},
                ],
            },
            {
                "role": "tool",
                "tool_responses": [  # Corresponding response for transit_function_1 removed
                    {"tool_call_id": "call_abc", "role": "tool", "content": "Normal result"},
                ],
            },
            # Message with only secret_handoff and its response are removed entirely
            {"role": "assistant", "content": "Final answer"},
        ]

        assert processed_messages == expected_messages
