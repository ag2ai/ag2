# Copyright (c) 2023 - 2025, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
import copy
import json
import re
import warnings
from dataclasses import dataclass
from enum import Enum
from inspect import signature
from types import MethodType
from typing import Any, Callable, Optional, Union

from pydantic import BaseModel

from autogen.oai import OpenAIWrapper

from ..agent import Agent
from ..chat import ChatResult
from ..conversable_agent import __CONTEXT_VARIABLES_PARAM_NAME__, ConversableAgent
from ..groupchat import GroupChat, GroupChatManager
from ..user_proxy_agent import UserProxyAgent


@dataclass
class UpdateCondition:
    """Update the condition string before they reply

    Args:
        update_function: The string or function to update the condition string. Can be a string or a Callable.
            If a string, it will be used as a template and substitute the context variables.
            If a Callable, it should have the signature:
                def my_update_function(agent: ConversableAgent, messages: List[Dict[str, Any]]) -> str
    """

    update_function: Union[Callable, str]

    def __post_init__(self):
        if isinstance(self.update_function, str):
            assert self.update_function.strip(), " please provide a non-empty string or a callable"
            # find all {var} in the string
            vars = re.findall(r"\{(\w+)\}", self.update_function)
            if len(vars) == 0:
                warnings.warn("Update function string contains no variables. This is probably unintended.")

        elif isinstance(self.update_function, Callable):
            sig = signature(self.update_function)
            if len(sig.parameters) != 2:
                raise ValueError(
                    "Update function must accept two parameters of type ConversableAgent and List[Dict[str, Any]], respectively"
                )
            if sig.return_annotation != str:
                raise ValueError("Update function must return a string")
        else:
            raise ValueError("Update function must be either a string or a callable")


# Created tool executor's name
__TOOL_EXECUTOR_NAME__ = "_Swarm_Tool_Executor"


class AfterWorkOption(Enum):
    TERMINATE = "TERMINATE"
    REVERT_TO_USER = "REVERT_TO_USER"
    STAY = "STAY"
    SWARM_MANAGER = "SWARM_MANAGER"


@dataclass
class AfterWork:
    """Handles the next step in the conversation when an agent doesn't suggest a tool call or a handoff

    Args:
        agent: The agent to hand off to or the after work option. Can be a ConversableAgent, a string name of a ConversableAgent, an AfterWorkOption, or a Callable.
            The Callable signature is:
                def my_after_work_func(last_speaker: ConversableAgent, messages: List[Dict[str, Any]], groupchat: GroupChat) -> Union[AfterWorkOption, ConversableAgent, str]:
    """

    agent: Union[AfterWorkOption, ConversableAgent, str, Callable]

    def __post_init__(self):
        if isinstance(self.agent, str):
            self.agent = AfterWorkOption(self.agent.upper())


class AFTER_WORK(AfterWork):  # noqa: N801
    """Deprecated: Use AfterWork instead. This class will be removed in a future version (TBD)."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "AFTER_WORK is deprecated and will be removed in a future version (TBD). Use AfterWork instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


@dataclass
class OnCondition:
    """Defines a condition for transitioning to another agent or nested chats

    Args:
        target: The agent to hand off to or the nested chat configuration. Can be a ConversableAgent or a Dict.
            If a Dict, it should follow the convention of the nested chat configuration, with the exception of a carryover configuration which is unique to Swarms.
            Swarm Nested chat documentation: https://docs.ag2.ai/docs/topics/swarm#registering-handoffs-to-a-nested-chat
        condition (str): The condition for transitioning to the target agent, evaluated by the LLM to determine whether to call the underlying function/tool which does the transition.
        available (Union[Callable, str]): Optional condition to determine if this OnCondition is available. Can be a Callable or a string.
            If a string, it will look up the value of the context variable with that name, which should be a bool.
    """

    target: Union[ConversableAgent, dict[str, Any]] = None
    condition: Union[str, UpdateCondition] = ""
    available: Optional[Union[Callable, str]] = None

    def __post_init__(self):
        # Ensure valid types
        if self.target is not None:
            assert isinstance(self.target, ConversableAgent) or isinstance(self.target, dict), (
                "'target' must be a ConversableAgent or a dict"
            )

        # Ensure they have a condition
        if isinstance(self.condition, str):
            assert self.condition.strip(), "'condition' must be a non-empty string"
        else:
            assert isinstance(self.condition, UpdateCondition), "'condition' must be a string or UpdateOnConditionStr"

        if self.available is not None:
            assert isinstance(self.available, (Callable, str)), "'available' must be a callable or a string"


class ON_CONDITION(OnCondition):  # noqa: N801
    """Deprecated: Use OnCondition instead. This class will be removed in a future version (TBD)."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "ON_CONDITION is deprecated and will be removed in a future version (TBD). Use OnCondition instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


def _establish_swarm_agent(agent: ConversableAgent):
    """Establish the swarm agent with the swarm-related attributes and hooks. Not for the tool executor.

    Args:
        agent (ConversableAgent): The agent to establish as a swarm agent.
    """

    def _swarm_agent_str(self: ConversableAgent) -> str:
        """Customise the __str__ method to show the agent name for transition messages."""
        return f"Swarm agent --> {self.name}"

    agent._swarm_after_work = None

    # Store nested chats hand offs as we'll establish these in the initiate_swarm_chat
    # List of Dictionaries containing the nested_chats and condition
    agent._swarm_nested_chat_handoffs = []

    # Store conditional functions (and their OnCondition instances) to add/remove later when transitioning to this agent
    agent._swarm_conditional_functions = {}

    # Register the hook to update agent state (except tool executor)
    agent.register_hook("update_agent_state", _update_conditional_functions)

    agent._get_display_name = MethodType(_swarm_agent_str, agent)

    # Mark this agent as established as a swarm agent
    agent._swarm_is_established = True


def _prepare_swarm_agents(
    initial_agent: ConversableAgent,
    agents: list[ConversableAgent],
) -> tuple[ConversableAgent, list[ConversableAgent]]:
    """Validates agents, create the tool executor, configure nested chats.

    Args:
        initial_agent (ConversableAgent): The first agent in the conversation.
        agents (list[ConversableAgent]): List of all agents in the conversation.

    Returns:
        ConversableAgent: The tool executor agent.
        list[ConversableAgent]: List of nested chat agents.
    """
    assert isinstance(initial_agent, ConversableAgent), "initial_agent must be a ConversableAgent"
    assert all(isinstance(agent, ConversableAgent) for agent in agents), "Agents must be a list of ConversableAgents"

    # Initialize all agents as swarm agents
    for agent in agents:
        if not hasattr(agent, "_swarm_is_established"):
            _establish_swarm_agent(agent)

    # Ensure all agents in hand-off after-works are in the passed in agents list
    for agent in agents:
        if agent._swarm_after_work is not None:
            if isinstance(agent._swarm_after_work.agent, ConversableAgent):
                assert agent._swarm_after_work.agent in agents, "Agent in hand-off must be in the agents list"

    tool_execution = ConversableAgent(
        name=__TOOL_EXECUTOR_NAME__,
        system_message="Tool Execution, do not use this agent directly.",
    )
    _set_to_tool_execution(tool_execution)

    nested_chat_agents = []
    for agent in agents:
        _create_nested_chats(agent, nested_chat_agents)

    # Update tool execution agent with all the functions from all the agents
    for agent in agents + nested_chat_agents:
        tool_execution._function_map.update(agent._function_map)
        # Add conditional functions to the tool_execution agent
        for func_name, (func, _) in agent._swarm_conditional_functions.items():
            tool_execution._function_map[func_name] = func

    return tool_execution, nested_chat_agents


def _create_nested_chats(agent: ConversableAgent, nested_chat_agents: list[ConversableAgent]):
    """Create nested chat agents and register nested chats.

    Args:
        agent (ConversableAgent): The agent to create nested chat agents for, including registering the hand offs.
        nested_chat_agents (list[ConversableAgent]): List for all nested chat agents, appends to this.
    """
    for i, nested_chat_handoff in enumerate(agent._swarm_nested_chat_handoffs):
        nested_chats: dict[str, Any] = nested_chat_handoff["nested_chats"]
        condition = nested_chat_handoff["condition"]
        available = nested_chat_handoff["available"]

        # Create a nested chat agent specifically for this nested chat
        nested_chat_agent = ConversableAgent(name=f"nested_chat_{agent.name}_{i + 1}")

        nested_chat_agent.register_nested_chats(
            nested_chats["chat_queue"],
            reply_func_from_nested_chats=nested_chats.get("reply_func_from_nested_chats")
            or "summary_from_nested_chats",
            config=nested_chats.get("config"),
            trigger=lambda sender: True,
            position=0,
            use_async=nested_chats.get("use_async", False),
        )

        # After the nested chat is complete, transfer back to the parent agent
        register_hand_off(nested_chat_agent, AfterWork(agent=agent))

        nested_chat_agents.append(nested_chat_agent)

        # Nested chat is triggered through an agent transfer to this nested chat agent
        register_hand_off(agent, OnCondition(nested_chat_agent, condition, available))


def _process_initial_messages(
    messages: Union[list[dict[str, Any]], str],
    user_agent: Optional[UserProxyAgent],
    agents: list[ConversableAgent],
    nested_chat_agents: list[ConversableAgent],
) -> tuple[list[dict], Optional[Agent], list[str], list[Agent]]:
    """Process initial messages, validating agent names against messages, and determining the last agent to speak.

    Args:
        messages: Initial messages to process.
        user_agent: Optional user proxy agent passed in to a_/initiate_swarm_chat.
        agents: Agents in swarm.
        nested_chat_agents: List of nested chat agents.

    Returns:
        list[dict]: Processed message(s).
        Agent: Last agent to speak.
        list[str]: List of agent names.
        list[Agent]: List of temporary user proxy agents to add to GroupChat.
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    swarm_agent_names = [agent.name for agent in agents + nested_chat_agents]

    # If there's only one message and there's no identified swarm agent
    # Start with a user proxy agent, creating one if they haven't passed one in
    temp_user_proxy = None
    temp_user_list = []
    if len(messages) == 1 and "name" not in messages[0] and not user_agent:
        temp_user_proxy = UserProxyAgent(name="_User")
        last_agent = temp_user_proxy
        temp_user_list.append(temp_user_proxy)
    else:
        last_message = messages[0]
        if "name" in last_message:
            if last_message["name"] in swarm_agent_names:
                last_agent = next(agent for agent in agents + nested_chat_agents if agent.name == last_message["name"])
            elif user_agent and last_message["name"] == user_agent.name:
                last_agent = user_agent
            else:
                raise ValueError(f"Invalid swarm agent name in last message: {last_message['name']}")
        else:
            last_agent = user_agent if user_agent else temp_user_proxy

    return messages, last_agent, swarm_agent_names, temp_user_list


def _setup_context_variables(
    tool_execution: ConversableAgent,
    agents: list[ConversableAgent],
    manager: GroupChatManager,
    context_variables: dict[str, Any],
) -> None:
    """Assign a common context_variables reference to all agents in the swarm, including the tool executor and group chat manager.

    Args:
        tool_execution: The tool execution agent.
        agents: List of all agents in the conversation.
        manager: GroupChatManager instance.
    """
    for agent in agents + [tool_execution] + [manager]:
        agent._context_variables = context_variables


def _cleanup_temp_user_messages(chat_result: ChatResult) -> None:
    """Remove temporary user proxy agent name from messages before returning.

    Args:
        chat_result: ChatResult instance.
    """
    for message in chat_result.chat_history:
        if "name" in message and message["name"] == "_User":
            del message["name"]


def _determine_next_agent(
    last_speaker: ConversableAgent,
    groupchat: GroupChat,
    initial_agent: ConversableAgent,
    use_initial_agent: bool,
    tool_execution: ConversableAgent,
    swarm_agent_names: list[str],
    user_agent: Optional[UserProxyAgent],
    swarm_after_work: Optional[Union[AfterWorkOption, Callable]],
) -> Optional[Agent]:
    """Determine the next agent in the conversation.

    Args:
        last_speaker (ConversableAgent): The last agent to speak.
        groupchat (GroupChat): GroupChat instance.
        initial_agent (ConversableAgent): The initial agent in the conversation.
        use_initial_agent (bool): Whether to use the initial agent straight away.
        tool_execution (ConversableAgent): The tool execution agent.
        swarm_agent_names (list[str]): List of agent names.
        user_agent (UserProxyAgent): Optional user proxy agent.
        swarm_after_work (Union[AfterWorkOption, Callable]): Method to handle conversation continuation when an agent doesn't select the next agent.
    """
    if use_initial_agent:
        return initial_agent

    if "tool_calls" in groupchat.messages[-1]:
        return tool_execution

    if tool_execution._swarm_next_agent is not None:
        next_agent = tool_execution._swarm_next_agent
        tool_execution._swarm_next_agent = None

        # Check for string, access agent from group chat.

        if isinstance(next_agent, str):
            if next_agent in swarm_agent_names:
                next_agent = groupchat.agent_by_name(name=next_agent)
            else:
                raise ValueError(f"No agent found with the name '{next_agent}'. Ensure the agent exists in the swarm.")

        return next_agent

    # get the last swarm agent
    last_swarm_speaker = None
    for message in reversed(groupchat.messages):
        if "name" in message and message["name"] in swarm_agent_names:
            agent = groupchat.agent_by_name(name=message["name"])
            if isinstance(agent, ConversableAgent):
                last_swarm_speaker = agent
                break
    if last_swarm_speaker is None:
        raise ValueError("No swarm agent found in the message history")

    # If the user last spoke, return to the agent prior
    if (user_agent and last_speaker == user_agent) or groupchat.messages[-1]["role"] == "tool":
        return last_swarm_speaker

    # Resolve after_work condition (agent-level overrides global)
    after_work_condition = (
        last_swarm_speaker._swarm_after_work if last_swarm_speaker._swarm_after_work is not None else swarm_after_work
    )
    if isinstance(after_work_condition, AfterWork):
        after_work_condition = after_work_condition.agent

    # Evaluate callable after_work
    if isinstance(after_work_condition, Callable):
        after_work_condition = after_work_condition(last_speaker, groupchat.messages, groupchat)

    if isinstance(after_work_condition, str):  # Agent name in a string
        if after_work_condition in swarm_agent_names:
            return groupchat.agent_by_name(name=after_work_condition)
        else:
            raise ValueError(f"Invalid agent name in after_work: {after_work_condition}")
    elif isinstance(after_work_condition, ConversableAgent):
        return after_work_condition
    elif isinstance(after_work_condition, AfterWorkOption):
        if after_work_condition == AfterWorkOption.TERMINATE:
            return None
        elif after_work_condition == AfterWorkOption.REVERT_TO_USER:
            return None if user_agent is None else user_agent
        elif after_work_condition == AfterWorkOption.STAY:
            return last_speaker
        elif after_work_condition == AfterWorkOption.SWARM_MANAGER:
            return "auto"
    else:
        raise ValueError("Invalid After Work condition or return value from callable")


def create_swarm_transition(
    initial_agent: ConversableAgent,
    tool_execution: ConversableAgent,
    swarm_agent_names: list[str],
    user_agent: Optional[UserProxyAgent],
    swarm_after_work: Optional[Union[AfterWorkOption, Callable]],
) -> Callable[[ConversableAgent, GroupChat], Optional[Agent]]:
    """Creates a transition function for swarm chat with enclosed state for the use_initial_agent.

    Args:
        initial_agent (ConversableAgent): The first agent to speak
        tool_execution (ConversableAgent): The tool execution agent
        swarm_agent_names (list[str]): List of all agent names
        user_agent (UserProxyAgent): Optional user proxy agent
        swarm_after_work (Union[AfterWorkOption, Callable]): Swarm-level after work

    Returns:
        Callable transition function (for sync and async swarm chats)
    """
    # Create enclosed state, this will be set once per creation so will only be True on the first execution
    # of swarm_transition
    state = {"use_initial_agent": True}

    def swarm_transition(last_speaker: ConversableAgent, groupchat: GroupChat) -> Optional[Agent]:
        result = _determine_next_agent(
            last_speaker=last_speaker,
            groupchat=groupchat,
            initial_agent=initial_agent,
            use_initial_agent=state["use_initial_agent"],
            tool_execution=tool_execution,
            swarm_agent_names=swarm_agent_names,
            user_agent=user_agent,
            swarm_after_work=swarm_after_work,
        )
        state["use_initial_agent"] = False
        return result

    return swarm_transition


def initiate_swarm_chat(
    initial_agent: ConversableAgent,
    messages: Union[list[dict[str, Any]], str],
    agents: list[ConversableAgent],
    user_agent: Optional[UserProxyAgent] = None,
    max_rounds: int = 20,
    context_variables: Optional[dict[str, Any]] = None,
    after_work: Optional[Union[AfterWorkOption, Callable]] = AfterWork(AfterWorkOption.TERMINATE),
) -> tuple[ChatResult, dict[str, Any], ConversableAgent]:
    """Initialize and run a swarm chat

    Args:
        initial_agent: The first receiving agent of the conversation.
        messages: Initial message(s).
        agents: List of swarm agents.
        user_agent: Optional user proxy agent for falling back to.
        max_rounds: Maximum number of conversation rounds.
        context_variables: Starting context variables.
        after_work: Method to handle conversation continuation when an agent doesn't select the next agent. If no agent is selected and no tool calls are output, we will use this method to determine the next agent.
            Must be a AfterWork instance (which is a dataclass accepting a ConversableAgent, AfterWorkOption, A str (of the AfterWorkOption)) or a callable.
            AfterWorkOption:
                - TERMINATE (Default): Terminate the conversation.
                - REVERT_TO_USER : Revert to the user agent if a user agent is provided. If not provided, terminate the conversation.
                - STAY : Stay with the last speaker.

            Callable: A custom function that takes the current agent, messages, and groupchat as arguments and returns an AfterWorkOption or a ConversableAgent (by reference or string name).
                ```python
                def custom_afterwork_func(last_speaker: ConversableAgent, messages: List[Dict[str, Any]], groupchat: GroupChat) -> Union[AfterWorkOption, ConversableAgent, str]:
                ```
    Returns:
        ChatResult:     Conversations chat history.
        Dict[str, Any]: Updated Context variables.
        ConversableAgent:     Last speaker.
    """
    tool_execution, nested_chat_agents = _prepare_swarm_agents(initial_agent, agents)

    processed_messages, last_agent, swarm_agent_names, temp_user_list = _process_initial_messages(
        messages, user_agent, agents, nested_chat_agents
    )

    # Create transition function (has enclosed state for initial agent)
    swarm_transition = create_swarm_transition(
        initial_agent=initial_agent,
        tool_execution=tool_execution,
        swarm_agent_names=swarm_agent_names,
        user_agent=user_agent,
        swarm_after_work=after_work,
    )

    groupchat = GroupChat(
        agents=[tool_execution] + agents + nested_chat_agents + ([user_agent] if user_agent else temp_user_list),
        messages=[],
        max_round=max_rounds,
        speaker_selection_method=swarm_transition,
    )

    manager = GroupChatManager(groupchat)

    # Point all ConversableAgent's context variables to this function's context_variables
    _setup_context_variables(tool_execution, agents, manager, context_variables or {})

    if len(processed_messages) > 1:
        last_agent, last_message = manager.resume(messages=processed_messages)
        clear_history = False
    else:
        last_message = processed_messages[0]
        clear_history = True

    chat_result = last_agent.initiate_chat(
        manager,
        message=last_message,
        clear_history=clear_history,
    )

    _cleanup_temp_user_messages(chat_result)

    return chat_result, context_variables, manager.last_speaker


async def a_initiate_swarm_chat(
    initial_agent: ConversableAgent,
    messages: Union[list[dict[str, Any]], str],
    agents: list[ConversableAgent],
    user_agent: Optional[UserProxyAgent] = None,
    max_rounds: int = 20,
    context_variables: Optional[dict[str, Any]] = None,
    after_work: Optional[Union[AfterWorkOption, Callable]] = AfterWork(AfterWorkOption.TERMINATE),
) -> tuple[ChatResult, dict[str, Any], ConversableAgent]:
    """Initialize and run a swarm chat asynchronously

    Args:
        initial_agent: The first receiving agent of the conversation.
        messages: Initial message(s).
        agents: List of swarm agents.
        user_agent: Optional user proxy agent for falling back to.
        max_rounds: Maximum number of conversation rounds.
        context_variables: Starting context variables.
        after_work: Method to handle conversation continuation when an agent doesn't select the next agent. If no agent is selected and no tool calls are output, we will use this method to determine the next agent.
            Must be a AfterWork instance (which is a dataclass accepting a ConversableAgent, AfterWorkOption, A str (of the AfterWorkOption)) or a callable.
            AfterWorkOption:
                - TERMINATE (Default): Terminate the conversation.
                - REVERT_TO_USER : Revert to the user agent if a user agent is provided. If not provided, terminate the conversation.
                - STAY : Stay with the last speaker.

            Callable: A custom function that takes the current agent, messages, and groupchat as arguments and returns an AfterWorkOption or a ConversableAgent (by reference or string name).
                ```python
                def custom_afterwork_func(last_speaker: ConversableAgent, messages: List[Dict[str, Any]], groupchat: GroupChat) -> Union[AfterWorkOption, ConversableAgent, str]:
                ```
    Returns:
        ChatResult:     Conversations chat history.
        Dict[str, Any]: Updated Context variables.
        ConversableAgent:     Last speaker.
    """
    tool_execution, nested_chat_agents = _prepare_swarm_agents(initial_agent, agents)

    processed_messages, last_agent, swarm_agent_names, temp_user_list = _process_initial_messages(
        messages, user_agent, agents, nested_chat_agents
    )

    # Create transition function (has enclosed state for initial agent)
    swarm_transition = create_swarm_transition(
        initial_agent=initial_agent,
        tool_execution=tool_execution,
        swarm_agent_names=swarm_agent_names,
        user_agent=user_agent,
        swarm_after_work=after_work,
    )

    groupchat = GroupChat(
        agents=[tool_execution] + agents + nested_chat_agents + ([user_agent] if user_agent else temp_user_list),
        messages=[],
        max_round=max_rounds,
        speaker_selection_method=swarm_transition,
    )

    manager = GroupChatManager(groupchat)

    # Point all ConversableAgent's context variables to this function's context_variables
    _setup_context_variables(tool_execution, agents, manager, context_variables or {})

    if len(processed_messages) > 1:
        last_agent, last_message = await manager.a_resume(messages=processed_messages)
        clear_history = False
    else:
        last_message = processed_messages[0]
        clear_history = True

    chat_result = await last_agent.a_initiate_chat(
        manager,
        message=last_message,
        clear_history=clear_history,
    )

    _cleanup_temp_user_messages(chat_result)

    return chat_result, context_variables, manager.last_speaker


class SwarmResult(BaseModel):
    """Encapsulates the possible return values for a swarm agent function.

    Args:
        values (str): The result values as a string.
        agent (ConversableAgent): The agent instance, if applicable.
        context_variables (dict): A dictionary of context variables.
    """

    values: str = ""
    agent: Optional[Union[ConversableAgent, str]] = None
    context_variables: dict[str, Any] = {}

    class Config:  # Add this inner class
        arbitrary_types_allowed = True

    def __str__(self):
        return self.values


def _set_to_tool_execution(agent: ConversableAgent):
    """Set to a special instance of ConversableAgent that is responsible for executing tool calls from other swarm agents.
    This agent will be used internally and should not be visible to the user.

    It will execute the tool calls and update the referenced context_variables and next_agent accordingly.
    """
    agent._swarm_next_agent = None
    agent._reply_func_list.clear()
    agent.register_reply([Agent, None], _generate_swarm_tool_reply)


def register_hand_off(
    agent: ConversableAgent,
    hand_to: Union[list[Union[OnCondition, AfterWork]], OnCondition, AfterWork],
):
    """Register a function to hand off to another agent.

    Args:
        agent: The agent to register the hand off with.
        hand_to: A list of OnCondition's and an, optional, AfterWork condition

    Hand off template:
    def transfer_to_agent_name() -> ConversableAgent:
        return agent_name
    1. register the function with the agent
    2. register the schema with the agent, description set to the condition
    """
    # If the agent hasn't been established as a swarm agent, do so first
    if not hasattr(agent, "_swarm_is_established"):
        _establish_swarm_agent(agent)

    # Ensure that hand_to is a list or OnCondition or AfterWork
    if not isinstance(hand_to, (list, OnCondition, AfterWork)):
        raise ValueError("hand_to must be a list of OnCondition or AfterWork")

    if isinstance(hand_to, (OnCondition, AfterWork)):
        hand_to = [hand_to]

    for transit in hand_to:
        if isinstance(transit, AfterWork):
            assert isinstance(transit.agent, (AfterWorkOption, ConversableAgent, str, Callable)), (
                "Invalid After Work value"
            )
            agent._swarm_after_work = transit
        elif isinstance(transit, OnCondition):
            if isinstance(transit.target, ConversableAgent):
                # Transition to agent

                # Create closure with current loop transit value
                # to ensure the condition matches the one in the loop
                def make_transfer_function(current_transit: OnCondition):
                    def transfer_to_agent() -> ConversableAgent:
                        return current_transit.target

                    return transfer_to_agent

                transfer_func = make_transfer_function(transit)

                # Store function to add/remove later based on it being 'available'
                # Function names are made unique and allow multiple OnCondition's to the same agent
                base_func_name = f"transfer_{agent.name}_to_{transit.target.name}"
                func_name = base_func_name
                count = 2
                while func_name in agent._swarm_conditional_functions:
                    func_name = f"{base_func_name}_{count}"
                    count += 1

                # Store function to add/remove later based on it being 'available'
                agent._swarm_conditional_functions[func_name] = (transfer_func, transit)

            elif isinstance(transit.target, dict):
                # Transition to a nested chat
                # We will store them here and establish them in the initiate_swarm_chat
                agent._swarm_nested_chat_handoffs.append(
                    {"nested_chats": transit.target, "condition": transit.condition, "available": transit.available}
                )

        else:
            raise ValueError("Invalid hand off condition, must be either OnCondition or AfterWork")


def _update_conditional_functions(agent: ConversableAgent, messages: Optional[list[dict]] = None) -> None:
    """Updates the agent's functions based on the OnCondition's available condition."""
    for func_name, (func, on_condition) in agent._swarm_conditional_functions.items():
        is_available = True

        if on_condition.available is not None:
            if isinstance(on_condition.available, Callable):
                is_available = on_condition.available(agent, next(iter(agent.chat_messages.values())))
            elif isinstance(on_condition.available, str):
                is_available = agent.get_context(on_condition.available) or False

        # first remove the function if it exists
        if func_name in agent._function_map:
            agent.update_tool_signature(func_name, is_remove=True)
            del agent._function_map[func_name]

        # then add the function if it is available, so that the function signature is updated
        if is_available:
            condition = on_condition.condition
            if isinstance(condition, UpdateCondition):
                if isinstance(condition.update_function, str):
                    condition = OpenAIWrapper.instantiate(
                        template=condition.update_function,
                        context=agent._context_variables,
                        allow_format_str_template=True,
                    )
                else:
                    condition = condition.update_function(agent, messages)
            agent._add_single_function(func, func_name, condition)


def _generate_swarm_tool_reply(
    agent: ConversableAgent,
    messages: Optional[list[dict]] = None,
    sender: Optional[Agent] = None,
    config: Optional[OpenAIWrapper] = None,
) -> tuple[bool, dict]:
    """Pre-processes and generates tool call replies.

    This function:
    1. Adds context_variables back to the tool call for the function, if necessary.
    2. Generates the tool calls reply.
    3. Updates context_variables and next_agent based on the tool call response."""

    if config is None:
        config = agent
    if messages is None:
        messages = agent._oai_messages[sender]

    message = messages[-1]
    if "tool_calls" in message:
        tool_call_count = len(message["tool_calls"])

        # Loop through tool calls individually (so context can be updated after each function call)
        next_agent = None
        tool_responses_inner = []
        contents = []
        for index in range(tool_call_count):
            # Deep copy to ensure no changes to messages when we insert the context variables
            message_copy = copy.deepcopy(message)

            # 1. add context_variables to the tool call arguments
            tool_call = message_copy["tool_calls"][index]

            if tool_call["type"] == "function":
                function_name = tool_call["function"]["name"]

                # Check if this function exists in our function map
                if function_name in agent._function_map:
                    func = agent._function_map[function_name]  # Get the original function

                    # Inject the context variables into the tool call if it has the parameter
                    sig = signature(func)
                    if __CONTEXT_VARIABLES_PARAM_NAME__ in sig.parameters:
                        current_args = json.loads(tool_call["function"]["arguments"])
                        current_args[__CONTEXT_VARIABLES_PARAM_NAME__] = agent._context_variables
                        tool_call["function"]["arguments"] = json.dumps(current_args)

            # Ensure we are only executing the one tool at a time
            message_copy["tool_calls"] = [tool_call]

            # 2. generate tool calls reply
            _, tool_message = agent.generate_tool_calls_reply([message_copy])

            # 3. update context_variables and next_agent, convert content to string
            for tool_response in tool_message["tool_responses"]:
                content = tool_response.get("content")
                if isinstance(content, SwarmResult):
                    if content.context_variables != {}:
                        agent._context_variables.update(content.context_variables)
                    if content.agent is not None:
                        next_agent = content.agent
                elif isinstance(content, Agent):
                    next_agent = content

                tool_responses_inner.append(tool_response)
                contents.append(str(tool_response["content"]))

        agent._swarm_next_agent = next_agent

        # Put the tool responses and content strings back into the response message
        # Caters for multiple tool calls
        tool_message["tool_responses"] = tool_responses_inner
        tool_message["content"] = "\n".join(contents)

        return True, tool_message
    return False, None


class SwarmAgent(ConversableAgent):
    """SwarmAgent is deprecated and has been incorporated into ConversableAgent, use ConversableAgent instead. SwarmAgent will be removed in a future version (TBD)"""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "SwarmAgent is deprecated and has been incorporated into ConversableAgent, use ConversableAgent instead. SwarmAgent will be removed in a future version (TBD).",
            DeprecationWarning,
            stacklevel=2,
        )

        super().__init__(*args, **kwargs)
