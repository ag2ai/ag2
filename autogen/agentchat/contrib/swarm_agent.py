# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import copy
import inspect
import threading
import warnings
from dataclasses import dataclass
from enum import Enum
from functools import partial
from types import MethodType
from typing import Annotated, Any, Callable, Literal, Optional, Union

from pydantic import BaseModel, field_serializer

from ...doc_utils import export_module
from ...events.agent_events import ErrorEvent, RunCompletionEvent
from ...io.base import IOStream
from ...io.run_response import AsyncRunResponse, AsyncRunResponseProtocol, RunResponse, RunResponseProtocol
from ...io.thread_io_stream import AsyncThreadIOStream, ThreadIOStream
from ...oai import OpenAIWrapper
from ...tools import Depends, Tool
from ...tools.dependency_injection import inject_params, on
from ..agent import Agent
from ..chat import ChatResult
from ..conversable_agent import ConversableAgent
from ..group.context_expression import ContextExpression
from ..group.context_str import ContextStr
from ..group.context_variables import __CONTEXT_VARIABLES_PARAM_NAME__, ContextVariables
from ..groupchat import SELECT_SPEAKER_PROMPT_TEMPLATE, GroupChat, GroupChatManager
from ..user_proxy_agent import UserProxyAgent

__all__ = [
    "AFTER_WORK",
    "ON_CONDITION",
    "AfterWork",
    "AfterWorkOption",
    "OnCondition",
    "OnContextCondition",
    "SwarmAgent",
    "a_initiate_swarm_chat",
    "create_swarm_transition",
    "initiate_swarm_chat",
    "register_hand_off",
    "run_swarm",
]


# Created tool executor's name
__TOOL_EXECUTOR_NAME__ = "_Swarm_Tool_Executor"


@export_module("autogen")
class AfterWorkOption(Enum):
    TERMINATE = "TERMINATE"
    REVERT_TO_USER = "REVERT_TO_USER"
    STAY = "STAY"
    SWARM_MANAGER = "SWARM_MANAGER"


@dataclass
@export_module("autogen")
class AfterWork:  # noqa: N801
    """Handles the next step in the conversation when an agent doesn't suggest a tool call or a handoff.

    Args:
        agent (Union[AfterWorkOption, ConversableAgent, str, Callable[..., Any]]): The agent to hand off to or the after work option. Can be a ConversableAgent, a string name of a ConversableAgent, an AfterWorkOption, or a Callable.
            The Callable signature is:
                def my_after_work_func(last_speaker: ConversableAgent, messages: list[dict[str, Any]], groupchat: GroupChat) -> Union[AfterWorkOption, ConversableAgent, str]:
        next_agent_selection_msg (Optional[Union[str, Callable[..., Any]]]): Optional message to use for the agent selection (in internal group chat), only valid for when agent is AfterWorkOption.SWARM_MANAGER.
            If a string, it will be used as a template and substitute the context variables.
            If a Callable, it should have the signature:
                def my_selection_message(agent: ConversableAgent, messages: list[dict[str, Any]]) -> str
    """

    agent: Union[AfterWorkOption, ConversableAgent, str, Callable[..., Any]]
    next_agent_selection_msg: Optional[
        Union[str, ContextStr, Callable[[ConversableAgent, list[dict[str, Any]]], str]]
    ] = None

    def __post_init__(self) -> None:
        if isinstance(self.agent, str):
            self.agent = AfterWorkOption(self.agent.upper())

        # next_agent_selection_msg is only valid for when agent is AfterWorkOption.SWARM_MANAGER, but isn't mandatory
        if self.next_agent_selection_msg is not None:
            if not (
                isinstance(self.next_agent_selection_msg, (str, ContextStr)) or callable(self.next_agent_selection_msg)
            ):
                raise ValueError("next_agent_selection_msg must be a string, ContextStr, or a Callable")

            if self.agent != AfterWorkOption.SWARM_MANAGER:
                warnings.warn(
                    "next_agent_selection_msg is only valid for agent=AfterWorkOption.SWARM_MANAGER. Ignoring the value.",
                    UserWarning,
                )
                self.next_agent_selection_msg = None


class AFTER_WORK(AfterWork):  # noqa: N801
    """Deprecated: Use AfterWork instead. This class will be removed in a future version (TBD)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "AFTER_WORK is deprecated and will be removed in a future version (TBD). Use AfterWork instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


@dataclass
@export_module("autogen")
class OnCondition:  # noqa: N801
    """Defines a condition for transitioning to another agent or nested chats.

    This is for LLM-based condition evaluation where these conditions are translated into tools and attached to the agent.

    These are evaluated after the OnCondition conditions but before the AfterWork conditions.

    Args:
        target (Optional[Union[ConversableAgent, dict[str, Any]]]): The agent to hand off to or the nested chat configuration. Can be a ConversableAgent or a Dict.
            If a Dict, it should follow the convention of the nested chat configuration, with the exception of a carryover configuration which is unique to Swarms.
            Swarm Nested chat documentation: https://docs.ag2.ai/docs/user-guide/advanced-concepts/swarm-deep-dive#registering-handoffs-to-a-nested-chat
        condition (Optional[Union[str, ContextStr, Callable[[ConversableAgent, list[dict[str, Any]]], str]]]): The condition for transitioning to the target agent, evaluated by the LLM.
            If a string or Callable, no automatic context variable substitution occurs.
            If a ContextStr, context variable substitution occurs.
            The Callable signature is:
                def my_condition_string(agent: ConversableAgent, messages: list[Dict[str, Any]]) -> str
        available (Optional[Union[Callable[[ConversableAgent, list[dict[str, Any]]], bool], str, ContextExpression]]): Optional condition to determine if this OnCondition is included for the LLM to evaluate.
            If a string, it will look up the value of the context variable with that name, which should be a bool, to determine whether it should include this condition.
            If a ContextExpression, it will evaluate the logical expression against the context variables. Can use not, and, or, and comparison operators (>, <, >=, <=, ==, !=).
                Example: ContextExpression("not(${logged_in} and ${is_admin}) or (${guest_checkout})")
                Example with comparison: ContextExpression("${attempts} >= 3 or ${is_premium} == True or ${tier} == 'gold'")
            The Callable signature is:
                def my_available_func(agent: ConversableAgent, messages: list[Dict[str, Any]]) -> bool
    """

    target: Optional[Union[ConversableAgent, dict[str, Any]]] = None
    condition: Optional[Union[str, ContextStr, Callable[[ConversableAgent, list[dict[str, Any]]], str]]] = None
    available: Optional[Union[Callable[[ConversableAgent, list[dict[str, Any]]], bool], str, ContextExpression]] = None

    def __post_init__(self) -> None:
        # Ensure valid types
        if (self.target is not None) and (not isinstance(self.target, (ConversableAgent, dict))):
            raise ValueError("'target' must be a ConversableAgent or a dict")

        # Ensure they have a condition
        if isinstance(self.condition, str):
            if not self.condition.strip():
                raise ValueError("'condition' must be a non-empty string")
        else:
            if not isinstance(self.condition, ContextStr) and not callable(self.condition):
                raise ValueError("'condition' must be a string, ContextStr, or callable")

        if (self.available is not None) and (
            not (isinstance(self.available, (str, ContextExpression)) or callable(self.available))
        ):
            raise ValueError("'available' must be a callable, a string, or a ContextExpression")


class ON_CONDITION(OnCondition):  # noqa: N801
    """Deprecated: Use OnCondition instead. This class will be removed in a future version (TBD)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "ON_CONDITION is deprecated and will be removed in a future version (TBD). Use OnCondition instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


@dataclass
@export_module("autogen")
class OnContextCondition:  # noqa: N801
    """Defines a condition for transitioning to another agent or nested chats using context variables and the ContextExpression class.

    This is for context variable-based condition evaluation (does not use the agent's LLM).

    These are evaluated before the OnCondition and AfterWork conditions.

    Args:
        target (Optional[Union[ConversableAgent, dict[str, Any]]]): The agent to hand off to or the nested chat configuration. Can be a ConversableAgent or a Dict.
            If a Dict, it should follow the convention of the nested chat configuration, with the exception of a carryover configuration which is unique to Swarms.
            Swarm Nested chat documentation: https://docs.ag2.ai/docs/user-guide/advanced-concepts/swarm-deep-dive#registering-handoffs-to-a-nested-chat
        condition (Optional[Union[str, ContextExpression]]): The condition for transitioning to the target agent, evaluated by the LLM.
            If a string, it needs to represent a context variable key and the value will be evaluated as a boolean
            If a ContextExpression, it will evaluate the logical expression against the context variables. If it is True, the transition will occur.
                Can use not, and, or, and comparison operators (>, <, >=, <=, ==, !=).
                Example: ContextExpression("not(${logged_in} and ${is_admin}) or (${guest_checkout})")
                Example with comparison: ContextExpression("${attempts} >= 3 or ${is_premium} == True or ${tier} == 'gold'")
        available (Optional[Union[Callable[[ConversableAgent, list[dict[str, Any]]], bool], str, ContextExpression]]): Optional condition to determine if this OnContextCondition is included for the LLM to evaluate.
            If a string, it will look up the value of the context variable with that name, which should be a bool, to determine whether it should include this condition.
            If a ContextExpression, it will evaluate the logical expression against the context variables. Can use not, and, or, and comparison operators (>, <, >=, <=, ==, !=).
            The Callable signature is:
                def my_available_func(agent: ConversableAgent, messages: list[Dict[str, Any]]) -> bool

    """

    target: Optional[Union[ConversableAgent, dict[str, Any]]] = None
    condition: Optional[Union[str, ContextExpression]] = None
    available: Optional[Union[Callable[[ConversableAgent, list[dict[str, Any]]], bool], str, ContextExpression]] = None

    def __post_init__(self) -> None:
        # Ensure valid types
        if (self.target is not None) and (not isinstance(self.target, (ConversableAgent, dict))):
            raise ValueError("'target' must be a ConversableAgent or a dict")

        # Ensure they have a condition
        if isinstance(self.condition, str):
            if not self.condition.strip():
                raise ValueError("'condition' must be a non-empty string")

            self._context_condition = ContextExpression("${" + self.condition + "}")
        else:
            if not isinstance(self.condition, ContextExpression):
                raise ValueError("'condition' must be a string on ContextExpression")

            self._context_condition = self.condition

        if (self.available is not None) and (
            not (isinstance(self.available, (str, ContextExpression)) or callable(self.available))
        ):
            raise ValueError("'available' must be a callable, a string, or a ContextExpression")


def _establish_swarm_agent(agent: ConversableAgent) -> None:
    """Establish the swarm agent with the swarm-related attributes and hooks. Not for the tool executor.

    Args:
        agent (ConversableAgent): The agent to establish as a swarm agent.
    """

    def _swarm_agent_str(self: ConversableAgent) -> str:
        """Customise the __str__ method to show the agent name for transition messages."""
        return f"Swarm agent --> {self.name}"

    agent._swarm_after_work = None  # type: ignore[attr-defined]
    agent._swarm_after_work_selection_msg = None  # type: ignore[attr-defined]

    # Store nested chats hand offs as we'll establish these in the initiate_swarm_chat
    # List of Dictionaries containing the nested_chats and condition
    agent._swarm_nested_chat_handoffs = []  # type: ignore[attr-defined]

    # Store conditional functions (and their OnCondition instances) to add/remove later when transitioning to this agent
    agent._swarm_conditional_functions = {}  # type: ignore[attr-defined]

    # Register the hook to update agent state (except tool executor)
    agent.register_hook("update_agent_state", _update_conditional_functions)

    # Store the OnContextConditions for evaluation (list[OnContextCondition])
    agent._swarm_oncontextconditions = []  # type: ignore[attr-defined]

    # Register a reply function to run Python function-based OnConditions before any other reply function
    agent.register_reply(trigger=([Agent, None]), reply_func=_run_oncontextconditions, position=0)

    agent._get_display_name = MethodType(_swarm_agent_str, agent)  # type: ignore[method-assign]

    # Mark this agent as established as a swarm agent
    agent._swarm_is_established = True  # type: ignore[attr-defined]


def _link_agents_to_swarm_manager(agents: list[Agent], group_chat_manager: Agent) -> None:
    """Link all agents to the GroupChatManager so they can access the underlying GroupChat and other agents.

    This is primarily used so that agents can set the tool executor's _swarm_next_agent attribute to control
    the next agent programmatically.

    Does not link the Tool Executor agent.
    """
    for agent in agents:
        agent._swarm_manager = group_chat_manager  # type: ignore[attr-defined]


def _run_oncontextconditions(
    agent: ConversableAgent,
    messages: Optional[list[dict[str, Any]]] = None,
    sender: Optional[Agent] = None,
    config: Optional[Any] = None,
) -> tuple[bool, Optional[Union[str, dict[str, Any]]]]:
    """Run OnContextConditions for an agent before any other reply function."""
    for on_condition in agent._swarm_oncontextconditions:  # type: ignore[attr-defined]
        is_available = True

        if on_condition.available is not None:
            if callable(on_condition.available):
                is_available = on_condition.available(agent, next(iter(agent.chat_messages.values())))
            elif isinstance(on_condition.available, str):
                is_available = agent.context_variables.get(on_condition.available) or False
            elif isinstance(on_condition.available, ContextExpression):
                is_available = on_condition.available.evaluate(agent.context_variables)

        if is_available and on_condition._context_condition.evaluate(agent.context_variables):
            # Condition has been met, we'll set the Tool Executor's _swarm_next_agent
            # attribute and that will be picked up on the next iteration when
            # _determine_next_agent is called
            for agent in agent._swarm_manager.groupchat.agents:  # type: ignore[attr-defined]
                if agent.name == __TOOL_EXECUTOR_NAME__:
                    agent._swarm_next_agent = on_condition.target  # type: ignore[attr-defined]
                    break

            if isinstance(on_condition.target, ConversableAgent):
                transfer_name = on_condition.target.name
            else:
                transfer_name = "a nested chat"

            return True, "[Handing off to " + transfer_name + "]"

    return False, None


def _modify_context_variables_param(f: Callable[..., Any], context_variables: ContextVariables) -> Callable[..., Any]:
    """Modifies the context_variables parameter to use dependency injection and link it to the swarm context variables.

    This essentially changes:
    def some_function(some_variable: int, context_variables: ContextVariables) -> str:

    to:

    def some_function(some_variable: int, context_variables: Annotated[ContextVariables, Depends(on(self.context_variables))]) -> str:
    """
    sig = inspect.signature(f)

    # Check if context_variables parameter exists and update it if so
    if __CONTEXT_VARIABLES_PARAM_NAME__ in sig.parameters:
        new_params = []
        for name, param in sig.parameters.items():
            if name == __CONTEXT_VARIABLES_PARAM_NAME__:
                # Replace with new annotation using Depends
                new_param = param.replace(annotation=Annotated[ContextVariables, Depends(on(context_variables))])
                new_params.append(new_param)
            else:
                new_params.append(param)

        # Update signature
        new_sig = sig.replace(parameters=new_params)
        f.__signature__ = new_sig  # type: ignore[attr-defined]

    return f


def _change_tool_context_variables_to_depends(
    agent: ConversableAgent, current_tool: Tool, context_variables: ContextVariables
) -> None:
    """Checks for the context_variables parameter in the tool and updates it to use dependency injection."""

    # If the tool has a context_variables parameter, remove the tool and reregister it without the parameter
    if __CONTEXT_VARIABLES_PARAM_NAME__ in current_tool.tool_schema["function"]["parameters"]["properties"]:
        # We'll replace the tool, so start with getting the underlying function
        tool_func = current_tool._func

        # Remove the Tool from the agent
        name = current_tool._name
        description = current_tool._description
        agent.remove_tool_for_llm(current_tool)

        # Recreate the tool without the context_variables parameter
        tool_func = _modify_context_variables_param(current_tool._func, context_variables)
        tool_func = inject_params(tool_func)
        new_tool = ConversableAgent._create_tool_if_needed(func_or_tool=tool_func, name=name, description=description)

        # Re-register with the agent
        agent.register_for_llm()(new_tool)


def _prepare_swarm_agents(
    initial_agent: ConversableAgent,
    agents: list[ConversableAgent],
    context_variables: ContextVariables,
    exclude_transit_message: bool = True,
) -> tuple[ConversableAgent, list[ConversableAgent]]:
    """Validates agents, create the tool executor, configure nested chats.

    Args:
        initial_agent (ConversableAgent): The first agent in the conversation.
        agents (list[ConversableAgent]): List of all agents in the conversation.
        context_variables (ContextVariables): Context variables to assign to all agents.
        exclude_transit_message (bool): Whether to exclude transit messages from the agents.

    Returns:
        ConversableAgent: The tool executor agent.
        list[ConversableAgent]: List of nested chat agents.
    """
    if not isinstance(initial_agent, ConversableAgent):
        raise ValueError("initial_agent must be a ConversableAgent")
    if not all(isinstance(agent, ConversableAgent) for agent in agents):
        raise ValueError("Agents must be a list of ConversableAgents")

    # Initialize all agents as swarm agents
    for agent in agents:
        if not hasattr(agent, "_swarm_is_established"):
            _establish_swarm_agent(agent)

    # Ensure all agents in hand-off after-works are in the passed in agents list
    for agent in agents:
        if (agent._swarm_after_work is not None and isinstance(agent._swarm_after_work.agent, ConversableAgent)) and (  # type: ignore[attr-defined]
            agent._swarm_after_work.agent not in agents  # type: ignore[attr-defined]
        ):
            raise ValueError("Agent in hand-off must be in the agents list")

    tool_execution = ConversableAgent(
        name=__TOOL_EXECUTOR_NAME__,
        system_message="Tool Execution, do not use this agent directly.",
    )
    _set_to_tool_execution(tool_execution)

    nested_chat_agents: list[ConversableAgent] = []
    for agent in agents:
        _create_nested_chats(agent, nested_chat_agents)

    # Update any agent's tools that have context_variables as a parameter
    # To use Dependency Injection

    # Update tool execution agent with all the functions from all the agents
    for agent in agents + nested_chat_agents:
        tool_execution._function_map.update(agent._function_map)

        # Add conditional functions to the tool_execution agent
        for func_name, (func, _) in agent._swarm_conditional_functions.items():  # type: ignore[attr-defined]
            tool_execution._function_map[func_name] = func

        # Update any agent tools that have context_variables parameters to use Dependency Injection
        for tool in agent.tools:
            _change_tool_context_variables_to_depends(agent, tool, context_variables)

        # Add all tools to the Tool Executor agent
        for tool in agent.tools:
            tool_execution.register_for_execution(serialize=False, silent_override=True)(tool)

    if exclude_transit_message:
        # get all transit functions names
        to_be_removed = []
        for agent in agents + nested_chat_agents:
            if hasattr(agent, "_swarm_conditional_functions"):
                to_be_removed += list(agent._swarm_conditional_functions.keys())

        # register hook to remove transit messages for swarm agents
        for agent in agents + nested_chat_agents:
            agent.register_hook("process_all_messages_before_reply", make_remove_function(to_be_removed))

    return tool_execution, nested_chat_agents


def _create_nested_chats(agent: ConversableAgent, nested_chat_agents: list[ConversableAgent]) -> None:
    """Create nested chat agents and register nested chats.

    Args:
        agent (ConversableAgent): The agent to create nested chat agents for, including registering the hand offs.
        nested_chat_agents (list[ConversableAgent]): List for all nested chat agents, appends to this.
    """

    def create_nested_chat_agent(agent: ConversableAgent, nested_chats: dict[str, Any]) -> ConversableAgent:
        """Create a nested chat agent for a nested chat configuration.

        Args:
            agent (ConversableAgent): The agent to create the nested chat agent for.
            nested_chats (dict[str, Any]): The nested chat configuration.

        Returns:
            The created nested chat agent.
        """
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

        return nested_chat_agent

    for i, nested_chat_handoff in enumerate(agent._swarm_nested_chat_handoffs):  # type: ignore[attr-defined]
        llm_nested_chats: dict[str, Any] = nested_chat_handoff["nested_chats"]

        # Create nested chat agent
        nested_chat_agent = create_nested_chat_agent(agent, llm_nested_chats)
        nested_chat_agents.append(nested_chat_agent)

        # Nested chat is triggered through an agent transfer to this nested chat agent
        condition = nested_chat_handoff["condition"]
        available = nested_chat_handoff["available"]
        register_hand_off(agent, OnCondition(target=nested_chat_agent, condition=condition, available=available))

    for i, nested_chat_context_handoff in enumerate(agent._swarm_oncontextconditions):  # type: ignore[attr-defined]
        if isinstance(nested_chat_context_handoff.target, dict):
            context_nested_chats: dict[str, Any] = nested_chat_context_handoff.target

            # Create nested chat agent
            nested_chat_agent = create_nested_chat_agent(agent, context_nested_chats)
            nested_chat_agents.append(nested_chat_agent)

            # Update the OnContextCondition, replacing the nested chat dictionary with the nested chat agent
            nested_chat_context_handoff.target = nested_chat_agent


def _process_initial_messages(
    messages: Union[list[dict[str, Any]], str],
    user_agent: Optional[UserProxyAgent],
    agents: list[ConversableAgent],
    nested_chat_agents: list[ConversableAgent],
) -> tuple[list[dict[str, Any]], Optional[Agent], list[str], list[Agent]]:
    """Process initial messages, validating agent names against messages, and determining the last agent to speak.

    Args:
        messages: Initial messages to process.
        user_agent: Optional user proxy agent passed in to a_/initiate_swarm_chat.
        agents: Agents in swarm.
        nested_chat_agents: List of nested chat agents.

    Returns:
        list[dict[str, Any]]: Processed message(s).
        Agent: Last agent to speak.
        list[str]: List of agent names.
        list[Agent]: List of temporary user proxy agents to add to GroupChat.
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    swarm_agent_names = [agent.name for agent in agents + nested_chat_agents]

    # If there's only one message and there's no identified swarm agent
    # Start with a user proxy agent, creating one if they haven't passed one in
    last_agent: Optional[Agent]
    temp_user_proxy: Optional[Agent] = None
    temp_user_list: list[Agent] = []
    if len(messages) == 1 and "name" not in messages[0] and not user_agent:
        temp_user_proxy = UserProxyAgent(name="_User", code_execution_config=False)
        last_agent = temp_user_proxy
        temp_user_list.append(temp_user_proxy)
    else:
        last_message = messages[0]
        if "name" in last_message:
            if last_message["name"] in swarm_agent_names:
                last_agent = next(agent for agent in agents + nested_chat_agents if agent.name == last_message["name"])  # type: ignore[assignment]
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
    context_variables: ContextVariables,
) -> None:
    """Assign a common context_variables reference to all agents in the swarm, including the tool executor and group chat manager.

    Args:
        tool_execution: The tool execution agent.
        agents: List of all agents in the conversation.
        manager: GroupChatManager instance.
        context_variables: Context variables to assign to all agents.
    """
    for agent in agents + [tool_execution] + [manager]:
        agent.context_variables = context_variables


def _cleanup_temp_user_messages(chat_result: ChatResult) -> None:
    """Remove temporary user proxy agent name from messages before returning.

    Args:
        chat_result: ChatResult instance.
    """
    for message in chat_result.chat_history:
        if "name" in message and message["name"] == "_User":
            del message["name"]


def _prepare_groupchat_auto_speaker(
    groupchat: GroupChat,
    last_swarm_agent: ConversableAgent,
    after_work_next_agent_selection_msg: Optional[
        Union[str, ContextStr, Callable[[ConversableAgent, list[dict[str, Any]]], str]]
    ],
) -> None:
    """Prepare the group chat for auto speaker selection, includes updating or restore the groupchat speaker selection message.

    Tool Executor and Nested Chat agents will be removed from the available agents list.

    Args:
        groupchat (GroupChat): GroupChat instance.
        last_swarm_agent (ConversableAgent): The last swarm agent for which the LLM config is used
        after_work_next_agent_selection_msg (Union[str, ContextStr, Callable[..., Any]]): Optional message to use for the agent selection (in internal group chat).
            if a string, it will be use the string a the prompt template, no context variable substitution however '{agentlist}' will be substituted for a list of agents.
            if a ContextStr, it will substitute the agentlist first and then the context variables
            if a Callable, it will not substitute the agentlist or context variables, signature:
                def my_selection_message(agent: ConversableAgent, messages: list[dict[str, Any]]) -> str
    """

    def substitute_agentlist(template: str) -> str:
        # Run through group chat's string substitution first for {agentlist}
        # We need to do this so that the next substitution doesn't fail with agentlist
        # and we can remove the tool executor and nested chats from the available agents list
        agent_list = [
            agent
            for agent in groupchat.agents
            if agent.name != __TOOL_EXECUTOR_NAME__ and not agent.name.startswith("nested_chat_")
        ]

        groupchat.select_speaker_prompt_template = template
        return groupchat.select_speaker_prompt(agent_list)

    if after_work_next_agent_selection_msg is None:
        # If there's no selection message, restore the default and filter out the tool executor and nested chat agents
        groupchat.select_speaker_prompt_template = substitute_agentlist(SELECT_SPEAKER_PROMPT_TEMPLATE)
    elif isinstance(after_work_next_agent_selection_msg, str):
        # No context variable substitution for string, but agentlist will be substituted
        groupchat.select_speaker_prompt_template = substitute_agentlist(after_work_next_agent_selection_msg)
    elif isinstance(after_work_next_agent_selection_msg, ContextStr):
        # Replace the agentlist in the string first, putting it into a new ContextStr
        agent_list_replaced_string = ContextStr(
            template=substitute_agentlist(after_work_next_agent_selection_msg.template)
        )

        # Then replace the context variables
        groupchat.select_speaker_prompt_template = agent_list_replaced_string.format(  # type: ignore[assignment]
            last_swarm_agent.context_variables
        )
    elif callable(after_work_next_agent_selection_msg):
        groupchat.select_speaker_prompt_template = substitute_agentlist(
            after_work_next_agent_selection_msg(last_swarm_agent, groupchat.messages)
        )


def _determine_next_agent(
    last_speaker: ConversableAgent,
    groupchat: GroupChat,
    initial_agent: ConversableAgent,
    use_initial_agent: bool,
    tool_execution: ConversableAgent,
    swarm_agent_names: list[str],
    user_agent: Optional[UserProxyAgent],
    swarm_after_work: Optional[Union[AfterWorkOption, Callable[..., Any]]],
) -> Optional[Union[Agent, Literal["auto"]]]:
    """Determine the next agent in the conversation.

    Args:
        last_speaker (ConversableAgent): The last agent to speak.
        groupchat (GroupChat): GroupChat instance.
        initial_agent (ConversableAgent): The initial agent in the conversation.
        use_initial_agent (bool): Whether to use the initial agent straight away.
        tool_execution (ConversableAgent): The tool execution agent.
        swarm_agent_names (list[str]): List of agent names.
        user_agent (UserProxyAgent): Optional user proxy agent.
        swarm_after_work (Union[AfterWorkOption, Callable[..., Any]]): Method to handle conversation continuation when an agent doesn't select the next agent.
    """
    if use_initial_agent:
        return initial_agent

    if "tool_calls" in groupchat.messages[-1]:
        return tool_execution

    after_work_condition = None

    if tool_execution._swarm_next_agent is not None:  # type: ignore[attr-defined]
        next_agent: Optional[Agent] = tool_execution._swarm_next_agent  # type: ignore[attr-defined]
        tool_execution._swarm_next_agent = None  # type: ignore[attr-defined]

        if not isinstance(next_agent, AfterWorkOption):
            # Check for string, access agent from group chat.

            if isinstance(next_agent, str):
                if next_agent in swarm_agent_names:
                    next_agent = groupchat.agent_by_name(name=next_agent)
                else:
                    raise ValueError(
                        f"No agent found with the name '{next_agent}'. Ensure the agent exists in the swarm."
                    )

            return next_agent
        else:
            after_work_condition = next_agent

    # get the last swarm agent
    last_swarm_speaker = None
    for message in reversed(groupchat.messages):
        if "name" in message and message["name"] in swarm_agent_names and message["name"] != __TOOL_EXECUTOR_NAME__:
            agent = groupchat.agent_by_name(name=message["name"])
            if isinstance(agent, ConversableAgent):
                last_swarm_speaker = agent
                break
    if last_swarm_speaker is None:
        raise ValueError("No swarm agent found in the message history")

    # If the user last spoke, return to the agent prior
    if after_work_condition is None and (
        (user_agent and last_speaker == user_agent) or groupchat.messages[-1]["role"] == "tool"
    ):
        return last_swarm_speaker

    after_work_next_agent_selection_msg = None

    if after_work_condition is None:
        # Resolve after_work condition if one hasn't been passed in (agent-level overrides global)
        after_work_condition = (
            last_swarm_speaker._swarm_after_work  # type: ignore[attr-defined]
            if last_swarm_speaker._swarm_after_work is not None  # type: ignore[attr-defined]
            else swarm_after_work
        )

    if isinstance(after_work_condition, AfterWork):
        after_work_next_agent_selection_msg = after_work_condition.next_agent_selection_msg
        after_work_condition = after_work_condition.agent

    # Evaluate callable after_work
    if callable(after_work_condition):
        after_work_condition = after_work_condition(last_swarm_speaker, groupchat.messages, groupchat)

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
            return last_swarm_speaker
        elif after_work_condition == AfterWorkOption.SWARM_MANAGER:
            _prepare_groupchat_auto_speaker(groupchat, last_swarm_speaker, after_work_next_agent_selection_msg)
            return "auto"
    else:
        raise ValueError("Invalid After Work condition or return value from callable")


def create_swarm_transition(
    initial_agent: ConversableAgent,
    tool_execution: ConversableAgent,
    swarm_agent_names: list[str],
    user_agent: Optional[UserProxyAgent],
    swarm_after_work: Optional[Union[AfterWorkOption, Callable[..., Any]]],
) -> Callable[[ConversableAgent, GroupChat], Optional[Union[Agent, Literal["auto"]]]]:
    """Creates a transition function for swarm chat with enclosed state for the use_initial_agent.

    Args:
        initial_agent (ConversableAgent): The first agent to speak
        tool_execution (ConversableAgent): The tool execution agent
        swarm_agent_names (list[str]): List of all agent names
        user_agent (UserProxyAgent): Optional user proxy agent
        swarm_after_work (Union[AfterWorkOption, Callable[..., Any]]): Swarm-level after work

    Returns:
        Callable transition function (for sync and async swarm chats)
    """
    # Create enclosed state, this will be set once per creation so will only be True on the first execution
    # of swarm_transition
    state = {"use_initial_agent": True}

    def swarm_transition(
        last_speaker: ConversableAgent, groupchat: GroupChat
    ) -> Optional[Union[Agent, Literal["auto"]]]:
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


def _create_swarm_manager(
    groupchat: GroupChat, swarm_manager_args: Optional[dict[str, Any]], agents: list[ConversableAgent]
) -> GroupChatManager:
    """Create a GroupChatManager for the swarm chat utilising any arguments passed in and ensure an LLM Config exists if needed

    Args:
        groupchat (GroupChat): Swarm groupchat.
        swarm_manager_args (dict[str, Any]): Swarm manager arguments to create the GroupChatManager.
        agents (list[ConversableAgent]): List of agents in the swarm.

    Returns:
        GroupChatManager: GroupChatManager instance.
    """
    manager_args = (swarm_manager_args or {}).copy()
    if "groupchat" in manager_args:
        raise ValueError("'groupchat' cannot be specified in swarm_manager_args as it is set by initiate_swarm_chat")
    manager = GroupChatManager(groupchat, **manager_args)

    # Ensure that our manager has an LLM Config if we have any AfterWorkOption.SWARM_MANAGER after works
    if manager.llm_config is False:
        for agent in agents:
            if (
                agent._swarm_after_work  # type: ignore[attr-defined]
                and isinstance(agent._swarm_after_work.agent, AfterWorkOption)  # type: ignore[attr-defined]
                and agent._swarm_after_work.agent == AfterWorkOption.SWARM_MANAGER  # type: ignore[attr-defined]
            ):
                raise ValueError(
                    "The swarm manager doesn't have an LLM Config and it is required for AfterWorkOption.SWARM_MANAGER. Use the swarm_manager_args to specify the LLM Config for the swarm manager."
                )

    return manager


def make_remove_function(tool_msgs_to_remove: list[str]) -> Callable[[list[dict[str, Any]]], list[dict[str, Any]]]:
    """Create a function to remove messages with tool calls from the messages list.

    The returned function can be registered as a hook to "process_all_messages_before_reply"" to remove messages with tool calls.
    """

    def remove_messages(messages: list[dict[str, Any]], tool_msgs_to_remove: list[str]) -> list[dict[str, Any]]:
        copied = copy.deepcopy(messages)
        new_messages = []
        removed_tool_ids = []
        for message in copied:
            # remove tool calls
            if message.get("tool_calls") is not None:
                filtered_tool_calls = []
                for tool_call in message["tool_calls"]:
                    if tool_call.get("function") is not None and tool_call["function"]["name"] in tool_msgs_to_remove:
                        # remove
                        removed_tool_ids.append(tool_call["id"])
                    else:
                        filtered_tool_calls.append(tool_call)
                if len(filtered_tool_calls) > 0:
                    message["tool_calls"] = filtered_tool_calls
                else:
                    del message["tool_calls"]
                    if (
                        message.get("content") is None
                        or message.get("content") == ""
                        or message.get("content") == "None"
                    ):
                        continue  # if no tool call and no content, skip this message
                    # else: keep the message with tool_calls removed
            # remove corresponding tool responses
            elif message.get("tool_responses") is not None:
                filtered_tool_responses = []
                for tool_response in message["tool_responses"]:
                    if tool_response["tool_call_id"] not in removed_tool_ids:
                        filtered_tool_responses.append(tool_response)

                if len(filtered_tool_responses) > 0:
                    message["tool_responses"] = filtered_tool_responses
                else:
                    continue

            new_messages.append(message)

        return new_messages

    return partial(remove_messages, tool_msgs_to_remove=tool_msgs_to_remove)


@export_module("autogen")
def initiate_swarm_chat(
    initial_agent: ConversableAgent,
    messages: Union[list[dict[str, Any]], str],
    agents: list[ConversableAgent],
    user_agent: Optional[UserProxyAgent] = None,
    swarm_manager_args: Optional[dict[str, Any]] = None,
    max_rounds: int = 20,
    context_variables: Optional[ContextVariables] = None,
    after_work: Optional[
        Union[
            AfterWorkOption,
            Callable[
                [ConversableAgent, list[dict[str, Any]], GroupChat], Union[AfterWorkOption, ConversableAgent, str]
            ],
        ]
    ] = AfterWorkOption.TERMINATE,
    exclude_transit_message: bool = True,
) -> tuple[ChatResult, ContextVariables, ConversableAgent]:
    """Initialize and run a swarm chat

    Args:
        initial_agent: The first receiving agent of the conversation.
        messages: Initial message(s).
        agents: list of swarm agents.
        user_agent: Optional user proxy agent for falling back to.
        swarm_manager_args: Optional group chat manager arguments used to establish the swarm's groupchat manager, required when AfterWorkOption.SWARM_MANAGER is used.
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
                def custom_afterwork_func(last_speaker: ConversableAgent, messages: list[dict[str, Any]], groupchat: GroupChat) -> Union[AfterWorkOption, ConversableAgent, str]:
                ```
        exclude_transit_message:  all registered handoff function call and responses messages will be removed from message list before calling an LLM.
            Note: only with transition functions added with `register_handoff` will be removed. If you pass in a function to manage workflow, it will not be removed. You may register a cumstomized hook to `process_all_messages_before_reply` to remove that.
    Returns:
        ChatResult:     Conversations chat history.
        ContextVariables: Updated Context variables.
        ConversableAgent:     Last speaker.
    """
    context_variables = context_variables or ContextVariables()

    tool_execution, nested_chat_agents = _prepare_swarm_agents(
        initial_agent, agents, context_variables, exclude_transit_message
    )

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

    manager = _create_swarm_manager(groupchat, swarm_manager_args, agents)

    # Point all ConversableAgent's context variables to this function's context_variables
    _setup_context_variables(tool_execution, agents, manager, context_variables)

    # Link all agents with the GroupChatManager to allow access to the group chat
    # and other agents, particularly the tool executor for setting _swarm_next_agent
    _link_agents_to_swarm_manager(groupchat.agents, manager)  # Commented out as the function is not defined

    if len(processed_messages) > 1:
        last_agent, last_message = manager.resume(messages=processed_messages)
        clear_history = False
    else:
        last_message = processed_messages[0]
        clear_history = True

    if last_agent is None:
        raise ValueError("No agent selected to start the conversation")

    chat_result = last_agent.initiate_chat(  # type: ignore[attr-defined]
        manager,
        message=last_message,
        clear_history=clear_history,
    )

    _cleanup_temp_user_messages(chat_result)

    return chat_result, context_variables, manager.last_speaker  # type: ignore[return-value]


@export_module("autogen")
def run_swarm(
    initial_agent: ConversableAgent,
    messages: Union[list[dict[str, Any]], str],
    agents: list[ConversableAgent],
    user_agent: Optional[UserProxyAgent] = None,
    swarm_manager_args: Optional[dict[str, Any]] = None,
    max_rounds: int = 20,
    context_variables: Optional[ContextVariables] = None,
    after_work: Optional[
        Union[
            AfterWorkOption,
            Callable[
                [ConversableAgent, list[dict[str, Any]], GroupChat], Union[AfterWorkOption, ConversableAgent, str]
            ],
        ]
    ] = AfterWorkOption.TERMINATE,
    exclude_transit_message: bool = True,
) -> RunResponseProtocol:
    iostream = ThreadIOStream()
    response = RunResponse(iostream, agents)  # type: ignore[arg-type]

    def stream_run(
        iostream: ThreadIOStream = iostream,
        response: RunResponse = response,
    ) -> None:
        with IOStream.set_default(iostream):
            try:
                chat_result, returned_context_variables, last_speaker = initiate_swarm_chat(
                    initial_agent=initial_agent,
                    messages=messages,
                    agents=agents,
                    user_agent=user_agent,
                    swarm_manager_args=swarm_manager_args,
                    max_rounds=max_rounds,
                    context_variables=context_variables,
                    after_work=after_work,
                    exclude_transit_message=exclude_transit_message,
                )

                IOStream.get_default().send(
                    RunCompletionEvent(  # type: ignore[call-arg]
                        history=chat_result.chat_history,
                        summary=chat_result.summary,
                        cost=chat_result.cost,
                        last_speaker=last_speaker.name,
                        context_variables=returned_context_variables,
                    )
                )
            except Exception as e:
                response.iostream.send(ErrorEvent(error=e))  # type: ignore[call-arg]

    threading.Thread(
        target=stream_run,
    ).start()

    return response


@export_module("autogen")
async def a_initiate_swarm_chat(
    initial_agent: ConversableAgent,
    messages: Union[list[dict[str, Any]], str],
    agents: list[ConversableAgent],
    user_agent: Optional[UserProxyAgent] = None,
    swarm_manager_args: Optional[dict[str, Any]] = None,
    max_rounds: int = 20,
    context_variables: Optional[ContextVariables] = None,
    after_work: Optional[
        Union[
            AfterWorkOption,
            Callable[
                [ConversableAgent, list[dict[str, Any]], GroupChat], Union[AfterWorkOption, ConversableAgent, str]
            ],
        ]
    ] = AfterWorkOption.TERMINATE,
    exclude_transit_message: bool = True,
) -> tuple[ChatResult, ContextVariables, ConversableAgent]:
    """Initialize and run a swarm chat asynchronously

    Args:
        initial_agent: The first receiving agent of the conversation.
        messages: Initial message(s).
        agents: List of swarm agents.
        user_agent: Optional user proxy agent for falling back to.
        swarm_manager_args: Optional group chat manager arguments used to establish the swarm's groupchat manager, required when AfterWorkOption.SWARM_MANAGER is used.
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
                def custom_afterwork_func(last_speaker: ConversableAgent, messages: list[dict[str, Any]], groupchat: GroupChat) -> Union[AfterWorkOption, ConversableAgent, str]:
                ```
        exclude_transit_message:  all registered handoff function call and responses messages will be removed from message list before calling an LLM.
            Note: only with transition functions added with `register_handoff` will be removed. If you pass in a function to manage workflow, it will not be removed. You may register a cumstomized hook to `process_all_messages_before_reply` to remove that.
    Returns:
        ChatResult:     Conversations chat history.
        ContextVariables: Updated Context variables.
        ConversableAgent:     Last speaker.
    """
    context_variables = context_variables or ContextVariables()
    tool_execution, nested_chat_agents = _prepare_swarm_agents(
        initial_agent, agents, context_variables, exclude_transit_message
    )

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

    manager = _create_swarm_manager(groupchat, swarm_manager_args, agents)

    # Point all ConversableAgent's context variables to this function's context_variables
    _setup_context_variables(tool_execution, agents, manager, context_variables)

    # Link all agents with the GroupChatManager to allow access to the group chat
    # and other agents, particularly the tool executor for setting _swarm_next_agent
    _link_agents_to_swarm_manager(groupchat.agents, manager)

    if len(processed_messages) > 1:
        last_agent, last_message = await manager.a_resume(messages=processed_messages)
        clear_history = False
    else:
        last_message = processed_messages[0]
        clear_history = True

    if last_agent is None:
        raise ValueError("No agent selected to start the conversation")

    chat_result = await last_agent.a_initiate_chat(  # type: ignore[attr-defined]
        manager,
        message=last_message,
        clear_history=clear_history,
    )

    _cleanup_temp_user_messages(chat_result)

    return chat_result, context_variables, manager.last_speaker  # type: ignore[return-value]


@export_module("autogen")
async def a_run_swarm(
    initial_agent: ConversableAgent,
    messages: Union[list[dict[str, Any]], str],
    agents: list[ConversableAgent],
    user_agent: Optional[UserProxyAgent] = None,
    swarm_manager_args: Optional[dict[str, Any]] = None,
    max_rounds: int = 20,
    context_variables: Optional[ContextVariables] = None,
    after_work: Optional[
        Union[
            AfterWorkOption,
            Callable[
                [ConversableAgent, list[dict[str, Any]], GroupChat], Union[AfterWorkOption, ConversableAgent, str]
            ],
        ]
    ] = AfterWorkOption.TERMINATE,
    exclude_transit_message: bool = True,
) -> AsyncRunResponseProtocol:
    iostream = AsyncThreadIOStream()
    response = AsyncRunResponse(iostream, agents)  # type: ignore[arg-type]

    async def stream_run(
        iostream: AsyncThreadIOStream = iostream,
        response: AsyncRunResponse = response,
    ) -> None:
        with IOStream.set_default(iostream):
            try:
                chat_result, returned_context_variables, last_speaker = await a_initiate_swarm_chat(
                    initial_agent=initial_agent,
                    messages=messages,
                    agents=agents,
                    user_agent=user_agent,
                    swarm_manager_args=swarm_manager_args,
                    max_rounds=max_rounds,
                    context_variables=context_variables,
                    after_work=after_work,
                    exclude_transit_message=exclude_transit_message,
                )

                IOStream.get_default().send(
                    RunCompletionEvent(  # type: ignore[call-arg]
                        history=chat_result.chat_history,
                        summary=chat_result.summary,
                        cost=chat_result.cost,
                        last_speaker=last_speaker.name,
                        context_variables=returned_context_variables,
                    )
                )
            except Exception as e:
                response.iostream.send(ErrorEvent(error=e))  # type: ignore[call-arg]

    asyncio.create_task(stream_run())

    return response


@export_module("autogen")
class SwarmResult(BaseModel):
    """Encapsulates the possible return values for a swarm agent function."""

    values: str = ""
    agent: Optional[Union[ConversableAgent, AfterWorkOption, str]] = None
    context_variables: Optional[ContextVariables] = None

    @field_serializer("agent", when_used="json")
    def serialize_agent(self, agent: Union[ConversableAgent, str]) -> str:
        if isinstance(agent, ConversableAgent):
            return agent.name
        return agent

    def model_post_init(self, __context: Any) -> None:
        # Initialise with a new ContextVariables object if not provided
        if self.context_variables is None:
            self.context_variables = ContextVariables()

    class Config:  # Add this inner class
        arbitrary_types_allowed = True

    def __str__(self) -> str:
        return self.values


def _set_to_tool_execution(agent: ConversableAgent) -> None:
    """Set to a special instance of ConversableAgent that is responsible for executing tool calls from other swarm agents.
    This agent will be used internally and should not be visible to the user.

    It will execute the tool calls and update the referenced context_variables and next_agent accordingly.
    """
    agent._swarm_next_agent = None  # type: ignore[attr-defined]
    agent._reply_func_list.clear()
    agent.register_reply([Agent, None], _generate_swarm_tool_reply)


@export_module("autogen")
def register_hand_off(
    agent: ConversableAgent,
    hand_to: Union[list[Union[OnCondition, OnContextCondition, AfterWork]], OnCondition, OnContextCondition, AfterWork],
) -> None:
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
    if not isinstance(hand_to, (list, OnCondition, OnContextCondition, AfterWork)):
        raise ValueError("hand_to must be a list of OnCondition, OnContextCondition, or AfterWork")

    if isinstance(hand_to, (OnCondition, OnContextCondition, AfterWork)):
        hand_to = [hand_to]

    for transit in hand_to:
        if isinstance(transit, AfterWork):
            if not (isinstance(transit.agent, (AfterWorkOption, ConversableAgent, str)) or callable(transit.agent)):
                raise ValueError(f"Invalid AfterWork agent: {transit.agent}")
            agent._swarm_after_work = transit  # type: ignore[attr-defined]
            agent._swarm_after_work_selection_msg = transit.next_agent_selection_msg  # type: ignore[attr-defined]
        elif isinstance(transit, OnCondition):
            if isinstance(transit.target, ConversableAgent):
                # Transition to agent

                # Create closure with current loop transit value
                # to ensure the condition matches the one in the loop
                def make_transfer_function(current_transit: OnCondition) -> Callable[[], ConversableAgent]:
                    def transfer_to_agent() -> ConversableAgent:
                        return current_transit.target  # type: ignore[return-value]

                    return transfer_to_agent

                transfer_func = make_transfer_function(transit)

                # Store function to add/remove later based on it being 'available'
                # Function names are made unique and allow multiple OnCondition's to the same agent
                base_func_name = f"transfer_{agent.name}_to_{transit.target.name}"
                func_name = base_func_name
                count = 2
                while func_name in agent._swarm_conditional_functions:  # type: ignore[attr-defined]
                    func_name = f"{base_func_name}_{count}"
                    count += 1

                # Store function to add/remove later based on it being 'available'
                agent._swarm_conditional_functions[func_name] = (transfer_func, transit)  # type: ignore[attr-defined]

            elif isinstance(transit.target, dict):
                # Transition to a nested chat
                # We will store them here and establish them in the initiate_swarm_chat
                agent._swarm_nested_chat_handoffs.append({  # type: ignore[attr-defined]
                    "nested_chats": transit.target,
                    "condition": transit.condition,
                    "available": transit.available,
                })

        elif isinstance(transit, OnContextCondition):
            agent._swarm_oncontextconditions.append(transit)  # type: ignore[attr-defined]

        else:
            raise ValueError("Invalid hand off condition, must be either OnCondition or AfterWork")


def _update_conditional_functions(agent: ConversableAgent, messages: Optional[list[dict[str, Any]]] = None) -> None:
    """Updates the agent's functions based on the OnCondition's available condition."""
    for func_name, (func, on_condition) in agent._swarm_conditional_functions.items():  # type: ignore[attr-defined]
        is_available = True

        if on_condition.available is not None:
            if callable(on_condition.available):
                is_available = on_condition.available(agent, next(iter(agent.chat_messages.values())))
            elif isinstance(on_condition.available, str):
                is_available = agent.context_variables.get(on_condition.available) or False
            elif isinstance(on_condition.available, ContextExpression):
                is_available = on_condition.available.evaluate(agent.context_variables)

        # first remove the function if it exists
        if func_name in agent._function_map:
            agent.update_tool_signature(func_name, is_remove=True)
            del agent._function_map[func_name]

        # then add the function if it is available, so that the function signature is updated
        if is_available:
            condition = on_condition.condition
            if isinstance(condition, ContextStr):
                condition = condition.format(context_variables=agent.context_variables)
            elif callable(condition):
                condition = condition(agent, messages)

            # TODO: Don't add it if it's already there
            agent._add_single_function(func, func_name, condition)


def _generate_swarm_tool_reply(
    agent: ConversableAgent,
    messages: Optional[list[dict[str, Any]]] = None,
    sender: Optional[Agent] = None,
    config: Optional[OpenAIWrapper] = None,
) -> tuple[bool, Optional[dict[str, Any]]]:
    """Pre-processes and generates tool call replies.

    This function:
    1. Adds context_variables back to the tool call for the function, if necessary.
    2. Generates the tool calls reply.
    3. Updates context_variables and next_agent based on the tool call response."""

    if config is None:
        config = agent  # type: ignore[assignment]
    if messages is None:
        messages = agent._oai_messages[sender]

    message = messages[-1]
    if "tool_calls" in message:
        tool_call_count = len(message["tool_calls"])

        # Loop through tool calls individually (so context can be updated after each function call)
        next_agent: Optional[Agent] = None
        tool_responses_inner = []
        contents = []
        for index in range(tool_call_count):
            message_copy = copy.deepcopy(message)

            # 1. add context_variables to the tool call arguments
            tool_call = message_copy["tool_calls"][index]

            # Ensure we are only executing the one tool at a time
            message_copy["tool_calls"] = [tool_call]

            # 2. generate tool calls reply
            _, tool_message = agent.generate_tool_calls_reply([message_copy])

            if tool_message is None:
                raise ValueError("Tool call did not return a message")

            # 3. update context_variables and next_agent, convert content to string
            for tool_response in tool_message["tool_responses"]:
                content = tool_response.get("content")

                if isinstance(content, SwarmResult):
                    if content.context_variables is not None and content.context_variables.to_dict() != {}:
                        agent.context_variables.update(content.context_variables.to_dict())
                    if content.agent is not None:
                        next_agent = content.agent  # type: ignore[assignment]
                elif isinstance(content, Agent):
                    next_agent = content

                # Serialize the content to a string
                if content is not None:
                    tool_response["content"] = str(content)

                tool_responses_inner.append(tool_response)
                contents.append(str(tool_response["content"]))

        agent._swarm_next_agent = next_agent  # type: ignore[attr-defined]

        # Put the tool responses and content strings back into the response message
        # Caters for multiple tool calls
        if tool_message is None:
            raise ValueError("Tool call did not return a message")

        tool_message["tool_responses"] = tool_responses_inner
        tool_message["content"] = "\n".join(contents)

        return True, tool_message
    return False, None


class SwarmAgent(ConversableAgent):
    """SwarmAgent is deprecated and has been incorporated into ConversableAgent, use ConversableAgent instead. SwarmAgent will be removed in a future version (TBD)"""

    def __init__(self, *args: Any, **kwargs: Any):
        """Initializes a new instance of the SwarmAgent class.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        warnings.warn(
            "SwarmAgent is deprecated and has been incorporated into ConversableAgent, use ConversableAgent instead. SwarmAgent will be removed in a future version (TBD).",
            DeprecationWarning,
            stacklevel=2,
        )

        super().__init__(*args, **kwargs)
