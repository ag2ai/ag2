# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
import copy
import json
import re
import warnings
from dataclasses import dataclass
from enum import Enum
from inspect import signature
from typing import Any, Callable, Literal, Optional, Union

from pydantic import BaseModel

from ...doc_utils import export_module
from ...oai import OpenAIWrapper
from ...tools import get_function_schema
from ..agent import Agent
from ..chat import ChatResult
from ..conversable_agent import ConversableAgent
from ..groupchat import GroupChat, GroupChatManager
from ..user_proxy_agent import UserProxyAgent

# Parameter name for context variables
# Use the value in functions and they will be substituted with the context variables:
# e.g. def my_function(context_variables: Dict[str, Any], my_other_parameters: Any) -> Any:
__CONTEXT_VARIABLES_PARAM_NAME__ = "context_variables"

__TOOL_EXECUTOR_NAME__ = "Tool_Execution"


@export_module("autogen")
class AfterWorkOption(Enum):
    TERMINATE = "TERMINATE"
    REVERT_TO_USER = "REVERT_TO_USER"
    STAY = "STAY"
    SWARM_MANAGER = "SWARM_MANAGER"


@dataclass
@export_module("autogen")
class AFTER_WORK:  # noqa: N801
    """Handles the next step in the conversation when an agent doesn't suggest a tool call or a handoff

    Args:
        agent (Union[AfterWorkOption, SwarmAgent, str, Callable]): The agent to hand off to or the after work option. Can be a SwarmAgent, a string name of a SwarmAgent, an AfterWorkOption, or a Callable.
            The Callable signature is:
                def my_after_work_func(last_speaker: SwarmAgent, messages: List[Dict[str, Any]], groupchat: GroupChat) -> Union[AfterWorkOption, SwarmAgent, str]:
    """

    agent: Union[AfterWorkOption, "SwarmAgent", str, Callable]

    def __post_init__(self):
        if isinstance(self.agent, str):
            self.agent = AfterWorkOption(self.agent.upper())


@dataclass
@export_module("autogen")
class ON_CONDITION:  # noqa: N801
    """Defines a condition for transitioning to another agent or nested chats

    Args:
        target (Union[SwarmAgent, dict[str, Any]]): The agent to hand off to or the nested chat configuration. Can be a SwarmAgent or a Dict.
            If a Dict, it should follow the convention of the nested chat configuration, with the exception of a carryover configuration which is unique to Swarms.
            Swarm Nested chat documentation: https://docs.ag2.ai/docs/topics/swarm#registering-handoffs-to-a-nested-chat
        condition (str): The condition for transitioning to the target agent, evaluated by the LLM to determine whether to call the underlying function/tool which does the transition.
        available (Union[Callable, str]): Optional condition to determine if this ON_CONDITION is available. Can be a Callable or a string.
            If a string, it will look up the value of the context variable with that name, which should be a bool.
    """

    target: Union["SwarmAgent", dict[str, Any]] = None
    condition: str = ""
    available: Optional[Union[Callable, str]] = None

    def __post_init__(self):
        # Ensure valid types
        if self.target is not None:
            assert isinstance(self.target, (SwarmAgent, dict)), "'target' must be a SwarmAgent or a Dict"

        # Ensure they have a condition
        assert isinstance(self.condition, str) and self.condition.strip(), "'condition' must be a non-empty string"

        if self.available is not None:
            assert isinstance(self.available, (Callable, str)), "'available' must be a callable or a string"


@dataclass
@export_module("autogen")
class UPDATE_SYSTEM_MESSAGE:  # noqa: N801
    """Update the agent's system message before they reply

    Args:
        update_function (Union[Callable, str]): The string or function to update the agent's system message. Can be a string or a Callable.
            If a string, it will be used as a template and substitute the context variables.
            If a Callable, it should have the signature:
                def my_update_function(agent: ConversableAgent, messages: List[Dict[str, Any]]) -> str
    """

    update_function: Union[Callable, str]

    def __post_init__(self):
        if isinstance(self.update_function, str):
            # find all {var} in the string
            vars = re.findall(r"\{(\w+)\}", self.update_function)
            if len(vars) == 0:
                warnings.warn("Update function string contains no variables. This is probably unintended.")

        elif isinstance(self.update_function, Callable):
            sig = signature(self.update_function)
            if len(sig.parameters) != 2:
                raise ValueError(
                    "Update function must accept two parameters of type ConversableAgent and List[Dict[str Any]], respectively"
                )
            if sig.return_annotation != str:
                raise ValueError("Update function must return a string")
        else:
            raise ValueError("Update function must be either a string or a callable")


def _prepare_swarm_agents(
    initial_agent: "SwarmAgent",
    agents: list["SwarmAgent"],
) -> tuple["SwarmAgent", list["SwarmAgent"]]:
    """Validates agents, create the tool executor, configure nested chats.

    Args:
        initial_agent (SwarmAgent): The first agent in the conversation.
        agents (list[SwarmAgent]): List of all agents in the conversation.

    Returns:
        SwarmAgent: The tool executor agent.
        list[SwarmAgent]: List of nested chat agents.
    """
    assert isinstance(initial_agent, SwarmAgent), "initial_agent must be a SwarmAgent"
    assert all(isinstance(agent, SwarmAgent) for agent in agents), "Agents must be a list of SwarmAgents"

    # Ensure all agents in hand-off after-works are in the passed in agents list
    for agent in agents:
        if agent.after_work is not None and isinstance(agent.after_work.agent, SwarmAgent):
            assert agent.after_work.agent in agents, "Agent in hand-off must be in the agents list"

    tool_execution = SwarmAgent(
        name=__TOOL_EXECUTOR_NAME__,
        system_message="Tool Execution",
    )
    tool_execution._set_to_tool_execution()

    nested_chat_agents = []
    for agent in agents:
        _create_nested_chats(agent, nested_chat_agents)

    # Update tool execution agent with all the functions from all the agents
    for agent in agents + nested_chat_agents:
        tool_execution._function_map.update(agent._function_map)
        # Add conditional functions to the tool_execution agent
        for func_name, (func, _) in agent._conditional_functions.items():
            tool_execution._function_map[func_name] = func

    return tool_execution, nested_chat_agents


def _create_nested_chats(agent: "SwarmAgent", nested_chat_agents: list["SwarmAgent"]):
    """Create nested chat agents and register nested chats.

    Args:
        agent (SwarmAgent): The agent to create nested chat agents for, including registering the hand offs.
        nested_chat_agents (list[SwarmAgent]): List for all nested chat agents, appends to this.
    """
    for i, nested_chat_handoff in enumerate(agent._nested_chat_handoffs):
        nested_chats: dict[str, Any] = nested_chat_handoff["nested_chats"]
        condition = nested_chat_handoff["condition"]
        available = nested_chat_handoff["available"]

        # Create a nested chat agent specifically for this nested chat
        nested_chat_agent = SwarmAgent(name=f"nested_chat_{agent.name}_{i + 1}")

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
        nested_chat_agent.register_hand_off(AFTER_WORK(agent=agent))

        nested_chat_agents.append(nested_chat_agent)

        # Nested chat is triggered through an agent transfer to this nested chat agent
        agent.register_hand_off(ON_CONDITION(nested_chat_agent, condition, available))


def _process_initial_messages(
    messages: Union[list[dict[str, Any]], str],
    user_agent: Optional[UserProxyAgent],
    agents: list["SwarmAgent"],
    nested_chat_agents: list["SwarmAgent"],
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
        temp_user_proxy = UserProxyAgent(name="_User", code_execution_config=False)
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
    tool_execution: "SwarmAgent",
    agents: list["SwarmAgent"],
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
    last_speaker: "SwarmAgent",
    groupchat: GroupChat,
    initial_agent: ConversableAgent,
    use_initial_agent: bool,
    tool_execution: "SwarmAgent",
    swarm_agent_names: list[str],
    user_agent: Optional[UserProxyAgent],
    swarm_after_work: Optional[Union[AfterWorkOption, Callable]],
) -> Optional[Agent]:
    """Determine the next agent in the conversation.

    Args:
        last_speaker (SwarmAgent): The last agent to speak.
        groupchat (GroupChat): GroupChat instance.
        initial_agent (ConversableAgent): The initial agent in the conversation.
        use_initial_agent (bool): Whether to use the initial agent straight away.
        tool_execution (SwarmAgent): The tool execution agent.
        swarm_agent_names (list[str]): List of agent names.
        user_agent (UserProxyAgent): Optional user proxy agent.
        swarm_after_work (Union[AfterWorkOption, Callable]): Method to handle conversation continuation when an agent doesn't select the next agent.
    """
    if use_initial_agent:
        return initial_agent

    if "tool_calls" in groupchat.messages[-1]:
        return tool_execution

    after_work_condition = None

    if tool_execution._next_agent is not None:
        next_agent = tool_execution._next_agent
        tool_execution._next_agent = None

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
            if isinstance(agent, SwarmAgent):
                last_swarm_speaker = agent
                break
    if last_swarm_speaker is None:
        raise ValueError("No swarm agent found in the message history")

    if after_work_condition is None:
        # If the user last spoke, return to the agent prior
        if (user_agent and last_speaker == user_agent) or groupchat.messages[-1]["role"] == "tool":
            return last_swarm_speaker

        # Resolve after_work condition (agent-level overrides global)
        after_work_condition = (
            last_swarm_speaker.after_work if last_swarm_speaker.after_work is not None else swarm_after_work
        )

        if isinstance(after_work_condition, AFTER_WORK):
            after_work_condition = after_work_condition.agent

        # Evaluate callable after_work
        if isinstance(after_work_condition, Callable):
            after_work_condition = after_work_condition(last_swarm_speaker, groupchat.messages, groupchat)

    if isinstance(after_work_condition, str):  # Agent name in a string
        if after_work_condition in swarm_agent_names:
            return groupchat.agent_by_name(name=after_work_condition)
        else:
            raise ValueError(f"Invalid agent name in after_work: {after_work_condition}")
    elif isinstance(after_work_condition, SwarmAgent):
        return after_work_condition
    elif isinstance(after_work_condition, AfterWorkOption):
        if after_work_condition == AfterWorkOption.TERMINATE:
            return None
        elif after_work_condition == AfterWorkOption.REVERT_TO_USER:
            return None if user_agent is None else user_agent
        elif after_work_condition == AfterWorkOption.STAY:
            return last_swarm_speaker
        elif after_work_condition == AfterWorkOption.SWARM_MANAGER:
            return "auto"
    else:
        raise ValueError("Invalid After Work condition or return value from callable")


def create_swarm_transition(
    initial_agent: "SwarmAgent",
    tool_execution: "SwarmAgent",
    swarm_agent_names: list[str],
    user_agent: Optional[UserProxyAgent],
    swarm_after_work: Optional[Union[AfterWorkOption, Callable]],
) -> Callable[["SwarmAgent", GroupChat], Optional[Agent]]:
    """Creates a transition function for swarm chat with enclosed state for the use_initial_agent.

    Args:
        initial_agent (SwarmAgent): The first agent to speak
        tool_execution (SwarmAgent): The tool execution agent
        swarm_agent_names (list[str]): List of all agent names
        user_agent (UserProxyAgent): Optional user proxy agent
        swarm_after_work (Union[AfterWorkOption, Callable]): Swarm-level after work

    Returns:
        Callable transition function (for sync and async swarm chats)
    """
    # Create enclosed state, this will be set once per creation so will only be True on the first execution
    # of swarm_transition
    state = {"use_initial_agent": True}

    def swarm_transition(last_speaker: SwarmAgent, groupchat: GroupChat) -> Optional[Agent]:
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


@export_module("autogen")
def initiate_swarm_chat(
    initial_agent: "SwarmAgent",
    messages: Union[list[dict[str, Any]], str],
    agents: list["SwarmAgent"],
    user_agent: Optional[UserProxyAgent] = None,
    max_rounds: int = 20,
    context_variables: Optional[dict[str, Any]] = None,
    after_work: Optional[Union[AfterWorkOption, Callable]] = AFTER_WORK(AfterWorkOption.TERMINATE),
) -> tuple[ChatResult, dict[str, Any], "SwarmAgent"]:
    """Initialize and run a swarm chat

    Args:
        initial_agent: The first receiving agent of the conversation.
        messages: Initial message(s).
        agents: List of swarm agents.
        user_agent: Optional user proxy agent for falling back to.
        max_rounds: Maximum number of conversation rounds.
        context_variables: Starting context variables.
        after_work: Method to handle conversation continuation when an agent doesn't select the next agent. If no agent is selected and no tool calls are output, we will use this method to determine the next agent.
            Must be a AFTER_WORK instance (which is a dataclass accepting a SwarmAgent, AfterWorkOption, A str (of the AfterWorkOption)) or a callable.
            AfterWorkOption:
                - TERMINATE (Default): Terminate the conversation.
                - REVERT_TO_USER : Revert to the user agent if a user agent is provided. If not provided, terminate the conversation.
                - STAY : Stay with the last speaker.

            Callable: A custom function that takes the current agent, messages, and groupchat as arguments and returns an AfterWorkOption or a SwarmAgent (by reference or string name).
                ```python
                def custom_afterwork_func(last_speaker: SwarmAgent, messages: List[Dict[str, Any]], groupchat: GroupChat) -> Union[AfterWorkOption, SwarmAgent, str]:
                ```
    Returns:
        ChatResult:     Conversations chat history.
        Dict[str, Any]: Updated Context variables.
        SwarmAgent:     Last speaker.
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

    # Point all SwarmAgent's context variables to this function's context_variables
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


@export_module("autogen")
async def a_initiate_swarm_chat(
    initial_agent: "SwarmAgent",
    messages: Union[list[dict[str, Any]], str],
    agents: list["SwarmAgent"],
    user_agent: Optional[UserProxyAgent] = None,
    max_rounds: int = 20,
    context_variables: Optional[dict[str, Any]] = None,
    after_work: Optional[Union[AfterWorkOption, Callable]] = AFTER_WORK(AfterWorkOption.TERMINATE),
) -> tuple[ChatResult, dict[str, Any], "SwarmAgent"]:
    """Initialize and run a swarm chat asynchronously

    Args:
        initial_agent: The first receiving agent of the conversation.
        messages: Initial message(s).
        agents: List of swarm agents.
        user_agent: Optional user proxy agent for falling back to.
        max_rounds: Maximum number of conversation rounds.
        context_variables: Starting context variables.
        after_work: Method to handle conversation continuation when an agent doesn't select the next agent. If no agent is selected and no tool calls are output, we will use this method to determine the next agent.
            Must be a AFTER_WORK instance (which is a dataclass accepting a SwarmAgent, AfterWorkOption, A str (of the AfterWorkOption)) or a callable.
            AfterWorkOption:
                - TERMINATE (Default): Terminate the conversation.
                - REVERT_TO_USER : Revert to the user agent if a user agent is provided. If not provided, terminate the conversation.
                - STAY : Stay with the last speaker.

            Callable: A custom function that takes the current agent, messages, and groupchat as arguments and returns an AfterWorkOption or a SwarmAgent (by reference or string name).
                ```python
                def custom_afterwork_func(last_speaker: SwarmAgent, messages: List[Dict[str, Any]], groupchat: GroupChat) -> Union[AfterWorkOption, SwarmAgent, str]:
                ```
    Returns:
        ChatResult:     Conversations chat history.
        Dict[str, Any]: Updated Context variables.
        SwarmAgent:     Last speaker.
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

    # Point all SwarmAgent's context variables to this function's context_variables
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


@export_module("autogen")
class SwarmAgent(ConversableAgent):
    """Swarm agent for participating in a swarm.

    SwarmAgent is a subclass of ConversableAgent.

    Additional args:
        functions (List[Callable]): A list of functions to register with the agent.
        update_agent_state_before_reply (List[Callable]): A list of functions, including UPDATE_SYSTEM_MESSAGEs, called to update the agent before it replies.
    """

    def __init__(
        self,
        name: str,
        system_message: Optional[str] = "You are a helpful AI Assistant.",
        llm_config: Optional[Union[dict, Literal[False]]] = None,
        functions: Union[list[Callable], Callable] = None,
        is_termination_msg: Optional[Callable[[dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Literal["ALWAYS", "NEVER", "TERMINATE"] = "NEVER",
        description: Optional[str] = None,
        code_execution_config=False,
        update_agent_state_before_reply: Optional[
            Union[list[Union[Callable, UPDATE_SYSTEM_MESSAGE]], Callable, UPDATE_SYSTEM_MESSAGE]
        ] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            name,
            system_message,
            is_termination_msg,
            max_consecutive_auto_reply,
            human_input_mode,
            llm_config=llm_config,
            description=description,
            code_execution_config=code_execution_config,
            **kwargs,
        )

        if isinstance(functions, list):
            if not all(isinstance(func, Callable) for func in functions):
                raise TypeError("All elements in the functions list must be callable")
            self.add_functions(functions)
        elif isinstance(functions, Callable):
            self.add_single_function(functions)
        elif functions is not None:
            raise TypeError("Functions must be a callable or a list of callables")

        self.after_work = None

        # Used in the tool execution agent to transfer to the next agent
        self._next_agent = None

        # Store nested chats hand offs as we'll establish these in the initiate_swarm_chat
        # List of Dictionaries containing the nested_chats and condition
        self._nested_chat_handoffs = []

        self.register_update_agent_state_before_reply(update_agent_state_before_reply)

        # Store conditional functions (and their ON_CONDITION instances) to add/remove later when transitioning to this agent
        self._conditional_functions = {}

        # Register the hook to update agent state (except tool executor)
        if name != __TOOL_EXECUTOR_NAME__:
            self.register_hook("update_agent_state", self._update_conditional_functions)

    def register_update_agent_state_before_reply(self, functions: Optional[Union[list[Callable], Callable]]):
        """Register functions that will be called when the agent is selected and before it speaks.
        You can add your own validation or precondition functions here.

        Args:
            functions (List[Callable[[], None]]): A list of functions to be registered. Each function
                is called when the agent is selected and before it speaks.
        """
        if functions is None:
            return
        if not isinstance(functions, list) and type(functions) not in [UPDATE_SYSTEM_MESSAGE, Callable]:
            raise ValueError("functions must be a list of callables")

        if not isinstance(functions, list):
            functions = [functions]

        for func in functions:
            if isinstance(func, UPDATE_SYSTEM_MESSAGE):
                # Wrapper function that allows this to be used in the update_agent_state hook
                # Its primary purpose, however, is just to update the agent's system message
                # Outer function to create a closure with the update function
                def create_wrapper(update_func: UPDATE_SYSTEM_MESSAGE):
                    def update_system_message_wrapper(
                        agent: ConversableAgent, messages: list[dict[str, Any]]
                    ) -> list[dict[str, Any]]:
                        if isinstance(update_func.update_function, str):
                            # Templates like "My context variable passport is {passport}" will
                            # use the context_variables for substitution
                            sys_message = OpenAIWrapper.instantiate(
                                template=update_func.update_function,
                                context=agent._context_variables,
                                allow_format_str_template=True,
                            )
                        else:
                            sys_message = update_func.update_function(agent, messages)

                        agent.update_system_message(sys_message)
                        return messages

                    return update_system_message_wrapper

                self.register_hook(hookable_method="update_agent_state", hook=create_wrapper(func))

            else:
                self.register_hook(hookable_method="update_agent_state", hook=func)

    def _set_to_tool_execution(self):
        """Set to a special instance of SwarmAgent that is responsible for executing tool calls from other swarm agents.
        This agent will be used internally and should not be visible to the user.

        It will execute the tool calls and update the referenced context_variables and next_agent accordingly.
        """
        self._next_agent = None
        self._reply_func_list.clear()
        self.register_reply([Agent, None], SwarmAgent.generate_swarm_tool_reply)

    def __str__(self):
        return f"SwarmAgent --> {self.name}"

    def register_hand_off(
        self,
        hand_to: Union[list[Union[ON_CONDITION, AFTER_WORK]], ON_CONDITION, AFTER_WORK],
    ):
        """Register a function to hand off to another agent.

        Args:
            hand_to: A list of ON_CONDITIONs and an, optional, AFTER_WORK condition

        Hand off template:
        def transfer_to_agent_name() -> SwarmAgent:
            return agent_name
        1. register the function with the agent
        2. register the schema with the agent, description set to the condition
        """
        # Ensure that hand_to is a list or ON_CONDITION or AFTER_WORK
        if not isinstance(hand_to, (list, ON_CONDITION, AFTER_WORK)):
            raise ValueError("hand_to must be a list of ON_CONDITION or AFTER_WORK")

        if isinstance(hand_to, (ON_CONDITION, AFTER_WORK)):
            hand_to = [hand_to]

        for transit in hand_to:
            if isinstance(transit, AFTER_WORK):
                assert isinstance(transit.agent, (AfterWorkOption, SwarmAgent, str, Callable)), (
                    "Invalid After Work value"
                )
                self.after_work = transit
            elif isinstance(transit, ON_CONDITION):
                if isinstance(transit.target, SwarmAgent):
                    # Transition to agent

                    # Create closure with current loop transit value
                    # to ensure the condition matches the one in the loop
                    def make_transfer_function(current_transit: ON_CONDITION):
                        def transfer_to_agent() -> "SwarmAgent":
                            return current_transit.target

                        return transfer_to_agent

                    transfer_func = make_transfer_function(transit)

                    # Store function to add/remove later based on it being 'available'
                    # Function names are made unique and allow multiple ON_CONDITIONS to the same agent
                    base_func_name = f"transfer_{self.name}_to_{transit.target.name}"
                    func_name = base_func_name
                    count = 2
                    while func_name in self._conditional_functions:
                        func_name = f"{base_func_name}_{count}"
                        count += 1

                    # Store function to add/remove later based on it being 'available'
                    self._conditional_functions[func_name] = (transfer_func, transit)

                elif isinstance(transit.target, dict):
                    # Transition to a nested chat
                    # We will store them here and establish them in the initiate_swarm_chat
                    self._nested_chat_handoffs.append(
                        {"nested_chats": transit.target, "condition": transit.condition, "available": transit.available}
                    )

            else:
                raise ValueError("Invalid hand off condition, must be either ON_CONDITION or AFTER_WORK")

    @staticmethod
    def _update_conditional_functions(agent: Agent, messages: Optional[list[dict]] = None) -> None:
        """Updates the agent's functions based on the ON_CONDITION's available condition."""
        for func_name, (func, on_condition) in agent._conditional_functions.items():
            is_available = True

            if on_condition.available is not None:
                if isinstance(on_condition.available, Callable):
                    is_available = on_condition.available(agent, next(iter(agent.chat_messages.values())))
                elif isinstance(on_condition.available, str):
                    is_available = agent.get_context(on_condition.available) or False

            if is_available:
                if func_name not in agent._function_map:
                    agent.add_single_function(func, func_name, on_condition.condition)
            else:
                # Remove function using the stored name
                if func_name in agent._function_map:
                    agent.update_tool_signature(func_name, is_remove=True)
                    del agent._function_map[func_name]

    def generate_swarm_tool_reply(
        self,
        messages: Optional[list[dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[OpenAIWrapper] = None,
    ) -> tuple[bool, dict]:
        """Pre-processes and generates tool call replies.

        This function:
        1. Adds context_variables back to the tool call for the function, if necessary.
        2. Generates the tool calls reply.
        3. Updates context_variables and next_agent based on the tool call response.
        """
        if config is None:
            config = self
        if messages is None:
            messages = self._oai_messages[sender]

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
                    if function_name in self._function_map:
                        func = self._function_map[function_name]  # Get the original function

                        # Inject the context variables into the tool call if it has the parameter
                        sig = signature(func)
                        if __CONTEXT_VARIABLES_PARAM_NAME__ in sig.parameters:
                            current_args = json.loads(tool_call["function"]["arguments"])
                            current_args[__CONTEXT_VARIABLES_PARAM_NAME__] = self._context_variables
                            tool_call["function"]["arguments"] = json.dumps(current_args)

                # Ensure we are only executing the one tool at a time
                message_copy["tool_calls"] = [tool_call]

                # 2. generate tool calls reply
                _, tool_message = self.generate_tool_calls_reply([message_copy])

                # 3. update context_variables and next_agent, convert content to string
                for tool_response in tool_message["tool_responses"]:
                    content = tool_response.get("content")
                    if isinstance(content, SwarmResult):
                        if content.context_variables != {}:
                            self._context_variables.update(content.context_variables)
                        if content.agent is not None:
                            next_agent = content.agent
                    elif isinstance(content, Agent):
                        next_agent = content

                    tool_responses_inner.append(tool_response)
                    contents.append(str(tool_response["content"]))

            self._next_agent = next_agent

            # Put the tool responses and content strings back into the response message
            # Caters for multiple tool calls
            tool_message["tool_responses"] = tool_responses_inner
            tool_message["content"] = "\n".join(contents)

            return True, tool_message
        return False, None

    def add_single_function(self, func: Callable, name=None, description=""):
        """Add a single function to the agent, removing context variables for LLM use"""
        if name:
            func._name = name
        else:
            func._name = func.__name__

        if description:
            func._description = description
        else:
            # Use function's docstring, strip whitespace, fall back to empty string
            func._description = (func.__doc__ or "").strip()

        f = get_function_schema(func, name=func._name, description=func._description)

        # Remove context_variables parameter from function schema
        f_no_context = f.copy()
        if __CONTEXT_VARIABLES_PARAM_NAME__ in f_no_context["function"]["parameters"]["properties"]:
            del f_no_context["function"]["parameters"]["properties"][__CONTEXT_VARIABLES_PARAM_NAME__]
        if "required" in f_no_context["function"]["parameters"]:
            required = f_no_context["function"]["parameters"]["required"]
            f_no_context["function"]["parameters"]["required"] = [
                param for param in required if param != __CONTEXT_VARIABLES_PARAM_NAME__
            ]
            # If required list is empty, remove it
            if not f_no_context["function"]["parameters"]["required"]:
                del f_no_context["function"]["parameters"]["required"]

        self.update_tool_signature(f_no_context, is_remove=False)
        self.register_function({func._name: func})

    def add_functions(self, func_list: list[Callable]):
        for func in func_list:
            self.add_single_function(func)

    @staticmethod
    def process_nested_chat_carryover(
        chat: dict[str, Any],
        recipient: ConversableAgent,
        messages: list[dict[str, Any]],
        sender: ConversableAgent,
        config: Any,
        trim_n_messages: int = 0,
    ) -> None:
        """Process carryover messages for a nested chat (typically for the first chat of a swarm)

        The carryover_config key is a dictionary containing:
            "summary_method": The method to use to summarise the messages, can be "all", "last_msg", "reflection_with_llm" or a Callable
            "summary_args": Optional arguments for the summary method

        Supported carryover 'summary_methods' are:
            "all" - all messages will be incorporated
            "last_msg" - the last message will be incorporated
            "reflection_with_llm" - an llm will summarise all the messages and the summary will be incorporated as a single message
            Callable - a callable with the signature: my_method(agent: ConversableAgent, messages: List[Dict[str, Any]]) -> str

        Args:
            chat: The chat dictionary containing the carryover configuration
            recipient: The recipient agent
            messages: The messages from the parent chat
            sender: The sender agent
            trim_n_messages: The number of latest messages to trim from the messages list
        """

        def concat_carryover(chat_message: str, carryover_message: Union[str, list[dict[str, Any]]]) -> str:
            """Concatenate the carryover message to the chat message."""
            prefix = f"{chat_message}\n" if chat_message else ""

            if isinstance(carryover_message, str):
                content = carryover_message
            elif isinstance(carryover_message, list):
                content = "\n".join(
                    msg["content"] for msg in carryover_message if "content" in msg and msg["content"] is not None
                )
            else:
                raise ValueError("Carryover message must be a string or a list of dictionaries")

            return f"{prefix}Context:\n{content}"

        carryover_config = chat["carryover_config"]

        if "summary_method" not in carryover_config:
            raise ValueError("Carryover configuration must contain a 'summary_method' key")

        carryover_summary_method = carryover_config["summary_method"]
        carryover_summary_args = carryover_config.get("summary_args") or {}

        chat_message = ""
        message = chat.get("message")

        # If the message is a callable, run it and get the result
        if message:
            chat_message = message(recipient, messages, sender, config) if callable(message) else message

        # deep copy and trim the latest messages
        content_messages = copy.deepcopy(messages)
        content_messages = content_messages[:-trim_n_messages]

        if carryover_summary_method == "all":
            # Put a string concatenated value of all parent messages into the first message
            # (e.g. message = <first nested chat message>\nContext: \n<swarm message 1>\n<swarm message 2>\n...)
            carry_over_message = concat_carryover(chat_message, content_messages)

        elif carryover_summary_method == "last_msg":
            # (e.g. message = <first nested chat message>\nContext: \n<last swarm message>)
            carry_over_message = concat_carryover(chat_message, content_messages[-1]["content"])

        elif carryover_summary_method == "reflection_with_llm":
            # (e.g. message = <first nested chat message>\nContext: \n<llm summary>)

            # Add the messages to the nested chat agent for reflection (we'll clear after reflection)
            chat["recipient"]._oai_messages[sender] = content_messages

            carry_over_message_llm = ConversableAgent._reflection_with_llm_as_summary(
                sender=sender,
                recipient=chat["recipient"],  # Chat recipient LLM config will be used for the reflection
                summary_args=carryover_summary_args,
            )

            recipient._oai_messages[sender] = []

            carry_over_message = concat_carryover(chat_message, carry_over_message_llm)

        elif isinstance(carryover_summary_method, Callable):
            # (e.g. message = <first nested chat message>\nContext: \n<function's return string>)
            carry_over_message_result = carryover_summary_method(recipient, content_messages, carryover_summary_args)

            carry_over_message = concat_carryover(chat_message, carry_over_message_result)

        chat["message"] = carry_over_message

    @staticmethod
    def _summary_from_nested_chats(
        chat_queue: list[dict[str, Any]], recipient: Agent, messages: Union[str, Callable], sender: Agent, config: Any
    ) -> tuple[bool, Union[str, None]]:
        """Overridden _summary_from_nested_chats method from ConversableAgent.
        This function initiates one or a sequence of chats between the "recipient" and the agents in the chat_queue.

        It extracts and returns a summary from the nested chat based on the "summary_method" in each chat in chat_queue.

        Swarm Updates:
        - the 'messages' parameter contains the parent chat's messages
        - the first chat in the queue can contain a 'carryover_config' which is a dictionary that denotes how to carryover messages from the swarm chat into the first chat of the nested chats). Only applies to the first chat.
            e.g.: carryover_summarize_chat_config = {"summary_method": "reflection_with_llm", "summary_args": None}
            summary_method can be "last_msg", "all", "reflection_with_llm", Callable
            The Callable signature: my_method(agent: ConversableAgent, messages: List[Dict[str, Any]]) -> str
            The summary will be concatenated to the message of the first chat in the queue.

        Returns:
            Tuple[bool, str]: A tuple where the first element indicates the completion of the chat, and the second element contains the summary of the last chat if any chats were initiated.
        """
        # Carryover configuration allowed on the first chat in the queue only, trim the last two messages specifically for swarm nested chat carryover as these are the messages for the transition to the nested chat agent
        restore_chat_queue_message = False
        if len(chat_queue) > 0 and "carryover_config" in chat_queue[0]:
            if "message" in chat_queue[0]:
                # As we're updating the message in the nested chat queue, we need to restore it after finishing this nested chat.
                restore_chat_queue_message = True
                original_chat_queue_message = chat_queue[0]["message"]
            SwarmAgent.process_nested_chat_carryover(chat_queue[0], recipient, messages, sender, config, 2)

        chat_to_run = ConversableAgent._get_chats_to_run(chat_queue, recipient, messages, sender, config)
        if not chat_to_run:
            return True, None
        res = sender.initiate_chats(chat_to_run)

        # We need to restore the chat queue message if it has been modified so that it will be the original message for subsequent uses
        if restore_chat_queue_message:
            chat_queue[0]["message"] = original_chat_queue_message

        return True, res[-1].summary


@export_module("autogen")
class SwarmResult(BaseModel):
    """Encapsulates the possible return values for a swarm agent function.

    Args:
        values (str): The result values as a string.
        agent (SwarmAgent): The swarm agent instance, if applicable.
        context_variables (dict): A dictionary of context variables.
    """

    values: str = ""
    agent: Optional[Union["SwarmAgent", str, AfterWorkOption]] = None
    context_variables: dict[str, Any] = {}

    class Config:  # Add this inner class
        arbitrary_types_allowed = True

    def __str__(self):
        return self.values


# Forward references for SwarmAgent in SwarmResult
SwarmResult.update_forward_refs()
