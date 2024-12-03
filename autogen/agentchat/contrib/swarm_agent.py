# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
import copy
import json
import re
import warnings
from dataclasses import dataclass
from enum import Enum
from inspect import signature
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel

from autogen.function_utils import get_function_schema
from autogen.oai import OpenAIWrapper

from ..agent import Agent
from ..chat import ChatResult
from ..conversable_agent import ConversableAgent
from ..groupchat import GroupChat, GroupChatManager
from ..user_proxy_agent import UserProxyAgent

# Parameter name for context variables
# Use the value in functions and they will be substituted with the context variables:
# e.g. def my_function(context_variables: Dict[str, Any], my_other_parameters: Any) -> Any:
__CONTEXT_VARIABLES_PARAM_NAME__ = "context_variables"


class AfterWorkOption(Enum):
    TERMINATE = "TERMINATE"
    REVERT_TO_USER = "REVERT_TO_USER"
    STAY = "STAY"


@dataclass
class AFTER_WORK:
    agent: Union[AfterWorkOption, "SwarmAgent", str, Callable]

    def __post_init__(self):
        if isinstance(self.agent, str):
            self.agent = AfterWorkOption(self.agent.upper())


@dataclass
class ON_CONDITION:
    agent: "SwarmAgent"
    condition: str = ""

    # Ensure that agent is a SwarmAgent
    def __post_init__(self):
        assert isinstance(self.agent, SwarmAgent), "Agent must be a SwarmAgent"


@dataclass
class UPDATE_SYSTEM_MESSAGE:
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
                raise ValueError("Update function must accept two parameters, context_variables and messages")
            if sig.return_annotation != str:
                raise ValueError("Update function must return a string")
        else:
            raise ValueError("Update function must be either a string or a callable")


def initiate_swarm_chat(
    initial_agent: "SwarmAgent",
    messages: Union[List[Dict[str, Any]], str],
    agents: List["SwarmAgent"],
    user_agent: Optional[UserProxyAgent] = None,
    max_rounds: int = 20,
    context_variables: Optional[Dict[str, Any]] = None,
    after_work: Optional[Union[AFTER_WORK, Callable]] = AFTER_WORK(AfterWorkOption.TERMINATE),
) -> Tuple[ChatResult, Dict[str, Any], "SwarmAgent"]:
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

            Callable: A custom function that takes the current agent, messages, groupchat, and context_variables as arguments and returns the next agent. The function should return None to terminate.
                ```python
                def custom_afterwork_func(last_speaker: SwarmAgent, messages: List[Dict[str, Any]], groupchat: GroupChat, context_variables: Optional[Dict[str, Any]]) -> Optional[SwarmAgent]:
                ```
    Returns:
        ChatResult:     Conversations chat history.
        Dict[str, Any]: Updated Context variables.
        SwarmAgent:     Last speaker.
    """
    assert isinstance(initial_agent, SwarmAgent), "initial_agent must be a SwarmAgent"
    assert all(isinstance(agent, SwarmAgent) for agent in agents), "Agents must be a list of SwarmAgents"
    # Ensure all agents in hand-off after-works are in the passed in agents list
    for agent in agents:
        if agent.after_work is not None:
            if isinstance(agent.after_work.agent, SwarmAgent):
                assert agent.after_work.agent in agents, "Agent in hand-off must be in the agents list"

    context_variables = context_variables or {}
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    swarm_agent_names = [agent.name for agent in agents]

    tool_execution = SwarmAgent(
        name="Tool_Execution",
        system_message="Tool Execution",
    )
    tool_execution._set_to_tool_execution()

    # Update tool execution agent with all the functions from all the agents
    for agent in agents:
        tool_execution._function_map.update(agent._function_map)

    # Point all SwarmAgent's context variables to this function's context_variables
    # providing a single (shared) context across all SwarmAgents in the swarm
    for agent in agents + [tool_execution]:
        agent._context_variables = context_variables

    INIT_AGENT_USED = False

    def swarm_transition(last_speaker: SwarmAgent, groupchat: GroupChat):
        """Swarm transition function to determine and prepare the next agent in the conversation"""
        next_agent = determine_next_agent(last_speaker, groupchat)

        return next_agent

    def determine_next_agent(last_speaker: SwarmAgent, groupchat: GroupChat):
        """Determine the next agent in the conversation"""
        nonlocal INIT_AGENT_USED
        if not INIT_AGENT_USED:
            INIT_AGENT_USED = True
            return initial_agent

        if "tool_calls" in groupchat.messages[-1]:
            return tool_execution
        if tool_execution._next_agent is not None:
            next_agent = tool_execution._next_agent
            tool_execution._next_agent = None

            # Check for string, access agent from group chat.

            if isinstance(next_agent, str):
                if next_agent in swarm_agent_names:
                    next_agent = groupchat.agent_by_name(name=next_agent)
                else:
                    raise ValueError(
                        f"No agent found with the name '{next_agent}'. Ensure the agent exists in the swarm."
                    )

            return next_agent

        # get the last swarm agent
        last_swarm_speaker = None
        for message in reversed(groupchat.messages):
            if "name" in message and message["name"] in swarm_agent_names:
                agent = groupchat.agent_by_name(name=message["name"])
                if isinstance(agent, SwarmAgent):
                    last_swarm_speaker = agent
                    break
        if last_swarm_speaker is None:
            raise ValueError("No swarm agent found in the message history")

        # If the user last spoke, return to the agent prior
        if (user_agent and last_speaker == user_agent) or groupchat.messages[-1]["role"] == "tool":
            return last_swarm_speaker

        # No agent selected via hand-offs (tool calls)
        # Assume the work is Done
        # override if agent-level after_work is defined, else use the global after_work
        tmp_after_work = last_swarm_speaker.after_work if last_swarm_speaker.after_work is not None else after_work
        if isinstance(tmp_after_work, AFTER_WORK):
            tmp_after_work = tmp_after_work.agent

        if isinstance(tmp_after_work, SwarmAgent):
            return tmp_after_work
        elif isinstance(tmp_after_work, AfterWorkOption):
            if tmp_after_work == AfterWorkOption.TERMINATE or (
                user_agent is None and tmp_after_work == AfterWorkOption.REVERT_TO_USER
            ):
                return None
            elif tmp_after_work == AfterWorkOption.REVERT_TO_USER:
                return user_agent
            elif tmp_after_work == AfterWorkOption.STAY:
                return last_speaker
        elif isinstance(tmp_after_work, Callable):
            return tmp_after_work(last_speaker, groupchat.messages, groupchat, context_variables)
        else:
            raise ValueError("Invalid After Work condition")

    # If there's only one message and there's no identified swarm agent
    # Start with a user proxy agent, creating one if they haven't passed one in
    if len(messages) == 1 and "name" not in messages[0] and not user_agent:
        temp_user_proxy = [UserProxyAgent(name="_User")]
    else:
        temp_user_proxy = []

    groupchat = GroupChat(
        agents=[tool_execution] + agents + ([user_agent] if user_agent is not None else temp_user_proxy),
        messages=[],  # Set to empty. We will resume the conversation with the messages
        max_round=max_rounds,
        speaker_selection_method=swarm_transition,
    )
    manager = GroupChatManager(groupchat)
    clear_history = True

    if len(messages) > 1:
        last_agent, last_message = manager.resume(messages=messages)
        clear_history = False
    else:
        last_message = messages[0]

        if "name" in last_message:
            if last_message["name"] in swarm_agent_names:
                # If there's a name in the message and it's a swarm agent, use that
                last_agent = groupchat.agent_by_name(name=last_message["name"])
            elif user_agent and last_message["name"] == user_agent.name:
                # If the user agent is passed in and is the first message
                last_agent = user_agent
            else:
                raise ValueError(f"Invalid swarm agent name in last message: {last_message['name']}")
        else:
            # No name, so we're using the user proxy to start the conversation
            if user_agent:
                last_agent = user_agent
            else:
                # If no user agent passed in, use our temporary user proxy
                last_agent = temp_user_proxy[0]

    chat_result = last_agent.initiate_chat(
        manager,
        message=last_message,
        clear_history=clear_history,
    )

    # Clear the temporary user proxy's name from messages
    if len(temp_user_proxy) == 1:
        for message in chat_result.chat_history:
            if "name" in message and message["name"] == "_User":
                # delete the name key from the message
                del message["name"]

    return chat_result, context_variables, manager.last_speaker


class SwarmResult(BaseModel):
    """
    Encapsulates the possible return values for a swarm agent function.

    Args:
        values (str): The result values as a string.
        agent (SwarmAgent): The swarm agent instance, if applicable.
        context_variables (dict): A dictionary of context variables.
    """

    values: str = ""
    agent: Optional[Union["SwarmAgent", str]] = None
    context_variables: Dict[str, Any] = {}

    class Config:  # Add this inner class
        arbitrary_types_allowed = True

    def __str__(self):
        return self.values


class SwarmAgent(ConversableAgent):
    """Swarm agent for participating in a swarm.

    SwarmAgent is a subclass of ConversableAgent.

    Additional args:
        functions (List[Callable]): A list of functions to register with the agent.
    """

    def __init__(
        self,
        name: str,
        system_message: Optional[str] = "You are a helpful AI Assistant.",
        llm_config: Optional[Union[Dict, Literal[False]]] = None,
        functions: Union[List[Callable], Callable] = None,
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Literal["ALWAYS", "NEVER", "TERMINATE"] = "NEVER",
        description: Optional[str] = None,
        code_execution_config=False,
        update_agent_before_reply: Optional[
            Union[List[Union[Callable, UPDATE_SYSTEM_MESSAGE]], Callable, UPDATE_SYSTEM_MESSAGE]
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

        # use in the tool execution agent to transfer to the next agent
        self._next_agent = None

        self.register_update_agent_before_reply(update_agent_before_reply)

    def register_update_agent_before_reply(self, functions: Optional[Union[List[Callable], Callable]]):
        """
        Register functions that will be called when the agent is selected and before it speaks.
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
                        agent: ConversableAgent, messages: List[Dict[str, Any]]
                    ) -> List[Dict[str, Any]]:
                        if isinstance(update_func.update_function, str):
                            # Templates like "My context variable passport is {passport}" will
                            # use the context_variables for substitution
                            sys_message = OpenAIWrapper.instantiate(
                                template=update_func.update_function,
                                context=agent._context_variables,
                                allow_format_str_template=True,
                            )
                        else:
                            sys_message = update_func.update_function(agent._context_variables, messages)

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
        hand_to: Union[List[Union[ON_CONDITION, AFTER_WORK]], ON_CONDITION, AFTER_WORK],
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
                assert isinstance(
                    transit.agent, (AfterWorkOption, SwarmAgent, str, Callable)
                ), "Invalid After Work value"
                self.after_work = transit
            elif isinstance(transit, ON_CONDITION):

                # Create closure with current loop transit value
                # to ensure the condition matches the one in the loop
                def make_transfer_function(current_transit):
                    def transfer_to_agent() -> "SwarmAgent":
                        return current_transit.agent

                    return transfer_to_agent

                transfer_func = make_transfer_function(transit)
                self.add_single_function(transfer_func, f"transfer_to_{transit.agent.name}", transit.condition)
            else:
                raise ValueError("Invalid hand off condition, must be either ON_CONDITION or AFTER_WORK")

    def generate_swarm_tool_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[OpenAIWrapper] = None,
    ) -> Tuple[bool, dict]:
        """Pre-processes and generates tool call replies.

        This function:
        1. Adds context_variables back to the tool call for the function, if necessary.
        2. Generates the tool calls reply.
        3. Updates context_variables and next_agent based on the tool call response."""

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

    def add_functions(self, func_list: List[Callable]):
        for func in func_list:
            self.add_single_function(func)


# Forward references for SwarmAgent in SwarmResult
SwarmResult.update_forward_refs()
