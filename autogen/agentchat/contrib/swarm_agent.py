# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
import json
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
class NESTED_CHAT_CONFIG:
    chat_list: List[Dict[str, Any]]
    starting_message_method: Optional[Union[str, Callable]] = None
    starting_llm_summary_args: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        assert isinstance(self.chat_list, list) and self.chat_list, "'chat_list' must be a non-empty list"
        assert all(isinstance(chat, dict) for chat in self.chat_list), "'chat_list' must be a list of dictionaries"
        assert isinstance(
            self.starting_message_method, (str, Callable)
        ), "'starting_message_method' must be a string or callable"

        if self.starting_llm_summary_args is not None:
            assert (
                self.starting_message_method == "llm_summary"
            ), "If 'starting_llm_summary_args' is provided, 'starting_message_method' must be 'carryover_llm_summary'"

        if isinstance(self.starting_message_method, str):
            assert self.starting_message_method in [
                "carryover",
                "carryover_last_msg",
                "carryover_llm_summary",
            ], "'starting_message_method' must be 'carryover', 'carryover_last_msg', 'carryover_llm_summary' or a callable"
            assert "message" in self.chat_list[0], "All carryovers need the first chat_list item to have a 'message'"

        if isinstance(self.starting_message_method, Callable):
            if "message" in self.chat_list[0]:
                raise ValueError(
                    "If 'starting_message_method' is a callable, the first chat_list item can not have a 'message'. The callable will return the message."
                )


@dataclass
class ON_CONDITION:
    agent: Optional["SwarmAgent"] = None
    nested_chat: Optional[NESTED_CHAT_CONFIG] = None
    condition: str = ""

    def __post_init__(self):
        # Ensure valid types
        if self.agent is not None:
            assert isinstance(self.agent, SwarmAgent), "'agent' must be a SwarmAgent"

        if self.nested_chat is not None:
            assert isinstance(self.nested_chat, NESTED_CHAT_CONFIG), "'nested_chat' must be a NESTED_CHAT_CONFIG"

        # Ensure they have an agent or nested_chat
        assert self.agent is not None or self.nested_chat is not None, "'agent' or 'nested_chat' must be provided"

        # Ensure they don't have both an agent and a nested_chat
        assert not (
            self.agent is not None and self.nested_chat is not None
        ), "'agent' and 'nested_chat' cannot both be provided"

        # Ensure they have a condition
        assert isinstance(self.condition, str) and self.condition.strip(), "'condition' must be a non-empty string"


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
    tool_execution._set_to_tool_execution(context_variables=context_variables)

    # Update tool execution agent with all the functions from all the agents
    for agent in agents:
        tool_execution._function_map.update(agent._function_map)

    INIT_AGENT_USED = False

    def swarm_transition(last_speaker: SwarmAgent, groupchat: GroupChat):
        """Swarm transition function to determine the next agent in the conversation"""
        nonlocal INIT_AGENT_USED
        if not INIT_AGENT_USED:
            INIT_AGENT_USED = True
            return initial_agent

        if "tool_calls" in groupchat.messages[-1]:
            return tool_execution
        if tool_execution._next_agent is not None:
            next_agent = tool_execution._next_agent
            tool_execution._next_agent = None
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

    # We associate the groupchat manager with SwarmAgents
    # to be able to access group messages, tool executor context variables
    for agent in agents:
        if isinstance(agent, SwarmAgent):
            agent.associate_groupchat(manager)

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
    agent: Optional["SwarmAgent"] = None
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

        # Used only in the tool execution agent for context and transferring to the next agent
        # Note: context variables are not stored for each agent
        self._context_variables = {}
        self._next_agent = None

    def _set_to_tool_execution(self, context_variables: Optional[Dict[str, Any]] = None):
        """Set to a special instance of SwarmAgent that is responsible for executing tool calls from other swarm agents.
        This agent will be used internally and should not be visible to the user.

        It will execute the tool calls and update the context_variables and next_agent accordingly.
        """
        self._next_agent = None
        self._context_variables = context_variables or {}
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

                if transit.agent:
                    # Transition to agent

                    # Create closure with current loop transit value
                    # to ensure the condition matches the one in the loop
                    def make_transfer_function(current_transit: ON_CONDITION):
                        def transfer_to_agent() -> "SwarmAgent":
                            return current_transit.agent

                        return transfer_to_agent

                    transfer_func = make_transfer_function(transit)
                    self.add_single_function(transfer_func, f"transfer_to_{transit.agent.name}", transit.condition)

                else:
                    # Transition to a nested chat

                    # Create closure (see above note)
                    def make_transfer_nested_function(nested_chat_config: NESTED_CHAT_CONFIG):
                        def transfer_to_nested_chat() -> str:

                            # All messages excluding the tool call message to get here
                            current_messages = self._groupchatmanager.groupchat.messages[:-1]
                            starting_message = [{"content": "", "role": "user"}]

                            if "message" in nested_chat_config.chat_list[0]:
                                starting_message[0]["content"] = nested_chat_config.chat_list[0]["message"]

                            carry_over_message = ""

                            if nested_chat_config.starting_message_method == "carryover":
                                # Carryovers put a string concatenated value of messages into the first message
                                # All carryovers need the "message" parameter as well
                                # (e.g. message = <first nested chat message>\nContext: \n<swarm message 1>\n<swarm message 2>\n...)
                                carry_over_message = current_messages

                            elif nested_chat_config.starting_message_method == "carryover_last_msg":
                                # (e.g. message = <first nested chat message>\nContext: \n<last swarm message>)
                                carry_over_message = current_messages[-1]["content"]

                            elif nested_chat_config.starting_message_method == "carryover_llm_summary":
                                # We need to remove the last tool message from the messages before running inference, as the last message can't be a tool call
                                last_tool_message = self._oai_messages[self._groupchatmanager].pop()

                                carry_over_message = ConversableAgent._reflection_with_llm_as_summary(
                                    sender=self._groupchatmanager,
                                    recipient=self,
                                    summary_args=(
                                        nested_chat_config.starting_llm_summary_args
                                        if nested_chat_config.starting_llm_summary_args
                                        else {}
                                    ),
                                )

                                self._oai_messages[self._groupchatmanager].append(
                                    last_tool_message
                                )  # Restore the tool message

                            elif isinstance(nested_chat_config.starting_message_method, Callable):
                                nested_chat_config.chat_list[0]["message"] = nested_chat_config.starting_message_method(
                                    context_variables=self.get_swarm_context_variables(),
                                    messages=self._groupchatmanager.groupchat.messages,
                                )

                            if carry_over_message:
                                nested_chat_config.chat_list[0]["carryover"] = carry_over_message

                            print("In transfer_to_nested_chat")
                            self.register_nested_chats(
                                nested_chat_config.chat_list, trigger=lambda sender: True, position=0
                            )

                            # Note: If we pass a list of messages in, the nested chat always
                            # extracts and uses just the last message. This is the reason we use carryovers.
                            reply = self.generate_reply(sender=self, messages=starting_message)

                            # Remove the registered nested chat we added
                            self._reply_func_list.pop(0)

                            return reply

                        return transfer_to_nested_chat

                    transfer_func = make_transfer_nested_function(transit.nested_chat)
                    self.add_single_function(
                        transfer_func, f"transfer_to_nested_chat_{len(self._function_map)}", transit.condition
                    )

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

            tool_calls = len(message["tool_calls"])

            # Loop through tool calls individually (so context can be updated after each function call)
            next_agent = None
            tool_responses_inner = []
            contents = []
            for index in range(tool_calls):

                # 1. add context_variables to the tool call arguments
                tool_call = message["tool_calls"][index]

                if tool_call["type"] == "function":
                    function_name = tool_call["function"]["name"]

                    # Check if this function exists in our function map
                    if function_name in self._function_map:
                        func = self._function_map[function_name]  # Get the original function

                        # Check if function has context_variables parameter
                        sig = signature(func)
                        if __CONTEXT_VARIABLES_PARAM_NAME__ in sig.parameters:
                            current_args = json.loads(tool_call["function"]["arguments"])
                            current_args[__CONTEXT_VARIABLES_PARAM_NAME__] = self._context_variables
                            # Update the tool call with new arguments
                            tool_call["function"]["arguments"] = json.dumps(current_args)

                # Copy the message
                message_copy = message.copy()
                tool_calls_copy = message_copy["tool_calls"]

                # remove all the tool calls except the one at the index
                message_copy["tool_calls"] = [tool_calls_copy[index]]

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

    def associate_groupchat(self, groupchatmanager: GroupChatManager):
        """Associate the group chat with an agent so we can access overall messages and other agents"""
        self._groupchatmanager = groupchatmanager

    def get_swarm_context_variables(self) -> Dict[str, Any]:
        """Returns the context variables from the tool execution agent"""
        for agent in self._groupchatmanager.groupchat.agents:
            if agent.name == "Tool_Execution":
                return agent._context_variables

        raise Exception("Tool Execution agent not found")


# Forward references for SwarmAgent in SwarmResult
SwarmResult.update_forward_refs()
