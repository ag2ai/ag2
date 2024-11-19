import json
from inspect import signature
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel

from autogen.agentchat import Agent, ChatResult, ConversableAgent, GroupChat, GroupChatManager, UserProxyAgent
from autogen.function_utils import get_function_schema
from autogen.oai import OpenAIWrapper

# Parameter name for context variables
# Use the value in functions and they will be substituted with the context variables:
# e.g. def my_function(context_variables: Dict[str, Any], my_other_parameters: Any) -> Any:
__CONTEXT_VARIABLES_PARAM_NAME__ = "context_variables"


def initialize_swarm_chat(
    init_agent: "SwarmAgent",
    messages: Union[List[Dict[str, Any]], str],
    agents: List["SwarmAgent"],
    user_agent: Optional[UserProxyAgent] = None,
    max_rounds: int = 20,
    context_variables: Optional[Dict[str, Any]] = None,
    fallback_method: Union[Literal["TERMINATE", "REVERT_TO_USER", "STAY"], Callable] = "REVERT_TO_USER",
) -> Tuple[ChatResult, Dict[str, Any], "SwarmAgent"]:
    """Initialize and run a swarm chat

    Args:
        init_agent: The initial agent of the conversation.
        messages: Initial message(s).
        agents: List of swarm agents.
        user_agent: Optional user proxy agent for falling back to.
        max_rounds: Maximum number of conversation rounds.
        context_variables: Starting context variables.
        fallback_method: Method to handle conversation continuation when an agent doesn't select the next agent. This fallback_method is considered after the speaking agent's fallback_method. Default is "REVERT_TO_USER".
            Could be any of the following (case insensitive):
            - "TERMINATE": End the conversation if no next agent is selected
            - "REVERT_TO_USER": Return to the passed in user_agent if no next agent is selected. Is equivalent to "TERMINATE" if no user_agent is passed in.
            - "STAY": Stay with the current agent if no next agent is selected
            - Callable: A custom function that takes the current agent, messages, groupchat, and context_variables as arguments and returns the next agent. The function should return None to terminate.
                ```python
                def custom_fallback_func(last_speaker: SwarmAgent, messages: List[Dict[str, Any]], groupchat: GroupChat, context_variables: Optional[Dict[str, Any]]) -> Optional[SwarmAgent]:
                ```
    Returns:
        ChatResult:     Conversations chat history.
        Dict[str, Any]: Updated Context variables.
        SwarmAgent:     Last speaker.
    """
    context_variables = context_variables or {}
    if isinstance(fallback_method, str):
        fallback_method = fallback_method.upper()

    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    swarm_agent_names = [agent.name for agent in agents]

    tool_execution = SwarmAgent(
        name="Tool_Execution",
        system_message="Tool Execution",
        human_input_mode="NEVER",
        code_execution_config=False,
        is_tool_execution=True,
        context_variables=context_variables,
    )

    # Update tool execution agent with all the functions from all the agents
    for agent in agents:
        tool_execution._function_map.update(agent._function_map)

    INIT_AGENT_USED = False

    def swarm_transition(last_speaker: SwarmAgent, groupchat: GroupChat):
        """Swarm transition function to determine the next agent in the conversation"""

        nonlocal INIT_AGENT_USED
        if not INIT_AGENT_USED:
            INIT_AGENT_USED = True
            return init_agent

        if "tool_calls" in messages[-1]:
            return tool_execution
        if tool_execution.next_agent is not None:
            next_agent = tool_execution.next_agent
            tool_execution.next_agent = None
            return next_agent

        last_swarm_speaker = get_last_swarm_speaker()

        # If the user last spoke, return to the agent prior
        if user_agent and last_speaker == user_agent:
            return last_swarm_speaker

        # No agent selected via hand-offs (tool calls)
        # Check the agent's fallback method
        if last_swarm_speaker.fallback_method:
            next_agent = get_fallback_agent(last_swarm_speaker.fallback_method, last_swarm_speaker)

            if next_agent is not None:
                return next_agent

        # Check the swarm's fallback method
        # Returns None, ending swarm, if still no agent selected
        return get_fallback_agent(fallback_method, last_swarm_speaker)

    def get_last_swarm_speaker() -> "SwarmAgent":
        """Get the last swarm agent that spoke in the message history"""
        for message in reversed(messages):
            if "name" in message and message["name"] in swarm_agent_names:
                agent = groupchat.agent_by_name(name=message["name"])
                if isinstance(agent, SwarmAgent):
                    return agent

        raise ValueError("No swarm agent found in the message history")

    def get_fallback_agent(method: Union[str, Callable], agent: "SwarmAgent") -> Optional["SwarmAgent"]:
        """Get the next agent based on the fallback method"""

        if method == "TERMINATE" or method == "REVERT_TO_USER" and not user_agent:
            return None
        elif method == "REVERT_TO_USER":
            return user_agent
        elif method == "STAY":
            return agent
        elif callable(method):
            return method(agent, messages, groupchat, context_variables)

    groupchat = GroupChat(
        agents=[tool_execution] + agents + ([user_agent] if user_agent is not None else []),
        messages=messages,
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
        last_agent = init_agent

    chat_history = last_agent.initiate_chat(
        manager,
        message=last_message,
        clear_history=clear_history,
    )
    return chat_history, context_variables, manager.last_speaker


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
        context_variables (dict): A dictionary of context variables.
        is_tool_execution (bool): A flag to indicate if the agent is a tool execution agent.
        fallback_method (str or Callable): Method to handle conversation continuation when an agent doesn't select the next agent. Default is None.

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
        context_variables: Optional[Dict[str, Any]] = None,
        is_tool_execution: Optional[bool] = False,
        fallback_method: Optional[Union[Literal["TERMINATE", "REVERT_TO_USER", "STAY"], Callable]] = None,
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
            **kwargs,
        )

        if isinstance(functions, list):
            self.add_functions(functions)
        elif isinstance(functions, Callable):
            self.add_single_function(functions)

        if is_tool_execution:
            self._reply_func_list.clear()
            self.register_reply([Agent, None], SwarmAgent.generate_swarm_tool_reply)

        self.fallback_method = fallback_method
        self.context_variables = context_variables or {}
        self.next_agent = None  # use in the tool execution agent to transfer to the next agent

    def update_context_variables(self, context_variables: Dict[str, Any]) -> None:
        self.context_variables.update(context_variables)

    def __str__(self):
        return f"SwarmAgent: {self.name}"

    def hand_off(
        self,
        agent: "SwarmAgent",
        condition: str = "",
    ):
        """Register a function to hand off to another agent.

        Hand off template:
        def transfer_to_agent_name() -> SwarmAgent:
            return agent_name
        1. register the function with the agent
        2. register the schema with the agent, description set to the condition
        """

        def transfer_to_agent() -> "SwarmAgent":
            return agent

        self.add_single_function(transfer_to_agent, f"transfer_to_{agent.name}", condition)

    def generate_swarm_tool_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[OpenAIWrapper] = None,
    ) -> Tuple[bool, dict]:

        if config is None:
            config = self
        if messages is None:
            messages = self._oai_messages[sender]

        message = messages[-1]
        if "tool_calls" in message:
            # 1. add context_variables to the tool call arguments
            for tool_call in message["tool_calls"]:
                if tool_call["type"] == "function":
                    function_name = tool_call["function"]["name"]

                    # Check if this function exists in our function map
                    if function_name in self._function_map:
                        func = self._function_map[function_name]  # Get the original function

                        # Check if function has context_variables parameter
                        sig = signature(func)
                        if __CONTEXT_VARIABLES_PARAM_NAME__ in sig.parameters:
                            current_args = json.loads(tool_call["function"]["arguments"])
                            current_args[__CONTEXT_VARIABLES_PARAM_NAME__] = self.context_variables
                            # Update the tool call with new arguments
                            tool_call["function"]["arguments"] = json.dumps(current_args)

            # 2. generate tool calls reply
            _, tool_message = self.generate_tool_calls_reply([message])

            # 3. update context_variables and next_agent, convert content to string
            for tool_response in tool_message["tool_responses"]:
                content = tool_response.get("content")
                if isinstance(content, SwarmResult):
                    if content.context_variables != {}:
                        self.context_variables.update(content.context_variables)
                    if content.agent is not None:
                        self.next_agent = content.agent
                elif isinstance(content, Agent):
                    self.next_agent = content
                tool_response["content"] = str(tool_response["content"])

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
