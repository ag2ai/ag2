import json
from inspect import signature
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel

from autogen.agentchat import Agent, ChatResult, ConversableAgent, GroupChat, GroupChatManager, UserProxyAgent
from autogen.function_utils import get_function_schema
from autogen.oai import OpenAIWrapper


def parse_json_object(response: str) -> dict:
    return json.loads(response)


# Parameter name for context variables
# Use the value in functions and they will be substituted with the context variables:
# e.g. def my_function(context_variables: Dict[str, Any], my_other_parameters: Any) -> Any:
__CONTEXT_VARIABLES_PARAM_NAME__ = "context_variables"


def initialize_swarm_chat(
    init_agent: "SwarmAgent",
    messages: Union[List[Dict[str, Any]], str],
    agents: List["SwarmAgent"],
    max_rounds: int = 20,
    context_variables: Optional[Dict[str, Any]] = {},
) -> Tuple[ChatResult, Dict[str, Any], "SwarmAgent"]:
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

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

        # No next agent has been selected, if last agent is the tool executor,
        # we need to go back to the agent before this last tool execution
        if last_speaker == tool_execution:
            return groupchat.agent_by_name(name=messages[-2].get("name", ""))

        return None

    groupchat = GroupChat(
        agents=[tool_execution] + agents,
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

    arguments:
        values (str): The result values as a string.
        agent (SwarmAgent): The swarm agent instance, if applicable.
        context_variables (dict): A dictionary of context variables.
    """

    values: str = ""
    agent: Optional["SwarmAgent"] = None
    context_variables: dict = {}

    class Config:  # Add this inner class
        arbitrary_types_allowed = True

    def __str__(self):
        return self.values


class SwarmAgent(ConversableAgent):
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

        self.context_variables = context_variables or {}
        self.next_agent = None  # use in the tool execution agent to transfer to the next agent

    def update_context_variables(self, context_variables: Dict[str, Any]) -> None:
        pass

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
