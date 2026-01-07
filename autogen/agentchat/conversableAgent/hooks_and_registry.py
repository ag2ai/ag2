# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from ...llm_config.client import ModelClient
from ...oai.client import OpenAIWrapper
from ...tools import ChatContext, Tool
from .types import UpdateSystemMessage

if TYPE_CHECKING:
    from .base import ConversableAgentBase


class HooksAndRegistryMixin:
    """Mixin class for hooks and registry functionality"""

    def register_hook(self: "ConversableAgentBase", hookable_method: str, hook: Callable):
        """Registers a hook to be called by a hookable method, in order to add a capability to the agent.
        Registered hooks are kept in lists (one per hookable method), and are called in their order of registration.

        Args:
            hookable_method: A hookable method name implemented by ConversableAgent.
            hook: A method implemented by a subclass of AgentCapability.
        """
        assert hookable_method in self.hook_lists, f"{hookable_method} is not a hookable method."
        hook_list = self.hook_lists[hookable_method]
        assert hook not in hook_list, f"{hook} is already registered as a hook."
        hook_list.append(hook)

    def update_agent_state_before_reply(self: "ConversableAgentBase", messages: list[dict[str, Any]]) -> None:
        """Calls any registered capability hooks to update the agent's state.
        Primarily used to update context variables.
        Will, potentially, modify the messages.
        """
        hook_list = self.hook_lists["update_agent_state"]

        # Call each hook (in order of registration) to process the messages.
        for hook in hook_list:
            hook(self, messages)

    def process_all_messages_before_reply(
        self: "ConversableAgentBase", messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Calls any registered capability hooks to process all messages, potentially modifying the messages."""
        hook_list = self.hook_lists["process_all_messages_before_reply"]
        # If no hooks are registered, or if there are no messages to process, return the original message list.
        if len(hook_list) == 0 or messages is None:
            return messages

        # Call each hook (in order of registration) to process the messages.
        processed_messages = messages
        for hook in hook_list:
            processed_messages = hook(processed_messages)
        return processed_messages

    def process_last_received_message(
        self: "ConversableAgentBase", messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Calls any registered capability hooks to use and potentially modify the text of the last message,
        as long as the last message is not a function call or exit command.
        """
        # If any required condition is not met, return the original message list.
        hook_list = self.hook_lists["process_last_received_message"]
        if len(hook_list) == 0:
            return messages  # No hooks registered.
        if messages is None:
            return None  # No message to process.
        if len(messages) == 0:
            return messages  # No message to process.
        last_message = messages[-1]
        if "function_call" in last_message:
            return messages  # Last message is a function call.
        if "context" in last_message:
            return messages  # Last message contains a context key.
        if "content" not in last_message:
            return messages  # Last message has no content.

        user_content = last_message["content"]
        if not isinstance(user_content, str) and not isinstance(user_content, list):
            # if the user_content is a string, it is for regular LLM
            # if the user_content is a list, it should follow the multimodal LMM format.
            return messages
        if user_content == "exit":
            return messages  # Last message is an exit command.

        # Call each hook (in order of registration) to process the user's message.
        processed_user_content = user_content
        for hook in hook_list:
            processed_user_content = hook(processed_user_content)

        if processed_user_content == user_content:
            return messages  # No hooks actually modified the user's message.

        # Replace the last user message with the expanded one.
        messages = messages.copy()
        messages[-1]["content"] = processed_user_content
        return messages

    def _add_functions(self: "ConversableAgentBase", func_list: list[Callable[..., Any]]):
        """Add (Register) a list of functions to the agent

        Args:
            func_list (list[Callable[..., Any]]): A list of functions to register with the agent.
        """
        for func in func_list:
            self._add_single_function(func)

    def _add_single_function(
        self: "ConversableAgentBase", func: Callable, name: str | None = None, description: str | None = ""
    ):
        """Add a single function to the agent

        Args:
            func (Callable): The function to register.
            name (str): The name of the function. If not provided, the function's name will be used.
            description (str): The description of the function, used by the LLM. If not provided, the function's docstring will be used.
        """
        if name:
            func._name = name
        elif not hasattr(func, "_name"):
            func._name = func.__name__

        if hasattr(func, "_description") and func._description and not description:
            # If the function already has a description, use it
            description = func._description
        else:
            if description:
                func._description = description
            else:
                # Use function's docstring, strip whitespace, fall back to empty string
                description = (func.__doc__ or "").strip()
                func._description = description

        # Register the function
        self.register_for_llm(name=name, description=description, silent_override=True)(func)

    def _register_update_agent_state_before_reply(
        self: "ConversableAgentBase", functions: list[Callable[..., Any]] | Callable[..., Any] | None
    ):
        """Register functions that will be called when the agent is selected and before it speaks.
        You can add your own validation or precondition functions here.

        Args:
            functions (List[Callable[[], None]]): A list of functions to be registered. Each function
                is called when the agent is selected and before it speaks.
        """
        if functions is None:
            return
        if not isinstance(functions, list) and type(functions) not in [UpdateSystemMessage, Callable[..., Any]]:
            raise ValueError("functions must be a list of callables")

        if not isinstance(functions, list):
            functions = [functions]

        for func in functions:
            if isinstance(func, UpdateSystemMessage):
                # Wrapper function that allows this to be used in the update_agent_state hook
                # Its primary purpose, however, is just to update the agent's system message
                # Outer function to create a closure with the update function
                def create_wrapper(update_func: UpdateSystemMessage):
                    def update_system_message_wrapper(
                        agent: "ConversableAgentBase", messages: list[dict[str, Any]]
                    ) -> list[dict[str, Any]]:
                        if isinstance(update_func.content_updater, str):
                            # Templates like "My context variable passport is {passport}" will
                            # use the context_variables for substitution
                            sys_message = OpenAIWrapper.instantiate(
                                template=update_func.content_updater,
                                context=agent.context_variables.to_dict(),
                                allow_format_str_template=True,
                            )
                        else:
                            sys_message = update_func.content_updater(agent, messages)

                        agent.update_system_message(sys_message)
                        return messages

                    return update_system_message_wrapper

                self.register_hook(hookable_method="update_agent_state", hook=create_wrapper(func))

            else:
                self.register_hook(hookable_method="update_agent_state", hook=func)

    def remove_tool_for_llm(self: "ConversableAgentBase", tool: Tool) -> None:
        """Remove a tool (register for LLM tool)"""
        try:
            self._register_for_llm(tool=tool, api_style="tool", is_remove=True)
            self._tools.remove(tool)
        except ValueError:
            raise ValueError(f"Tool {tool} not found in collection")

    def set_ui_tools(self: "ConversableAgentBase", tools: list[Tool]) -> None:
        """Set the UI tools for the agent.

        Args:
            tools: a list of tools to be set.
        """
        # Unset the previous UI tools
        self._unset_previous_ui_tools()

        # Set the new UI tools
        for tool in tools:
            # Register the tool for LLM
            self._register_for_llm(tool, api_style="tool", silent_override=True)
            if tool not in self._tools:
                self._tools.append(tool)

            # Register for execution
            self.register_for_execution(serialize=False, silent_override=True)(tool)

        # Set the current UI tools
        self._ui_tools = tools

    def unset_ui_tools(self: "ConversableAgentBase", tools: list[Tool]) -> None:
        """Unset the UI tools for the agent.

        Args:
            tools: a list of tools to be unset.
        """
        for tool in tools:
            self.remove_tool_for_llm(tool)

    def _unset_previous_ui_tools(self: "ConversableAgentBase") -> None:
        """Unset the previous UI tools for the agent.

        This is used to remove UI tools that were previously registered for LLM.
        """
        self.unset_ui_tools(self._ui_tools)
        for tool in self._ui_tools:
            if tool in self._tools:
                self._tools.remove(tool)

            # Unregister the function from the function map
            if tool.name in self._function_map:
                del self._function_map[tool.name]

        self._ui_tools = []

    def register_for_llm(
        self: "ConversableAgentBase",
        *,
        name: str | None = None,
        description: str | None = None,
        api_style: Literal["function", "tool"] = "tool",
        silent_override: bool = False,
    ) -> Callable:
        """Decorator factory for registering a function to be used by an agent.

        It's return value is used to decorate a function to be registered to the agent. The function uses type hints to
        specify the arguments and return type. The function name is used as the default name for the function,
        but a custom name can be provided. The function description is used to describe the function in the
        agent's configuration.

        Args:
            name (optional(str)): name of the function. If None, the function name will be used (default: None).
            description (optional(str)): description of the function (default: None). It is mandatory
                for the initial decorator, but the following ones can omit it.
            api_style: (literal): the API style for function call.
                For Azure OpenAI API, use version 2023-12-01-preview or later.
                `"function"` style will be deprecated. For earlier version use
                `"function"` if `"tool"` doesn't work.
                See [Azure OpenAI documentation](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/function-calling?tabs=python) for details.
            silent_override (bool): whether to suppress any override warning messages.

        Returns:
            The decorator for registering a function to be used by an agent.

        Examples:
            ```
            @user_proxy.register_for_execution()
            @agent2.register_for_llm()
            @agent1.register_for_llm(description="This is a very useful function")
            def my_function(a: Annotated[str, "description of a parameter"] = "a", b: int, c=3.14) -> str:
                 return a + str(b * c)
            ```

            For Azure OpenAI versions prior to 2023-12-01-preview, set `api_style`
            to `"function"` if `"tool"` doesn't work:
            ```
            @agent2.register_for_llm(api_style="function")
            def my_function(a: Annotated[str, "description of a parameter"] = "a", b: int, c=3.14) -> str:
                 return a + str(b * c)
            ```

        """

        def _decorator(func_or_tool, name: str | None = name, description: str | None = description) -> Tool:
            """Decorator for registering a function to be used by an agent.

            Args:
                func_or_tool: The function or the tool to be registered.
                name: The name of the function or the tool.
                description: The description of the function or the tool.

            Returns:
                The function to be registered, with the _description attribute set to the function description.

            Raises:
                ValueError: if the function description is not provided and not propagated by a previous decorator.
                RuntimeError: if the LLM config is not set up before registering a function.

            """
            tool = self._create_tool_if_needed(func_or_tool, name, description)

            self._register_for_llm(tool, api_style, silent_override=silent_override)
            if tool not in self._tools:
                self._tools.append(tool)

            return tool

        return _decorator

    def _register_for_llm(
        self: "ConversableAgentBase",
        tool: Tool,
        api_style: Literal["tool", "function"],
        is_remove: bool = False,
        silent_override: bool = False,
    ) -> None:
        """Register a tool for LLM.

        Args:
            tool: the tool to be registered.
            api_style: the API style for function call ("tool" or "function").
            is_remove: whether to remove the function or tool.
            silent_override: whether to suppress any override warning messages.

        Returns:
            None
        """
        # register the function to the agent if there is LLM config, raise an exception otherwise
        if self.llm_config is None:
            raise RuntimeError("LLM config must be setup before registering a function for LLM.")

        if api_style == "function":
            self.update_function_signature(tool.function_schema, is_remove=is_remove, silent_override=silent_override)
        elif api_style == "tool":
            self.update_tool_signature(tool.tool_schema, is_remove=is_remove, silent_override=silent_override)
        else:
            raise ValueError(f"Unsupported API style: {api_style}")

    def register_for_execution(
        self: "ConversableAgentBase",
        name: str | None = None,
        description: str | None = None,
        *,
        serialize: bool = True,
        silent_override: bool = False,
    ) -> Callable:
        """Decorator factory for registering a function to be executed by an agent.

        It's return value is used to decorate a function to be registered to the agent.

        Args:
            name: name of the function. If None, the function name will be used (default: None).
            description: description of the function (default: None).
            serialize: whether to serialize the return value
            silent_override: whether to suppress any override warning messages

        Returns:
            The decorator for registering a function to be used by an agent.

        Examples:
            ```
            @user_proxy.register_for_execution()
            @agent2.register_for_llm()
            @agent1.register_for_llm(description="This is a very useful function")
            def my_function(a: Annotated[str, "description of a parameter"] = "a", b: int, c=3.14):
                 return a + str(b * c)
            ```

        """

        def _decorator(func_or_tool, name: str | None = name, description: str | None = description) -> Tool:
            """Decorator for registering a function to be used by an agent.

            Args:
                func_or_tool: the function or the tool to be registered.
                name: the name of the function.
                description: the description of the function.

            Returns:
                The tool to be registered.

            """
            tool = self._create_tool_if_needed(func_or_tool, name, description)
            chat_context = ChatContext(self)
            chat_context_params = dict.fromkeys(tool._chat_context_param_names, chat_context)

            self.register_function(
                {tool.name: self._wrap_function(tool.func, chat_context_params, serialize=serialize)},
                silent_override=silent_override,
            )

            return tool

        return _decorator

    def register_model_client(self: "ConversableAgentBase", model_client_cls: ModelClient, **kwargs: Any):
        """Register a model client.

        Args:
            model_client_cls: A custom client class that follows the Client interface
            **kwargs: The kwargs for the custom client class to be initialized with
        """
        self.client.register_model_client(model_client_cls, **kwargs)

    def _process_tool_input(self: "ConversableAgentBase", tool_input: dict[str, Any]) -> dict[str, Any] | None:
        """Process tool input through registered hooks."""
        hook_list = self.hook_lists["safeguard_tool_inputs"]

        # If no hooks are registered, allow the tool input
        if len(hook_list) == 0:
            return tool_input

        # Process through each hook
        processed_input = tool_input
        for hook in hook_list:
            processed_input = hook(processed_input)
            if processed_input is None:
                return None

        return processed_input

    def _process_tool_output(self: "ConversableAgentBase", response: dict[str, Any]) -> dict[str, Any]:
        """Process tool output through registered hooks"""
        hook_list = self.hook_lists["safeguard_tool_outputs"]

        # If no hooks are registered, return original response
        if len(hook_list) == 0:
            return response

        # Process through each hook
        processed_response = response
        for hook in hook_list:
            processed_response = hook(processed_response)

        return processed_response

    def _process_human_input(self: "ConversableAgentBase", human_input: str) -> str | None:
        """Process human input through registered hooks."""
        hook_list = self.hook_lists["safeguard_human_inputs"]

        # If no hooks registered, allow the input through
        if len(hook_list) == 0:
            return human_input

        # Process through each hook
        processed_input = human_input
        for hook in hook_list:
            processed_input = hook(processed_input)
            if processed_input is None:
                return None

        return processed_input
