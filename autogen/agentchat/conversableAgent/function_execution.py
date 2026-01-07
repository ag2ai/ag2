# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import functools
import inspect
import json
import threading
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from ...events.agent_events import (
    ExecuteFunctionEvent,
    ExecutedFunctionEvent,
)
from ...fast_depends.utils import is_coroutine_callable
from ...io.base import IOStream
from ...llm_config import LLMConfig
from ...oai.client import OpenAIWrapper
from ...runtime_logging import log_function_use, logging_enabled
from ...tools import ChatContext, Tool, load_basemodels_if_needed, serialize_to_str
from ..agent import Agent

if TYPE_CHECKING:
    from .base import ConversableAgentBase


class FunctionExecutionMixin:
    """Mixin class for function and tool execution functionality"""

    @staticmethod
    def _format_json_str(jstr):
        """Remove newlines outside of quotes, and handle JSON escape sequences.

        1. this function removes the newline in the query outside of quotes otherwise json.loads(s) will fail.
            Ex 1:
            "{\n"tool": "python",\n"query": "print('hello')\nprint('world')"\n}" -> "{"tool": "python","query": "print('hello')\nprint('world')"}"
            Ex 2:
            "{\n  \"location\": \"Boston, MA\"\n}" -> "{"location": "Boston, MA"}"

        2. this function also handles JSON escape sequences inside quotes.
            Ex 1:
            '{"args": "a\na\na\ta"}' -> '{"args": "a\\na\\na\\ta"}'
        """
        result = []
        inside_quotes = False
        last_char = " "
        for char in jstr:
            if last_char != "\\" and char == '"':
                inside_quotes = not inside_quotes
            last_char = char
            if not inside_quotes and char == "\n":
                continue
            if inside_quotes and char == "\n":
                char = "\\n"
            if inside_quotes and char == "\t":
                char = "\\t"
            result.append(char)
        return "".join(result)

    def execute_function(
        self: "ConversableAgentBase", func_call: dict[str, Any], call_id: str | None = None, verbose: bool = False
    ) -> tuple[bool, dict[str, Any]]:
        """Execute a function call and return the result.

        Override this function to modify the way to execute function and tool calls.

        Args:
            func_call: a dictionary extracted from openai message at "function_call" or "tool_calls" with keys "name" and "arguments".
            call_id: a string to identify the tool call.
            verbose (bool): Whether to send messages about the execution details to the
                output stream. When True, both the function call arguments and the execution
                result will be displayed. Defaults to False.


        Returns:
            A tuple of (is_exec_success, result_dict).
            is_exec_success (boolean): whether the execution is successful.
            result_dict: a dictionary with keys "name", "role", and "content". Value of "role" is "function".

        "function_call" deprecated as of [OpenAI API v1.1.0](https://github.com/openai/openai-python/releases/tag/v1.1.0)
        See https://platform.openai.com/docs/api-reference/chat/create#chat-create-function_call
        """
        iostream = IOStream.get_default()

        func_name = func_call.get("name", "")
        func = self._function_map.get(func_name, None)

        is_exec_success = False
        if func is not None:
            # Extract arguments from a json-like string and put it into a dict.
            input_string = self._format_json_str(func_call.get("arguments", "{}"))
            try:
                arguments = json.loads(input_string)
            except json.JSONDecodeError as e:
                arguments = None
                content = f"Error: {e}\n The argument must be in JSON format."

            # Try to execute the function
            if arguments is not None:
                iostream.send(
                    ExecuteFunctionEvent(func_name=func_name, call_id=call_id, arguments=arguments, recipient=self)
                )
                try:
                    content = func(**arguments)
                    if inspect.isawaitable(content):

                        async def _await_result(awaitable):
                            return await awaitable

                        content = self._run_async_in_thread(_await_result(content))
                    is_exec_success = True
                except Exception as e:
                    content = f"Error: {e}"
        else:
            arguments = {}
            content = f"Error: Function {func_name} not found."

        iostream.send(
            ExecutedFunctionEvent(
                func_name=func_name,
                call_id=call_id,
                arguments=arguments,
                content=content,
                recipient=self,
                is_exec_success=is_exec_success,
            )
        )

        return is_exec_success, {
            "name": func_name,
            "role": "function",
            "content": content,
        }

    async def a_execute_function(
        self: "ConversableAgentBase", func_call: dict[str, Any], call_id: str | None = None, verbose: bool = False
    ) -> tuple[bool, dict[str, Any]]:
        """Execute an async function call and return the result.

        Override this function to modify the way async functions and tools are executed.

        Args:
            func_call: a dictionary extracted from openai message at key "function_call" or "tool_calls" with keys "name" and "arguments".
            call_id: a string to identify the tool call.
            verbose (bool): Whether to send messages about the execution details to the
                output stream. When True, both the function call arguments and the execution
                result will be displayed. Defaults to False.

        Returns:
            A tuple of (is_exec_success, result_dict).
            is_exec_success (boolean): whether the execution is successful.
            result_dict: a dictionary with keys "name", "role", and "content". Value of "role" is "function".

        "function_call" deprecated as of [OpenAI API v1.1.0](https://github.com/openai/openai-python/releases/tag/v1.1.0)
        See https://platform.openai.com/docs/api-reference/chat/create#chat-create-function_call
        """
        iostream = IOStream.get_default()

        func_name = func_call.get("name", "")
        func = self._function_map.get(func_name, None)

        is_exec_success = False
        if func is not None:
            # Extract arguments from a json-like string and put it into a dict.
            input_string = self._format_json_str(func_call.get("arguments", "{}"))
            try:
                arguments = json.loads(input_string)
            except json.JSONDecodeError as e:
                arguments = None
                content = f"Error: {e}\n The argument must be in JSON format."

            # Try to execute the function
            if arguments is not None:
                iostream.send(
                    ExecuteFunctionEvent(func_name=func_name, call_id=call_id, arguments=arguments, recipient=self)
                )
                try:
                    if inspect.iscoroutinefunction(func):
                        content = await func(**arguments)
                    else:
                        # Fallback to sync function if the function is not async
                        content = func(**arguments)
                    is_exec_success = True
                except Exception as e:
                    content = f"Error: {e}"
        else:
            arguments = {}
            content = f"Error: Function {func_name} not found."

        iostream.send(
            ExecutedFunctionEvent(
                func_name=func_name,
                call_id=call_id,
                arguments=arguments,
                content=content,
                recipient=self,
                is_exec_success=is_exec_success,
            )
        )

        return is_exec_success, {
            "name": func_name,
            "role": "function",
            "content": content,
        }

    def _run_async_in_thread(self: "ConversableAgentBase", coro):
        """Run an async coroutine in a separate thread with its own event loop."""
        result = {}

        def runner():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result["value"] = loop.run_until_complete(coro)
            loop.close()

        t = threading.Thread(target=runner)
        t.start()
        t.join()
        return result["value"]

    def generate_function_call_reply(
        self: "ConversableAgentBase",
        messages: list[dict[str, Any]] | None = None,
        sender: Agent | None = None,
        config: Any | None = None,
    ) -> tuple[bool, dict[str, Any] | None]:
        """Generate a reply using function call.

        "function_call" replaced by "tool_calls" as of [OpenAI API v1.1.0](https://github.com/openai/openai-python/releases/tag/v1.1.0)
        See https://platform.openai.com/docs/api-reference/chat/create#chat-create-functions
        """
        if config is None:
            config = self
        if messages is None:
            messages = self._oai_messages[sender]
        message = messages[-1]
        if message.get("function_call"):
            call_id = message.get("id", None)
            func_call = message["function_call"]
            func = self._function_map.get(func_call.get("name", None), None)
            if is_coroutine_callable(func):
                coro = self.a_execute_function(func_call, call_id=call_id)
                _, func_return = self._run_async_in_thread(coro)
            else:
                _, func_return = self.execute_function(message["function_call"], call_id=call_id)
            return True, func_return
        return False, None

    async def a_generate_function_call_reply(
        self: "ConversableAgentBase",
        messages: list[dict[str, Any]] | None = None,
        sender: Agent | None = None,
        config: Any | None = None,
    ) -> tuple[bool, dict[str, Any] | None]:
        """Generate a reply using async function call.

        "function_call" replaced by "tool_calls" as of [OpenAI API v1.1.0](https://github.com/openai/openai-python/releases/tag/v1.1.0)
        See https://platform.openai.com/docs/api-reference/chat/create#chat-create-functions
        """
        if config is None:
            config = self
        if messages is None:
            messages = self._oai_messages[sender]
        message = messages[-1]
        if "function_call" in message:
            call_id = message.get("id", None)
            func_call = message["function_call"]
            func_name = func_call.get("name", "")
            func = self._function_map.get(func_name, None)
            if func and is_coroutine_callable(func):
                _, func_return = await self.a_execute_function(func_call, call_id=call_id)
            else:
                _, func_return = self.execute_function(func_call, call_id=call_id)
            return True, func_return

        return False, None

    def _str_for_tool_response(self: "ConversableAgentBase", tool_response):
        return str(tool_response.get("content", ""))

    def generate_tool_calls_reply(
        self: "ConversableAgentBase",
        messages: list[dict[str, Any]] | None = None,
        sender: Agent | None = None,
        config: Any | None = None,
    ) -> tuple[bool, dict[str, Any] | None]:
        """Generate a reply using tool call."""
        if config is None:
            config = self
        if messages is None:
            messages = self._oai_messages[sender]
        message = messages[-1]
        tool_returns = []
        for tool_call in message.get("tool_calls", []):
            function_call = tool_call.get("function", {})
            function_name = function_call.get("name", "")
            # Special case for __structured_output
            if function_name == "__structured_output":
                return True, function_call.get("arguments", {})

            # Hook: Process tool input before execution
            processed_call = self._process_tool_input(function_call)
            if processed_call is None:
                raise ValueError("safeguard_tool_inputs hook returned None")

            tool_call_id = tool_call.get("id", None)
            func = self._function_map.get(processed_call.get("name", None), None)
            if is_coroutine_callable(func):
                coro = self.a_execute_function(processed_call, call_id=tool_call_id)
                _, func_return = self._run_async_in_thread(coro)
            else:
                _, func_return = self.execute_function(processed_call, call_id=tool_call_id)

            # Hook: Process tool output before returning
            processed_return = self._process_tool_output(func_return)
            if processed_return is None:
                raise ValueError("safeguard_tool_outputs hook returned None")

            content = processed_return.get("content", "")
            if content is None:
                content = ""

            if tool_call_id is not None:
                tool_call_response = {
                    "tool_call_id": tool_call_id,
                    "role": "tool",
                    "content": content,
                }
            else:
                # Do not include tool_call_id if it is not present.
                # This is to make the tool call object compatible with Mistral API.
                tool_call_response = {
                    "role": "tool",
                    "content": content,
                }
            tool_returns.append(tool_call_response)
        if tool_returns:
            return True, {
                "role": "tool",
                "tool_responses": tool_returns,
                "content": "\n\n".join([self._str_for_tool_response(tool_return) for tool_return in tool_returns]),
            }
        return False, None

    async def _a_execute_tool_call(self: "ConversableAgentBase", tool_call):
        tool_call_id = tool_call["id"]
        function_call = tool_call.get("function", {})
        function_name = function_call.get("name", "")
        # Special case for __structured_output
        if function_name == "__structured_output":
            return {
                "tool_call_id": tool_call_id,
                "role": "tool",
                "content": function_call.get("arguments", {}),
            }

        # Hook: Process tool input before execution
        processed_call = self._process_tool_input(function_call)
        if processed_call is None:
            raise ValueError("safeguard_tool_inputs hook returned None")

        _, func_return = await self.a_execute_function(processed_call, call_id=tool_call_id)

        # Hook: Process tool output before returning
        processed_return = self._process_tool_output(func_return)
        if processed_return is None:
            raise ValueError("safeguard_tool_outputs hook returned None")

        return {
            "tool_call_id": tool_call_id,
            "role": "tool",
            "content": processed_return.get("content", ""),
        }

    async def a_generate_tool_calls_reply(
        self: "ConversableAgentBase",
        messages: list[dict[str, Any]] | None = None,
        sender: Agent | None = None,
        config: Any | None = None,
    ) -> tuple[bool, dict[str, Any] | None]:
        """Generate a reply using async function call."""
        if config is None:
            config = self
        if messages is None:
            messages = self._oai_messages[sender]
        message = messages[-1]
        async_tool_calls = []
        for tool_call in message.get("tool_calls", []):
            async_tool_calls.append(self._a_execute_tool_call(tool_call))
        if async_tool_calls:
            tool_returns = await asyncio.gather(*async_tool_calls)
            return True, {
                "role": "tool",
                "tool_responses": tool_returns,
                "content": "\n\n".join([self._str_for_tool_response(tool_return) for tool_return in tool_returns]),
            }

        return False, None

    def register_function(
        self: "ConversableAgentBase", function_map: dict[str, Callable[..., Any]], silent_override: bool = False
    ):
        """Register functions to the agent.

        Args:
            function_map: a dictionary mapping function names to functions. if function_map[name] is None, the function will be removed from the function_map.
            silent_override: whether to print warnings when overriding functions.
        """
        for name, func in function_map.items():
            self._assert_valid_name(name)
            if func is None and name not in self._function_map:
                warnings.warn(f"The function {name} to remove doesn't exist", name)
            if not silent_override and name in self._function_map:
                warnings.warn(f"Function '{name}' is being overridden.", UserWarning)
        self._function_map.update(function_map)
        self._function_map = {k: v for k, v in self._function_map.items() if v is not None}

    def update_function_signature(
        self: "ConversableAgentBase",
        func_sig: str | dict[str, Any],
        is_remove: bool = False,
        silent_override: bool = False,
    ):
        """Update a function_signature in the LLM configuration for function_call.

        Args:
            func_sig (str or dict): description/name of the function to update/remove to the model. See: https://platform.openai.com/docs/api-reference/chat/create#chat/create-functions
            is_remove: whether removing the function from llm_config with name 'func_sig'
            silent_override: whether to print warnings when overriding functions.

        Deprecated as of [OpenAI API v1.1.0](https://github.com/openai/openai-python/releases/tag/v1.1.0)
        See https://platform.openai.com/docs/api-reference/chat/create#chat-create-function_call
        """
        import logging

        logger = logging.getLogger(__name__)

        if not isinstance(self.llm_config, (dict, LLMConfig)):
            error_msg = "To update a function signature, agent must have an llm_config"
            logger.error(error_msg)
            raise AssertionError(error_msg)

        if is_remove:
            if "functions" not in self.llm_config or len(self.llm_config["functions"]) == 0:
                error_msg = f"The agent config doesn't have function {func_sig}."
                logger.error(error_msg)
                raise AssertionError(error_msg)
            else:
                self.llm_config["functions"] = [
                    func for func in self.llm_config["functions"] if func["name"] != func_sig
                ]
        else:
            if not isinstance(func_sig, dict):
                raise ValueError(
                    f"The function signature must be of the type dict. Received function signature type {type(func_sig)}"
                )
            if "name" not in func_sig:
                raise ValueError(f"The function signature must have a 'name' key. Received: {func_sig}")
            self._assert_valid_name(func_sig["name"]), func_sig
            if "functions" in self.llm_config:
                if not silent_override and any(
                    func["name"] == func_sig["name"] for func in self.llm_config["functions"]
                ):
                    warnings.warn(f"Function '{func_sig['name']}' is being overridden.", UserWarning)

                self.llm_config["functions"] = [
                    func for func in self.llm_config["functions"] if func.get("name") != func_sig["name"]
                ] + [func_sig]
            else:
                self.llm_config["functions"] = [func_sig]

        # Do this only if llm_config is a dict. If llm_config is LLMConfig, LLMConfig will handle this.
        if len(self.llm_config["functions"]) == 0 and isinstance(self.llm_config, dict):
            del self.llm_config["functions"]

        self.client = OpenAIWrapper(**self.llm_config)

    def update_tool_signature(
        self: "ConversableAgentBase", tool_sig: str | dict[str, Any], is_remove: bool, silent_override: bool = False
    ):
        """Update a tool_signature in the LLM configuration for tool_call.

        Args:
            tool_sig (str or dict): description/name of the tool to update/remove to the model. See: https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools
            is_remove: whether removing the tool from llm_config with name 'tool_sig'
            silent_override: whether to print warnings when overriding functions.
        """
        import logging

        logger = logging.getLogger(__name__)

        if not self.llm_config:
            error_msg = "To update a tool signature, agent must have an llm_config"
            logger.error(error_msg)
            raise AssertionError(error_msg)

        if is_remove:
            if "tools" not in self.llm_config or len(self.llm_config["tools"]) == 0:
                error_msg = f"The agent config doesn't have tool {tool_sig}."
                logger.error(error_msg)
                raise AssertionError(error_msg)
            else:
                current_tools = self.llm_config["tools"]
                filtered_tools = []

                # Loop through and rebuild tools list without the tool to remove
                for tool in current_tools:
                    tool_name = tool["function"]["name"]

                    # Match by tool name, or by tool signature
                    is_different = tool_name != tool_sig if isinstance(tool_sig, str) else tool != tool_sig

                    if is_different:
                        filtered_tools.append(tool)

                self.llm_config["tools"] = filtered_tools
        else:
            if not isinstance(tool_sig, dict):
                raise ValueError(
                    f"The tool signature must be of the type dict. Received tool signature type {type(tool_sig)}"
                )
            self._assert_valid_name(tool_sig["function"]["name"])
            if "tools" in self.llm_config and len(self.llm_config["tools"]) > 0:
                if not silent_override and any(
                    tool["function"]["name"] == tool_sig["function"]["name"] for tool in self.llm_config["tools"]
                ):
                    warnings.warn(f"Function '{tool_sig['function']['name']}' is being overridden.", UserWarning)
                self.llm_config["tools"] = [
                    tool
                    for tool in self.llm_config["tools"]
                    if tool.get("function", {}).get("name") != tool_sig["function"]["name"]
                ] + [tool_sig]
            else:
                self.llm_config["tools"] = [tool_sig]

        # Do this only if llm_config is a dict. If llm_config is LLMConfig, LLMConfig will handle this.
        if len(self.llm_config["tools"]) == 0 and isinstance(self.llm_config, dict):
            del self.llm_config["tools"]

        self.client = OpenAIWrapper(**self.llm_config)

    def can_execute_function(self: "ConversableAgentBase", name: list[str] | str) -> bool:
        """Whether the agent can execute the function."""
        names = name if isinstance(name, list) else [name]
        return all(n in self._function_map for n in names)

    def _wrap_function(
        self: "ConversableAgentBase", func, inject_params: dict[str, Any] = {}, *, serialize: bool = True
    ):
        """Wrap the function inject chat context parameters and to dump the return value to json.

        Handles both sync and async functions.

        Args:
            func: the function to be wrapped.
            inject_params: the chat context parameters which will be passed to the function.
            serialize: whether to serialize the return value

        Returns:
            The wrapped function.
        """

        @load_basemodels_if_needed
        @functools.wraps(func)
        def _wrapped_func(*args, **kwargs):
            retval = func(*args, **kwargs, **inject_params)
            if logging_enabled():
                log_function_use(self, func, kwargs, retval)
            return serialize_to_str(retval) if serialize else retval

        @load_basemodels_if_needed
        @functools.wraps(func)
        async def _a_wrapped_func(*args, **kwargs):
            retval = await func(*args, **kwargs, **inject_params)
            if logging_enabled():
                log_function_use(self, func, kwargs, retval)
            return serialize_to_str(retval) if serialize else retval

        wrapped_func = _a_wrapped_func if inspect.iscoroutinefunction(func) else _wrapped_func

        # needed for testing
        wrapped_func._origin = func

        return wrapped_func

    @staticmethod
    def _create_tool_if_needed(
        func_or_tool,
        name: str | None,
        description: str | None,
    ) -> Tool:
        if isinstance(func_or_tool, Tool):
            tool: Tool = func_or_tool
            # create new tool object if name or description is not None
            if name or description:
                tool = Tool(func_or_tool=tool, name=name, description=description)
        elif inspect.isfunction(func_or_tool):
            function: Callable[..., Any] = func_or_tool
            tool = Tool(func_or_tool=function, name=name, description=description)
        else:
            raise TypeError(f"'func_or_tool' must be a function or a Tool object, got '{type(func_or_tool)}' instead.")
        return tool

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
