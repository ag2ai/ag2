# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
import re
import threading
from collections.abc import Callable, Generator, Iterable
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Literal

from ...exception_utils import InvalidCarryOverTypeError
from ...tools import Tool
from ..chat import ChatResult, _post_process_carryover_item

if TYPE_CHECKING:
    from ..agent import Agent
    from .base import ConversableAgentBase

F = Callable[..., Any]


class ConversableUtilsMixin:
    """Mixin class for utility functions."""

    @staticmethod
    def _normalize_name(name):
        """LLMs sometimes ask functions while ignoring their own format requirements, this function should be used to replace invalid characters with "_".

        Prefer _assert_valid_name for validating user configuration or input
        """
        return re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:64]

    @staticmethod
    def _assert_valid_name(name):
        """Ensure that configured names are valid, raises ValueError if not.

        For munging LLM responses use _normalize_name to ensure LLM specified names don't break the API.
        """
        if not re.match(r"^[a-zA-Z0-9_-]+$", name):
            raise ValueError(f"Invalid name: {name}. Only letters, numbers, '_' and '-' are allowed.")
        if len(name) > 64:
            raise ValueError(f"Invalid name: {name}. Name must be less than 64 characters.")
        return name

    @staticmethod
    def _is_silent(agent: "Agent", silent: bool | None = False) -> bool:
        return agent.silent if agent.silent is not None else silent

    def _should_terminate_chat(
        self: "ConversableAgentBase", recipient: "ConversableAgentBase", message: dict[str, Any]
    ) -> bool:
        """
        Determines whether the chat should be terminated based on the message content
        and the recipient's termination condition.

        Args:
            recipient (ConversableAgent): The agent to check for termination condition.
            message (dict[str, Any]): The message dictionary to evaluate for termination.

        Returns:
            bool: True if the chat should be terminated, False otherwise.
        """
        content = message.get("content")
        return (
            isinstance(recipient, ConversableAgentBase)
            and content is not None
            and hasattr(recipient, "_is_termination_msg")
            and recipient._is_termination_msg(message)
        )

    def _check_chat_queue_for_sender(
        self: "ConversableAgentBase", chat_queue: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Check the chat queue and add the "sender" key if it's missing.

        Args:
            chat_queue (List[Dict[str, Any]]): A list of dictionaries containing chat information.

        Returns:
            List[Dict[str, Any]]: The updated chat queue with sender information.
        """
        for chat in chat_queue:
            if "sender" not in chat:
                chat["sender"] = self
        return chat_queue

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

    def _str_for_tool_response(self: "ConversableAgentBase", tool_response):
        return str(tool_response.get("content", ""))

    def _process_carryover(self: "ConversableAgentBase", content: str, kwargs: dict) -> str:
        # Makes sure there's a carryover
        if not kwargs.get("carryover"):
            return content

        # if carryover is string
        if isinstance(kwargs["carryover"], str):
            content += "\nContext: \n" + kwargs["carryover"]
        elif isinstance(kwargs["carryover"], list):
            content += "\nContext: \n" + ("\n").join([_post_process_carryover_item(t) for t in kwargs["carryover"]])
        else:
            raise InvalidCarryOverTypeError(
                "Carryover should be a string or a list of strings. Not adding carryover to the message."
            )
        return content

    def _process_multimodal_carryover(
        self: "ConversableAgentBase", content: list[dict[str, Any]], kwargs: dict
    ) -> list[dict[str, Any]]:
        """Prepends the context to a multimodal message."""
        # Makes sure there's a carryover
        if not kwargs.get("carryover"):
            return content

        return [{"type": "text", "text": self._process_carryover("", kwargs)}] + content

    @staticmethod
    def _create_tool_if_needed(
        func_or_tool: F | Tool,
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

    @contextmanager
    def _create_or_get_executor(
        self: "ConversableAgentBase",
        executor_kwargs: dict[str, Any] | None = None,
        tools: Tool | Iterable[Tool] | None = None,
        agent_name: str = "executor",
        agent_human_input_mode: str = "NEVER",
    ) -> Generator["ConversableAgentBase", None, None]:
        """Creates a user proxy / tool executor agent.

        Note: Code execution is not enabled by default. Pass the code execution config into executor_kwargs, if needed.

        Args:
            executor_kwargs: agent's arguments.
            tools: tools to register for execution with the agent.
            agent_name: agent's name, defaults to 'executor'.
            agent_human_input_mode: agent's human input mode, defaults to 'NEVER'.
        """
        if executor_kwargs is None:
            executor_kwargs = {}
        if "is_termination_msg" not in executor_kwargs:
            executor_kwargs["is_termination_msg"] = lambda x: (x["content"] is not None) and "TERMINATE" in x["content"]

        try:
            if not self.run_executor:
                from .conversable_agent import ConversableAgent

                self.run_executor = ConversableAgent(
                    name=agent_name,
                    human_input_mode=agent_human_input_mode,
                    **executor_kwargs,
                )

            # Combine agent's existing tools with passed tools
            agent_tools = self._tools.copy()  # Get agent's pre-registered tools
            passed_tools = [] if tools is None else tools
            passed_tools = [passed_tools] if isinstance(passed_tools, Tool) else passed_tools

            # Combine both sets of tools (avoid duplicates)
            all_tools = agent_tools.copy()
            for tool in passed_tools:
                if tool not in all_tools:
                    all_tools.append(tool)

            # Register all tools with the executor
            for tool in all_tools:
                tool.register_for_execution(self.run_executor)

            # Register only newly passed tools for LLM (agent's pre-existing tools are already registered)
            for tool in passed_tools:
                tool.register_for_llm(self)
            yield self.run_executor
        finally:
            # Clean up only newly passed tools (not agent's pre-existing tools)
            if "passed_tools" in locals():
                for tool in passed_tools:
                    self.update_tool_signature(tool_sig=tool.tool_schema["function"]["name"], is_remove=True)

    def _deprecated_run(
        self: "ConversableAgentBase",
        message: str,
        *,
        tools: Tool | Iterable[Tool] | None = None,
        executor_kwargs: dict[str, Any] | None = None,
        max_turns: int | None = None,
        msg_to: Literal["agent", "user"] = "agent",
        clear_history: bool = False,
        user_input: bool = True,
        summary_method: str | Callable[..., Any] | None = None,
    ) -> ChatResult:
        """Run a chat with the agent using the given message.

        A second agent will be created to represent the user, this agent will by known by the name 'user'. This agent does not have code execution enabled by default, if needed pass the code execution config in with the executor_kwargs parameter.

        The user can terminate the conversation when prompted or, if agent's reply contains 'TERMINATE', it will terminate.

        Args:
            message: the message to be processed.
            tools: the tools to be used by the agent.
            executor_kwargs: the keyword arguments for the executor.
            max_turns: maximum number of turns (a turn is equivalent to both agents having replied), defaults no None which means unlimited. The original message is included.
            msg_to: which agent is receiving the message and will be the first to reply, defaults to the agent.
            clear_history: whether to clear the chat history.
            user_input: the user will be asked for input at their turn.
            summary_method: the method to summarize the chat.
        """
        with self._create_or_get_executor(
            executor_kwargs=executor_kwargs,
            tools=tools,
            agent_name="user",
            agent_human_input_mode="ALWAYS" if user_input else "NEVER",
        ) as executor:
            if msg_to == "agent":
                return executor.initiate_chat(
                    self,
                    message=message,
                    clear_history=clear_history,
                    max_turns=max_turns,
                    summary_method=summary_method,
                )
            else:
                return self.initiate_chat(
                    executor,
                    message=message,
                    clear_history=clear_history,
                    max_turns=max_turns,
                    summary_method=summary_method,
                )

    async def _deprecated_a_run(
        self: "ConversableAgentBase",
        message: str,
        *,
        tools: Tool | Iterable[Tool] | None = None,
        executor_kwargs: dict[str, Any] | None = None,
        max_turns: int | None = None,
        msg_to: Literal["agent", "user"] = "agent",
        clear_history: bool = False,
        user_input: bool = True,
        summary_method: str | Callable[..., Any] | None = None,
    ) -> ChatResult:
        """Run a chat asynchronously with the agent using the given message.

        A second agent will be created to represent the user, this agent will by known by the name 'user'.

        The user can terminate the conversation when prompted or, if agent's reply contains 'TERMINATE', it will terminate.

        Args:
            message: the message to be processed.
            tools: the tools to be used by the agent.
            executor_kwargs: the keyword arguments for the executor.
            max_turns: maximum number of turns (a turn is equivalent to both agents having replied), defaults no None which means unlimited. The original message is included.
            msg_to: which agent is receiving the message and will be the first to reply, defaults to the agent.
            clear_history: whether to clear the chat history.
            user_input: the user will be asked for input at their turn.
            summary_method: the method to summarize the chat.
        """
        with self._create_or_get_executor(
            executor_kwargs=executor_kwargs,
            tools=tools,
            agent_name="user",
            agent_human_input_mode="ALWAYS" if user_input else "NEVER",
        ) as executor:
            if msg_to == "agent":
                return await executor.a_initiate_chat(
                    self,
                    message=message,
                    clear_history=clear_history,
                    max_turns=max_turns,
                    summary_method=summary_method,
                )
            else:
                return await self.a_initiate_chat(
                    executor,
                    message=message,
                    clear_history=clear_history,
                    max_turns=max_turns,
                    summary_method=summary_method,
                )
