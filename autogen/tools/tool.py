# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Any, Callable, Literal

if TYPE_CHECKING:
    from ..agentchat.conversable_agent import ConversableAgent

__all__ = ["Tool"]


class Tool:
    """
    A class representing a Tool that can be used by an agent for various tasks.

    This class encapsulates a tool with a name, description, and an executable function.
    The tool can be registered with a ConversableAgent for use either with an LLM or for direct execution.

    Attributes:
        name (str): The name of the tool.
        description (str): A brief description of the tool's purpose or function.
        func (Callable[..., Any]): The function to be executed when the tool is called.
    """

    def __init__(self, *, name: str, description: str, func: Callable[..., Any]) -> None:
        """Create a new Tool object.

        Args:
            name (str): The name of the tool.
            description (str): The description of the tool.
            func (Callable[..., Any]): The function that will be executed when the tool is called.
        """
        self._name = name
        self._description = description
        self._func = func

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def func(self) -> Callable[..., Any]:
        return self._func

    def register_for_llm(self, agent: "ConversableAgent") -> None:
        """
        Registers the tool for use with a ConversableAgent's language model (LLM).

        This method registers the tool so that it can be invoked by the agent during
        interactions with the language model.

        Args:
            agent (ConversableAgent): The agent to which the tool will be registered.
        """
        agent.register_for_llm()(self)

    def register_for_execution(self, agent: "ConversableAgent") -> None:
        """
        Registers the tool for direct execution by a ConversableAgent.

        This method registers the tool so that it can be executed by the agent,
        typically outside of the context of an LLM interaction.

        Args:
            agent (ConversableAgent): The agent to which the tool will be registered.
        """
        agent.register_for_execution()(self)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Execute the tool by calling its underlying function with the provided arguments.

        Args:
            *args: Positional arguments to pass to the tool
            **kwargs: Keyword arguments to pass to the tool

        Returns:
            The result of executing the tool's function.
        """
        return self._func(*args, **kwargs)
