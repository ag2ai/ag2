# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from ..tools.function_utils import get_function_schema
from .dependency_injection import ChatContext, get_context_params, inject_params

if TYPE_CHECKING:
    from ..agentchat.conversable_agent import ConversableAgent

__all__ = ["Tool"]


class Tool:
    """A class representing a Tool that can be used by an agent for various tasks.

    This class encapsulates a tool with a name, description, and an executable function.
    The tool can be registered with a ConversableAgent for use either with an LLM or for direct execution.

    Attributes:
        name (str): The name of the tool.
        description (str): A brief description of the tool's purpose or function.
        func (Callable[..., Any]): The function to be executed when the tool is called.
    """

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        func_or_tool: Union["Tool", Callable[..., Any]],
    ) -> None:
        """Create a new Tool object.

        Args:
            name (str): The name of the tool.
            description (str): The description of the tool.
            func_or_tool (Union[Tool, Callable[..., Any]]): The function or Tool instance to create a Tool from.
        """
        if isinstance(func_or_tool, Tool):
            self._name: str = name or func_or_tool.name
            self._description: str = description or func_or_tool.description
            self._func: Callable[..., Any] = func_or_tool.func
            self._chat_context_param_names: list[str] = func_or_tool._chat_context_param_names
        elif inspect.isfunction(func_or_tool) or inspect.ismethod(func_or_tool):
            self._chat_context_param_names = get_context_params(func_or_tool, subclass=ChatContext)
            self._func = inject_params(func_or_tool)
            self._name = name or func_or_tool.__name__
            self._description = description or func_or_tool.__doc__ or ""
        else:
            raise ValueError(
                f"Parameter 'func_or_tool' must be a function, method or a Tool instance, it is '{type(func_or_tool)}' instead."
            )

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
        """Registers the tool for use with a ConversableAgent's language model (LLM).

        This method registers the tool so that it can be invoked by the agent during
        interactions with the language model.

        Args:
            agent (ConversableAgent): The agent to which the tool will be registered.
        """
        agent.register_for_llm()(self)

    def register_for_execution(self, agent: "ConversableAgent") -> None:
        """Registers the tool for direct execution by a ConversableAgent.

        This method registers the tool so that it can be executed by the agent,
        typically outside of the context of an LLM interaction.

        Args:
            agent (ConversableAgent): The agent to which the tool will be registered.
        """
        agent.register_for_execution()(self)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool by calling its underlying function with the provided arguments.

        Args:
            *args: Positional arguments to pass to the tool
            **kwargs: Keyword arguments to pass to the tool

        Returns:
            The result of executing the tool's function.
        """
        return self._func(*args, **kwargs)

    @property
    def tool_schema(self) -> dict[str, Any]:
        """Get the schema for the tool.

        This is the preferred way of handling function calls with OpeaAI and compatible frameworks.

        """
        return get_function_schema(self.func, name=self.name, description=self.description)

    @property
    def function_schema(self) -> dict[str, Any]:
        """Get the schema for the function.

        This is the old way of handling function calls with OpenAI and compatible frameworks.
        It is provided for backward compatibility.

        """
        schema = get_function_schema(self.func, name=self.name, description=self.description)
        return schema["function"]  # type: ignore[no-any-return]

    @property
    def realtime_tool_schema(self) -> dict[str, Any]:
        """Get the schema for the tool.

        This is the preferred way of handling function calls with OpeaAI and compatible frameworks.

        """
        schema = get_function_schema(self.func, name=self.name, description=self.description)
        schema = {"type": schema["type"], **schema["function"]}

        return schema
