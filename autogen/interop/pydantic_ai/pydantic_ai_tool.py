# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable

from ...agentchat.conversable_agent import ConversableAgent
from ...tools import Tool

__all__ = ["PydanticAITool"]


class PydanticAITool(Tool):
    """A class representing a Pydantic AI Tool that extends the general Tool functionality
    with additional functionality specific to Pydantic AI tools.

    This class inherits from the Tool class and adds functionality for registering
    tools with a ConversableAgent, along with providing additional schema information
    specific to Pydantic AI tools, such as parameters and function signatures.

    Attributes:
        parameters_json_schema (Dict[str, Any]): A schema describing the parameters
                                                 that the tool's function expects.
    """

    def __init__(
        self, name: str, description: str, func: Callable[..., Any], parameters_json_schema: dict[str, Any]
    ) -> None:
        """Initializes a PydanticAITool object with the provided name, description,
        function, and parameter schema.

        Args:
            name (str): The name of the tool.
            description (str): A description of what the tool does.
            func (Callable[..., Any]): The function that is executed when the tool is called.
            parameters_json_schema (Dict[str, Any]): A schema describing the parameters
                                                     that the function accepts.
        """
        super().__init__(name=name, description=description, func_or_tool=func)
        self._func_schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters_json_schema,
            },
        }

    def register_for_llm(self, agent: ConversableAgent) -> None:
        """Registers the tool with the ConversableAgent for use with a language model (LLM).

        This method updates the agent's tool signature to include the function schema,
        allowing the agent to invoke the tool correctly during interactions with the LLM.

        Args:
            agent (ConversableAgent): The agent with which the tool will be registered.
        """
        agent.update_tool_signature(self._func_schema, is_remove=False)
