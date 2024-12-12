# Copyright (c) 2023 - 2024, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict
from unittest.mock import MagicMock

from autogen.agentchat.conversable_agent import ConversableAgent

try:
    from crewai.tools import Tool as CrewAITool
except ImportError:
    CrewAITool = MagicMock()

__all__ = ["Tool"]


class Tool:
    def __init__(self, name: str, description: str, func: Callable[..., Any], kwargs: Dict[str, Any]):
        self._name = name
        self._description = description
        self._func = func
        self._kwargs = kwargs

    @classmethod
    def from_crewai_tool(cls, tool: CrewAITool) -> "Tool":
        name = tool.name.replace(" ", "_")
        description = tool.description.split("Tool Description: ")[-1]

        def func(args: tool.args_schema) -> Any:
            return tool.run(**args.model_dump())

        return Tool(
            name=name,
            description=description,
            func=func,
            kwargs={},
        )

    def register_for_llm(self, agent: ConversableAgent) -> None:
        agent.register_for_llm(name=self._name, description=self._description)(self._func)

    def register_for_execution(self, agent: ConversableAgent) -> None:
        agent.register_for_execution(name=self._name)(self._func)
