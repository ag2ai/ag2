# Copyright (c) 2023 - 2025, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Any, Callable, Optional

from autogen.agentchat.assistant_agent import ConversableAgent


@dataclass
class ToolSpecs:
    tool_func: Callable[..., Any]
    caller: ConversableAgent
    executor: ConversableAgent
    tool_description: str
    tool_name: Optional[str] = None


class ToolsCapability:
    """Adding a list of tools as composable capabilities to an agent."""

    def __init__(self, tool_list: list[ToolSpecs]):
        self.tools = [specs for specs in tool_list]

    def add_to_agent(self, agent: ConversableAgent):
        """
        Add tools to the given agent.
        """
        for specs in self.tools:
            agent.register_function(
                f=specs.tool_func,
                caller=specs.caller,
                executor=specs.executor,
                tool_description=specs.tool_description,
                tool_name=specs.tool_name,
            )
