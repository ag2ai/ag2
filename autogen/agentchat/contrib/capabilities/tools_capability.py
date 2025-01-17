# Copyright (c) 2023 - 2025, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Any, Callable, Optional

from autogen.agentchat import ConversableAgent, register_function


@dataclass
class ToolSpecs:
    tool_func: Callable[..., Any]
    tool_description: str
    tool_name: Optional[str] = None


class ToolsCapability:
    """Adding a list of tools as composable capabilities to a single agent.
    Note: both caller and executor of the tools are the same agent.
    """

    def __init__(self, tool_list: list[ToolSpecs]):
        self.tools = [specs for specs in tool_list]

    def add_to_agent(self, agent: ConversableAgent):
        """
        Add tools to the given agent.
        """
        for specs in self.tools:
            register_function(
                f=specs.tool_func,
                caller=agent,
                executor=agent,
                description=specs.tool_description,
                name=specs.tool_name,
            )
