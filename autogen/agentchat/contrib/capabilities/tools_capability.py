# Copyright (c) 2023 - 2025, Owners of https://github.com/ag2ai
#
# SPDX-License-Identifier: Apache-2.0

from autogen.agentchat import ConversableAgent
from autogen.tools import Tool


class ToolsCapability:
    """Adding a list of tools as composable capabilities to a single agent.
    Note: both caller and executor of the tools are the same agent.
    """

    def __init__(self, tool_list: list[Tool]):
        self.tools = [tool for tool in tool_list]

    def add_to_agent(self, agent: ConversableAgent):
        """
        Add tools to the given agent.
        """
        for tool in self.tools:
            tool.register_tool(agent=agent)
