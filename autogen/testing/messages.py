# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from uuid import uuid4

from pydantic_core import to_json

from autogen.events.agent_events import FunctionCall
from autogen.events.agent_events import ToolCall as RawToolCall


class ToolCall:
    """Represents a tool call with a specified tool name and arguments.

    Args:
        tool_name: Tool name to call. Tool should be rigestered in Agent you send message.
        arguments: keyword arguments to pass to the tool.
    """

    def __init__(self, tool_name: str, /, **arguments: Any) -> None:
        self.tool_message = RawToolCall(
            id=f"call_{uuid4()}",
            type="function",
            function=FunctionCall(name=tool_name, arguments=to_json(arguments).decode()),
        ).model_dump()

    def to_message(self) -> dict[str, Any]:
        """Convert the tool call to a message format suitable for API calls.

        Returns:
            A dictionary containing the tool call in message format,
            ready to be used in API requests or message queues.
        """
        return tools_message(self)


def tools_message(*tool_calls: ToolCall) -> dict[str, Any]:
    """Convert multiple tool calls into a message format suitable for API calls.

    Args:
        *tool_calls: One or more ToolCall objects to convert.
    """
    return {"content": None, "tool_calls": [c.tool_message for c in tool_calls]}
