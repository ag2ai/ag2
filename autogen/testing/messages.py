# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from pydantic_core import to_json

from autogen.events.agent_events import FunctionCall
from autogen.events.agent_events import ToolCall as RawToolCall


def tools_message(*tool_calls: dict[str, Any]) -> dict[str, Any]:
    return {"content": None, "tool_calls": list(tool_calls)}


def ToolCall(  # noqa: N802
    tool_name: str,
    /,
    **arguments: Any,
) -> dict[str, Any]:
    return RawToolCall(
        type="function", function=FunctionCall(name=tool_name, arguments=to_json(arguments).decode())
    ).model_dump()
