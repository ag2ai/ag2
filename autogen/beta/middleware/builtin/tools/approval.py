# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.annotations import Context
from autogen.beta.events import ToolCallEvent, ToolResultEvent
from autogen.beta.middleware.base import ToolExecution, ToolMiddleware, ToolResultType


def approval_required(
    message: str = "Agent wants to call the tool:\n`{tool_name}`, {tool_arguments}\nPlease approve or deny this request.\nY/N?\n",
    denied_message: str = "User denied the tool call request",
    timeout: int = 30,
) -> ToolMiddleware:
    """Tool middleware that requests human approval before executing a tool call.

    Args:
        message: Prompt template shown to the user. Supports ``{tool_name}`` and
            ``{tool_arguments}`` placeholders.
        denied_message: Message shown to the LLM after the tool call is denied.
        timeout: Seconds to wait for user input before timing out.

    Returns:
        A tool middleware hook that can be passed to the ``middleware``
        parameter of :func:`~autogen.beta.tool`.
    """

    async def hitl_hook(
        call_next: ToolExecution,
        event: ToolCallEvent,
        context: Context,
    ) -> ToolResultType:
        user_result = await context.input(
            message.format(tool_name=event.name, tool_arguments=event.arguments),
            timeout=timeout,
        )

        if user_result.lower() in ("y", "yes", "1"):
            return await call_next(event, context)

        return ToolResultEvent.from_call(event, result=denied_message)

    return hitl_hook
