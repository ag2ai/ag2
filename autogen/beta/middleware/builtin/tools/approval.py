# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.annotations import Context
from autogen.beta.events import ToolCallEvent, ToolResultEvent
from autogen.beta.middleware.base import ToolExecution, ToolMiddleware, ToolResultType


def approval_required(
    message: str = "Agent wants to call the tool:\n`{tool_name}`, {tool_arguments}\nPlease approve or deny this request.\nY/N/Always?\n",
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

    The user can respond with:
    - ``y`` / ``yes`` / ``1`` to approve the current tool call.
    - ``always`` to approve the current and all subsequent tool calls in the
      same context.
    - Any other input to deny the tool call.
    """

    bypass_key = "approval_required:always"

    async def hitl_hook(
        call_next: ToolExecution,
        event: ToolCallEvent,
        context: Context,
    ) -> ToolResultType:
        bypass_dict = context.variables.get(bypass_key, {})
        if bypass_dict.get(event.name):
            return await call_next(event, context)

        user_result = (
            await context.input(
                message.format(tool_name=event.name, tool_arguments=event.arguments),
                timeout=timeout,
            )
        ).lower()

        if user_result == "always":
            bypass_dict[event.name] = True
            context.variables[bypass_key] = bypass_dict
            return await call_next(event, context)

        elif user_result in ("y", "yes", "1"):
            return await call_next(event, context)

        return ToolResultEvent.from_call(event, result=denied_message)

    return hitl_hook
