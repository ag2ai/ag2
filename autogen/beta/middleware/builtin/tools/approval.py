# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.annotations import Context
from autogen.beta.events import ToolCallEvent, ToolResultEvent
from autogen.beta.middleware.base import ToolExecution, ToolMiddleware, ToolResultType

_DEFAULT_MESSAGE = (
    "Agent wants to call the tool:\n`{tool_name}`, {tool_arguments}\nPlease approve or deny this request.\nY/N?\n"
)
_DEFAULT_MESSAGE_ALWAYS = "Agent wants to call the tool:\n`{tool_name}`, {tool_arguments}\nPlease approve or deny this request.\nY/N/Always?\n"
BYPASS_KEY = "approval_required:always"


def approval_required(
    message: str | None = None,
    denied_message: str = "User denied the tool call request",
    *,
    timeout: int = 30,
    allow_always: bool = True,
) -> ToolMiddleware:
    """Tool middleware that requests human approval before executing a tool call.

    Args:
        message: Prompt template shown to the user. Supports ``{tool_name}`` and
            ``{tool_arguments}`` placeholders. Defaults to a built-in prompt
            that includes "Always" when *allow_always* is enabled.
        denied_message: Message shown to the LLM after the tool call is denied.
        timeout: Seconds to wait for user input before timing out.
        allow_always: When ``True``, the user can respond with ``always`` to
            approve the current and all subsequent calls of the same tool in the
            same context.

    Returns:
        A tool middleware hook that can be passed to the ``middleware``
        parameter of :func:`~autogen.beta.tool`.
    """

    prompt = message if message is not None else (_DEFAULT_MESSAGE_ALWAYS if allow_always else _DEFAULT_MESSAGE)

    async def hitl_hook(
        call_next: ToolExecution,
        event: ToolCallEvent,
        context: Context,
    ) -> ToolResultType:
        if allow_always:
            bypass_dict = context.variables.get(BYPASS_KEY, {})
            if bypass_dict.get(event.name):
                return await call_next(event, context)

        user_result = (
            await context.input(
                prompt.format(tool_name=event.name, tool_arguments=event.arguments),
                timeout=timeout,
            )
        ).lower()

        if allow_always and user_result == "always":
            bypass_dict = context.variables.get(BYPASS_KEY, {})
            bypass_dict[event.name] = True
            context.variables[BYPASS_KEY] = bypass_dict
            return await call_next(event, context)

        elif user_result in ("y", "yes", "1"):
            return await call_next(event, context)

        return ToolResultEvent.from_call(event, result=denied_message)

    return hitl_hook
