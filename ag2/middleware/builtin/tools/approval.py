# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.annotations import Context
from ag2.events import ToolCallEvent, ToolResultEvent
from ag2.middleware.base import ToolExecution, ToolMiddleware, ToolResultType
from ag2.middleware.describe import MiddlewareDescription

_DEFAULT_MESSAGE = (
    "Agent wants to call the tool:\n`{tool_name}`, {tool_arguments}\nPlease approve or deny this request.\nY/N?\n"
)
_DEFAULT_MESSAGE_ALWAYS = "Agent wants to call the tool:\n`{tool_name}`, {tool_arguments}\nPlease approve or deny this request.\nY/N/Always?\n"
BYPASS_KEY = "ag:approval_required:always"


class ApprovalRequired:
    """Tool middleware that requests human approval before executing a tool call.

    Callable, so it satisfies :data:`~ag2.middleware.ToolMiddleware` wherever a
    hook is accepted. Approval state lives in ``context.variables``, not here.
    """

    def __init__(
        self,
        message: str | None = None,
        denied_message: str = "User denied the tool call request",
        *,
        timeout: int = 30,
        allow_always: bool = True,
    ) -> None:
        self._prompt = (
            message if message is not None else (_DEFAULT_MESSAGE_ALWAYS if allow_always else _DEFAULT_MESSAGE)
        )
        self._denied_message = denied_message
        self._timeout = timeout
        self._allow_always = allow_always

    def describe(self) -> MiddlewareDescription:
        return MiddlewareDescription(
            kind=type(self).__qualname__,
            config={
                "message": self._prompt,
                "denied_message": self._denied_message,
                "timeout": self._timeout,
                "allow_always": self._allow_always,
            },
        )

    async def __call__(
        self,
        call_next: ToolExecution,
        event: ToolCallEvent,
        context: Context,
    ) -> ToolResultType:
        if self._allow_always:
            bypass_dict = context.variables.get(BYPASS_KEY, {})
            if bypass_dict.get(event.name):
                return await call_next(event, context)

        user_result = (
            await context.input(
                self._prompt.format(tool_name=event.name, tool_arguments=event.arguments),
                timeout=self._timeout,
            )
        ).lower()

        if self._allow_always and user_result == "always":
            bypass_dict = context.variables.get(BYPASS_KEY, {})
            bypass_dict[event.name] = True
            context.variables[BYPASS_KEY] = bypass_dict
            return await call_next(event, context)

        elif user_result in ("y", "yes", "1"):
            return await call_next(event, context)

        return ToolResultEvent.from_call(event, result=self._denied_message)


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
        An :class:`ApprovalRequired` hook that can be passed to the
        ``middleware`` parameter of :func:`~ag2.tool`.
    """

    return ApprovalRequired(
        message,
        denied_message,
        timeout=timeout,
        allow_always=allow_always,
    )
