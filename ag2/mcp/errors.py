# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.exceptions import AG2Error, HumanInputNotProvidedError


class MCPServerError(AG2Error):
    """Base error for the ``ag2.mcp`` server wrapper."""


class MCPAgentConfigError(MCPServerError):
    """Raised when an agent without a model config is served over MCP."""

    def __init__(self, agent_name: str) -> None:
        super().__init__(
            f"Agent {agent_name!r} has no model config; set `Agent(config=...)` before serving it over MCP."
        )


class MCPToolNameConflictError(MCPServerError):
    """Raised when a custom tool's name collides with the agent's ``ask`` tool or another custom tool."""

    def __init__(self, name: str, *, reserved: bool = True) -> None:
        if reserved:
            message = (
                f"Custom tool {name!r} conflicts with the agent's conversational tool; "
                "rename the tool or pass a different `tool_name=` to MCPServer."
            )
        else:
            message = f"Duplicate custom tool name {name!r}; tool names must be unique."
        super().__init__(message)


class MCPResourceNotFoundError(MCPServerError):
    """Raised when a ``resources/read`` targets an unknown URI."""

    def __init__(self, uri: str) -> None:
        super().__init__(f"No resource matches URI {uri!r}.")


class MCPPromptNotFoundError(MCPServerError):
    """Raised when a ``prompts/get`` targets an unknown prompt name."""

    def __init__(self, name: str) -> None:
        super().__init__(f"No prompt named {name!r}.")


class UnknownConversationError(MCPServerError):
    """Raised when a presented conversation handle names no live conversation.

    Reported to the caller as a *tool execution* error rather than a JSON-RPC
    one: the protocol draws that line so the model can recover by starting a new
    conversation instead of failing the turn. A handle created by a different
    principal raises this too, so the error does not disclose that it exists.
    """

    def __init__(self) -> None:
        super().__init__(
            "Unknown or expired conversation handle. Omit the 'conversation' argument to start a new conversation."
        )


class MCPElicitationDeclinedError(HumanInputNotProvidedError):
    """Raised when the calling MCP client refused a served agent's question.

    A subclass of the existing "requested but not provided" failure rather than a
    new type: the channel worked and the question was put, but no answer came
    back, which is the same outcome the human-input model already names. Kept
    distinct only so a host that wants to tell a refusal from an absent channel
    still can.
    """

    def __init__(self, action: str) -> None:
        super().__init__(
            f"The calling MCP client answered the agent's question with {action!r}, "
            "so there is no answer to continue from."
        )
        self.action = action
