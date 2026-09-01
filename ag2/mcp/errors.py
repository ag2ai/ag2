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


class MCPSamplingError(MCPServerError):
    """Base error for a served agent whose model is the calling client's."""


class MCPSamplingUnavailableError(MCPSamplingError):
    """Raised when the caller cannot lend the model this server was told to borrow.

    The turn fails rather than quietly answering some other way: a deployment
    configured to run on its caller's model has said it has no model of its own,
    and inventing one — or returning a degraded answer — would hide from the
    caller that the agent never reasoned at all. Pass ``ClientModel(fallback=
    True)`` to use the agent's own configured model instead.
    """

    def __init__(self) -> None:
        super().__init__(
            "This server runs the agent on the calling client's model, and this client advertised no "
            "sampling capability. Connect with sampling enabled, or ask the operator to configure a model "
            "for the agent to fall back to."
        )


class MCPSamplingRefusedError(MCPSamplingError):
    """Raised when a turn needs more of a model than a borrowed one can give.

    Tools and structured output both need something ``sampling/createMessage``
    does not carry here, and an agent that lost either without being told would
    answer as though it had never had them.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(f"Cannot run this turn on the calling MCP client's model: {reason}.")
