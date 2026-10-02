# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from mcp.shared.exceptions import MCPError
from mcp.types import (
    INVALID_PARAMS,
    MISSING_REQUIRED_CLIENT_CAPABILITY,
    ClientCapabilities,
    MissingRequiredClientCapabilityErrorData,
    SamplingCapability,
)

from ag2.exceptions import AG2Error, HumanInputNotProvidedError


class MCPServerError(AG2Error):
    """Base error for the ``ag2.mcp`` server wrapper."""


class MCPAgentConfigError(MCPServerError):
    """Raised when an agent without a model config is served over MCP."""

    def __init__(self, agent_name: str) -> None:
        super().__init__(
            f"Agent {agent_name!r} has no model config. Give it one with `Agent(config=...)`, or run it on "
            "the calling client's model with `MCPServer(..., client_model=True)`."
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


class MCPResourceNotFoundError(MCPServerError, MCPError):
    """Raised when a ``resources/read`` targets an unknown URI; reaches the client as ``-32602``."""

    def __init__(self, uri: str) -> None:
        MCPError.__init__(self, code=INVALID_PARAMS, message=f"No resource matches URI {uri!r}.", data={"uri": uri})


class MCPPromptNotFoundError(MCPServerError, MCPError):
    """Raised when a ``prompts/get`` targets an unknown prompt name; reaches the client as ``-32602``."""

    def __init__(self, name: str) -> None:
        MCPError.__init__(self, code=INVALID_PARAMS, message=f"No prompt named {name!r}.")


class UnknownConversationError(MCPServerError):
    """Raised when a presented conversation handle names no live conversation.

    Reported as a *tool execution* error, so the model can recover by starting a
    new conversation instead of failing the turn. A handle created by a
    different principal raises this too.
    """

    def __init__(self) -> None:
        super().__init__(
            "Unknown or expired conversation handle. Omit the 'conversation' argument to start a new conversation."
        )


class MCPElicitationDeclinedError(HumanInputNotProvidedError):
    """Raised when the calling MCP client refused a served agent's question.

    A :class:`HumanInputNotProvidedError` subclass, distinct only so a host can
    tell a refusal from an absent channel.
    """

    def __init__(self, action: str) -> None:
        super().__init__(
            f"The calling MCP client answered the agent's question with {action!r}, "
            "so there is no answer to continue from."
        )
        self.action = action


class MCPSamplingError(MCPServerError):
    """Base error for a served agent whose model is the calling client's."""


class MCPSamplingUnavailableError(MCPSamplingError, MCPError):
    """Raised when the caller cannot lend the model this server was told to borrow.

    Reaches the client as ``-32021`` naming the ``sampling`` capability. An agent
    that has a ``config`` of its own falls back to it instead.
    """

    def __init__(self) -> None:
        data = MissingRequiredClientCapabilityErrorData(
            required_capabilities=ClientCapabilities(sampling=SamplingCapability())
        )
        MCPError.__init__(
            self,
            code=MISSING_REQUIRED_CLIENT_CAPABILITY,
            message=(
                "This server runs the agent on the calling client's model, and this client advertised no "
                "sampling capability. Connect with sampling enabled, or ask the operator to configure a model "
                "for the agent to fall back to."
            ),
            data=data.model_dump(by_alias=True, mode="json", exclude_none=True),
        )


class MCPSamplingRefusedError(MCPSamplingError):
    """Raised when a turn needs more of a model than a borrowed one can give."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Cannot run this turn on the calling MCP client's model: {reason}.")


class MCPAppURIError(MCPServerError):
    """Raised when an app's document URI does not use the ``ui://`` scheme."""

    def __init__(self, uri: str) -> None:
        super().__init__(f"An app document URI must use the ui:// scheme, got {uri!r}; a host discards anything else.")


class MCPDuplicateAppURIError(MCPServerError):
    """Raised when two apps registered on one server claim the same document URI."""

    def __init__(self, uri: str) -> None:
        super().__init__(
            f"Duplicate app document URI {uri!r}; one document would shadow the other and the losing "
            "app's tools would point at a body nobody wrote."
        )


class MCPAppFrozenError(MCPServerError):
    """Raised when a tool is declared on an app already registered with a server.

    Registration is what reads an app's tool composition, so a tool added after
    it would exist and never be served. Failing here names the tool rather than
    leaving a silently absent one to be found on the wire.
    """

    def __init__(self, uri: str, name: str | None = None) -> None:
        tool = f"Tool {name!r} cannot" if name else "A tool cannot"
        super().__init__(
            f"{tool} be declared on app {uri!r}: it is already registered with an MCPServer, which has "
            "read its tools. Declare every tool before constructing the server."
        )
