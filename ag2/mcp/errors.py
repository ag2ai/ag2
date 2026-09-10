# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.exceptions import AG2Error


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
