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


class MCPResourceNotFoundError(MCPServerError):
    """Raised when a ``resources/read`` targets an unknown URI."""

    def __init__(self, uri: str) -> None:
        super().__init__(f"No resource matches URI {uri!r}.")


class MCPPromptNotFoundError(MCPServerError):
    """Raised when a ``prompts/get`` targets an unknown prompt name."""

    def __init__(self, name: str) -> None:
        super().__init__(f"No prompt named {name!r}.")
