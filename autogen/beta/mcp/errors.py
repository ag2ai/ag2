# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.exceptions import AG2Error


class MCPServerError(AG2Error):
    """Base error for the ``autogen.beta.mcp`` server wrapper."""


class MCPAgentConfigError(MCPServerError):
    """Raised when an agent without a model config is served over MCP."""

    def __init__(self, agent_name: str) -> None:
        super().__init__(
            f"Agent {agent_name!r} has no model config; set `Agent(config=...)` before serving it over MCP."
        )
