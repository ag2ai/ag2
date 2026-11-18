# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Privatus Analytics toolkit for AG2.

Privatus Analytics is cookieless website analytics. Its hosted MCP server lets
an agent read website stats (visitors, top pages, referrers, campaigns, Core
Web Vitals) and manage goals, alerts and uptime checks. This toolkit is a thin
preset over :class:`ag2.tools.MCPToolkit` that points at that server and sends
the API token as a bearer token.

Maintainer: ja-andrews
Docs: https://docs.ag2.ai/docs/user-guide/extensions/privatus/
"""

import os
from collections.abc import Iterable

from ag2.annotations import Variable
from ag2.middleware import ToolMiddleware
from ag2.tools.toolkits.mcp_server import MCPAnswerPolicy, MCPServerConfig, MCPToolkit

PRIVATUS_MCP_URL = "https://privatusanalytics.com/mcp"
"""The hosted Privatus Analytics MCP endpoint."""

_API_TOKEN_ENV_VAR = "PRIVATUS_API_TOKEN"  # pragma: allowlist secret

_DEFAULT_LABEL = "privatus_analytics"

_DEFAULT_DESCRIPTION = (
    "Privatus Analytics website stats: visitors, pages, referrers, campaigns, goals, "
    "funnels, Core Web Vitals, uptime checks and alerts."
)


class PrivatusAnalyticsToolkit(MCPToolkit):
    """Expose the Privatus Analytics MCP server's tools to an agent.

    The server lists only the tools the token's permissions allow, so a token
    with just ``analytics.read`` gives a read-only agent. Hard-to-undo actions
    (deleting data, billing, access settings) are never run by the server; it
    returns a link for a person to finish them instead.

    Example::

        from ag2 import Agent
        from ag2.extensions.privatus import PrivatusAnalyticsToolkit

        agent = Agent(
            "analyst",
            config=config,
            tools=[PrivatusAnalyticsToolkit(allowed_tools=["sites_list", "stats_aggregate"])],
        )

    Args:
        api_token: A Privatus Analytics API token (it starts with ``pat_``),
            sent as ``Authorization: Bearer <token>``. Falls back to the
            ``PRIVATUS_API_TOKEN`` environment variable, read when the toolkit
            is built. A :class:`~ag2.annotations.Variable` is resolved from the
            agent context on each discovery instead.
        server_url: The MCP endpoint. Change it only to reach a staging server.
        allowed_tools: Server tool names to expose. With none set, every tool
            the token may use is exposed.
        blocked_tools: Server tool names to hide, applied after ``allowed_tools``.
        tool_name_prefix: Prefix for the agent-visible tool names.
        connection_timeout: Seconds to wait on the server before giving up.
        middleware: Tool middleware applied to every call made through this toolkit.
        answering: Which of the operator's own resources the server may use.

    Raises:
        ValueError: If no token is passed and ``PRIVATUS_API_TOKEN`` is unset.
    """

    __slots__ = ()

    def __init__(
        self,
        api_token: str | Variable | None = None,
        *,
        server_url: str = PRIVATUS_MCP_URL,
        allowed_tools: list[str] | Variable | None = None,
        blocked_tools: list[str] | Variable | None = None,
        tool_name_prefix: str | Variable = "",
        connection_timeout: float = 30.0,
        middleware: Iterable[ToolMiddleware] = (),
        answering: MCPAnswerPolicy | None = None,
    ) -> None:
        token = api_token if api_token is not None else os.environ.get(_API_TOKEN_ENV_VAR)
        if not token:
            raise ValueError(
                "PrivatusAnalyticsToolkit needs an API token. Pass api_token=... or set "
                f"{_API_TOKEN_ENV_VAR}. Create one under Workspace settings, API tokens "
                "at https://privatusanalytics.com."
            )

        super().__init__(
            MCPServerConfig(
                server_url=server_url,
                authorization_token=token,
                server_label=_DEFAULT_LABEL,
                description=_DEFAULT_DESCRIPTION,
                allowed_tools=allowed_tools,
                blocked_tools=blocked_tools,
                tool_name_prefix=tool_name_prefix,
                connection_timeout=connection_timeout,
            ),
            middleware=middleware,
            answering=answering,
        )
