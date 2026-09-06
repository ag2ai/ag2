# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Webz.io contextual news search Extension for AG2.

Wraps the hosted News Search MCP server so agents can search global news
in natural language. Filter schemas come from MCP ``tools/list`` at runtime
and are not hardcoded in this module.

Maintainer: Webz.io (ShakedDegani)
Docs: https://docs.ag2.ai/docs/user-guide/extensions/tools/search/webzio/
"""

from __future__ import annotations

import os
from collections.abc import Iterable

from ag2.middleware import ToolMiddleware
from ag2.tools import MCPServerConfig, MCPToolkit

DEFAULT_MCP_URL = "https://news-search-mcp.webz.io/mcp"
TOKEN_ENV_NAME = "WEBZ_API_TOKEN"
MCP_URL_ENV_NAME = "WEBZ_MCP_URL"
PREFERRED_TOOL_NAME = "news_search_by_webz"
SERVER_LABEL = "webzio-news-search"


class WebzioConfigError(ValueError):
    """Raised when the Webzio MCP client cannot be configured."""


def resolve_api_token(api_token: str | None = None) -> str:
    """Return a Webz API token from the argument or ``WEBZ_API_TOKEN``.

    Args:
        api_token: Explicit token. When omitted, ``WEBZ_API_TOKEN`` is used.

    Returns:
        The stripped API token.

    Raises:
        WebzioConfigError: If neither the argument nor the environment is set.
    """
    token = (api_token or os.getenv(TOKEN_ENV_NAME) or "").strip()
    if not token:
        raise WebzioConfigError(f"missing Webz API token. set {TOKEN_ENV_NAME} or pass api_token.")
    return token


def resolve_mcp_url(mcp_url: str | None = None) -> str:
    """Return the MCP endpoint from the argument, env, or the production default.

    Args:
        mcp_url: Explicit MCP URL. When omitted, ``WEBZ_MCP_URL`` or the
            hosted production endpoint is used.

    Returns:
        The MCP URL without a trailing slash.

    Raises:
        WebzioConfigError: If the resolved value is empty.
    """
    url = (mcp_url or os.getenv(MCP_URL_ENV_NAME) or DEFAULT_MCP_URL).strip()
    if not url:
        raise WebzioConfigError("missing MCP url.")
    return url.rstrip("/")


def build_mcp_server_config(
    api_token: str | None = None,
    *,
    mcp_url: str | None = None,
) -> MCPServerConfig:
    """Build the MCP client config for the hosted Webz News Search server.

    Args:
        api_token: Webz API token. Defaults to ``WEBZ_API_TOKEN``.
        mcp_url: MCP endpoint. Defaults to the hosted production URL.

    Returns:
        An ``MCPServerConfig`` that authenticates with a Bearer token and
        registers ``news_search_by_webz``.
    """
    return MCPServerConfig(
        server_url=resolve_mcp_url(mcp_url),
        authorization_token=resolve_api_token(api_token),
        allowed_tools=[PREFERRED_TOOL_NAME],
        server_label=SERVER_LABEL,
    )


class WebzioNewsSearchToolkit(MCPToolkit):
    """Toolkit that exposes Webz.io contextual news search to AG2 agents.

    Connects to the hosted News Search MCP server and registers
    ``news_search_by_webz`` with its live input schema from ``tools/list``.
    Filter fields are not hardcoded — new server filters appear at runtime.

    Pass the toolkit to an agent to register news search::

        toolkit = WebzioNewsSearchToolkit(api_token=...)
        agent = Agent("researcher", config=config, tools=[toolkit])

    Call ``search()`` when you want to register only the news search tool::

        agent = Agent("researcher", config=config, tools=[toolkit.search()])

    Reads ``WEBZ_API_TOKEN`` from the environment when ``api_token`` is omitted.
    """

    def __init__(
        self,
        api_token: str | None = None,
        *,
        mcp_url: str | None = None,
        middleware: Iterable[ToolMiddleware] = (),
    ) -> None:
        """Connect to the hosted Webz News Search MCP server.

        Args:
            api_token: Webz API token. Defaults to ``WEBZ_API_TOKEN``.
            mcp_url: MCP endpoint override. Defaults to the hosted production URL.
            middleware: Middleware applied to every tool in the toolkit.
        """
        super().__init__(
            build_mcp_server_config(api_token, mcp_url=mcp_url),
            middleware=middleware,
        )

    def search(self) -> WebzioNewsSearchToolkit:
        """Return this toolkit, registering only the news search tool.

        Returns:
            This toolkit instance.
        """
        return self
