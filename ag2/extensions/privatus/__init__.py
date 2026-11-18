# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Privatus Analytics extension for AG2.

Gives agents the tools of the hosted Privatus Analytics MCP server, so they can
read website stats and manage goals, alerts and uptime checks.

Maintainer: ja-andrews
Docs: https://docs.ag2.ai/docs/user-guide/extensions/privatus/
"""

from ag2.exceptions import missing_optional_dependency

try:
    from .toolkit import PRIVATUS_MCP_URL, PrivatusAnalyticsToolkit
except ImportError as e:
    PRIVATUS_MCP_URL = "https://privatusanalytics.com/mcp"
    PrivatusAnalyticsToolkit = missing_optional_dependency("PrivatusAnalyticsToolkit", "mcp", e)  # type: ignore[misc]

__all__ = (
    "PRIVATUS_MCP_URL",
    "PrivatusAnalyticsToolkit",
)
