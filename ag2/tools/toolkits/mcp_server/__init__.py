# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from ag2.exceptions import missing_optional_dependency

from .types import MCPServerConfig, MCPStdioServerConfig

try:
    from .answering import MCPAnswerPolicy
    from .toolkit import MCPToolkit
except ImportError as e:
    MCPAnswerPolicy = missing_optional_dependency("MCPAnswerPolicy", "mcp", e)  # type: ignore[misc]
    MCPToolkit = missing_optional_dependency("MCPToolkit", "mcp", e)  # type: ignore[misc]


__all__ = (
    "MCPAnswerPolicy",
    "MCPServerConfig",
    "MCPStdioServerConfig",
    "MCPToolkit",
)
