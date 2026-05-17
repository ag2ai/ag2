# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.exceptions import missing_optional_dependency

from .filesystem import FilesystemToolkit
from .mcp_server import MCPServer, MCPServerConfig, MCPStdioServerConfig

try:
    from .crawl4ai import Crawl4AIToolkit
except ImportError as e:
    Crawl4AIToolkit = missing_optional_dependency("Crawl4AIToolkit", "crawl4ai", e)  # type: ignore[misc]

__all__ = (
    "Crawl4AIToolkit",
    "FilesystemToolkit",
    "MCPServer",
    "MCPServerConfig",
    "MCPStdioServerConfig",
)
