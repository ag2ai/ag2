# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from ag2._lazy import LazyExport, lazy_exports
from ag2.events import ToolResult

from .builtin import (
    AnthropicBashTool,
    CodeExecutionTool,
    ContainerAutoEnvironment,
    ContainerReferenceEnvironment,
    FileSearchTool,
    GoogleMapsTool,
    ImageGenerationTool,
    MCPServerTool,
    MemoryTool,
    NetworkPolicy,
    ShellTool,
    Skill,
    SkillsTool,
    UserLocation,
    WebFetchTool,
    WebSearchTool,
    XSearchTool,
)
from .code import SandboxCodeTool
from .final import Toolkit, tool
from .sandbox import LocalEnvironment
from .shell import SandboxShellTool
from .skills import MemorySkill, SkillPlugin, SkillSearchToolkit, SkillsToolkit

if TYPE_CHECKING:
    from .search import DuckDuckSearchTool, PerplexitySearchToolkit, TavilySearchTool
    from .toolkits import FilesystemToolkit, MCPAnswerPolicy, MCPServerConfig, MCPStdioServerConfig, MCPToolkit

# Search backends and the MCP client load on first use; see docs/adr/0036-provider-exports-load-on-first-use.md.
_SEARCH = LazyExport(".search")
_TOOLKITS = LazyExport(".toolkits")
_LAZY_EXPORTS = {
    "DuckDuckSearchTool": _SEARCH,
    "FilesystemToolkit": _TOOLKITS,
    "MCPAnswerPolicy": _TOOLKITS,
    "MCPServerConfig": _TOOLKITS,
    "MCPStdioServerConfig": _TOOLKITS,
    "MCPToolkit": _TOOLKITS,
    "PerplexitySearchToolkit": _SEARCH,
    "TavilySearchTool": _SEARCH,
}

__getattr__ = lazy_exports(__name__, globals(), _LAZY_EXPORTS)

__all__ = (
    "AnthropicBashTool",
    "CodeExecutionTool",
    "ContainerAutoEnvironment",
    "ContainerReferenceEnvironment",
    "DuckDuckSearchTool",
    "FileSearchTool",
    "FilesystemToolkit",
    "GoogleMapsTool",
    "ImageGenerationTool",
    "LocalEnvironment",
    "MCPAnswerPolicy",
    "MCPServerConfig",
    "MCPServerTool",
    "MCPStdioServerConfig",
    "MCPToolkit",
    "MemorySkill",
    "MemoryTool",
    "NetworkPolicy",
    "PerplexitySearchToolkit",
    "SandboxCodeTool",
    "SandboxShellTool",
    "ShellTool",
    "Skill",
    "SkillPlugin",
    "SkillSearchToolkit",
    "SkillsTool",
    "SkillsToolkit",
    "TavilySearchTool",
    "ToolResult",
    "Toolkit",
    "UserLocation",
    "WebFetchTool",
    "WebSearchTool",
    "XSearchTool",
    "tool",
)
