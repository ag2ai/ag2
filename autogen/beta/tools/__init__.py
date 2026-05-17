# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.events import ToolResult

from .builtin import (
    CodeExecutionTool,
    ContainerAutoEnvironment,
    ContainerReferenceEnvironment,
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
)
from .code import SandboxCodeTool
from .final import Toolkit, tool
from .search import DuckDuckSearchTool, ExaToolkit, PerplexitySearchToolkit, TavilySearchTool
from .shell import LocalShellTool
from .skills import SkillSearchToolkit, SkillsToolkit
from .toolkits import FilesystemToolkit, MCPServer, MCPServerConfig, MCPStdioServerConfig

__all__ = (
    "CodeExecutionTool",
    "ContainerAutoEnvironment",
    "ContainerReferenceEnvironment",
    "DuckDuckSearchTool",
    "ExaToolkit",
    "FilesystemToolkit",
    "ImageGenerationTool",
    "LocalShellTool",
    "MCPServer",
    "MCPServerConfig",
    "MCPServerTool",
    "MCPStdioServerConfig",
    "MemoryTool",
    "NetworkPolicy",
    "PerplexitySearchToolkit",
    "SandboxCodeTool",
    "ShellTool",
    "Skill",
    "SkillSearchToolkit",
    "SkillsPlugin",  # noqa: F822 — lazy-loaded via __getattr__ below
    "SkillsTool",
    "SkillsToolkit",
    "TavilySearchTool",
    "ToolResult",
    "Toolkit",
    "UserLocation",
    "WebFetchTool",
    "WebSearchTool",
    "tool",
)


def __getattr__(name: str) -> object:
    if name == "SkillsPlugin":
        from .skills import SkillsPlugin

        return SkillsPlugin
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
