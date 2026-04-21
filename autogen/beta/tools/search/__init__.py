# Copyright (c) 2026, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0

from autogen.beta.exceptions import missing_optional_dependency

try:
    from .tavily import TavilySearchTool
except ImportError as e:
    TavilySearchTool = missing_optional_dependency("TavilySearchTool", "tavily", e)  # type: ignore[misc]

try:
    from .duckduckgo import DuckDuckSearchTool
except ImportError as e:
    DuckDuckSearchTool = missing_optional_dependency("DuckDuckSearchTool", "ddgs", e)  # type: ignore[misc]

__all__ = (
    "DuckDuckSearchTool",
    "TavilySearchTool",
)
