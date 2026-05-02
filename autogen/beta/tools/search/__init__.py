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

try:
    from .exa import ExaToolkit
except ImportError as e:
    ExaToolkit = missing_optional_dependency("ExaToolkit", "exa", e)  # type: ignore[misc]

try:
    from .perplexity import PerplexitySearchTool
except ImportError as e:
    PerplexitySearchTool = missing_optional_dependency("PerplexitySearchTool", "perplexity", e)  # type: ignore[misc]

try:
    from .perplexity_search_api import PerplexitySearchAPITool
except ImportError as e:
    PerplexitySearchAPITool = missing_optional_dependency("PerplexitySearchAPITool", "perplexity", e)  # type: ignore[misc]

__all__ = (
    "DuckDuckSearchTool",
    "ExaToolkit",
    "PerplexitySearchAPITool",
    "PerplexitySearchTool",
    "TavilySearchTool",
)
